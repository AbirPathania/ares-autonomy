"""The node that makes the speed limit binding rather than advisory.

Build spec section 09 says to wire confidence to speed from the start. It is
easy to read that as "publish a speed limit" and stop there, and that is what
I did first: the traversability node published /speed_limit, nothing
subscribed to it, and the rover drove through the governing trench at full
speed while a topic next to it said 0.18 km/h.

A limit nothing applies is not a safety mechanism, it is a log line.

So the velocity path is:

    planner or operator  ->  /cmd_vel_raw
                                  |
                             this node, clamped by /speed_limit
                                  |
                               /cmd_vel  ->  the vehicle

Nothing publishes /cmd_vel except this node. That is the whole point, and it
is checkable in one grep rather than by reading every node.

Two behaviours worth stating, because both are the conservative choice and
both cost performance:

    a stale speed limit is treated as no speed limit, and no speed limit
    means stop. Perception that has died looks exactly like perception that
    sees clear ground, if you let it.

    the clamp is on magnitude, not a scale factor, so a command of zero stays
    zero and reverse is limited by the same number as forward. The vehicle is
    blind behind, and section 04 puts reversing in the CONOPS as a manoeuvre
    along a mapped path, not as a fast one.
"""

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from std_msgs.msg import Float32

from ares_math.constants import LIMITS

__all__ = ["SpeedGovernor"]


class SpeedGovernor(Node):
    def __init__(self):
        super().__init__("speed_governor")

        self.declare_parameter("limit_timeout", 1.0)
        self.declare_parameter("ceiling", LIMITS.v_autonomous)

        self.limit = 0.0
        self.limit_stamp = None
        self.ceiling = float(self.get_parameter("ceiling").value)
        self.timeout = float(self.get_parameter("limit_timeout").value)

        self.create_subscription(TwistStamped, "/speed_limit", self.on_limit, 10)
        self.create_subscription(Twist, "/cmd_vel_raw", self.on_command, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.applied = self.create_publisher(Float32, "/speed_applied", 10)

        self.clamped_count = 0
        self.passed_count = 0
        self.create_timer(5.0, self.report)

        self.get_logger().info(
            f"speed_governor up. Ceiling {self.ceiling * 3.6:.1f} km/h, and a "
            f"speed limit older than {self.timeout:.1f} s means stop."
        )

    def on_limit(self, msg):
        self.limit = float(msg.twist.linear.x)
        self.limit_stamp = self.get_clock().now()

    def _current_limit(self):
        if self.limit_stamp is None:
            return 0.0, "no speed limit received yet"
        age = (self.get_clock().now() - self.limit_stamp).nanoseconds * 1e-9
        if age > self.timeout:
            return 0.0, f"speed limit is {age:.1f} s stale"
        return min(self.limit, self.ceiling), ""

    def on_command(self, msg):
        limit, reason = self._current_limit()

        out = Twist()
        out.angular = msg.angular
        requested = msg.linear.x

        if abs(requested) <= limit:
            out.linear.x = requested
            self.passed_count += 1
        else:
            out.linear.x = limit if requested >= 0.0 else -limit
            self.clamped_count += 1
            if reason:
                self.get_logger().warn(
                    f"commanded {requested:.2f} m/s, allowing "
                    f"{out.linear.x:.2f}: {reason}",
                    throttle_duration_sec=2.0,
                )

        self.cmd.publish(out)
        self.applied.publish(Float32(data=float(out.linear.x)))

    def report(self):
        total = self.clamped_count + self.passed_count
        if total == 0:
            return
        limit, reason = self._current_limit()
        self.get_logger().info(
            f"limit {limit * 3.6:.2f} km/h{': ' + reason if reason else ''}, "
            f"clamped {self.clamped_count} of {total} commands"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SpeedGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
