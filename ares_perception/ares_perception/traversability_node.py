"""Traversability, and the speed that falls out of it.

Build spec section 09. Two heads, not one. Off-road, "identify a tree" is the
wrong question. The useful question is "can I drive on this pixel", and this
node answers it from geometry. The semantic head answers the other half and is
fused in when it is running.

This node publishes two things and the second is the one that keeps the
vehicle alive:

    /traversability     the three state grid: traversable, obstacle, unknown
    /speed_limit        the speed the vehicle may drive, and why

The speed limit is not advisory. It is derived from the range the perception
has actually decided out to, through the same equation that turns mast height
into 4 km/h. When depth collapses on textureless snow, the trusted range
shortens, and the speed follows it down without anyone writing a rule about
snow.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

from ares_math.constants import CHASSIS, LIMITS

from .governor import speed_limit
from .traversability import STATE_NAMES, GridSpec, estimate

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class TraversabilityNode(Node):
    def __init__(self):
        super().__init__("traversability_node")

        self.declare_parameter("unit", "mast_cam")
        self.declare_parameter("x_max", 20.0)
        self.declare_parameter("half_width", 8.0)
        self.declare_parameter("confidence_floor", 0.5)

        unit = self.get_parameter("unit").value
        self.spec = GridSpec(
            x_min=0.0,
            x_max=float(self.get_parameter("x_max").value),
            y_min=-float(self.get_parameter("half_width").value),
            y_max=float(self.get_parameter("half_width").value),
        )
        self.confidence_floor = float(self.get_parameter("confidence_floor").value)
        self.bridge = CvBridge()

        # The cloud arrives in the camera optical frame: Z forward, X right,
        # Y down. This grid is in base_link: X forward, Y left, Z up. Treating
        # one as the other is the bug section 08 warns about, and it does not
        # look like a bug, it looks like the rover seeing an obstacle two
        # centimetres in front of itself. Look the transform up, never assume
        # it, and refuse to publish if it is not available yet.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.warned_about_tf = False

        self.create_subscription(
            PointCloud2, f"/{unit}/points", self.on_cloud, qos_profile=SENSOR_QOS
        )

        self.grid_pub = self.create_publisher(Image, "/traversability", SENSOR_QOS)
        self.limit_pub = self.create_publisher(TwistStamped, "/speed_limit", 10)
        self.reason_pub = self.create_publisher(String, "/speed_limit_reason", 10)
        self.metrics_pub = self.create_publisher(
            Float32MultiArray, "/traversability_metrics", 10
        )

        self.last_reason = None
        self.get_logger().info(
            f"traversability_node up. Grid {self.spec.shape[0]} by "
            f"{self.spec.shape[1]} at {self.spec.cell * 1000:.0f} mm cells, "
            f"ceiling {LIMITS.v_autonomous * 3.6:.1f} km/h."
        )

    def on_cloud(self, msg):
        points = _read_xyz(msg)
        if points.size:
            points = self.to_base_link(points, msg.header)
        if points is None:
            self.publish_limit(msg.header, None)
            return
        if points.size == 0:
            # No points is not empty ground. It is no information, and the
            # governor treats it as such.
            self.publish_limit(msg.header, None)
            return

        state, surface, _ = estimate(
            points, spec=self.spec, confidence_floor=self.confidence_floor
        )
        limit = speed_limit(state, self.spec)

        grid = self.bridge.cv2_to_imgmsg(state.astype(np.uint8), "mono8")
        grid.header = msg.header
        grid.header.frame_id = "base_link"
        self.grid_pub.publish(grid)

        self.publish_limit(msg.header, limit)
        self.publish_metrics(state, surface, limit)

    def to_base_link(self, points, header):
        """Transform the cloud out of the optical frame. Returns None if TF
        is not ready, because guessing is how a ground plane becomes a wall."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", header.frame_id, rclpy.time.Time()
            )
        except Exception as exc:
            if not self.warned_about_tf:
                self.get_logger().warn(
                    f"no transform from {header.frame_id} to base_link yet "
                    f"({exc}). Publishing a stop until there is one."
                )
                self.warned_about_tf = True
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        rotation = _quaternion_matrix(q.x, q.y, q.z, q.w)
        moved = points @ rotation.T + np.array([t.x, t.y, t.z])

        # Drop anything inside the vehicle's own footprint. The hull and the
        # wheels are in the camera's field of view, and a rover that treats
        # its own bumper as an obstacle at 0.02 m never moves.
        ahead = CHASSIS.axle_pitch_front + CHASSIS.rolling_radius
        beside = CHASSIS.track / 2.0
        own_body = (
            (moved[:, 0] < ahead)
            & (moved[:, 0] > -(CHASSIS.axle_pitch_rear + CHASSIS.rolling_radius))
            & (np.abs(moved[:, 1]) < beside)
        )
        return moved[~own_body]

    def publish_limit(self, header, limit):
        command = TwistStamped()
        command.header = header
        command.header.frame_id = "base_link"

        if limit is None:
            command.twist.linear.x = 0.0
            reason = "no points received, stop"
        else:
            command.twist.linear.x = float(limit.speed)
            reason = limit.reason

        self.limit_pub.publish(command)

        if reason != self.last_reason:
            self.reason_pub.publish(String(data=reason))
            speed = command.twist.linear.x * 3.6
            # A vehicle that slows without saying why is indistinguishable
            # from one that is broken.
            if speed < 0.1:
                self.get_logger().warn(f"stopping: {reason}")
            elif speed < LIMITS.v_autonomous * 3.6 - 0.05:
                self.get_logger().info(f"limiting to {speed:.2f} km/h: {reason}")
            else:
                self.get_logger().info(f"{speed:.2f} km/h: {reason}")
            self.last_reason = reason

    def publish_metrics(self, state, surface, limit):
        counts = {
            name: float(np.mean(state == value)) for value, name in STATE_NAMES.items()
        }
        message = Float32MultiArray()
        message.data = [
            counts["unknown"],
            counts["traversable"],
            counts["obstacle"],
            float(np.mean(surface.confidence)),
            float(limit.trusted_range),
            float(limit.speed),
        ]
        self.metrics_pub.publish(message)


def _quaternion_matrix(x, y, z, w):
    """Rotation matrix from a quaternion. Same convention as rpy_matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _read_xyz(msg):
    """Points from a PointCloud2 with float32 x, y, z at the front."""
    if msg.point_step < 12 or msg.width * msg.height == 0:
        return np.empty((0, 3))
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    count = msg.width * msg.height
    raw = raw[: count * msg.point_step].reshape(count, msg.point_step)
    return raw[:, :12].copy().view(np.float32).reshape(count, 3)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
