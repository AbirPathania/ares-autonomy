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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

from ares_math.constants import CAMERA, CHASSIS, LIMITS, SENSORS
from ares_math.optics import focal_px

from .governor import CORRIDOR_WIDTHS, speed_limit
from .traversability import (
    STATE_NAMES,
    GridSpec,
    depth_noise_grid,
    estimate,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class TraversabilityNode(Node):
    def __init__(self):
        super().__init__("traversability_node")

        # Both units, fused. Section 04: this is a coverage scheme, not a
        # redundancy scheme. The mast unit is blind inside 0.81 m and the deck
        # unit exists to fill exactly that, so a grid built from the mast
        # alone is unknown right in front of the wheels and the vehicle stops
        # for its own blind spot.
        self.declare_parameter("units", ["mast_cam", "deck_cam"])
        self.declare_parameter("x_max", 20.0)
        self.declare_parameter("half_width", 8.0)
        self.declare_parameter("confidence_floor", 0.5)
        self.declare_parameter("baseline", SENSORS.mast_baseline)

        units = list(self.get_parameter("units").value)
        self.clouds = {}
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
        # Thresholds widen with range by the depth uncertainty the optics
        # predict there. Read from CameraInfo so it follows the real camera.
        self.noise = None
        self.fx = focal_px(CAMERA.sim_width_px, CAMERA.hfov_deg)
        self.baseline = float(self.get_parameter("baseline").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.warned_about_tf = False

        self.create_subscription(
            CameraInfo,
            f"/{units[0]}/camera_info",
            self.on_camera_info,
            qos_profile=SENSOR_QOS,
        )
        for unit in units:
            self.create_subscription(
                PointCloud2,
                f"/{unit}/points",
                lambda msg, name=unit: self.on_cloud(msg, name),
                qos_profile=SENSOR_QOS,
            )

        self.grid_pub = self.create_publisher(Image, "/traversability", SENSOR_QOS)
        self.limit_pub = self.create_publisher(TwistStamped, "/speed_limit", 10)
        self.reason_pub = self.create_publisher(String, "/speed_limit_reason", 10)
        self.metrics_pub = self.create_publisher(
            Float32MultiArray, "/traversability_metrics", 10
        )

        self.last_reason = None
        # Periodic telemetry. The reason string is only logged when it
        # changes, which is right for an operator and useless when the answer
        # is stuck: "nothing measured ahead" printed once looks the same
        # whether it happened for a second or for the whole run.
        self.last_stats = None
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f"traversability_node up on {', '.join(units)}. Grid "
            f"{self.spec.shape[0]} by {self.spec.shape[1]} at "
            f"{self.spec.cell * 1000:.0f} mm cells, ceiling "
            f"{LIMITS.v_autonomous * 3.6:.1f} km/h."
        )

    def on_camera_info(self, msg):
        if msg.k[0] > 0.0 and abs(msg.k[0] - self.fx) > 1e-9:
            self.fx = float(msg.k[0])
            self.noise = None

    def on_cloud(self, msg, unit):
        points = _read_xyz(msg)
        if points.size:
            points = self.to_base_link(points, msg.header)
        if points is None:
            self.publish_limit(msg.header, None)
            return

        self.clouds[unit] = points
        fused = [c for c in self.clouds.values() if c is not None and c.size]
        if not fused:
            # No points is not empty ground. It is no information, and the
            # governor treats it as such.
            self.publish_limit(msg.header, None)
            return
        points = np.concatenate(fused, axis=0)

        if self.noise is None:
            self.noise = depth_noise_grid(self.spec, self.fx, self.baseline)
        state, surface, _ = estimate(
            points,
            spec=self.spec,
            confidence_floor=self.confidence_floor,
            noise=self.noise,
        )
        limit = speed_limit(state, self.spec, noise=self.noise)

        grid = self.bridge.cv2_to_imgmsg(state.astype(np.uint8), "mono8")
        grid.header = msg.header
        grid.header.frame_id = "base_link"
        self.grid_pub.publish(grid)

        self.publish_limit(msg.header, limit)
        self.publish_metrics(state, surface, limit)
        self.last_stats = (points, state, limit)

    def report(self):
        if self.last_stats is None:
            self.get_logger().warn("no cloud processed yet")
            return
        points, state, limit = self.last_stats
        observed = float(np.mean(state != 0))
        traversable = float(np.mean(state == 1))
        obstacle = float(np.mean(state == 2))
        ahead = points[points[:, 0] > 0]
        reach = float(np.max(ahead[:, 0])) if ahead.size else 0.0
        nearest = float(np.min(ahead[:, 0])) if ahead.size else 0.0
        self.get_logger().info(
            f"{len(points)} points, {nearest:.2f} to {reach:.2f} m. Grid: "
            f"{observed * 100:.1f}% decided, {traversable * 100:.1f}% "
            f"traversable, {obstacle * 100:.1f}% obstacle. Trusted range "
            f"{limit.trusted_range:.2f} m, allowing {limit.kmh:.2f} km/h "
            f"({limit.reason})"
        )
        # Corridor coverage, band by band. "Trusted range 0" is a conclusion,
        # not a diagnosis, and this is the row of numbers that distinguishes
        # a camera that sees nothing from a corridor defined in the wrong
        # place from a density threshold nothing meets.
        xs, ys = self.spec.centres()
        corridor = np.abs(ys) <= CORRIDOR_WIDTHS * CHASSIS.track / 2.0
        profile = []
        for i, x in enumerate(xs):
            if x < 0.3 or x > 4.0 or i % 4:
                continue
            band = state[i, corridor]
            profile.append(f"{x:.1f}:{np.mean(band != 0) * 100:.0f}%")
        self.get_logger().info(
            f"corridor is {int(corridor.sum())} cells wide, decided by range: "
            + " ".join(profile)
        )

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
