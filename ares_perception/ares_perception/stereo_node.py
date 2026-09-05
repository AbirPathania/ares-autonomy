"""Stereo depth from two rendered RGB cameras. Build spec sections 09 and 10.

Subscribes ONLY to the two image topics. It never subscribes to a depth
camera, and the ground truth topic is named so that doing so by accident is
not possible. Gazebo will happily give you a geometrically perfect depth
image, and validating the autonomy stack against that would prove nothing at
all, because the entire risk of dropping the LiDAR lives in the ways stereo
depth fails.

The node is deliberately thin. The matcher, the confidence model and the
synchroniser are plain Python in this package, tested without ROS, and
exercised end to end by the phase 0 ablation in eval/. What is here is the
wiring, the timing, and the parts that must not be quietly wrong:

    the two eyes are synchronised, and unsynchronised pairs are refused
    every depth pixel carries a confidence
    the point cloud is published in the optical frame it was measured in
    the pipeline reports its own latency, because latency is stopping distance
"""

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray

from ares_math.constants import CAMERA, LIMITS, SENSORS
from ares_math.optics import focal_px
from ares_math.safety import stopping_distance

from .matcher import Intrinsics, StereoMatcher
from .sync import StereoSynchroniser

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=1,
)


class StereoNode(Node):
    def __init__(self):
        super().__init__("stereo_node")

        self.declare_parameter("unit", "mast_cam")
        self.declare_parameter("baseline", SENSORS.mast_baseline)
        self.declare_parameter("confidence_floor", 0.5)
        self.declare_parameter("max_skew", 0.005)
        self.declare_parameter("publish_cloud", True)

        unit = self.get_parameter("unit").value
        baseline = float(self.get_parameter("baseline").value)
        self.confidence_floor = float(self.get_parameter("confidence_floor").value)

        # Intrinsics start from the working case and are replaced the moment
        # a real CameraInfo arrives. Nothing downstream caches them.
        f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
        self.intrinsics = Intrinsics(
            fx=f,
            fy=f,
            cx=CAMERA.width_px / 2.0,
            cy=CAMERA.height_px / 2.0,
            width=CAMERA.width_px,
            height=CAMERA.height_px,
            baseline=baseline,
        )
        self.matcher = StereoMatcher(self.intrinsics)
        self.sync = StereoSynchroniser(
            focal_px=self.intrinsics.fx,
            max_skew=float(self.get_parameter("max_skew").value),
        )
        self.bridge = CvBridge()
        self.optical_frame = f"{unit}_optical"

        self.create_subscription(
            Image, f"/{unit}/image_raw", self.on_left, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            Image, f"/{unit}_right/image_raw", self.on_right, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            CameraInfo, f"/{unit}/camera_info", self.on_camera_info, qos_profile=SENSOR_QOS
        )
        # Motion tightens the sync budget. Without it the node would accept
        # pairs during a turn that it should refuse.
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile=SENSOR_QOS)

        self.depth_pub = self.create_publisher(Image, f"/{unit}/depth", SENSOR_QOS)
        self.confidence_pub = self.create_publisher(
            Image, f"/{unit}/depth_confidence", SENSOR_QOS
        )
        self.cloud_pub = self.create_publisher(PointCloud2, f"/{unit}/points", SENSOR_QOS)
        self.diagnostics_pub = self.create_publisher(
            Float32MultiArray, f"/{unit}/stereo_diagnostics", 10
        )

        self.create_timer(2.0, self.report)
        self._latencies = []

        budget_us = self.sync.current_budget
        self.get_logger().info(
            f"stereo_node up on {unit}, baseline {baseline} m, "
            f"f {self.intrinsics.fx:.1f} px. Sync ceiling "
            f"{self.sync.max_skew * 1e3:.1f} ms, and tighter under rotation: "
            f"{'unbounded while stationary' if budget_us == float('inf') else budget_us}"
        )

    # ------------------------------------------------------------ callbacks

    def on_camera_info(self, msg):
        if msg.k[0] <= 0.0:
            return
        if abs(msg.k[0] - self.intrinsics.fx) < 1e-9:
            return
        self.intrinsics = Intrinsics(
            fx=float(msg.k[0]),
            fy=float(msg.k[4]),
            cx=float(msg.k[2]),
            cy=float(msg.k[5]),
            width=int(msg.width),
            height=int(msg.height),
            baseline=self.intrinsics.baseline,
        )
        self.matcher = StereoMatcher(self.intrinsics)
        self.sync.focal_px = self.intrinsics.fx
        self.get_logger().info(
            f"intrinsics from CameraInfo: f {self.intrinsics.fx:.2f} px, "
            f"{self.intrinsics.width} by {self.intrinsics.height}"
        )

    def on_odom(self, msg):
        self.sync.set_motion(
            speed=abs(msg.twist.twist.linear.x),
            yaw_rate=abs(msg.twist.twist.angular.z),
        )

    def on_left(self, msg):
        self.on_pair(self.sync.push_left(_stamp(msg), msg))

    def on_right(self, msg):
        self.on_pair(self.sync.push_right(_stamp(msg), msg))

    # -------------------------------------------------------------- matching

    # Not called handle. rclpy.node.Node.handle is a property rclpy uses
    # internally, and shadowing it makes Node.__init__ fail with
    # "'method' object does not support the context manager protocol",
    # which names neither the attribute nor this file.
    def on_pair(self, pair):
        if pair is None:
            return
        stamp, left_msg, right_msg, skew = pair

        # Wall clock, deliberately. Under use_sim_time the ROS clock only
        # advances when /clock ticks, so this would read exactly zero and
        # the latency budget in section 03.3 would look free.
        started = time.monotonic()
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")

        depth, confidence, _ = self.matcher(left, right)

        header = left_msg.header
        header.frame_id = self.optical_frame

        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), "32FC1")
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)

        # Confidence is not a diagnostic, it is a first class output. A
        # consumer that ignores it will treat unknown as free space.
        confidence_msg = self.bridge.cv2_to_imgmsg(
            confidence.astype(np.float32), "32FC1"
        )
        confidence_msg.header = header
        self.confidence_pub.publish(confidence_msg)

        if self.get_parameter("publish_cloud").value:
            self.publish_cloud(header, depth, confidence)

        latency = time.monotonic() - started
        self._latencies.append(latency)
        self.publish_diagnostics(stamp, skew, confidence, latency)

    def publish_cloud(self, header, depth, confidence):
        from .matcher import depth_to_points

        # Identity: the cloud is published in the frame it was measured in,
        # and TF does the rest. Baking a transform in here is how a cloud ends
        # up rotated by 90 degrees.
        points, _ = depth_to_points(
            depth,
            self.intrinsics,
            confidence=np.where(confidence >= self.confidence_floor, confidence, 0.0),
            camera_to_body=(np.eye(3), np.zeros(3)),
        )
        self.cloud_pub.publish(_cloud(header, points))

    def publish_diagnostics(self, stamp, skew, confidence, latency):
        stats = self.sync.stats
        trusted = float(np.mean(confidence >= self.confidence_floor))
        msg = Float32MultiArray()
        msg.data = [
            float(stamp),
            float(skew),
            float(stats.last_budget),
            float(stats.rejection_rate),
            trusted,
            float(latency),
        ]
        self.diagnostics_pub.publish(msg)

    # ------------------------------------------------------------ reporting

    def report(self):
        stats = self.sync.stats
        if not self._latencies:
            self.get_logger().warn(
                "no stereo pairs matched. Check that both image topics are "
                "publishing and that their timestamps come from the same clock."
            )
            return

        latency = float(np.mean(self._latencies))
        worst_latency = float(np.max(self._latencies))
        self._latencies = self._latencies[-100:]

        # Latency is stopping distance. At 4 km/h every 100 ms of pipeline
        # delay costs 111 mm, and section 03.3 budgets 300 ms for the whole
        # chain: capture, rectify, match, segment, fuse, plan, actuate.
        cost = stopping_distance(
            LIMITS.v_autonomous, LIMITS.decel, latency
        ) - stopping_distance(LIMITS.v_autonomous, LIMITS.decel, 0.0)

        message = (
            f"paired {stats.paired}, refused on skew {stats.rejected_skew} "
            f"({stats.rejection_rate * 100:.1f}%), worst skew "
            f"{stats.worst_skew * 1e6:.0f} us, match latency {latency * 1e3:.1f} ms "
            f"(worst {worst_latency * 1e3:.1f}) costing {cost * 1000:.0f} mm of "
            "stopping distance"
        )
        if latency > LIMITS.latency:
            self.get_logger().error(
                f"{message}. The matcher alone is over the whole {LIMITS.latency} s "
                "sense-to-brake budget from section 03.3."
            )
        elif stats.rejection_rate > 0.2:
            self.get_logger().warn(
                f"{message}. More than a fifth of pairs refused: check the "
                "camera trigger, not the matcher."
            )
        else:
            self.get_logger().info(message)


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _cloud(header, points):
    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = 12 * len(points)
    cloud.is_dense = True
    cloud.data = np.asarray(points, dtype=np.float32).tobytes()
    return cloud


def main(args=None):
    rclpy.init(args=args)
    node = StereoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
