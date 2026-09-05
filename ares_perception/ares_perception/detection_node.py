"""Object detection. Boxes for person, vehicle, animal, structure.

Build spec section 09. This is the head that carries the safety case and the
SAR product feature, and it is the reason resolution is not the problem:
section 03.5 says a standing person subtends 27 pixels at 40 m on the working
case, well past the 10 to 20 a detector needs to fire.

Every 2D box becomes a 3D object through ares_math.optics.unproject and
nothing else. That is not a style preference. A detector that reports range by
any other path is a second, unvalidated projection model, and the two will
disagree quietly.

Range comes from the stereo depth already published, sampled inside the box
with its confidence. A box over a region the matcher declined to resolve gets
a bearing and no range, and says so, rather than inventing a distance.
"""

from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from ares_math.constants import CAMERA
from ares_math.optics import focal_px, pixels_on_target, unproject

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# Nominal heights, used only to sanity check a detection's range against the
# pixels it actually occupies. A box that claims 5 m but subtends the pixels
# of something at 40 m is a false positive, and this catches a class of them
# for free.
NOMINAL_HEIGHT = {
    "person": 1.70,
    "vehicle": 1.60,
    "animal": 1.00,
    "structure": 2.00,
}

# How far the pixel height may disagree with the depth-derived range before
# the detection is dropped. Generous, because people crouch and animals vary.
HEIGHT_TOLERANCE = 2.5


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        self.declare_parameter("unit", "mast_cam")
        self.declare_parameter("weights", "")
        self.declare_parameter("confidence_floor", 0.4)
        self.declare_parameter("depth_confidence_floor", 0.5)

        unit = self.get_parameter("unit").value
        self.bridge = CvBridge()
        self.depth = None
        self.depth_confidence = None

        f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
        self.fx = self.fy = f
        self.cx = CAMERA.width_px / 2.0
        self.cy = CAMERA.height_px / 2.0

        self.model = self._load(str(self.get_parameter("weights").value))

        self.create_subscription(
            Image, f"/{unit}/image_raw", self.on_image, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            Image, f"/{unit}/depth", self.on_depth, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            Image,
            f"/{unit}/depth_confidence",
            self.on_depth_confidence,
            qos_profile=SENSOR_QOS,
        )
        self.create_subscription(
            CameraInfo, f"/{unit}/camera_info", self.on_camera_info, qos_profile=SENSOR_QOS
        )
        self.detections_pub = self.create_publisher(
            Detection3DArray, f"/{unit}/detections", 10
        )
        self.optical_frame = f"{unit}_optical"

    def _load(self, weights):
        if not weights:
            raise RuntimeError(
                "detection_node has no weights and will not start.\n\n"
                "Person detection is the safety case. A detector that loads no "
                "model and publishes an empty array every frame is "
                "indistinguishable from one that is working and sees nobody, "
                "which is exactly the failure you cannot afford here.\n\n"
                "Pass weights:=/path/to/model.pt once one is trained."
            )
        path = Path(weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"detector weights not found at {path}")

        from ultralytics import YOLO

        model = YOLO(str(path))
        self.get_logger().info(f"detector loaded from {path}")
        return model

    def on_camera_info(self, msg):
        if msg.k[0] > 0.0:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])

    def on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def on_depth_confidence(self, msg):
        self.depth_confidence = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def on_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        results = self.model.predict(
            image, conf=float(self.get_parameter("confidence_floor").value), verbose=False
        )

        array = Detection3DArray()
        array.header = msg.header
        array.header.frame_id = self.optical_frame

        for result in results:
            for box in result.boxes:
                detection = self._to_3d(box, result.names)
                if detection is not None:
                    array.detections.append(detection)

        self.detections_pub.publish(array)

    def _to_3d(self, box, names):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        label = names[int(box.cls[0])]
        score = float(box.conf[0])

        depth = self._range_in(x1, y1, x2, y2)
        if depth is None:
            # A bearing with no range. Published as a zero range detection
            # would be a lie, so it is dropped here and the region is already
            # UNKNOWN in the traversability grid, which slows the vehicle.
            self.get_logger().debug(
                f"{label} at ({x1:.0f}, {y1:.0f}) has no trusted depth, dropped"
            )
            return None

        if not self._height_is_consistent(label, y2 - y1, depth):
            return None

        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        x, y, z = unproject(u, v, depth, self.fx, self.fy, self.cx, self.cy)

        detection = Detection3D()
        detection.bbox.center.position.x = float(x)
        detection.bbox.center.position.y = float(y)
        detection.bbox.center.position.z = float(z)
        # Extent from the box, projected out at the measured range.
        detection.bbox.size.x = float((x2 - x1) * depth / self.fx)
        detection.bbox.size.y = float((y2 - y1) * depth / self.fy)
        detection.bbox.size.z = detection.bbox.size.y

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = label
        hypothesis.hypothesis.score = score
        hypothesis.pose.pose.position = detection.bbox.center.position
        detection.results.append(hypothesis)
        return detection

    def _range_in(self, x1, y1, x2, y2):
        """Median depth inside the box, over pixels the matcher trusted."""
        if self.depth is None:
            return None
        h, w = self.depth.shape
        a, b = int(max(y1, 0)), int(min(y2, h))
        c, d = int(max(x1, 0)), int(min(x2, w))
        if b <= a or d <= c:
            return None

        patch = self.depth[a:b, c:d]
        valid = np.isfinite(patch)
        if self.depth_confidence is not None:
            floor = float(self.get_parameter("depth_confidence_floor").value)
            valid &= self.depth_confidence[a:b, c:d] >= floor
        if valid.sum() < 16:
            return None
        return float(np.median(patch[valid]))

    def _height_is_consistent(self, label, pixel_height, depth):
        """Cross check the range against the pixels the object occupies.

        Section 03.5: n_px = H * f / Z. If a detection's apparent size and its
        measured range disagree by more than a factor, one of them is wrong,
        and dropping it costs less than acting on it.
        """
        nominal = NOMINAL_HEIGHT.get(label)
        if nominal is None or pixel_height <= 0:
            return True
        expected = pixels_on_target(nominal, self.fy, depth)
        ratio = pixel_height / expected
        if not 1.0 / HEIGHT_TOLERANCE <= ratio <= HEIGHT_TOLERANCE:
            self.get_logger().debug(
                f"{label} dropped: {pixel_height:.0f} px at {depth:.1f} m, "
                f"expected about {expected:.0f} px"
            )
            return False
        return True


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DetectionNode()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\n{exc}\n")
        rclpy.try_shutdown()
        raise SystemExit(1) from exc
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
