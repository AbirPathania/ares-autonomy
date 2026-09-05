"""Terrain segmentation. Per-pixel class, which is what actually drives.

Build spec section 09. Two heads, not one. Off-road, "identify a tree" is the
wrong question, and "can I drive on this pixel" is the useful one. This is the
head that answers the second question by class rather than by geometry, and it
is fused with the geometric traversability rather than replacing it.

    model class     SegFormer-B0 or DDRNet-23-slim
    output          per-pixel: rock, scree, grass, snow, mud, water, bush,
                    sky, man-made
    why             small models, real-time on Orin

This node does not ship weights, and it does not pretend to. Section 09 names
the training data: RELLIS-3D for the ablation, GOOSE for broader class
coverage, and your own BlenderProc renders for high altitude snow, scree and
the Ladakh palette, because no public dataset contains them.

Until weights exist it refuses to start rather than publishing a class map of
zeros. A segmentation head that silently outputs "grass" everywhere is worse
than no segmentation head, because the fusion layer downstream cannot tell the
difference between a confident answer and an absent one.
"""

from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# The ontology the elevation map's semantic layers expect. Kept here and in
# ares_bringup/config/elevation_map.yaml, which is generated, so a class added
# in one place shows up as a diff in the other.
CLASSES = (
    "rock",
    "scree",
    "grass",
    "snow",
    "mud",
    "water",
    "bush",
    "sky",
    "man_made",
)

# Classes whose surface the camera cannot see the bottom of. Not obstacles,
# and not free space either. They reduce confidence, which reduces speed
# through the governor, which is the honest handling.
REDUCED_CONFIDENCE = ("snow", "water", "mud")


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("segmentation_node")

        self.declare_parameter("unit", "mast_cam")
        self.declare_parameter("weights", "")
        self.declare_parameter("device", "cpu")

        unit = self.get_parameter("unit").value
        weights = str(self.get_parameter("weights").value)

        self.model = self._load(weights)
        self.bridge = CvBridge()

        self.create_subscription(
            Image, f"/{unit}/image_raw", self.on_image, qos_profile=SENSOR_QOS
        )
        self.class_pub = self.create_publisher(Image, f"/{unit}/terrain_class", SENSOR_QOS)
        self.confidence_pub = self.create_publisher(
            Image, f"/{unit}/terrain_confidence", SENSOR_QOS
        )

    def _load(self, weights):
        if not weights:
            raise RuntimeError(
                "segmentation_node has no weights and will not start.\n\n"
                "This is deliberate. A segmentation head with no weights "
                "publishes a class map of zeros, which reads downstream as a "
                "confident 'rock everywhere' and is worse than no head at all.\n\n"
                "Train one first. Section 09 names the data: RELLIS-3D for the "
                "ablation, GOOSE for class coverage, and phase 6's BlenderProc "
                "renders for snow, scree and the Ladakh palette, which no "
                "public dataset contains. Then pass weights:=/path/to/model."
            )

        path = Path(weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"segmentation weights not found at {path}")

        import torch

        device = self.get_parameter("device").value
        # Section 06.3: on a Mac use mps, in a Codespace there is no GPU at all
        # and cpu is the only correct answer. On the vehicle it is the Orin.
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        self.get_logger().info(
            f"segmentation weights loaded from {path} onto {device}, "
            f"{len(CLASSES)} classes"
        )
        return model

    def on_image(self, msg):
        import torch

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32)
            / 255.0
        ).to(self.get_parameter("device").value)

        with torch.no_grad():
            logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, classes = probabilities.max(dim=0)

        classes = classes.cpu().numpy().astype(np.uint8)
        confidence = confidence.cpu().numpy().astype(np.float32)

        # A class the camera cannot see the bottom of does not get full
        # confidence no matter how sure the network is that it is snow. Being
        # certain it is snow and being able to measure the ground under it are
        # different claims, and only the second one licenses speed.
        for name in REDUCED_CONFIDENCE:
            confidence[classes == CLASSES.index(name)] *= 0.5

        class_msg = self.bridge.cv2_to_imgmsg(classes, "mono8")
        class_msg.header = msg.header
        self.class_pub.publish(class_msg)

        confidence_msg = self.bridge.cv2_to_imgmsg(confidence, "32FC1")
        confidence_msg.header = msg.header
        self.confidence_pub.publish(confidence_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SegmentationNode()
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
