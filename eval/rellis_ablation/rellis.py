"""RELLIS-3D loader.

Off-road, multimodal, synchronised stereo and LiDAR from the same platform.
That last property is the whole reason phase 0 uses it: the two depth sources
saw the same scene at the same instant, so the difference between them is the
sensor and not the drive.

A warning about this file, kept deliberately visible.

The directory names and calibration keys below are written from the published
layout. They have not been run against a real download, because RELLIS-3D is
tens of gigabytes behind a registration form and this repository does not
carry the data. If the layout differs, this loader raises with the tree it
actually found rather than silently loading the wrong thing, and the fix is
one edit to LAYOUT.

That is the honest position. A loader that quietly assumed a layout, found
nothing, and reported an empty ablation would be worse than one that stops.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .depth import Intrinsics

# Every path fragment the loader expects, in one place, so a layout that
# turns out to differ is a single edit rather than a search.
LAYOUT = {
    "sequences": ["00000", "00001", "00002", "00003", "00004"],
    "left_image": "pylon_camera_node",
    "right_image": "pylon_camera_node_right",
    "image_labels": "pylon_camera_node_label_id",
    "lidar_points": "os1_cloud_node_kitti_bin",
    "lidar_labels": "os1_cloud_node_semantickitti_label_id",
    "calibration": "transforms.yaml",
    "camera_info": "camera_info.txt",
}

# RELLIS-3D ontology, and which classes a 25 kg rover can drive on.
# Traversable means the surface supports the vehicle, not that a planner
# should prefer it. Puddle is deliberately not traversable: standing water
# on an unknown bottom is exactly the case where looking is not enough.
CLASS_NAMES = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}

TRAVERSABLE_CLASSES = {"dirt", "grass", "asphalt", "concrete", "mud", "rubble"}


@dataclass
class Frame:
    sequence: str
    stamp: str
    left: np.ndarray = None
    right: np.ndarray = None
    image_labels: np.ndarray = None
    lidar_points: np.ndarray = None
    lidar_labels: np.ndarray = None
    meta: dict = field(default_factory=dict)


class Rellis3D:
    """Reads frames, and refuses to guess when the layout is not what it expects."""

    def __init__(self, root, sequences=None, baseline=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} does not exist. See eval/rellis_ablation/README.md "
                "for how to obtain RELLIS-3D."
            )
        self.sequences = sequences or [
            s for s in LAYOUT["sequences"] if (self.root / s).is_dir()
        ]
        if not self.sequences:
            raise FileNotFoundError(self._tree_message("no sequence directories"))
        self.baseline = baseline
        self._intrinsics = None
        self._camera_to_body = None

    def _tree_message(self, problem):
        found = sorted(p.name for p in self.root.iterdir())[:40]
        return (
            f"{problem} under {self.root}.\n"
            f"Found: {found}\n"
            "The expected layout is in LAYOUT at the top of "
            "eval/rellis_ablation/rellis.py. If the real download differs, "
            "correct LAYOUT rather than working around it here."
        )

    def _require(self, path, what):
        if not path.exists():
            raise FileNotFoundError(self._tree_message(f"missing {what} at {path}"))
        return path

    @property
    def intrinsics(self):
        if self._intrinsics is None:
            self._intrinsics = self._read_camera_info()
        return self._intrinsics

    @property
    def camera_to_body(self):
        """Rotation and translation from the left optical frame to base_link."""
        if self._camera_to_body is None:
            self._camera_to_body = self._read_calibration()
        return self._camera_to_body

    def _read_camera_info(self):
        path = self._require(
            self.root / self.sequences[0] / LAYOUT["camera_info"], "camera_info"
        )
        values = {}
        for line in path.read_text().splitlines():
            if ":" in line:
                key, _, rest = line.partition(":")
                values[key.strip()] = rest.strip()
        try:
            k = [float(v) for v in values["K"].replace("[", " ").replace("]", " ").split()]
            width = int(float(values.get("width", 1920)))
            height = int(float(values.get("height", 1200)))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"could not read intrinsics from {path}. Keys found: "
                f"{sorted(values)}. Correct the parser rather than assuming values."
            ) from exc
        if self.baseline is None:
            raise ValueError(
                "the stereo baseline is a property of the RELLIS-3D platform and "
                "must be passed explicitly. Read it from the dataset's own "
                "calibration and pass baseline=. Guessing it would silently scale "
                "every depth in the study."
            )
        return Intrinsics(
            fx=k[0], fy=k[4], cx=k[2], cy=k[5],
            width=width, height=height, baseline=self.baseline,
        )

    def _read_calibration(self):
        path = self._require(
            self.root / self.sequences[0] / LAYOUT["calibration"], "transforms.yaml"
        )
        data = yaml.safe_load(path.read_text())
        raise NotImplementedError(
            "the camera to body extrinsic must be read from the dataset's own "
            f"calibration, not assumed. {path} parsed with top level keys "
            f"{sorted(data) if isinstance(data, dict) else type(data).__name__}. "
            "Map them onto a (rotation, translation) pair here once the real file "
            "is in hand. Section 08 is explicit that a guessed optical transform "
            "is the bug that costs everyone two days."
        )

    def frames(self):
        out = []
        for sequence in self.sequences:
            left_dir = self._require(
                self.root / sequence / LAYOUT["left_image"], "left images"
            )
            for image in sorted(left_dir.glob("*.jpg")) + sorted(left_dir.glob("*.png")):
                out.append(Frame(sequence=sequence, stamp=image.stem))
        if not out:
            raise FileNotFoundError(self._tree_message("no images"))
        return out

    def load(self, frame):
        import cv2

        base = self.root / frame.sequence
        left_path = self._find(base / LAYOUT["left_image"], frame.stamp)
        right_path = self._find(base / LAYOUT["right_image"], frame.stamp)
        frame.left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)[:, :, ::-1]
        frame.right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)[:, :, ::-1]

        label_path = self._find(base / LAYOUT["image_labels"], frame.stamp, optional=True)
        if label_path:
            frame.image_labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        points_path = self._find(
            base / LAYOUT["lidar_points"], frame.stamp, suffixes=(".bin",)
        )
        frame.lidar_points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)[
            :, :3
        ]
        labels_path = self._find(
            base / LAYOUT["lidar_labels"], frame.stamp, suffixes=(".label",),
            optional=True,
        )
        if labels_path:
            frame.lidar_labels = np.fromfile(labels_path, dtype=np.uint32) & 0xFFFF
        return frame

    def _find(self, directory, stamp, suffixes=(".jpg", ".png"), optional=False):
        for suffix in suffixes:
            candidate = directory / f"{stamp}{suffix}"
            if candidate.exists():
                return candidate
        if optional:
            return None
        raise FileNotFoundError(
            self._tree_message(f"no file for stamp {stamp} in {directory}")
        )

    @staticmethod
    def class_name(label):
        return CLASS_NAMES.get(int(label), f"class_{int(label)}")

    def truth_grid(self, frame, spec):
        """Semantic ground truth on the estimator's grid, plus the class map.

        Uses the labelled LiDAR points, because they carry both a position and
        a class, which is what a body frame grid needs. Returns
        (traversable bool grid, class id grid with -1 where unlabelled).
        """
        if frame.lidar_labels is None:
            raise ValueError(
                "semantic ground truth needs the labelled point cloud. "
                f"Expected {LAYOUT['lidar_labels']} beside the points."
            )
        points = frame.lidar_points
        labels = frame.lidar_labels
        ix, iy, inside = spec.index(points[:, 0], points[:, 1])
        nx, ny = spec.shape

        classes = np.full((nx, ny), -1, dtype=np.int32)
        traversable = np.zeros((nx, ny), dtype=bool)
        blocked = np.zeros((nx, ny), dtype=bool)

        for index, label in zip(
            (ix[inside] * ny + iy[inside]), labels[inside], strict=True
        ):
            name = self.class_name(label)
            row, col = divmod(int(index), ny)
            classes[row, col] = label
            if name in TRAVERSABLE_CLASSES:
                traversable[row, col] = True
            else:
                blocked[row, col] = True

        # A cell holding anything non traversable is non traversable. A tree
        # trunk in a cell of grass does not make the cell drivable.
        return traversable & ~blocked, classes
