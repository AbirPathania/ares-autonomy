"""The two depth sources under test, and the confidence that separates them.

LiDAR: project the point cloud into the camera frame. Sparse, accurate, and
the thing the study is trying to justify removing.

Stereo: a real matcher on the real image pair. Semi-global block matching,
the classical baseline the build spec section 09 asks for, so there is a
floor to compare a learned matcher against later. Dense, cheap, and wrong in
a specific way that matters.

The confidence output is not a nice-to-have. The camera-only bet fails in
exactly one way: textureless terrain returns bad depth that looks like good
depth. A matcher that says when it does not know is the mitigation, so the
confidence here is computed from things that actually predict failure, not
from the disparity value itself.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ares_math.constants import CAMERA
from ares_math.optics import unproject
from ares_math.stereo import depth_error, depth_from_disparity

__all__ = [
    "Intrinsics",
    "StereoMatcher",
    "lidar_to_points",
    "depth_to_points",
]


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline: float

    @classmethod
    def working_case(cls, baseline):
        """The section 03 working case, for synthetic scenes and sanity runs."""
        from ares_math.optics import focal_px

        f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
        return cls(
            fx=f,
            fy=f,
            cx=CAMERA.width_px / 2.0,
            cy=CAMERA.height_px / 2.0,
            width=CAMERA.width_px,
            height=CAMERA.height_px,
            baseline=baseline,
        )


class StereoMatcher:
    """Semi-global block matching, with a confidence the stack can act on.

    Four things drive the confidence down, and each of them is a way real
    stereo fails rather than a heuristic:

      texture      local intensity variance. No texture, nothing to match,
                   and this is the snow failure mode in one number.
      uniqueness   the ratio between the best and second best cost. A
                   repeating pattern matches everywhere equally well.
      consistency  left-right agreement. Occlusions fail this.
      precision    the depth error the geometry predicts at that range,
                   from ares_math.stereo.depth_error, relative to the
                   accuracy the vehicle needs to plan at all.

    The product is the per-pixel confidence. Nothing here reads the disparity
    value and decides it looks plausible, which is the trap.
    """

    def __init__(self, intrinsics, max_disparity=None, block=7):
        self.k = intrinsics
        # The disparity search scales with image width, and it has to.
        #
        # CAMERA.max_disparity_px is 192 for the 1280 wide working case, and
        # what it really fixes is the closest range the matcher can resolve:
        # Z_min = f*B/d_max. Halve the width and f halves too, so keeping
        # d_max at 192 does not see any closer, it just searches twice as far
        # for nothing. On four cores with two matchers running that surplus
        # is enough memory to get a node killed, which is how this was found.
        if max_disparity is None:
            scale = intrinsics.width / CAMERA.width_px
            max_disparity = CAMERA.max_disparity_px * scale
        # SGBM needs a multiple of 16.
        self.max_disparity = max(16, int(np.ceil(max_disparity / 16.0) * 16))
        self.block = block
        channels = 1
        self.left = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.max_disparity,
            blockSize=block,
            P1=8 * channels * block * block,
            P2=32 * channels * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.right = (
            cv2.ximgproc_createRightMatcher(self.left) if _has_ximgproc() else None
        )

    @property
    def minimum_range(self):
        """Closest range this matcher can resolve, metres.

        The invariant the disparity scaling protects. It should not change
        when the image is resized.
        """
        return self.k.fx * self.k.baseline / self.max_disparity

    def __call__(self, left_image, right_image, texture_floor=8.0):
        """Returns (depth, confidence, disparity), all (H, W) float arrays.

        Depth is NaN where the matcher declined to answer.
        """
        left_gray = _gray(left_image)
        right_gray = _gray(right_image)

        disparity = self.left.compute(left_gray, right_gray).astype(np.float32) / 16.0
        valid = disparity > 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                valid, self.k.fx * self.k.baseline / np.maximum(disparity, 1e-6), np.nan
            )

        confidence = self._confidence(
            left_gray, right_gray, disparity, valid, texture_floor
        )
        depth = np.where(confidence > 0.0, depth, np.nan)
        return depth, confidence, np.where(valid, disparity, np.nan)

    def _confidence(self, left_gray, right_gray, disparity, valid, texture_floor):
        # Texture. Local standard deviation of intensity. This is the term
        # that collapses on fresh snow, which is the entire reason the
        # confidence output exists.
        blur = cv2.blur(left_gray.astype(np.float32), (self.block, self.block))
        blur_sq = cv2.blur(left_gray.astype(np.float32) ** 2, (self.block, self.block))
        texture = np.sqrt(np.maximum(blur_sq - blur**2, 0.0))
        texture_score = np.clip(texture / texture_floor, 0.0, 1.0)

        # Left-right consistency. Warp the disparity by itself and check it
        # lands back where it started. Occlusion boundaries fail this, and
        # they are where a confident wrong depth is most dangerous.
        consistency = self._left_right_consistency(left_gray, right_gray, disparity)

        # Geometric precision. The error the section 03.2 curve predicts at
        # this range, against the accuracy needed to plan a path. Beyond that
        # the depth is real but not useful, and saying so is not pessimism.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.k.fx * self.k.baseline / np.maximum(disparity, 1e-6)
        planning_tolerance = 0.10
        sigma = np.where(
            valid,
            depth**2 * CAMERA.disparity_sigma_px / (self.k.fx * self.k.baseline),
            np.inf,
        )
        precision = np.clip(planning_tolerance / np.maximum(sigma, 1e-9), 0.0, 1.0)

        confidence = texture_score * consistency * precision
        return np.where(valid, confidence, 0.0)

    def _left_right_consistency(self, left_gray, right_gray, disparity):
        if self.right is None:
            # Without the contrib matcher, fall back to a direct check:
            # sample the right image at the matched column and compare.
            h, w = disparity.shape
            cols = np.arange(w)[None, :] - disparity
            cols = np.clip(cols, 0, w - 1)
            sampled = np.take_along_axis(
                right_gray.astype(np.float32), cols.astype(np.int64), axis=1
            )
            residual = np.abs(sampled - left_gray.astype(np.float32))
            return np.clip(1.0 - residual / 32.0, 0.0, 1.0)

        right_disp = self.right.compute(right_gray, left_gray).astype(np.float32) / 16.0
        h, w = disparity.shape
        cols = np.clip(np.arange(w)[None, :] - disparity, 0, w - 1).astype(np.int64)
        matched = np.take_along_axis(-right_disp, cols, axis=1)
        return np.clip(1.0 - np.abs(matched - disparity) / 2.0, 0.0, 1.0)


def _has_ximgproc():
    return hasattr(cv2, "ximgproc_createRightMatcher")


def _gray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def depth_to_points(depth, intrinsics, confidence=None, camera_to_body=None):
    """Depth image to a body frame point cloud.

    Every 2D measurement becomes a 3D point through ares_math.optics and
    nothing else. Build spec section 08.
    """
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    finite = np.isfinite(depth)
    if confidence is not None:
        finite &= confidence > 0.0

    u = us[finite].astype(float)
    v = vs[finite].astype(float)
    z = depth[finite].astype(float)

    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    optical = np.stack([x, y, z], axis=1)

    body = _optical_to_body(optical, camera_to_body)
    conf = confidence[finite] if confidence is not None else None
    return body, conf


def lidar_to_points(cloud, camera_to_body=None, sensor_to_body=None):
    """LiDAR returns to the same body frame, through the same transform path.

    Passing a sensor_to_body keeps the LiDAR honest: it is measured from the
    same datum as the cameras, so neither source gets a free extrinsic.
    """
    del camera_to_body
    points = np.asarray(cloud, dtype=float).reshape(-1, 3)
    if sensor_to_body is None:
        return points
    rotation, translation = sensor_to_body
    return points @ np.asarray(rotation).T + np.asarray(translation)


def _optical_to_body(points, camera_to_body):
    """Optical frame, Z forward X right Y down, to body frame X forward Y left Z up.

    This is the transform section 08 warns about. It is applied here, once,
    from the URDF's own rotation rather than from a launch file argument.
    """
    if camera_to_body is None:
        raise ValueError(
            "camera_to_body is required. Guessing the optical rotation is the "
            "bug that costs everyone two days."
        )
    rotation, translation = camera_to_body
    return points @ np.asarray(rotation).T + np.asarray(translation)


def predicted_depth_error(depth, intrinsics, sigma_px=None):
    """The section 03.2 curve, evaluated for these optics.

    Used to check that measured stereo error actually tracks the theory,
    which is the phase 4 gate and a good sanity check on phase 0 data.
    """
    sigma = CAMERA.disparity_sigma_px if sigma_px is None else sigma_px
    depth = np.asarray(depth, dtype=float)
    out = np.full(depth.shape, np.nan)
    finite = np.isfinite(depth) & (depth > 0)
    flat_in = depth[finite]
    out[finite] = [
        depth_error(Z=float(z), f=intrinsics.fx, B=intrinsics.baseline, d_disp=sigma)
        for z in flat_in
    ]
    return out


def sanity_depth_from_disparity(disparity, intrinsics):
    """Single value path through ares_math, used by the tests as a cross check."""
    return depth_from_disparity(intrinsics.fx, intrinsics.baseline, disparity)


def sanity_unproject(u, v, z, intrinsics):
    return unproject(u, v, z, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
