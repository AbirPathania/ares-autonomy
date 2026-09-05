"""A synthetic scene with known geometry, so the pipeline is testable today.

RELLIS-3D is tens of gigabytes behind a registration form. Waiting for it to
verify the ablation code would mean writing the whole phase blind, which is
the failure mode the build spec's gate structure exists to prevent.

So this renders a stereo pair by ray casting a height field, with a texture
that is a function of world position rather than of pixel position. That last
detail is what makes it a real stereo problem: the same ground point looks
the same from both eyes, so a matcher has something true to find, and the
disparity it recovers can be checked against geometry that is known exactly.

It also gives the two things no public dataset provides:

    a textureless region, which is the snow failure mode in miniature
    a trench of exactly the governing dimensions from section 02

Nothing here is a substitute for real data. It is a test fixture.
"""

from dataclasses import dataclass

import numpy as np

from ares_math.constants import CAMERA, MOBILITY, SENSORS, TRENCH
from ares_math.kinematics import rpy_matrix
from ares_math.optics import focal_px

from .depth import Intrinsics

RENDER_WIDTH = 640
RENDER_HEIGHT = 360


def render_intrinsics(baseline=None, width=RENDER_WIDTH, height=RENDER_HEIGHT):
    """Working case optics at render resolution.

    Section 06.2 keeps simulated cameras at 640 by 360 because the renderer
    is software. The field of view is unchanged, so the focal length scales
    with the width and every section 03 relation still holds.
    """
    f = focal_px(width, CAMERA.hfov_deg)
    return Intrinsics(
        fx=f,
        fy=f,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
        baseline=SENSORS.mast_baseline if baseline is None else baseline,
    )


@dataclass
class Scene:
    """A ground height field plus a texture field, both over world x and y."""

    name: str
    height: callable
    texture: callable
    traversable: callable


def _noise(x, y, scale, seed=0):
    """Deterministic band limited texture. Same value for the same world point."""
    out = np.zeros_like(x, dtype=float)
    rng = np.random.default_rng(seed)
    for octave in range(4):
        k = scale * (2.0**octave)
        phase = rng.uniform(0, 2 * np.pi, size=4)
        out += (
            np.sin(k * x + phase[0]) * np.cos(k * y + phase[1])
            + np.sin(k * (x + y) * 0.7 + phase[2]) * np.cos(k * (x - y) * 0.7 + phase[3])
        ) / (2.0**octave)
    return out


def flat_textured():
    """Gate 1. A flat plane, fully textured. Depth must track the theory."""

    def height(x, y):
        return np.zeros_like(x)

    def texture(x, y):
        return 128.0 + 60.0 * _noise(x, y, 6.0, seed=1)

    def traversable(x, y):
        return np.ones_like(x, dtype=bool)

    return Scene("flat_textured", height, texture, traversable)


def step_field():
    """Gate 2. A step at the obstacle claim, and one well over it."""
    climbable = MOBILITY.max_step * 0.8
    blocking = MOBILITY.max_step * 2.0

    def height(x, y):
        z = np.zeros_like(x)
        z = np.where(x > 4.0, climbable, z)
        z = np.where(x > 8.0, blocking, z)
        return z

    def texture(x, y):
        return 128.0 + 60.0 * _noise(x, y, 5.0, seed=2)

    def traversable(x, y):
        return x <= 8.0

    return Scene("step_field", height, texture, traversable)


def trench_field():
    """Gate 3. The governing case: a 0.5 m by 0.3 m trench.

    Placed at the range the mast camera can just see into, so the scene
    exercises the exact geometry section 02 is about.
    """
    near = 3.0

    def height(x, y):
        inside = (x > near) & (x < near + TRENCH.width)
        return np.where(inside, -TRENCH.depth, 0.0)

    def texture(x, y):
        return 128.0 + 60.0 * _noise(x, y, 5.0, seed=3)

    def traversable(x, y):
        return ~((x > near - 0.2) & (x < near + TRENCH.width + 0.2))

    return Scene("trench_field", height, texture, traversable)


def snow_scree():
    """Gate 4. The real risk. Textured scree in front, textureless snow beyond.

    The snow is geometrically flat and genuinely traversable. The point is
    not whether the estimator calls it an obstacle. The point is whether it
    admits it cannot tell, rather than reporting confident wrong depth.
    """
    snow_starts = 6.0

    def height(x, y):
        rough = 0.04 * _noise(x, y, 9.0, seed=4)
        return np.where(x < snow_starts, rough, 0.0)

    def texture(x, y):
        textured = 128.0 + 60.0 * _noise(x, y, 7.0, seed=5)
        # Fresh snow in full sun. Bright, and almost no contrast to match on.
        snow = 236.0 + 0.6 * _noise(x, y, 7.0, seed=6)
        return np.where(x < snow_starts, textured, snow)

    def traversable(x, y):
        return np.ones_like(x, dtype=bool)

    return Scene("snow_scree", height, texture, traversable)


SCENES = {
    scene.name: scene
    for scene in (flat_textured(), step_field(), trench_field(), snow_scree())
}


def camera_pose(height=None, tilt_deg=None):
    """Camera to body transform for the mast unit in its default pose.

    Composition of the downward tilt and the optical frame rotation, taken
    from the same convention the URDF and the solver use.
    """
    from ares_math.constants import OPTICAL_RPY

    height = SENSORS.mast_height if height is None else height
    tilt = SENSORS.mast_tilt_deg if tilt_deg is None else tilt_deg
    mount = rpy_matrix(0.0, np.radians(tilt), 0.0)
    optical = rpy_matrix(*OPTICAL_RPY)
    return mount @ optical, np.array([0.0, 0.0, height])


def ray_cast(scene, origin, directions, t_max=40.0, steps=400):
    """March each ray until it crosses the height field, then refine.

    Returns range along the ray, NaN where the ray never lands.
    """
    t = np.linspace(0.05, t_max, steps)[:, None]
    p = origin[None, None, :] + directions[None, :, :] * t[:, :, None]
    ground = scene.height(p[..., 0], p[..., 1])
    below = p[..., 2] <= ground

    first = np.argmax(below, axis=0)
    hit = below.any(axis=0)

    lo = np.where(first > 0, t[np.maximum(first - 1, 0), 0], t[0, 0])
    hi = t[np.minimum(first, steps - 1), 0]
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        q = origin[None, :] + directions * mid[:, None]
        under = q[:, 2] <= scene.height(q[:, 0], q[:, 1])
        hi = np.where(under, mid, hi)
        lo = np.where(under, lo, mid)
    out = 0.5 * (lo + hi)
    return np.where(hit, out, np.nan)


def render(scene, intrinsics=None, pose=None, eye="left"):
    """Render one eye. Returns (image uint8, true depth float, world points)."""
    k = intrinsics or render_intrinsics()
    rotation, translation = pose or camera_pose()

    offset = {"left": +k.baseline / 2.0, "right": -k.baseline / 2.0}[eye]
    origin = translation + rotation @ np.array([0.0, 0.0, 0.0]) + np.array(
        [0.0, offset, 0.0]
    )

    vs, us = np.mgrid[0 : k.height, 0 : k.width]
    x = (us.ravel() - k.cx) / k.fx
    y = (vs.ravel() - k.cy) / k.fy
    optical = np.stack([x, y, np.ones_like(x)], axis=1)
    optical /= np.linalg.norm(optical, axis=1, keepdims=True)
    directions = optical @ rotation.T

    ranges = ray_cast(scene, origin, directions)
    points = origin[None, :] + directions * ranges[:, None]

    intensity = np.where(
        np.isfinite(ranges),
        scene.texture(points[:, 0], points[:, 1]),
        20.0,  # sky
    )
    image = np.clip(intensity, 0, 255).astype(np.uint8).reshape(k.height, k.width)

    # Depth along the optical axis, which is what disparity measures, not
    # range along the ray.
    axis_depth = ranges * optical[:, 2]
    return (
        image,
        axis_depth.reshape(k.height, k.width),
        points.reshape(k.height, k.width, 3),
    )


def stereo_pair(scene, intrinsics=None, pose=None):
    """Both eyes plus the true depth of the left one."""
    k = intrinsics or render_intrinsics()
    p = pose or camera_pose()
    left, depth, points = render(scene, k, p, eye="left")
    right, _, _ = render(scene, k, p, eye="right")
    return left, right, depth, points


def lidar_sample(scene, pose=None, rings=64, azimuths=512, fov_deg=45.0, sigma=0.02):
    """A sparse spinning LiDAR over the same scene, as the reference source.

    Sparse and accurate, which is the trade the ablation is measuring. The
    noise is a plausible range sigma for a small unit, not a datasheet value,
    and it is only used for the synthetic path.
    """
    rotation, translation = pose or camera_pose()
    del rotation
    elevation = np.radians(np.linspace(-fov_deg, 2.0, rings))
    azimuth = np.radians(np.linspace(-fov_deg, fov_deg, azimuths))
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    directions = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
    ).reshape(-1, 3)

    ranges = ray_cast(scene, translation, directions)
    rng = np.random.default_rng(7)
    ranges = ranges + rng.normal(0.0, sigma, size=ranges.shape)
    hit = np.isfinite(ranges)
    return translation[None, :] + directions[hit] * ranges[hit, None]


def truth_grid(scene, spec):
    """Ground truth traversability on the estimator's own grid."""
    xs, ys = spec.centres()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return scene.traversable(gx, gy)
