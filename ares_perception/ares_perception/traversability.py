"""The traversability estimator, shared by both depth sources.

This is the piece that makes the study an ablation rather than a comparison
of two different systems. The LiDAR run and the stereo run put their points
through this identical code. Only the depth source changes.

Three states, and the third one is the whole point:

    TRAVERSABLE   measured, and within the vehicle's limits
    OBSTACLE      measured, and outside them
    UNKNOWN       not measured well enough to say

A stack that folds UNKNOWN into TRAVERSABLE will drive into a snowdrift at
full speed. Build spec section 09. Unknown is a first class output here, it
is scored separately, and downstream it caps speed rather than being ignored.

Every vehicle limit comes from ares_math. Nothing in this file is a physical
number.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from ares_math.constants import CHASSIS, MOBILITY
from ares_math.kinematics import stability_limits

UNKNOWN = 0
TRAVERSABLE = 1
OBSTACLE = 2

STATE_NAMES = {UNKNOWN: "unknown", TRAVERSABLE: "traversable", OBSTACLE: "obstacle"}

# A cell must be smaller than the wheel contact patch, or a step the wheel
# would climb gets averaged away with the ground beside it. Derived from the
# track rather than chosen, so it follows the vehicle if the hull revises.
CELL_SIZE = CHASSIS.track / 7.0

# A cell needs enough returns for its height statistics to mean anything.
# Below this it is UNKNOWN, which is the honest answer.
MIN_POINTS_PER_CELL = 3


@dataclass(frozen=True)
class GridSpec:
    """A body frame 2.5D grid. X forward, Y left, metres."""

    x_min: float = 0.0
    x_max: float = 20.0
    y_min: float = -10.0
    y_max: float = 10.0
    cell: float = CELL_SIZE

    @property
    def shape(self):
        nx = int(round((self.x_max - self.x_min) / self.cell))
        ny = int(round((self.y_max - self.y_min) / self.cell))
        return nx, ny

    def index(self, x, y):
        """Cell indices for point arrays. Returns (ix, iy, inside)."""
        ix = np.floor((x - self.x_min) / self.cell).astype(np.int64)
        iy = np.floor((y - self.y_min) / self.cell).astype(np.int64)
        nx, ny = self.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return ix, iy, inside

    def centres(self):
        nx, ny = self.shape
        xs = self.x_min + (np.arange(nx) + 0.5) * self.cell
        ys = self.y_min + (np.arange(ny) + 0.5) * self.cell
        return xs, ys

    def ranges(self):
        """Radial range of each cell centre from the vehicle datum."""
        xs, ys = self.centres()
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.hypot(gx, gy)


@dataclass
class Surface:
    """Rasterised height statistics. Everything downstream reads from here."""

    spec: GridSpec
    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    confidence: np.ndarray

    @property
    def observed(self):
        return self.count >= MIN_POINTS_PER_CELL


def rasterise(points, spec=None, confidence=None):
    """Points to a 2.5D height grid.

    points: (N, 3) in the body frame, metres, X forward, Y left, Z up.
    confidence: (N,) in [0, 1], or None to treat every point as certain.

    Cells with too few returns stay UNKNOWN. That is what makes a sparse
    LiDAR and a dense but unreliable stereo cloud comparable: both are
    judged on whether they actually measured the cell.
    """
    spec = spec or GridSpec()
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if confidence is None:
        confidence = np.ones(len(points))
    confidence = np.asarray(confidence, dtype=float).reshape(-1)

    nx, ny = spec.shape
    count = np.zeros((nx, ny), dtype=np.int64)
    z_sum = np.zeros((nx, ny))
    conf_sum = np.zeros((nx, ny))
    z_min = np.full((nx, ny), np.inf)
    z_max = np.full((nx, ny), -np.inf)

    if len(points):
        ix, iy, inside = spec.index(points[:, 0], points[:, 1])
        ix, iy = ix[inside], iy[inside]
        z = points[inside, 2]
        c = confidence[inside]
        flat = ix * ny + iy
        np.add.at(count.reshape(-1), flat, 1)
        np.add.at(z_sum.reshape(-1), flat, z)
        np.add.at(conf_sum.reshape(-1), flat, c)
        np.minimum.at(z_min.reshape(-1), flat, z)
        np.maximum.at(z_max.reshape(-1), flat, z)

    with np.errstate(invalid="ignore", divide="ignore"):
        z_mean = np.where(count > 0, z_sum / np.maximum(count, 1), np.nan)
        conf = np.where(count > 0, conf_sum / np.maximum(count, 1), 0.0)
    z_min = np.where(count > 0, z_min, np.nan)
    z_max = np.where(count > 0, z_max, np.nan)

    return Surface(spec, count, z_min, z_max, z_mean, conf)


def _neighbour_differences(z, observed, cell):
    """Height jump and slope to each observed neighbour.

    Only observed pairs count. The obvious implementation fills the gaps with
    something plausible, a median or the nearest edge value, and then takes
    gradients over the filled grid. That manufactures a cliff at every
    boundary between measured and unmeasured ground, and since a camera
    measures a wedge and the grid is a rectangle, almost every measured cell
    sits on such a boundary. Measured in the simulator: 2.5 per cent of the
    grid came back OBSTACLE against 0.6 per cent traversable, on flat ground.

    A cell with no observed neighbour yields NaN, and NaN fails every
    threshold comparison, so it is not blocked. That is the intended reading:
    a lone measured cell surrounded by nothing is not evidence of an
    obstacle, it is evidence of not much.

    Returns (step in metres, slope in degrees), both NaN where unjudgeable.
    """
    masked = np.where(observed, z, np.nan)
    padded = np.pad(masked, 1, mode="constant", constant_values=np.nan)
    ahead = padded[2:, 1:-1]
    behind = padded[:-2, 1:-1]
    left = padded[1:-1, 2:]
    right = padded[1:-1, :-2]

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An all-NaN slice is the expected answer for a cell with no observed
        # neighbour, not something to warn about once per cell per frame.
        warnings.simplefilter("ignore", RuntimeWarning)
        # Step: the largest jump to any observed neighbour. This is the test
        # for a kerb or a rock edge, and one cell is the right separation.
        step = np.nanmax(
            np.abs(np.stack([ahead, behind, left, right]) - masked), axis=0
        )

        # Slope: a central difference over two cells where both sides are
        # observed, falling back to one sided where only one is. Two cells
        # rather than one on purpose. A single cell separation doubles the
        # gradient a given height difference produces, and at a 50 mm cell
        # that turns 20 mm of stereo noise into a 22 degree slope, which is
        # most of the way to the vehicle's 26.6 degree limit.
        dzdx = _gradient(ahead, behind, masked, cell)
        dzdy = _gradient(left, right, masked, cell)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    return step, slope


def _gradient(plus, minus, centre, cell):
    """Central difference where both sides are observed, one sided otherwise."""
    central = (plus - minus) / (2.0 * cell)
    forward = (plus - centre) / cell
    backward = (centre - minus) / cell
    out = np.where(np.isfinite(central), central, np.nan)
    out = np.where(np.isfinite(out), out, forward)
    out = np.where(np.isfinite(out), out, backward)
    return np.nan_to_num(out, nan=0.0)


def classify(
    surface,
    confidence_floor=0.5,
    slope_limit=None,
    step_limit=None,
    noise=None,
):
    """Height statistics to the three way traversability decision.

    slope_limit defaults to the vehicle's working slope, derived from its own
    centre of mass and track by ares_math.kinematics.stability_limits.
    step_limit defaults to the obstacle claim in constants.py.

    noise is the expected one sigma depth error at each cell, which the caller
    computes from its own optics through ares_math.stereo.depth_error. It
    matters: depth error grows with the square of range, so at 6 m on the
    simulator's optics a cell's height spread is around 0.14 m from
    measurement noise alone. Judged against a fixed threshold, distant ground
    becomes an obstacle field made entirely of stereo noise. The thresholds
    are widened by twice the local sigma so that a cell is only blocked when
    the geometry, and not the uncertainty, says so.

    Returns (state grid, diagnostics dict).
    """
    if slope_limit is None:
        slope_limit = stability_limits()["working"]
    if step_limit is None:
        step_limit = MOBILITY.max_step

    cell = surface.spec.cell
    z = np.where(surface.observed, surface.z_mean, np.nan)

    step, slope = _neighbour_differences(z, surface.observed, cell)
    # Within-cell spread. A rock and the ground beside it land in one cell at
    # range, and this is what catches that.
    with np.errstate(invalid="ignore"):
        roughness = np.where(
            surface.observed, surface.z_max - surface.z_min, np.nan
        )

    tolerance = np.zeros_like(z) if noise is None else 2.0 * np.asarray(noise)
    step_threshold = step_limit + tolerance
    slope_threshold = slope_limit + np.degrees(np.arctan(tolerance / cell))

    state = np.full(surface.count.shape, UNKNOWN, dtype=np.uint8)
    measured = surface.observed & (surface.confidence >= confidence_floor)

    # NaN fails every comparison, which is the behaviour wanted here: a cell
    # with nothing to compare against is not blocked.
    with np.errstate(invalid="ignore"):
        blocked = (
            (slope > slope_threshold)
            | (step > step_threshold)
            | (roughness > step_threshold)
        )
    state[measured & ~blocked] = TRAVERSABLE
    state[measured & blocked] = OBSTACLE

    return state, {
        "slope_deg": slope,
        "step_m": step,
        "roughness_m": roughness,
        "measured": measured,
        "slope_limit_deg": slope_limit,
        "step_limit_m": step_limit,
        "noise_tolerance_m": tolerance,
    }


def estimate(points, spec=None, confidence=None, confidence_floor=0.5, noise=None):
    """Points to a traversability grid. The whole estimator, one call."""
    surface = rasterise(points, spec=spec, confidence=confidence)
    state, diagnostics = classify(
        surface, confidence_floor=confidence_floor, noise=noise
    )
    return state, surface, diagnostics


def depth_noise_grid(spec, f_px, baseline, sigma_px=None):
    """Expected one sigma depth error at each cell, metres.

    The section 03.2 curve evaluated over the grid. Passed to classify so the
    thresholds widen with range instead of turning distant stereo noise into
    an obstacle field.
    """
    from ares_math.constants import CAMERA
    from ares_math.stereo import depth_error_array

    sigma = CAMERA.disparity_sigma_px if sigma_px is None else sigma_px
    ranges = np.maximum(spec.ranges(), 1e-3)
    return depth_error_array(ranges, f_px, baseline, sigma)
