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


def _neighbour_slope(z, cell):
    """Steepest local gradient in degrees, from a central difference."""
    padded = np.pad(z, 1, mode="edge")
    dzdx = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * cell)
    dzdy = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * cell)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def _neighbour_step(z, cell):
    """Largest height jump to any of the four neighbours, metres."""
    del cell
    padded = np.pad(z, 1, mode="edge")
    steps = [
        np.abs(padded[2:, 1:-1] - z),
        np.abs(padded[:-2, 1:-1] - z),
        np.abs(padded[1:-1, 2:] - z),
        np.abs(padded[1:-1, :-2] - z),
    ]
    return np.nanmax(np.stack(steps), axis=0)


def classify(surface, confidence_floor=0.5, slope_limit=None, step_limit=None):
    """Height statistics to the three way traversability decision.

    slope_limit defaults to the vehicle's working slope, derived from its own
    centre of mass and track by ares_math.kinematics.stability_limits.
    step_limit defaults to the obstacle claim in constants.py.

    Returns (state grid, diagnostics dict).
    """
    if slope_limit is None:
        slope_limit = stability_limits()["working"]
    if step_limit is None:
        step_limit = MOBILITY.max_step

    cell = surface.spec.cell
    z = np.where(surface.observed, surface.z_mean, np.nan)
    filled = np.where(np.isnan(z), np.nanmedian(z) if np.any(~np.isnan(z)) else 0.0, z)

    slope = _neighbour_slope(filled, cell)
    step = _neighbour_step(filled, cell)
    # Within-cell spread. A rock and the ground beside it land in one cell at
    # range, and this is what catches that.
    with np.errstate(invalid="ignore"):
        roughness = np.where(
            surface.observed, surface.z_max - surface.z_min, np.nan
        )

    state = np.full(surface.count.shape, UNKNOWN, dtype=np.uint8)
    measured = surface.observed & (surface.confidence >= confidence_floor)

    blocked = (
        (slope > slope_limit)
        | (step > step_limit)
        | (np.nan_to_num(roughness, nan=0.0) > step_limit)
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
    }


def estimate(points, spec=None, confidence=None, confidence_floor=0.5):
    """Points to a traversability grid. The whole estimator, one call."""
    surface = rasterise(points, spec=spec, confidence=confidence)
    state, diagnostics = classify(surface, confidence_floor=confidence_floor)
    return state, surface, diagnostics
