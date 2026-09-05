"""Scoring. What accuracy you lose without LiDAR, by terrain class.

The headline number this whole phase exists to produce is a delta, not an
absolute. Both depth sources run through the same estimator on the same
frames, so the difference between them is attributable to the sensor and to
nothing else.

One metric matters more than the rest and it is not accuracy. It is
false free space: a cell that is really an obstacle, called traversable.
Accuracy averages that away against a large easy background. This is a
vehicle, so it is reported on its own and it is never averaged.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from ares_math.stereo import depth_error

from .traversability import OBSTACLE, TRAVERSABLE, UNKNOWN


@dataclass
class Confusion:
    """Traversability outcome against ground truth, cell counts."""

    true_traversable_called_traversable: int = 0
    true_traversable_called_obstacle: int = 0
    true_traversable_called_unknown: int = 0
    true_obstacle_called_obstacle: int = 0
    true_obstacle_called_traversable: int = 0
    true_obstacle_called_unknown: int = 0

    @property
    def total(self):
        return sum(asdict(self).values())

    @property
    def decided(self):
        return self.total - self.true_traversable_called_unknown - (
            self.true_obstacle_called_unknown
        )

    @property
    def accuracy(self):
        """Over cells the estimator was willing to decide."""
        if self.decided == 0:
            return float("nan")
        correct = (
            self.true_traversable_called_traversable + self.true_obstacle_called_obstacle
        )
        return correct / self.decided

    @property
    def obstacle_recall(self):
        """Of the obstacles that were decided, how many were called obstacles."""
        decided = (
            self.true_obstacle_called_obstacle + self.true_obstacle_called_traversable
        )
        if decided == 0:
            return float("nan")
        return self.true_obstacle_called_obstacle / decided

    @property
    def false_free_space_rate(self):
        """An obstacle called traversable. The one that ends a vehicle.

        Measured against every true obstacle cell, including the ones the
        estimator declined to decide, because declining is safe and being
        wrong is not.
        """
        obstacles = (
            self.true_obstacle_called_obstacle
            + self.true_obstacle_called_traversable
            + self.true_obstacle_called_unknown
        )
        if obstacles == 0:
            return float("nan")
        return self.true_obstacle_called_traversable / obstacles

    @property
    def unknown_rate(self):
        if self.total == 0:
            return float("nan")
        return (
            self.true_traversable_called_unknown + self.true_obstacle_called_unknown
        ) / self.total

    def as_dict(self):
        return {
            **asdict(self),
            "accuracy": self.accuracy,
            "obstacle_recall": self.obstacle_recall,
            "false_free_space_rate": self.false_free_space_rate,
            "unknown_rate": self.unknown_rate,
        }


def score_traversability(state, truth, mask=None):
    """Compare a state grid to a boolean truth grid of the same shape."""
    state = np.asarray(state)
    truth = np.asarray(truth, dtype=bool)
    if state.shape != truth.shape:
        raise ValueError(f"shape mismatch, {state.shape} against {truth.shape}")
    if mask is None:
        mask = np.ones_like(truth, dtype=bool)

    t = truth & mask
    o = (~truth) & mask
    return Confusion(
        true_traversable_called_traversable=int(np.sum(t & (state == TRAVERSABLE))),
        true_traversable_called_obstacle=int(np.sum(t & (state == OBSTACLE))),
        true_traversable_called_unknown=int(np.sum(t & (state == UNKNOWN))),
        true_obstacle_called_obstacle=int(np.sum(o & (state == OBSTACLE))),
        true_obstacle_called_traversable=int(np.sum(o & (state == TRAVERSABLE))),
        true_obstacle_called_unknown=int(np.sum(o & (state == UNKNOWN))),
    )


def score_by_range(state, truth, spec, edges=(0, 2, 4, 6, 8, 12, 20)):
    """The same score, banded by range, because that is where stereo differs.

    Depth error grows with the square of range, so a single number over the
    whole grid hides exactly the effect being measured.
    """
    ranges = spec.ranges()
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = (ranges >= lo) & (ranges < hi)
        if not band.any():
            continue
        out[f"{lo}-{hi} m"] = score_traversability(state, truth, mask=band).as_dict()
    return out


@dataclass
class DepthBand:
    low: float
    high: float
    count: int
    bias: float
    sigma: float
    predicted_sigma: float

    @property
    def ratio(self):
        if self.predicted_sigma == 0:
            return float("nan")
        return self.sigma / self.predicted_sigma


def score_depth(measured, truth, intrinsics, confidence=None, floor=0.3,
                edges=(1, 2, 3, 5, 7, 10, 15, 20), sigma_px=0.25):
    """Measured depth error against range, and against what theory predicts.

    This is the phase 4 gate brought forward: if the measured error does not
    track the section 03.2 curve, either the matcher or the calibration is
    wrong, and no amount of downstream tuning will fix it.
    """
    measured = np.asarray(measured, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(truth) & (truth > 0)
    if confidence is not None:
        valid &= np.asarray(confidence) >= floor

    err = measured[valid] - truth[valid]
    rng = truth[valid]

    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (rng >= lo) & (rng < hi)
        if sel.sum() < 100:
            continue
        centre = float(np.mean(rng[sel]))
        bands.append(
            DepthBand(
                low=float(lo),
                high=float(hi),
                count=int(sel.sum()),
                bias=float(np.mean(err[sel])),
                sigma=float(np.std(err[sel])),
                predicted_sigma=float(
                    depth_error(
                        Z=centre,
                        f=intrinsics.fx,
                        B=intrinsics.baseline,
                        d_disp=sigma_px,
                    )
                ),
            )
        )
    return bands


@dataclass
class Ablation:
    """The result the build spec asks phase 0 to produce."""

    scene: str
    lidar: dict
    stereo: dict
    lidar_by_range: dict = field(default_factory=dict)
    stereo_by_range: dict = field(default_factory=dict)
    depth_bands: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)

    @property
    def accuracy_lost(self):
        """The headline. Positive means the LiDAR was better."""
        return self.lidar["accuracy"] - self.stereo["accuracy"]

    @property
    def false_free_space_added(self):
        """The number that decides whether this vehicle is safe without LiDAR."""
        return (
            self.stereo["false_free_space_rate"] - self.lidar["false_free_space_rate"]
        )

    def as_dict(self):
        return {
            "scene": self.scene,
            "lidar": self.lidar,
            "stereo": self.stereo,
            "lidar_by_range": self.lidar_by_range,
            "stereo_by_range": self.stereo_by_range,
            "depth_bands": [asdict(b) | {"ratio": b.ratio} for b in self.depth_bands],
            "confidence": self.confidence,
            "accuracy_lost": self.accuracy_lost,
            "false_free_space_added": self.false_free_space_added,
        }
