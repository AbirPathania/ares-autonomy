"""Stereo depth, its error, baseline selection, and the sync budget.

Build spec section 03.2.

Depth error grows with the square of range. That single fact governs baseline
selection, and it is why "how far can this camera see" is a meaningless
question without an accuracy target attached.

Parameter names here are the symbols the build spec writes the formulas in
(Z, f, B, d_disp) rather than prose names, so a reader can check the code
against the document line by line.
"""

import math

from .units import MPS, PX, RADPS, M, S, units

__all__ = [
    "depth_from_disparity",
    "disparity_from_depth",
    "depth_error",
    "depth_error_array",
    "range_at_error",
    "min_range",
    "baseline_for_error_at_range",
    "sync_skew_budget",
    "skew_disparity_error",
    "skew_depth_bias",
]


@units(None, (PX, M, PX))
def depth_from_disparity(f, B, disp):
    """Depth from disparity.

        Z = f * B / disp

    f: focal length, pixels. B: baseline, metres. disp: disparity, pixels.
    Returns metres.
    """
    if disp <= 0:
        raise ValueError("disp must be positive")
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    return f * B / disp


@units(None, (PX, M, M))
def disparity_from_depth(f, B, Z):
    """Inverse of depth_from_disparity. Returns pixels."""
    if Z <= 0:
        raise ValueError("Z must be positive")
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    return f * B / Z


@units(None, (M, PX, M, PX))
def depth_error(Z, f, B, d_disp):
    """One sigma depth error from one sigma disparity uncertainty.

        dZ = Z**2 * d_disp / (f * B)

    Follows from differentiating Z = f*B/disp with respect to disp and
    substituting disp = f*B/Z. Derived symbolically in
    tools/derive_symbolic.py and asserted equal to this closed form.

    Returns metres.
    """
    if Z <= 0:
        raise ValueError("Z must be positive")
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    if d_disp < 0:
        raise ValueError("d_disp must not be negative")
    return Z**2 * d_disp / (f * B)


def depth_error_array(Z, f, B, d_disp):
    """depth_error over an array of ranges.

    Same relation, elementwise, for callers that need it per grid cell rather
    than per number. It lives here rather than being written out at the call
    site because section 11 does not make an exception for a loop being slow:
    the traversability layer needs the section 03.2 curve evaluated over a
    hundred thousand cells at 5 Hz, and calling the scalar form that many
    times is not an argument for having a second copy of the equation.

    Z is array-like, metres. Returns a numpy array, metres.
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    if d_disp < 0:
        raise ValueError("d_disp must not be negative")
    if np.any(Z <= 0):
        raise ValueError("every Z must be positive")
    return Z**2 * d_disp / (f * B)


@units(None, (M, PX, M, PX))
def range_at_error(target_error, f, B, d_disp):
    """Range at which depth error reaches target_error.

        Z = sqrt(target_error * f * B / d_disp)

    Returns metres.
    """
    if target_error <= 0:
        raise ValueError("target_error must be positive")
    if d_disp <= 0:
        raise ValueError("d_disp must be positive to have a finite answer")
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    return math.sqrt(target_error * f * B / d_disp)


@units(None, (PX, M, PX))
def min_range(f, B, max_disparity_px):
    """Closest range a matcher with a bounded disparity search can see.

        Z_min = f * B / d_max

    A 0.30 m baseline with a 192 pixel search cannot see anything closer
    than 1.00 m, which is why the second camera unit exists.
    Returns metres.
    """
    if max_disparity_px <= 0:
        raise ValueError("max_disparity_px must be positive")
    if f <= 0 or B <= 0:
        raise ValueError("f and B must be positive")
    return f * B / max_disparity_px


@units(None, (M, M, PX, PX))
def baseline_for_error_at_range(target_error, Z, f, d_disp):
    """Baseline needed to hold target_error at range Z.

        B = Z**2 * d_disp / (f * target_error)

    The inverse of depth_error, solved for B. Returns metres.
    """
    if target_error <= 0:
        raise ValueError("target_error must be positive")
    if Z <= 0:
        raise ValueError("Z must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    return Z**2 * d_disp / (f * target_error)


# --------------------------------------------------------------------------
# Stereo synchronisation.
#
# Two eyes only fuse if they look at the same instant. If the right camera
# exposes dt after the left while the vehicle is moving, the pair no longer
# shares a rigid baseline and the matcher solves a geometry that never
# existed. Both error terms below are exact for a rigid vehicle motion,
# and both are derived in tools/derive_symbolic.py.
# --------------------------------------------------------------------------


@units(None, (MPS, S))
def skew_depth_bias(v, skew):
    """Depth bias from exposure skew under pure forward translation.

        bias = v * skew

    If the right camera fires dt late, its optical centre has advanced by
    v*dt, so a point on the optical axis is reconstructed at Z - v*dt.
    The bias is independent of range, focal length and baseline.
    Returns metres.
    """
    if skew < 0:
        raise ValueError("skew must not be negative, order is handled by the caller")
    return v * skew


@units(None, (PX, RADPS, S))
def skew_disparity_error(f, yaw_rate, skew):
    """Disparity error from exposure skew under vehicle rotation.

        d_disp = f * tan(yaw_rate * skew)

    A yaw of yaw_rate*skew between the two exposures shifts the image, and
    the matcher reads that shift as disparity. Exact on the optical axis,
    derived in tools/derive_symbolic.py. For the angles that matter here it
    is indistinguishable from the small angle form f*yaw_rate*skew, but the
    exact version inverts cleanly against sync_skew_budget.

    This is the dominant sync term: it enters depth multiplied by
    Z**2 / (f*B), while the translation term does not scale with range at
    all. At f=640, B=0.20, Z=5 and a 0.5 rad/s turn, one millisecond of
    skew costs 62 mm of depth. Returns pixels.
    """
    angle = abs(yaw_rate) * skew
    if not -math.pi / 2.0 < angle < math.pi / 2.0:
        raise ValueError("yaw_rate * skew must be less than a quarter turn")
    return f * math.tan(angle)


@units(None, (PX, PX, RADPS))
def sync_skew_budget(disp_budget_px, f, yaw_rate):
    """Largest tolerable inter camera exposure skew.

        skew_max = atan(disp_budget / f) / yaw_rate

    The exact inverse of skew_disparity_error. Invert it with the disparity
    budget you are already spending on the matcher, typically a fraction of
    the sigma in section 03.2, and it gives the hardware trigger requirement
    in seconds. This is the number that decides whether a software timestamp
    match is good enough or whether the two cameras need a shared trigger
    line. Returns seconds, or math.inf when the vehicle is not rotating.
    """
    if disp_budget_px <= 0:
        raise ValueError("disp_budget_px must be positive")
    if f <= 0:
        raise ValueError("f must be positive")
    if yaw_rate == 0.0:
        return math.inf
    return math.atan(disp_budget_px / f) / abs(yaw_rate)
