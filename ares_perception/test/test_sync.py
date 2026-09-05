"""Stereo synchronisation, the two-eyes property.

Build spec section 03.2 and the sync derivation in tools/derive_symbolic.py.
No ROS here: the synchroniser is plain Python so the behaviour that keeps the
two cameras honest is testable in microseconds.
"""

import math

import pytest

from ares_math.constants import CAMERA, SENSORS
from ares_math.optics import focal_px
from ares_math.stereo import depth_error, skew_disparity_error
from ares_perception.sync import (
    SKEW_DISPARITY_SHARE,
    StereoSynchroniser,
    skew_budget_for,
)

F = focal_px(CAMERA.width_px, CAMERA.hfov_deg)


def make(max_skew=0.005):
    return StereoSynchroniser(focal_px=F, max_skew=max_skew)


def test_a_matched_pair_comes_out_once():
    s = make()
    assert s.push_left(1.000, "L") is None
    out = s.push_right(1.0001, "R")
    assert out is not None
    stamp, left, right, skew = out
    assert (left, right) == ("L", "R")
    assert stamp == pytest.approx(1.00005)
    assert skew == pytest.approx(1e-4, abs=1e-12)
    assert s.stats.paired == 1


def test_a_pair_beyond_the_hard_ceiling_is_refused():
    """A gap larger than a frame interval is a bookkeeping error, not motion."""
    s = make(max_skew=0.005)
    s.push_left(1.0, "L")
    assert s.push_right(1.020, "R") is None
    assert s.stats.rejected_skew == 1
    assert s.stats.paired == 0


def test_turning_tightens_the_budget():
    """Standing still any skew is harmless. Turning, a millisecond is not."""
    still = skew_budget_for(0.0, F)
    slow = skew_budget_for(0.1, F)
    brisk = skew_budget_for(1.0, F)
    assert still == math.inf
    assert slow > brisk
    assert brisk == pytest.approx(
        math.atan(SKEW_DISPARITY_SHARE * CAMERA.disparity_sigma_px / F) / 1.0,
        rel=1e-9,
    )


def test_a_pair_that_is_fine_when_straight_is_refused_in_a_turn():
    """The behaviour the whole module exists for."""
    skew = 6e-5

    straight = make()
    straight.set_motion(speed=1.111, yaw_rate=0.0)
    straight.push_left(1.0, "L")
    assert straight.push_right(1.0 + skew, "R") is not None

    turning = make()
    turning.set_motion(speed=1.111, yaw_rate=1.0)
    turning.push_left(1.0, "L")
    assert turning.push_right(1.0 + skew, "R") is None
    assert turning.stats.rejected_skew == 1


def test_the_budget_is_a_tenth_of_the_matcher_sigma():
    """Skew must be a rounding error against the matcher, not a rival to it."""
    yaw = 1.0
    budget = skew_budget_for(yaw, F)
    spent = skew_disparity_error(F, yaw, budget)
    assert spent == pytest.approx(
        SKEW_DISPARITY_SHARE * CAMERA.disparity_sigma_px, rel=1e-9
    )
    assert spent < CAMERA.disparity_sigma_px / 5.0


def test_rejecting_is_cheaper_than_accepting():
    """States the cost of the alternative, so the threshold is arguable.

    One millisecond of skew in a half radian per second turn, at 5 m, on the
    mast baseline. If this ever stops being large, the threshold can be
    relaxed on evidence rather than on feel.
    """
    s = make()
    s.set_motion(speed=1.111, yaw_rate=0.5)
    cost = s.error_if_accepted(skew=1e-3, depth=5.0, baseline=SENSORS.mast_baseline)
    assert cost["translation_bias"] == pytest.approx(1.111e-3, rel=1e-9)
    assert cost["rotation_depth_error"] > 0.05
    assert cost["rotation_depth_error"] > 50 * cost["translation_bias"]


def test_rotation_dominates_translation_at_range():
    """Why the budget is written against yaw rate and not against speed."""
    skew = 1e-3
    disparity = skew_disparity_error(F, 0.5, skew)
    rotation = depth_error(
        Z=5.0, f=F, B=SENSORS.mast_baseline, d_disp=disparity
    )
    translation = 1.111 * skew
    assert rotation > 50 * translation


def test_the_closest_pair_wins_not_the_newest():
    s = make()
    s.push_left(1.000, "L0")
    s.push_left(1.010, "L1")
    out = s.push_right(1.0099, "R")
    assert out is not None
    assert out[1] == "L1"


def test_an_unmatched_backlog_is_counted_not_hidden():
    """A stack that silently drops half its pairs looks fine from outside."""
    s = StereoSynchroniser(focal_px=F, queue_size=3)
    for i in range(6):
        s.push_left(float(i), f"L{i}")
    assert s.stats.dropped_unmatched == 3


def test_the_rejection_rate_is_observable():
    s = make(max_skew=1e-4)
    s.push_left(1.0, "L")
    s.push_right(1.00005, "R")
    s.push_left(2.0, "L")
    s.push_right(2.010, "R")
    assert s.stats.paired == 1
    assert s.stats.rejected_skew == 1
    assert s.stats.rejection_rate == pytest.approx(0.5)
    assert "worst_skew" in s.stats.as_dict()


def test_a_ten_microsecond_hardware_trigger_clears_the_worst_case():
    """The hardware requirement, stated as a test so it survives procurement.

    At one radian per second, a brisk turn for a 4 km/h vehicle, and spending
    a tenth of the matcher's own disparity sigma, the budget is 39
    microseconds. A shared hardware trigger line clears that with about four
    times to spare. Software timestamp matching over a network is a
    millisecond at best, which is 25 times over budget, and that is the whole
    reason this number is written down rather than assumed.
    """
    budget = skew_budget_for(1.0, F)
    assert budget == pytest.approx(3.90625e-5, rel=1e-4)
    assert budget > 3 * 10e-6, "a 10 us hardware trigger clears it"
    assert budget < 1e-3, "a 1 ms software timestamp match does not"
