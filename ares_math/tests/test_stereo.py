"""Stereo depth, its error, and the sync budget. Build spec section 03.2.

Golden assertions come from golden_values.yaml, which was generated from the
document by tools/generate_golden.py, not from this package.
"""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pint import DimensionalityError

from ares_math.optics import focal_px
from ares_math.stereo import (
    baseline_for_error_at_range,
    depth_error,
    depth_from_disparity,
    disparity_from_depth,
    min_range,
    range_at_error,
    skew_depth_bias,
    skew_disparity_error,
    sync_skew_budget,
)
from ares_math.units import Q_

from .conftest import GOLD, REL


def test_focal_matches_document():
    assert focal_px(1280, 90.0) == pytest.approx(640.0, rel=REL)


@pytest.mark.parametrize("row", GOLD["depth_error_table"])
def test_depth_error_table(row):
    got = depth_error(Z=row["Z"], f=row["f"], B=row["B"], d_disp=row["d_disp"])
    assert got == pytest.approx(row["expected"], rel=REL)


@pytest.mark.parametrize("row", GOLD["depth_error_table"])
def test_depth_error_table_rounds_to_the_printed_document(row):
    assert round(row["expected"], 3) == pytest.approx(row["document"], abs=5e-4)


def test_error_grows_as_range_squared():
    a = depth_error(Z=5.0, f=640.0, B=0.2, d_disp=0.25)
    b = depth_error(Z=10.0, f=640.0, B=0.2, d_disp=0.25)
    assert b / a == pytest.approx(4.0, rel=REL)


@pytest.mark.parametrize("row", GOLD["baseline_table"])
def test_range_at_ten_centimetre_error(row):
    got = range_at_error(row["target_error"], row["f"], row["B"], 0.25)
    assert got == pytest.approx(row["range_at_10cm"], rel=REL)


@pytest.mark.parametrize("row", GOLD["baseline_table"])
def test_range_at_ten_centimetre_error_pessimistic(row):
    got = range_at_error(row["target_error"], row["f"], row["B"], 0.50)
    assert got == pytest.approx(row["range_at_10cm_pessimistic"], rel=REL)


@pytest.mark.parametrize("row", GOLD["baseline_table"])
def test_minimum_range(row):
    got = min_range(row["f"], row["B"], row["max_disparity_px"])
    assert got == pytest.approx(row["min_range"], rel=REL)


def test_the_reason_the_second_camera_exists():
    """A 0.30 m baseline cannot see closer than 1.00 m with a 192 px search.

    Build spec section 03.2, and the whole justification for the deck unit.
    """
    assert min_range(640.0, 0.30, 192) == pytest.approx(1.00, rel=REL)
    assert min_range(640.0, 0.12, 192) == pytest.approx(0.40, rel=REL)


def test_depth_and_disparity_are_inverses():
    z = depth_from_disparity(640.0, 0.20, 64.0)
    assert disparity_from_depth(640.0, 0.20, z) == pytest.approx(64.0, rel=REL)


def test_baseline_inverts_depth_error():
    b = baseline_for_error_at_range(0.10, 5.0, 640.0, 0.25)
    assert depth_error(Z=5.0, f=640.0, B=b, d_disp=0.25) == pytest.approx(0.10, rel=REL)


# --------------------------------------------------------------------------
# Units carried, not assumed. Build spec section 11, mechanism 2.
# --------------------------------------------------------------------------


def test_a_baseline_in_millimetres_is_not_a_baseline_in_metres():
    metres = depth_error(Z=5.0, f=640.0, B=0.20, d_disp=0.25)
    millimetres = depth_error(Z=5.0, f=640.0, B=Q_(200, "mm"), d_disp=0.25)
    assert millimetres == pytest.approx(metres, rel=REL)


def test_a_length_where_a_disparity_belongs_is_rejected():
    with pytest.raises(DimensionalityError):
        depth_error(Z=5.0, f=640.0, B=0.20, d_disp=Q_(0.25, "meter"))


def test_a_range_in_feet_converts():
    assert depth_error(Z=Q_(1.0, "meter"), f=640.0, B=0.20, d_disp=0.25) == pytest.approx(
        depth_error(Z=Q_(100.0, "cm"), f=640.0, B=0.20, d_disp=0.25), rel=REL
    )


# --------------------------------------------------------------------------
# Synchronisation. Two eyes only fuse if they look at the same instant.
# --------------------------------------------------------------------------


def test_sync_budget_inverts_the_skew_error_exactly():
    budget_px = 0.05
    yaw = 0.5
    dt = sync_skew_budget(budget_px, 640.0, yaw)
    assert skew_disparity_error(640.0, yaw, dt) == pytest.approx(budget_px, rel=REL)


def test_a_stationary_vehicle_has_no_rotation_sync_requirement():
    assert sync_skew_budget(0.05, 640.0, 0.0) == math.inf


def test_rotation_dominates_translation_at_range():
    """The number that decides hardware trigger versus software timestamp.

    At the ARES working case, one millisecond of skew during a half radian
    per second turn costs far more depth at 5 m than the same millisecond of
    forward motion at 4 km/h does.
    """
    skew = 1e-3
    translation = skew_depth_bias(1.111, skew)
    disp = skew_disparity_error(640.0, 0.5, skew)
    rotation = depth_error(Z=5.0, f=640.0, B=0.20, d_disp=disp)
    assert translation == pytest.approx(1.111e-3, rel=REL)
    assert rotation > 50 * translation


def test_a_ten_microsecond_trigger_holds_a_tenth_pixel():
    """States the hardware requirement as a test so it cannot be forgotten."""
    worst_case_yaw = 1.0  # rad/s, a brisk turn for a 4 km/h vehicle
    dt = sync_skew_budget(0.1, 640.0, worst_case_yaw)
    assert dt == pytest.approx(1.5625e-4, rel=1e-6)
    assert dt > 1e-5, "a 10 us hardware trigger clears this with an order to spare"


# --------------------------------------------------------------------------
# Property tests. Build spec section 11, mechanism 4.
# --------------------------------------------------------------------------

positive_range = st.floats(min_value=0.1, max_value=500.0, allow_nan=False)
focal = st.floats(min_value=100.0, max_value=4000.0, allow_nan=False)
baseline = st.floats(min_value=0.01, max_value=2.0, allow_nan=False)
sigma = st.floats(min_value=1e-3, max_value=5.0, allow_nan=False)


@given(z1=positive_range, z2=positive_range, f=focal, b=baseline, s=sigma)
def test_depth_error_is_strictly_increasing_in_range(z1, z2, f, b, s):
    assume(abs(z1 - z2) > 1e-6)
    lo, hi = sorted((z1, z2))
    assert depth_error(Z=hi, f=f, B=b, d_disp=s) > depth_error(Z=lo, f=f, B=b, d_disp=s)


@given(z=positive_range, f=focal, b=baseline, s=sigma)
def test_depth_error_scales_as_the_square_of_range(z, f, b, s):
    single = depth_error(Z=z, f=f, B=b, d_disp=s)
    double = depth_error(Z=2 * z, f=f, B=b, d_disp=s)
    assert double == pytest.approx(4.0 * single, rel=1e-12)


@given(z=positive_range, f=focal, b=baseline)
def test_depth_disparity_round_trip(z, f, b):
    disp = disparity_from_depth(f, b, z)
    assume(disp > 1e-9)
    assert depth_from_disparity(f, b, disp) == pytest.approx(z, rel=1e-9)


@given(
    err=st.floats(min_value=1e-3, max_value=10.0),
    f=focal,
    b=baseline,
    s=sigma,
)
@settings(max_examples=200)
def test_range_at_error_inverts_depth_error(err, f, b, s):
    z = range_at_error(err, f, b, s)
    assert depth_error(Z=z, f=f, B=b, d_disp=s) == pytest.approx(err, rel=1e-9)


@given(
    yaw=st.floats(min_value=1e-3, max_value=3.0),
    dt=st.floats(min_value=1e-6, max_value=0.05),
    f=focal,
)
def test_sync_budget_round_trip(yaw, dt, f):
    disp = skew_disparity_error(f, yaw, dt)
    assume(disp > 1e-12)
    assert sync_skew_budget(disp, f, yaw) == pytest.approx(dt, rel=1e-9)
