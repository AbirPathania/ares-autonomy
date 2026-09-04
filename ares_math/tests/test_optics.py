"""Pinhole optics. Build spec sections 03.1, 03.4, 03.5 and 08."""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pint import DimensionalityError

from ares_math.constants import CAMERA, OPTICAL_RPY, SENSORS
from ares_math.optics import (
    far_edge_range,
    focal_px,
    hfov_from_focal,
    near_edge_range,
    pixels_on_target,
    project,
    range_for_pixels,
    unproject,
    vfov_from_focal,
)
from ares_math.units import Q_

from .conftest import GOLD, REL


@pytest.mark.parametrize("row", GOLD["focal_table"])
def test_focal_table(row):
    got = focal_px(row["width_px"], row["hfov_deg"])
    assert got == pytest.approx(row["expected"], rel=REL)


@pytest.mark.parametrize("row", GOLD["focal_table"])
def test_focal_table_matches_the_printed_document(row):
    assert round(row["expected"], 1) == pytest.approx(row["document"], abs=5e-2)


def test_focal_inverts_field_of_view():
    f = focal_px(1280, 90.0)
    assert hfov_from_focal(1280, f) == pytest.approx(90.0, rel=REL)


def test_a_field_of_view_in_radians_converts():
    assert focal_px(1280, Q_(math.pi / 2, "radian")) == pytest.approx(640.0, rel=REL)


def test_a_length_where_an_angle_belongs_is_rejected():
    with pytest.raises(DimensionalityError):
        focal_px(1280, Q_(90.0, "meter"))


@pytest.mark.parametrize("row", GOLD["blind_zone_table"])
def test_blind_zone_table(row):
    got = near_edge_range(row["h"], row["tilt_deg"], row["vfov_deg"])
    assert got == pytest.approx(row["expected"], rel=REL)


def test_the_blind_zone_the_deck_unit_exists_to_fill():
    """Mast at 0.95 m tilted 20 degrees sees nothing closer than 0.81 m.

    A hole big enough to drive a rover into. Build spec section 03.4.
    """
    mast = near_edge_range(SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg)
    deck = near_edge_range(SENSORS.deck_height, SENSORS.deck_tilt_deg, CAMERA.vfov_deg)
    assert mast == pytest.approx(0.8113766511902933, rel=REL)
    assert deck == pytest.approx(0.17171119177133773, rel=REL)
    assert deck < mast, "the deck unit must start inside the mast unit's blind zone"


def test_the_two_units_overlap():
    """Overlap between 0.81 and 1.0 m gives extrinsic calibration something
    to agree on. Build spec section 04.
    """
    from ares_math.stereo import min_range

    mast_blind = near_edge_range(
        SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg
    )
    mast_min = min_range(
        focal_px(CAMERA.width_px, CAMERA.hfov_deg),
        SENSORS.mast_baseline,
        CAMERA.max_disparity_px,
    )
    deck_blind = near_edge_range(
        SENSORS.deck_height, SENSORS.deck_tilt_deg, CAMERA.vfov_deg
    )
    near_limit = max(mast_blind, mast_min)
    assert near_limit == pytest.approx(0.8113766511902933, rel=1e-6)
    assert deck_blind < near_limit, "no overlap means nothing to calibrate against"


def test_a_camera_looking_at_the_horizon_has_no_far_edge():
    assert far_edge_range(0.95, 10.0, 59.0) == math.inf


def test_far_edge_is_finite_when_tilted_below_the_frustum_top():
    assert far_edge_range(0.95, 40.0, 59.0) == pytest.approx(
        0.95 / math.tan(math.radians(40.0 - 29.5)), rel=REL
    )


@pytest.mark.parametrize("row", GOLD["pixels_table"])
def test_pixels_on_target_table(row):
    got = pixels_on_target(row["H"], row["f"], row["Z"])
    assert got == pytest.approx(row["expected"], rel=REL)


@pytest.mark.parametrize("row", GOLD["pixels_table"])
def test_pixels_on_target_matches_the_printed_document(row):
    assert round(row["expected"]) == row["document"]


def test_resolution_is_not_the_problem():
    """Person detection is comfortable well past 40 m at the working case."""
    assert pixels_on_target(1.70, 640.0, 40.0) > 20.0
    assert pixels_on_target(0.30, 640.0, 20.0) >= 9.5


def test_pixels_and_range_are_inverses():
    n = pixels_on_target(1.70, 640.0, 12.0)
    assert range_for_pixels(1.70, 640.0, n) == pytest.approx(12.0, rel=REL)


def test_the_optical_frame_rotation_is_the_one_from_section_08():
    """Z forward, X right, Y down, relative to X forward, Y left, Z up.

    Wrong by 90 degrees is the bug that costs everyone two days.
    """
    roll, pitch, yaw = OPTICAL_RPY
    assert roll == pytest.approx(-math.pi / 2, abs=1e-9)
    assert pitch == 0.0
    assert yaw == pytest.approx(-math.pi / 2, abs=1e-9)


def test_vertical_field_of_view_of_the_working_case():
    f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
    assert vfov_from_focal(CAMERA.height_px, f) == pytest.approx(58.7156, abs=1e-3)


# --------------------------------------------------------------------------
# Property tests. Build spec section 11, mechanism 4.
# --------------------------------------------------------------------------

coord = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False)
depth = st.floats(min_value=0.05, max_value=200.0, allow_nan=False)
focal = st.floats(min_value=50.0, max_value=5000.0, allow_nan=False)
centre = st.floats(min_value=0.0, max_value=4000.0, allow_nan=False)


@given(x=coord, y=coord, z=depth, fx=focal, fy=focal, cx=centre, cy=centre)
def test_project_then_unproject_is_identity(x, y, z, fx, fy, cx, cy):
    """The single test that catches almost every intrinsics error.

    Build spec section 08: project a known 3D point to pixels, unproject it
    back, and assert it returns to within floating point tolerance.
    """
    u, v = project(x, y, z, fx, fy, cx, cy)
    x2, y2, z2 = unproject(u, v, z, fx, fy, cx, cy)
    assert x2 == pytest.approx(x, rel=1e-9, abs=1e-9)
    assert y2 == pytest.approx(y, rel=1e-9, abs=1e-9)
    assert z2 == pytest.approx(z, rel=1e-12)


@given(u=centre, v=centre, z=depth, fx=focal, fy=focal, cx=centre, cy=centre)
def test_unproject_then_project_is_identity(u, v, z, fx, fy, cx, cy):
    x, y, zz = unproject(u, v, z, fx, fy, cx, cy)
    u2, v2 = project(x, y, zz, fx, fy, cx, cy)
    assert u2 == pytest.approx(u, rel=1e-9, abs=1e-9)
    assert v2 == pytest.approx(v, rel=1e-9, abs=1e-9)


@given(
    width=st.integers(min_value=64, max_value=8192),
    hfov=st.floats(min_value=1.0, max_value=179.0),
)
def test_focal_and_field_of_view_round_trip(width, hfov):
    f = focal_px(width, hfov)
    assert hfov_from_focal(width, f) == pytest.approx(hfov, rel=1e-9)


@given(
    h=st.floats(min_value=0.05, max_value=3.0),
    tilt=st.floats(min_value=0.0, max_value=50.0),
    vfov=st.floats(min_value=5.0, max_value=70.0),
)
def test_raising_the_sensor_pushes_the_blind_zone_out(h, tilt, vfov):
    assume(tilt + vfov / 2.0 < 89.0)
    assert near_edge_range(h * 2.0, tilt, vfov) > near_edge_range(h, tilt, vfov)


@given(
    h=st.floats(min_value=0.05, max_value=3.0),
    t1=st.floats(min_value=0.0, max_value=40.0),
    t2=st.floats(min_value=0.0, max_value=40.0),
    vfov=st.floats(min_value=5.0, max_value=70.0),
)
def test_tilting_down_shrinks_the_blind_zone(h, t1, t2, vfov):
    assume(abs(t1 - t2) > 1e-6)
    assume(max(t1, t2) + vfov / 2.0 < 89.0)
    lo, hi = sorted((t1, t2))
    assert near_edge_range(h, hi, vfov) < near_edge_range(h, lo, vfov)
