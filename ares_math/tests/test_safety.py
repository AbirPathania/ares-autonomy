"""Stopping distance, negative obstacles, and the speed limit.

Build spec sections 02 and 03.3. These are the tests that hold the vehicle's
one hard safety claim in place: at 4 km/h, with the camera at 0.95 m, a
0.5 m by 0.3 m trench is seen in time to stop.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pint import DimensionalityError

from ares_math.constants import LIMITS, SENSORS, TRENCH
from ares_math.safety import (
    braking_distance,
    detects_in_time,
    max_speed_for_range,
    max_speed_for_sensor_height,
    negative_obstacle_range,
    reaction_distance,
    required_sensor_height,
    stopping_distance,
)
from ares_math.units import Q_

from .conftest import GOLD, REL


@pytest.mark.parametrize("row", GOLD["stopping_table"])
def test_reaction_distance_table(row):
    got = reaction_distance(row["v"], row["t_latency"])
    assert got == pytest.approx(row["reaction"], rel=REL)


@pytest.mark.parametrize("row", GOLD["stopping_table"])
def test_braking_distance_table(row):
    got = braking_distance(row["v"], row["a"])
    assert got == pytest.approx(row["braking"], rel=REL)


@pytest.mark.parametrize("row", GOLD["stopping_table"])
def test_stopping_distance_table(row):
    got = stopping_distance(row["v"], row["a"], row["t_latency"])
    assert got == pytest.approx(row["total"], rel=REL)


@pytest.mark.parametrize("row", GOLD["stopping_table"])
def test_stopping_distance_with_safety_factor(row):
    got = stopping_distance(row["v"], row["a"], row["t_latency"], row["safety_factor"])
    assert got == pytest.approx(row["total_x_safety"], rel=REL)


@pytest.mark.parametrize("row", GOLD["stopping_x_safety_table"])
def test_stopping_distance_by_speed(row):
    v = row["speed_kmh"] / 3.6
    got = stopping_distance(v, row["a"], row["t_latency"], row["safety_factor"])
    assert got == pytest.approx(row["expected"], rel=REL)


@pytest.mark.parametrize("row", GOLD["negative_obstacle_table"])
def test_negative_obstacle_range_table(row):
    got = negative_obstacle_range(row["w"], row["h"], row["d"])
    assert got == pytest.approx(row["expected"], rel=REL)


@pytest.mark.parametrize("row", GOLD["negative_obstacle_table"])
def test_detection_verdicts_match_the_document(row):
    for speed_kmh, verdict in row["verdicts"].items():
        v = speed_kmh / 3.6
        got = detects_in_time(row["w"], row["h"], row["d"], v, 1.0, 0.30, 1.5)
        assert got == (verdict == "OK"), (
            f"h={row['h']} at {speed_kmh} km/h: document says {verdict}"
        )


@pytest.mark.parametrize("row", GOLD["mast_height_table"])
def test_required_sensor_height_table(row):
    got = required_sensor_height(row["w"], row["d"], row["stop_distance"])
    assert got == pytest.approx(row["expected"], rel=REL)


def test_no_sensor_type_appears_in_the_equation():
    """The finding that changes the design, stated as a test.

    R_max depends on w, h and d and nothing else. A LiDAR mounted at the same
    height has exactly the same blind geometry, so dropping it costs nothing
    on negative obstacles.
    """
    import inspect

    source = inspect.getsource(negative_obstacle_range)
    body = source.split('"""')[-1]
    for word in ("lidar", "laser", "radar", "camera", "stereo"):
        assert word not in body.lower(), f"{word} leaked into the geometry"


def test_the_mast_height_that_section_02_derives():
    """Solving h = d * D_stop / w at 4 km/h gives 0.86 m."""
    stop = stopping_distance(4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor)
    h = required_sensor_height(TRENCH.width, TRENCH.depth, stop)
    assert h == pytest.approx(0.8555555555555556, rel=1e-9)
    assert SENSORS.mast_height > h, "the built mast must clear the derived requirement"


def test_the_mast_clears_four_kilometres_per_hour_with_margin():
    """0.95 m supports 4.26 km/h against the governing trench.

    This is the single number the whole camera-only architecture rests on.
    """
    v_max = max_speed_for_sensor_height(
        SENSORS.mast_height,
        TRENCH.width,
        TRENCH.depth,
        LIMITS.decel,
        LIMITS.latency,
        LIMITS.safety_factor,
    )
    assert v_max * 3.6 == pytest.approx(4.261011140224293, rel=1e-9)
    assert v_max > 4 / 3.6, "0.95 m must clear 4 km/h"
    assert v_max * 3.6 < 5.0, "and it must not be mistaken for clearing much more"


def test_eight_kilometres_per_hour_cannot_be_autonomous():
    """Not with cameras, not with LiDAR, not at any mast height that fits.

    Build spec section 02. The height required at 8 km/h is 2.82 m, which no
    25 kg two-person-carry vehicle carries.
    """
    stop = stopping_distance(8 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor)
    h = required_sensor_height(TRENCH.width, TRENCH.depth, stop)
    assert h == pytest.approx(2.8222222, rel=1e-6)
    assert h > 2.5, "transit speed is a teleoperated mode, put it in the CONOPS"


def test_the_deck_unit_cannot_carry_the_speed_limit():
    """0.36 m of sensor height only supports 2.3 km/h.

    Which is why the deck unit fills the mast unit's blind zone and does not
    replace it. Build spec section 04.
    """
    v_max = max_speed_for_sensor_height(
        SENSORS.deck_height,
        TRENCH.width,
        TRENCH.depth,
        LIMITS.decel,
        LIMITS.latency,
        LIMITS.safety_factor,
    )
    assert v_max * 3.6 == pytest.approx(2.316233207540577, rel=1e-9)
    assert v_max * 3.6 < 4.0, "the deck unit alone cannot carry the speed limit"


def test_latency_is_the_term_you_control():
    """At 4 km/h every 100 ms of pipeline delay costs 111 mm."""
    v = 4 / 3.6
    base = stopping_distance(v, 1.0, 0.30)
    slower = stopping_distance(v, 1.0, 0.40)
    assert slower - base == pytest.approx(0.1111111, rel=1e-6)


def test_a_speed_in_kilometres_per_hour_converts():
    metric = stopping_distance(Q_(4.0, "km/hour"), 1.0, 0.30)
    plain = stopping_distance(4 / 3.6, 1.0, 0.30)
    assert metric == pytest.approx(plain, rel=REL)


def test_a_time_where_an_acceleration_belongs_is_rejected():
    with pytest.raises(DimensionalityError):
        stopping_distance(1.111, Q_(1.0, "second"), 0.30)


# --------------------------------------------------------------------------
# Property tests. Build spec section 11, mechanism 4.
# --------------------------------------------------------------------------

speed = st.floats(min_value=0.01, max_value=30.0, allow_nan=False)
accel = st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
latency = st.floats(min_value=0.0, max_value=2.0, allow_nan=False)
length = st.floats(min_value=0.01, max_value=10.0, allow_nan=False)
factor = st.floats(min_value=1.0, max_value=3.0, allow_nan=False)


@given(v1=speed, v2=speed, a=accel, t=latency)
def test_stopping_distance_is_strictly_increasing_in_speed(v1, v2, a, t):
    assume(abs(v1 - v2) > 1e-9)
    lo, hi = sorted((v1, v2))
    assert stopping_distance(hi, a, t) > stopping_distance(lo, a, t)


@given(v=speed, a=accel, t=latency, sf=factor)
def test_max_speed_inverts_stopping_distance(v, a, t, sf):
    r = stopping_distance(v, a, t, sf)
    assert max_speed_for_range(r, a, t, sf) == pytest.approx(v, rel=1e-9)


@given(w=length, h=length, d=length)
def test_raising_the_sensor_always_helps(w, h, d):
    assert negative_obstacle_range(w, h * 2.0, d) > negative_obstacle_range(w, h, d)


@given(w=length, h=length, d=length)
def test_a_deeper_hole_is_harder_to_see_into(w, h, d):
    assert negative_obstacle_range(w, h, d * 2.0) < negative_obstacle_range(w, h, d)


@given(w=length, h=length, d=length)
def test_a_wider_hole_is_easier_to_see_into(w, h, d):
    assert negative_obstacle_range(w, h * 1.0, d) < negative_obstacle_range(
        w * 2.0, h, d
    )


@given(w=length, d=length, stop=length)
def test_required_height_inverts_detection_range(w, d, stop):
    h = required_sensor_height(w, d, stop)
    assert negative_obstacle_range(w, h, d) == pytest.approx(stop, rel=1e-9)


@given(w=length, h=length, d=length, a=accel, t=latency, sf=factor)
def test_the_speed_cap_is_exactly_the_speed_that_stops_in_time(w, h, d, a, t, sf):
    v = max_speed_for_sensor_height(h, w, d, a, t, sf)
    assert detects_in_time(w, h, d, v * (1 - 1e-9), a, t, sf)
