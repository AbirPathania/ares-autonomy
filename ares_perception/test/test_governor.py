"""Confidence to speed. Build spec section 09.

The behaviour under test is the one sentence the whole camera-only
architecture rests on: low confidence over a region means slow down or stop,
exactly like an obstacle.

These are the tests that would have to fail before this stack could drive into
a snowdrift at full speed.
"""

import numpy as np
import pytest

from ares_math.constants import CHASSIS, LIMITS, TRENCH
from ares_math.safety import max_speed_for_range, stopping_distance
from ares_perception.governor import (
    PLANNING_HORIZON_FACTOR,
    speed_limit,
    trusted_range,
)
from ares_perception.traversability import OBSTACLE, TRAVERSABLE, UNKNOWN, GridSpec

SPEC = GridSpec(x_min=0.0, x_max=12.0, y_min=-3.0, y_max=3.0)

# Ranges out of the governor are measured ahead of the front of the vehicle,
# not from the datum at the middle axle.
FRONT = CHASSIS.axle_pitch_front + CHASSIS.rolling_radius


def grid(fill=UNKNOWN):
    return np.full(SPEC.shape, fill, dtype=np.uint8)


def clear_to(state, distance, value=TRAVERSABLE):
    xs, _ = SPEC.centres()
    state[xs <= distance, :] = value
    return state


def test_seeing_nothing_means_stopping():
    """The failure mode this module exists to prevent, at its extreme."""
    limit = speed_limit(grid(UNKNOWN), SPEC)
    assert limit.speed == 0.0
    assert "stop" in limit.reason


def test_clear_ground_gives_the_autonomous_ceiling_and_no_more():
    """4 km/h is the cap section 02 derives. Seeing further does not earn more."""
    limit = speed_limit(clear_to(grid(), 12.0), SPEC)
    assert limit.speed == pytest.approx(LIMITS.v_autonomous)
    assert limit.kmh == pytest.approx(4.0, abs=0.01)
    assert "autonomous limit" in limit.reason


def test_a_short_trusted_range_caps_the_speed():
    """The whole idea, in one assertion.

    Not a confidence score multiplied by a speed. A speed derived through the
    same equation that turns mast height into 4 km/h, from the range the
    perception actually decided out to.
    """
    reach = 0.8 + FRONT
    limit = speed_limit(clear_to(grid(), reach), SPEC)
    expected = max_speed_for_range(
        limit.trusted_range / PLANNING_HORIZON_FACTOR,
        LIMITS.decel,
        LIMITS.latency,
        LIMITS.safety_factor,
    )
    assert limit.speed == pytest.approx(expected, rel=1e-9)
    assert limit.speed < LIMITS.v_autonomous
    # And it stops well inside what it can see, with the planning margin.
    assert stopping_distance(
        limit.speed, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    ) < limit.limiting_band


def test_five_metres_of_horizon_buys_the_full_autonomous_speed():
    """Section 03.2: about 1.5 m to stop, 5 to 8 m to plan a decent path.

    The governor is built so that 5 m of trusted range and 4 km/h coincide.
    If that stops being true, either the factor or the document moved.
    """
    at_five = speed_limit(clear_to(grid(), 5.0 + FRONT), SPEC)
    assert at_five.kmh == pytest.approx(4.0, abs=0.05)
    assert speed_limit(clear_to(grid(), 6.0 + FRONT), SPEC).speed == pytest.approx(
        LIMITS.v_autonomous
    )


def test_the_horizon_requirement_degrades_smoothly():
    """No cliff. A vehicle that jumps between full speed and stopped is
    harder to plan around than one that eases off."""
    speeds = [
        speed_limit(clear_to(grid(), float(r) + FRONT), SPEC).speed
        for r in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    assert speeds == sorted(speeds)
    assert all(b - a < 0.25 for a, b in zip(speeds, speeds[1:], strict=False))


def test_the_speed_always_stops_inside_the_trusted_range():
    """Swept, because this is the property that has to hold everywhere."""
    for reach in np.arange(0.5 + FRONT, 11.0, 0.25):
        limit = speed_limit(clear_to(grid(), float(reach)), SPEC)
        needed = stopping_distance(
            limit.speed, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
        )
        assert needed <= limit.limiting_band + 1e-9, (
            f"at {reach:.2f} m of trusted range it chose {limit.kmh:.2f} km/h, "
            f"which needs {needed:.3f} m to stop"
        )


def test_unknown_ground_beyond_a_point_shortens_the_range():
    state = clear_to(grid(), 4.0)
    reach, decided = trusted_range(state, SPEC)
    assert reach == pytest.approx(4.0 - FRONT, abs=SPEC.cell * 1.5)
    assert decided < 0.6


def test_the_range_is_measured_from_the_bumper_not_the_datum():
    """The datum is the middle axle. A stopping distance measured from there
    lets the front wheels reach the hazard while the arithmetic still says
    there is room."""
    reach, _ = trusted_range(clear_to(grid(), 5.0), SPEC)
    assert reach == pytest.approx(5.0 - FRONT, abs=SPEC.cell * 1.5)
    assert FRONT == pytest.approx(0.385, abs=1e-9)


def test_snow_that_reads_as_unknown_slows_the_vehicle_down():
    """Gate 4, expressed as a unit test.

    Textured ground to 6 m, then nothing the matcher could resolve. The
    vehicle must slow, and it must say that depth is the reason rather than
    an obstacle, because there is no obstacle.
    """
    state = clear_to(grid(UNKNOWN), 3.0 + FRONT)
    limit = speed_limit(state, SPEC)
    assert 0.0 < limit.speed < LIMITS.v_autonomous
    assert limit.kmh == pytest.approx(2.90, abs=0.05)
    assert "obstacle" not in limit.reason, "there is no obstacle, only no data"
    assert "horizon" in limit.reason or "not trusted" in limit.reason


def test_an_obstacle_binds_before_the_range_does():
    state = clear_to(grid(), 12.0)
    xs, _ = SPEC.centres()
    near = int(np.argmin(np.abs(xs - (1.0 + FRONT))))
    state[near, :] = OBSTACLE
    limit = speed_limit(state, SPEC)
    assert "obstacle" in limit.reason
    assert limit.limiting_band == pytest.approx(xs[near] - FRONT, abs=1e-9)
    assert limit.speed < LIMITS.v_autonomous
    assert stopping_distance(
        limit.speed, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    ) == pytest.approx(limit.limiting_band, rel=1e-9)


def test_an_obstacle_further_than_the_stopping_distance_does_not_slow_anything():
    """At 4 km/h the vehicle stops in 1.43 m. Something at 3 m is not a
    reason to slow, and treating it as one would make the vehicle useless."""
    state = clear_to(grid(), 12.0)
    xs, _ = SPEC.centres()
    state[int(np.argmin(np.abs(xs - (3.0 + FRONT)))), :] = OBSTACLE
    assert speed_limit(state, SPEC).speed == pytest.approx(LIMITS.v_autonomous)


def test_an_obstacle_off_to_the_side_is_not_in_the_way():
    """Stopping for something you would drive past is its own failure."""
    state = clear_to(grid(), 12.0)
    state[:, 0] = OBSTACLE
    state[:, -1] = OBSTACLE
    limit = speed_limit(state, SPEC)
    assert limit.speed == pytest.approx(LIMITS.v_autonomous)


def test_the_governing_trench_at_four_kilometres_per_hour():
    """Ties the governor to section 02's actual case.

    The trench is seen at 1.583 m from the mast. At 4 km/h the vehicle needs
    1.426 m to stop. So seeing it at that range must still permit roughly the
    autonomous speed, and no more.
    """
    from ares_math.constants import SENSORS
    from ares_math.safety import negative_obstacle_range

    seen_at = negative_obstacle_range(
        TRENCH.width, SENSORS.mast_height, TRENCH.depth
    )
    allowed = max_speed_for_range(
        seen_at, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    assert allowed > 4 / 3.6
    assert allowed * 3.6 == pytest.approx(4.26, abs=0.01)


def test_a_single_unresolved_cell_does_not_stop_the_vehicle():
    """Demanding perfection would park it permanently."""
    state = clear_to(grid(), 12.0)
    rng = np.random.default_rng(0)
    holes = rng.random(state.shape) < 0.2
    state[holes] = UNKNOWN
    limit = speed_limit(state, SPEC)
    assert limit.speed == pytest.approx(LIMITS.v_autonomous)


def test_the_reason_is_always_populated():
    for state in (
        grid(UNKNOWN),
        clear_to(grid(), 3.0 + FRONT),
        clear_to(grid(), 12.0),
    ):
        limit = speed_limit(state, SPEC)
        assert limit.reason
        assert "trusted_range" in limit.as_dict()
