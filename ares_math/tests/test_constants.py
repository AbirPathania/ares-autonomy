"""The constants file holds itself together.

Build spec section 07: these constants are final, and there is exactly one
copy of them. The project has already lost time to a stale copy, so these
tests exist to make a divergence loud rather than quiet.
"""

import dataclasses

import pytest

from ares_math import constants
from ares_math.constants import (
    AXLE_X,
    BOGIE,
    CAMERA,
    CHASSIS,
    LIMITS,
    MASS,
    REVISION,
    SENSORS,
    TRENCH,
    WHEEL_Y,
)

REL = 1e-9


def test_the_source_of_truth_is_named():
    assert REVISION == "H-2"
    assert "H-2" in constants.__doc__


def test_every_constant_block_is_frozen():
    for cls in (
        constants.Chassis,
        constants.MassBudget,
        constants.Sensors,
        constants.Limits,
        constants.Camera,
        constants.GoverningObstacle,
        constants.RockerBogie,
    ):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


def test_constants_cannot_be_reassigned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CHASSIS.wheel_diameter = 0.2


def test_wheel_diameter_and_rolling_radius_are_consistent():
    """The belt adds 10 mm of radius over the bare wheel. H-2 s01."""
    bare = CHASSIS.wheel_diameter / 2.0
    assert CHASSIS.rolling_radius > bare
    assert CHASSIS.rolling_radius - bare == pytest.approx(0.010, abs=1e-12)


def test_the_datum_is_the_middle_axle_station():
    assert AXLE_X[1] == 0.0
    assert AXLE_X[0] == CHASSIS.axle_pitch_front
    assert AXLE_X[2] == -CHASSIS.axle_pitch_rear
    assert WHEEL_Y == pytest.approx(CHASSIS.track / 2.0, rel=REL)


def test_the_trunnion_sits_inside_the_hull():
    assert 0.0 < CHASSIS.trunnion_z < CHASSIS.deck_z
    assert abs(CHASSIS.trunnion_y) < CHASSIS.track / 2.0
    assert abs(CHASSIS.trunnion_x) < CHASSIS.hull_length / 2.0


def test_the_v_apex_sits_above_the_wheel_centres():
    """The V-shaped belly apex is 115 mm up, the axles are at 105 mm.

    Blast deflection wants the apex high, and 10 mm above the axle line is
    what H-2 landed on. Recorded so that a future revision lowering the apex
    below the axles has to be a deliberate change.
    """
    assert CHASSIS.ground_clearance > CHASSIS.rolling_radius
    assert CHASSIS.ground_clearance - CHASSIS.rolling_radius == pytest.approx(
        0.010, abs=1e-12
    )
    assert CHASSIS.ground_clearance < CHASSIS.deck_z


def test_the_stowed_height_assumes_a_retracted_mast():
    """H-2 quotes 455 mm stowed, which already assumes a retracted mast.

    The deployed mast at 0.95 m is more than double that, which is exactly
    the open item the build spec flags: section 02 sized the mast, and the
    hull has not yet budgeted it.
    """
    assert CHASSIS.stowed_height == 0.455
    assert SENSORS.mast_height > CHASSIS.stowed_height


def test_the_mass_ladder_is_still_over_the_ceiling():
    """Honest disclosure held as a test. Build spec section 04 and 12.

    The tracked plus drone bay configuration is over by 1.10 kg. Until the
    ladder is re-run with the LiDAR deleted and the cameras and mast added,
    the full ground-air configuration is unresolved, not solved.
    """
    assert MASS.tracked_plus_drone > MASS.ceiling
    assert MASS.tracked_plus_drone - MASS.ceiling == pytest.approx(1.10, abs=1e-9)
    assert MASS.wheeled_mine < MASS.ceiling
    assert MASS.tracked_mine < MASS.ceiling


def test_the_speed_limits_are_the_documented_kilometres_per_hour():
    """v_autonomous is the document's three place quote of 4 km/h.

    Recorded as a test so that if anyone tightens it to 4/3.6 the change is
    deliberate. The golden tables use full precision, this constant does not.
    """
    assert LIMITS.v_autonomous == pytest.approx(round(4 / 3.6, 3), abs=1e-12)
    assert LIMITS.v_transit == pytest.approx(round(8 / 3.6, 3), abs=1e-12)
    assert LIMITS.v_transit > LIMITS.v_autonomous


def test_the_two_camera_units_do_two_different_jobs():
    """Section 04: this is a coverage scheme, not a redundancy scheme."""
    assert SENSORS.mast_height > SENSORS.deck_height
    assert SENSORS.deck_tilt_deg > SENSORS.mast_tilt_deg
    assert SENSORS.mast_baseline > SENSORS.deck_baseline


def test_the_mast_baseline_meets_the_section_03_floor():
    """Baseline at or above 0.20 m for the mast unit. Non-negotiable."""
    assert SENSORS.mast_baseline >= 0.20


def test_the_governing_obstacle_is_the_one_the_document_names():
    assert TRENCH.width == 0.5
    assert TRENCH.depth == 0.3


def test_the_camera_working_case_is_the_one_the_tables_used():
    assert CAMERA.width_px == 1280
    assert CAMERA.hfov_deg == 90.0
    assert CAMERA.vfov_deg == 59.0
    assert CAMERA.max_disparity_px == 192
    assert CAMERA.disparity_sigma_px == 0.25
    assert CAMERA.disparity_sigma_px_pessimistic == 0.50


def test_the_bogie_assumption_is_derived_and_not_invented():
    assert BOGIE.bogie_pivot_x == pytest.approx(-CHASSIS.axle_pitch_rear / 2.0, rel=REL)
    assert BOGIE.bogie_pivot_z == CHASSIS.trunnion_z
    assert "MODEL ASSUMPTION" in constants.RockerBogie.__doc__


def test_no_module_outside_constants_defines_a_physical_number():
    """Build spec section 11: no physical constant is written anywhere else.

    Scans the package for bare float literals that look like lengths. The
    allowed set is small and every entry is a mathematical constant, an
    index, a tolerance, or a unit conversion, never a piece of the vehicle.
    """
    import ast
    import pathlib

    # Mathematical constants, unit conversions, indices and solver
    # tolerances. Nothing in this set is a piece of the vehicle.
    allowed = {
        0.0, 1.0, 2.0, 0.5, 3.6, 90.0, 180.0, 100.0, 10.0, 4.0,
        0.1, 60.0, 1000.0,
        1e-6, 1e-9, 1e-12, 1e-14,
    }
    package = pathlib.Path(constants.__file__).parent
    offenders = []
    for path in package.glob("*.py"):
        if path.name == "constants.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value not in allowed:
                    offenders.append(f"{path.name}:{node.lineno} {node.value}")
    assert not offenders, "physical numbers outside constants.py: " + ", ".join(
        offenders
    )
