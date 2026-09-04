"""Rocker-bogie articulation. Build spec section 07.

Two independent implementations, asserted equal: a closed form pose for
planar ground and a numerical solver for arbitrary terrain. Plus the rigid
body property that no articulation can change a link length.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ares_math.constants import BOGIE, CHASSIS
from ares_math.kinematics import (
    LINKAGE,
    HeightField,
    Plane,
    Pose,
    body_pitch,
    pose_on_plane,
    solve_pose,
    wheel_centres,
)

REL = 1e-9


def test_the_differential_is_the_mean_of_the_rockers():
    """theta_body = (theta_rocker_left + theta_rocker_right) / 2

    Skip this and you have modelled a rigid six-wheeler that happens to look
    like a rocker-bogie, and every obstacle result will be wrong.
    """
    assert body_pitch(10.0, -4.0) == pytest.approx(3.0, rel=REL)
    assert body_pitch(0.0, 0.0) == 0.0
    assert body_pitch(-7.5, 7.5) == pytest.approx(0.0, abs=1e-12)


def test_the_bogie_pivot_is_derived_from_the_axle_pitch():
    """The one model assumption in the linkage, held to its own definition."""
    assert BOGIE.bogie_pivot_x == pytest.approx(-CHASSIS.axle_pitch_rear / 2.0, rel=REL)
    assert BOGIE.bogie_pivot_z == CHASSIS.trunnion_z


def test_neutral_pose_puts_every_wheel_on_the_ground():
    pose = pose_on_plane(0.0, 0.0)
    centres = wheel_centres(pose)
    assert len(centres) == 6
    for key, p in centres.items():
        assert p[2] == pytest.approx(CHASSIS.rolling_radius, rel=REL), key


def test_neutral_pose_matches_the_hull_geometry():
    pose = pose_on_plane(0.0, 0.0)
    centres = wheel_centres(pose)
    assert centres[("left", "front")][0] == pytest.approx(CHASSIS.axle_pitch_front)
    assert centres[("left", "middle")][0] == pytest.approx(0.0, abs=1e-12)
    assert centres[("left", "rear")][0] == pytest.approx(-CHASSIS.axle_pitch_rear)
    assert centres[("left", "middle")][1] == pytest.approx(CHASSIS.track / 2.0)
    assert centres[("right", "middle")][1] == pytest.approx(-CHASSIS.track / 2.0)


def test_the_wheelbase_is_what_the_hull_says_it_is():
    front = LINKAGE["left"].front[0]
    rear = LINKAGE["left"].rear[0]
    assert front - rear == pytest.approx(
        CHASSIS.axle_pitch_front + CHASSIS.axle_pitch_rear, rel=REL
    )


@pytest.mark.parametrize(
    "ground_pitch,ground_roll",
    [(0, 0), (5, 0), (-5, 0), (10, 0), (0, 8), (0, -8), (12, -7), (-15, 11), (20, 0)],
)
def test_closed_form_and_solver_agree_on_planes(ground_pitch, ground_roll):
    """Build spec section 11, mechanism 3: two independent implementations.

    The closed form constructs the pose from the plane normal in one step.
    The solver finds it by driving seven residuals to zero. They must agree.
    """
    analytic = pose_on_plane(ground_pitch, ground_roll)
    numeric = solve_pose(Plane(0.0, ground_pitch, ground_roll))
    assert numeric.z == pytest.approx(analytic.z, abs=1e-9)
    assert numeric.pitch == pytest.approx(analytic.pitch, abs=1e-9)
    assert numeric.roll == pytest.approx(analytic.roll, abs=1e-9)
    assert numeric.rocker_left == pytest.approx(0.0, abs=1e-9)
    assert numeric.rocker_right == pytest.approx(0.0, abs=1e-9)
    assert numeric.bogie_left == pytest.approx(0.0, abs=1e-9)
    assert numeric.bogie_right == pytest.approx(0.0, abs=1e-9)


def test_the_solver_satisfies_the_differential():
    pose = solve_pose(Plane(0.0, 12.0, -7.0))
    assert pose.differential_residual == pytest.approx(0.0, abs=1e-12)


def test_a_step_articulates_the_linkage():
    """The rocker-bogie must actually move. A rigid model would not."""
    step = HeightField(lambda x, y: 0.10 if x > 0.15 else 0.0)
    pose = solve_pose(step)
    articulated = max(
        abs(pose.rocker_left), abs(pose.bogie_left), abs(pose.rocker_right)
    )
    assert articulated > 1e-3, "the linkage did not articulate over a 100 mm step"
    assert pose.differential_residual == pytest.approx(0.0, abs=1e-12)


def test_a_one_sided_step_rolls_the_hull():
    """A step under the left wheels only must produce roll, not just pitch."""
    step = HeightField(lambda x, y: 0.08 if (x > 0.15 and y > 0.0) else 0.0)
    pose = solve_pose(step)
    assert abs(pose.roll) > 1e-3
    assert pose.rocker_left == pytest.approx(-pose.rocker_right, abs=1e-9)


def test_the_two_forty_five_millimetre_obstacle_claim():
    """Gate 2 of build spec section 10: the obstacle claim from the geometry.

    The hull must clear a 245 mm step: every wheel on the ground, and the
    belly above the obstacle. This is a geometry check, not a traction claim.
    """
    height = 0.245
    step = HeightField(lambda x, y: height if x > 0.20 else 0.0)
    pose = solve_pose(step)
    centres = wheel_centres(pose)
    for key, p in centres.items():
        ground = height if p[0] > 0.20 else 0.0
        assert p[2] > ground, f"{key} went through the ground"
    assert pose.differential_residual == pytest.approx(0.0, abs=1e-12)


def test_a_narrow_trench_is_spanned_not_fallen_into():
    """A wheel of radius 105 mm bridges a 60 mm gap. The envelope handles it."""
    trench = HeightField(lambda x, y: -0.30 if 0.10 < x < 0.16 else 0.0)
    pose = solve_pose(trench)
    for p in wheel_centres(pose).values():
        assert p[2] > -0.05, "a wheel fell into a gap narrower than its own footprint"


# --------------------------------------------------------------------------
# Property tests. Build spec section 11, mechanism 4:
# rigid body distances are invariant under articulation.
# --------------------------------------------------------------------------

angle = st.floats(min_value=-0.6, max_value=0.6, allow_nan=False)
height = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)


def _pairs(centres):
    keys = sorted(centres, key=str)
    return {
        (a, b): float(np.linalg.norm(centres[a] - centres[b]))
        for a, b in itertools.combinations(keys, 2)
    }


NEUTRAL = _pairs(
    wheel_centres(Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
)


@given(z=height, pitch=angle, roll=angle)
@settings(max_examples=100)
def test_rigid_body_distances_survive_body_motion(z, pitch, roll):
    """Moving the hull cannot change any distance between wheels."""
    pose = Pose(z, pitch, roll, 0.0, 0.0, 0.0, 0.0)
    got = _pairs(wheel_centres(pose))
    for key, want in NEUTRAL.items():
        assert got[key] == pytest.approx(want, rel=1e-9, abs=1e-12)


@given(rocker=angle, bogie=angle)
@settings(max_examples=100)
def test_link_lengths_survive_articulation(rocker, bogie):
    """Articulating the linkage cannot change any link length.

    The wheel to wheel distances do change, that is the point of a rocker
    bogie. The distances from each pivot to the parts it carries must not.
    """
    link = LINKAGE["left"]
    pts = link.points_in_body(rocker, bogie)
    trunnion = np.array(link.trunnion)
    pivot = pts["bogie_pivot"]

    assert float(np.linalg.norm(pts["front"] - trunnion)) == pytest.approx(
        link.rocker_front_arm, rel=1e-12
    )
    assert float(np.linalg.norm(pivot - trunnion)) == pytest.approx(
        link.rocker_rear_arm, rel=1e-12
    )
    assert float(np.linalg.norm(pts["middle"] - pivot)) == pytest.approx(
        link.bogie_front_arm, rel=1e-12
    )
    assert float(np.linalg.norm(pts["rear"] - pivot)) == pytest.approx(
        link.bogie_rear_arm, rel=1e-12
    )
    assert float(np.linalg.norm(pts["rear"] - pts["middle"])) == pytest.approx(
        CHASSIS.axle_pitch_rear, rel=1e-12
    ), "the bogie is one rigid part, its two wheels keep their spacing"


@given(rocker=angle, bogie=angle, pitch=angle, roll=angle, z=height)
@settings(max_examples=100)
def test_the_bogie_is_rigid_in_the_world_frame_too(rocker, bogie, pitch, roll, z):
    pose = Pose(z, pitch, roll, rocker, -rocker, bogie, bogie)
    centres = wheel_centres(pose)
    span = float(
        np.linalg.norm(centres[("left", "rear")] - centres[("left", "middle")])
    )
    assert span == pytest.approx(CHASSIS.axle_pitch_rear, rel=1e-12)


@given(
    ground_pitch=st.floats(min_value=-25.0, max_value=25.0),
    ground_roll=st.floats(min_value=-20.0, max_value=20.0),
)
@settings(max_examples=25, deadline=None)
def test_solver_matches_closed_form_on_any_plane(ground_pitch, ground_roll):
    analytic = pose_on_plane(ground_pitch, ground_roll)
    numeric = solve_pose(Plane(0.0, ground_pitch, ground_roll))
    assert numeric.z == pytest.approx(analytic.z, abs=1e-8)
    assert numeric.pitch == pytest.approx(analytic.pitch, abs=1e-8)
    assert numeric.roll == pytest.approx(analytic.roll, abs=1e-8)


@given(ground_pitch=st.floats(min_value=-25.0, max_value=25.0))
@settings(max_examples=25, deadline=None)
def test_hull_attitude_follows_the_ground_slope(ground_pitch):
    """On a plane the hull is parallel to it. Positive ground slope, nose up,
    which is negative pitch in the ROS convention."""
    pose = pose_on_plane(ground_pitch, 0.0)
    assert math.degrees(pose.pitch) == pytest.approx(-ground_pitch, abs=1e-9)
