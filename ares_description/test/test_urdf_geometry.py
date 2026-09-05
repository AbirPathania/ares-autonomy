"""The generated URDF says what constants.py says.

Build spec phase 2 gate: the rover displays correctly, the differential is
enforced, there are no TF warnings, and the optical frame test passes.

These tests parse the URDF as plain XML and compose the fixed transform chain
with the same rotation convention ares_math.kinematics uses. That is
deliberate: it means the geometry gate runs on a laptop in milliseconds with
no ROS, no container and no simulator, so a bad number cannot survive long
enough to reach RViz.

The RViz screenshot and the live TF tree are a separate check that needs the
container. This file covers everything that can be settled without one.
"""

import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from ares_math.constants import (
    CAMERA,
    CHASSIS,
    LIMITS,
    LINK_MASS,
    OPTICAL_RPY,
    REVISION,
    SENSORS,
)
from ares_math.kinematics import LINKAGE, rpy_matrix
from ares_math.optics import focal_px, near_edge_range

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "ares_description" / "urdf" / "ares_g.urdf.xacro"
GENERATOR = ROOT / "tools" / "generate_urdf.py"

REL = 1e-9


@pytest.fixture(scope="module")
def robot():
    if not URDF.exists():
        pytest.fail(f"{URDF} is missing. Run tools/generate_urdf.py.")
    return ET.parse(URDF).getroot()


@pytest.fixture(scope="module")
def joints(robot):
    return {j.get("name"): j for j in robot.findall("joint")}


@pytest.fixture(scope="module")
def links(robot):
    return {ln.get("name"): ln for ln in robot.findall("link")}


def parse_origin(joint):
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz").split()]
    rpy = [float(v) for v in origin.get("rpy").split()]
    return np.array(xyz), rpy


def transform_to(joints, link, stop="base_link"):
    """Compose the fixed transform chain from base_link down to link.

    Returns (translation, rotation) in the base_link frame, with every
    articulated joint at zero, which is the default pose.
    """
    chain = []
    current = link
    guard = 0
    while current != stop:
        guard += 1
        if guard > 50:
            raise AssertionError(f"no path from {link} to {stop}, or a cycle")
        for joint in joints.values():
            if joint.find("child").get("link") == current:
                chain.append(joint)
                current = joint.find("parent").get("link")
                break
        else:
            raise AssertionError(f"{current} has no parent joint")

    translation = np.zeros(3)
    rotation = np.eye(3)
    for joint in reversed(chain):
        xyz, rpy = parse_origin(joint)
        translation = translation + rotation @ xyz
        rotation = rotation @ rpy_matrix(*rpy)
    return translation, rotation


# --------------------------------------------------------------------------
# The gate the build spec names: the mast camera sits at 0.95 m.
# --------------------------------------------------------------------------


def test_the_mast_camera_optical_frame_sits_at_the_derived_height(joints):
    """The single number the camera-only architecture rests on.

    Section 02 derives 0.86 m as the requirement at 4 km/h and the mast puts
    the camera at 0.95 m. If this drifts, the speed limit is wrong and the
    trench gate in phase 5 will pass for the wrong reason.
    """
    translation, _ = transform_to(joints, "mast_cam_optical")
    assert translation[2] == pytest.approx(SENSORS.mast_height, abs=1e-9)
    assert translation[2] == pytest.approx(0.95, abs=1e-9)


def test_the_deck_camera_optical_frame_sits_at_its_derived_height(joints):
    translation, _ = transform_to(joints, "deck_cam_optical")
    assert translation[2] == pytest.approx(SENSORS.deck_height, abs=1e-9)


def test_both_eyes_of_a_pair_are_at_the_same_height(joints):
    """A stereo pair whose eyes sit at different heights has a vertical
    disparity the matcher cannot explain."""
    for unit in ("mast_cam", "deck_cam"):
        left, _ = transform_to(joints, f"{unit}_optical")
        right, _ = transform_to(joints, f"{unit}_right_optical")
        assert left[2] == pytest.approx(right[2], abs=1e-12)
        assert left[0] == pytest.approx(right[0], abs=1e-12)


@pytest.mark.parametrize(
    "unit,baseline",
    [("mast_cam", SENSORS.mast_baseline), ("deck_cam", SENSORS.deck_baseline)],
)
def test_the_baseline_is_the_one_section_03_requires(joints, unit, baseline):
    left, _ = transform_to(joints, f"{unit}_optical")
    right, _ = transform_to(joints, f"{unit}_right_optical")
    assert float(np.linalg.norm(left - right)) == pytest.approx(baseline, abs=1e-12)


def test_the_right_eye_is_to_the_right(joints):
    """Swap these and every disparity comes out negative."""
    for unit in ("mast_cam", "deck_cam"):
        left, _ = transform_to(joints, f"{unit}_optical")
        right, _ = transform_to(joints, f"{unit}_right_optical")
        assert right[1] < left[1], f"{unit} right eye is on the left"


# --------------------------------------------------------------------------
# The optical frame rotation. Section 08, the bug that costs two days.
# --------------------------------------------------------------------------


def test_the_optical_frame_is_z_forward_x_right_y_down(joints):
    """Relative to its mount link, before the camera tilt is applied."""
    joint = joints["mast_cam_optical_joint"]
    _, rpy = parse_origin(joint)
    assert rpy == pytest.approx(list(OPTICAL_RPY), abs=1e-12)

    rotation = rpy_matrix(*rpy)
    assert rotation @ np.array([0, 0, 1]) == pytest.approx([1, 0, 0], abs=1e-9)
    assert rotation @ np.array([1, 0, 0]) == pytest.approx([0, -1, 0], abs=1e-9)
    assert rotation @ np.array([0, 1, 0]) == pytest.approx([0, 0, -1], abs=1e-9)


def test_the_optical_axis_points_down_by_the_camera_tilt(joints):
    """In base_link, the mast camera looks forward and 20 degrees down.

    Publish a cloud in the wrong frame and it appears rotated by 90 degrees
    and, worse, sometimes almost right, so you tune Nav2 for a day before
    noticing the ground plane is a wall.
    """
    _, rotation = transform_to(joints, "mast_cam_optical")
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    tilt = math.radians(SENSORS.mast_tilt_deg)
    assert forward == pytest.approx([math.cos(tilt), 0.0, -math.sin(tilt)], abs=1e-9)

    down = rotation @ np.array([0.0, 1.0, 0.0])
    assert down[2] < 0.0, "the optical Y axis must point downward"
    right = rotation @ np.array([1.0, 0.0, 0.0])
    assert right == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_the_deck_camera_is_tilted_further_down_than_the_mast(joints):
    _, mast = transform_to(joints, "mast_cam_optical")
    _, deck = transform_to(joints, "deck_cam_optical")
    mast_z = (mast @ np.array([0.0, 0.0, 1.0]))[2]
    deck_z = (deck @ np.array([0.0, 0.0, 1.0]))[2]
    assert deck_z < mast_z, "the deck unit must look more steeply down"


def test_the_camera_placement_reproduces_the_blind_zone_table(joints):
    """Geometry in the URDF agrees with the coverage argument in section 04."""
    mast_t, _ = transform_to(joints, "mast_cam_optical")
    deck_t, _ = transform_to(joints, "deck_cam_optical")
    mast_blind = near_edge_range(mast_t[2], SENSORS.mast_tilt_deg, CAMERA.vfov_deg)
    deck_blind = near_edge_range(deck_t[2], SENSORS.deck_tilt_deg, CAMERA.vfov_deg)
    assert mast_blind == pytest.approx(0.8113766511902933, rel=1e-9)
    assert deck_blind < mast_blind


# --------------------------------------------------------------------------
# The linkage, and the differential.
# --------------------------------------------------------------------------


def test_every_wheel_touches_the_ground_in_the_default_pose(joints):
    for side in ("left", "right"):
        for name in ("front", "middle", "rear"):
            translation, _ = transform_to(joints, f"wheel_{name}_{side}")
            assert translation[2] == pytest.approx(CHASSIS.rolling_radius, abs=1e-12)


def test_wheel_positions_match_the_hull_geometry(joints):
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for name in ("front", "middle", "rear"):
            translation, _ = transform_to(joints, f"wheel_{name}_{side}")
            expected = getattr(LINKAGE[side], name)
            assert translation == pytest.approx(list(expected), abs=1e-12)
            assert translation[1] == pytest.approx(
                sign * CHASSIS.track / 2.0, abs=1e-12
            )


def test_the_front_wheel_hangs_off_the_rocker_and_the_others_off_the_bogie(joints):
    """The topology that makes it a rocker-bogie rather than a six-wheeler."""
    for side in ("left", "right"):
        front = joints[f"wheel_front_{side}_joint"].find("parent").get("link")
        assert front == f"rocker_{side}"
        for name in ("middle", "rear"):
            parent = joints[f"wheel_{name}_{side}_joint"].find("parent").get("link")
            assert parent == f"bogie_{side}", f"{name} must hang off the bogie"
        assert joints[f"bogie_{side}_joint"].find("parent").get("link") == (
            f"rocker_{side}"
        )
        assert joints[f"rocker_{side}_joint"].find("parent").get("link") == "base_link"


def test_the_articulated_joints_rotate_about_the_pitch_axis(joints):
    for side in ("left", "right"):
        for name in (f"rocker_{side}_joint", f"bogie_{side}_joint"):
            axis = joints[name].find("axis").get("xyz")
            assert [float(v) for v in axis.split()] == [0.0, 1.0, 0.0]
            assert joints[name].get("type") == "continuous"


def test_the_differential_is_present_and_is_not_decorative(robot):
    """Both forms of the constraint are declared.

    A gearbox joint for engines that implement one, and a plugin that holds
    theta_body = (theta_left + theta_right) / 2 for those that do not.
    """
    xml = ET.tostring(robot, encoding="unicode")
    gearbox = robot.findall(".//joint[@type='gearbox']")
    assert len(gearbox) == 1, "the rocker differential gearbox joint is missing"
    joint = gearbox[0]
    # SDFormat spells these as element text, not attributes. The attribute
    # form parses as valid XML, is silently invalid SDF, and takes the whole
    # model down with it.
    assert joint.find("parent").text == "rocker_left"
    assert joint.find("child").text == "rocker_right"
    assert joint.find("gearbox_reference_body").text == "base_link"
    assert joint.find("parent").get("link") is None, "SDF wants text, not an attribute"
    assert float(joint.find("gearbox_ratio").text) == -1.0
    assert "ares_differential" in xml, "the differential plugin is missing"


# --------------------------------------------------------------------------
# Sensors, and the one honesty rule.
# --------------------------------------------------------------------------


def test_four_rgb_cameras_are_simulated_not_one_depth_camera(robot):
    """Build spec section 10. Simulate two RGB cameras per unit and run the
    real matcher on the rendered images. A geometrically perfect DepthCamera
    in the perception path proves nothing at all.
    """
    rgb = robot.findall(".//sensor[@type='camera']")
    assert len(rgb) == 4, "two units of two eyes each"
    names = {s.get("name") for s in rgb}
    assert names == {"mast_cam", "mast_cam_right", "deck_cam", "deck_cam_right"}


def test_ground_truth_depth_is_named_so_it_cannot_be_used_by_accident(robot):
    depth = robot.findall(".//sensor[@type='depth_camera']")
    assert len(depth) == 1
    topic = depth[0].find("topic").text
    assert topic == "/sim/ground_truth/depth"
    assert "ground_truth" in topic, "the name is the safeguard, keep it"


def test_the_simulated_intrinsics_are_the_ones_the_tables_used(robot):
    expected = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
    for sensor in robot.findall(".//sensor[@type='camera']"):
        intrinsics = sensor.find("camera/lens/intrinsics")
        assert float(intrinsics.find("fx").text) == pytest.approx(expected, rel=1e-8)
        assert float(intrinsics.find("fy").text) == pytest.approx(expected, rel=1e-8)
        image = sensor.find("camera/image")
        assert int(image.find("width").text) == CAMERA.width_px
        assert int(image.find("height").text) == CAMERA.height_px


def test_the_drive_plugin_carries_the_autonomous_speed_cap(robot):
    plugin = robot.find(".//plugin[@filename='gz-sim-diff-drive-system']")
    assert plugin is not None
    cap = float(plugin.find("max_linear_velocity").text)
    assert cap == pytest.approx(LIMITS.v_autonomous, abs=1e-12)
    assert cap * 3.6 < 5.0, "8 km/h is teleoperated only, never the autonomous cap"


# --------------------------------------------------------------------------
# The file is generated, and stays generated.
# --------------------------------------------------------------------------


def test_the_urdf_is_a_tree(robot, links, joints):
    assert len(joints) == len(links) - 1, "a URDF must be a tree, not a graph"
    children = [j.find("child").get("link") for j in joints.values()]
    assert len(children) == len(set(children)), "a link has two parents"
    parents = {j.find("parent").get("link") for j in joints.values()}
    roots = set(links) - set(children)
    assert roots == {"base_link"}, f"expected one root, got {roots}"
    assert parents <= set(links), "a joint references a link that does not exist"


def test_every_link_has_mass_or_is_a_frame(links):
    """Frames carry no mass. Everything else must, or the physics is wrong."""
    massless = {
        name
        for name, link in links.items()
        if link.find("inertial") is None
    }
    for name in massless:
        assert name.endswith("_link") or name.endswith("_optical"), (
            f"{name} has no inertial and is not a frame"
        )
    total = sum(
        float(link.find("inertial/mass").get("value"))
        for link in links.values()
        if link.find("inertial") is not None
    )
    expected = (
        6 * LINK_MASS.wheel
        + 2 * LINK_MASS.rocker
        + 2 * LINK_MASS.bogie
        + LINK_MASS.mast
        + 2 * LINK_MASS.camera_unit
        + LINK_MASS.hull
    )
    assert total == pytest.approx(expected, abs=1e-6)


def test_the_file_says_it_is_generated(robot):
    text = URDF.read_text()
    assert "GENERATED FILE" in text
    assert "Do not hand edit" in text
    assert f"hull revision {REVISION}" in text


def test_regenerating_reproduces_the_committed_file():
    """The committed URDF is what constants.py currently produces.

    Change a constant without re-running the generator and this fails, which
    is the whole point of generating it rather than writing it.
    """
    before = URDF.read_text()
    subprocess.run(
        [sys.executable, str(GENERATOR)], check=True, capture_output=True, cwd=ROOT
    )
    after = URDF.read_text()
    assert before == after, (
        "ares_g.urdf.xacro is stale. Run tools/generate_urdf.py and commit it."
    )


# --------------------------------------------------------------------------
# The gate 3 failure case. Section 10: gate 3 should fail at the lower mast.
# --------------------------------------------------------------------------

GATE_URDF = ROOT / "ares_description" / "urdf" / "ares_g_mast_050.urdf.xacro"


@pytest.fixture(scope="module")
def gate_joints():
    if not GATE_URDF.exists():
        pytest.fail(f"{GATE_URDF} is missing. Run tools/generate_urdf.py.")
    root = ET.parse(GATE_URDF).getroot()
    return {j.get("name"): j for j in root.findall("joint")}


def test_the_failure_case_model_puts_the_camera_at_half_a_metre(gate_joints):
    translation, _ = transform_to(gate_joints, "mast_cam_optical")
    assert translation[2] == pytest.approx(0.50, abs=1e-9)


def test_the_failure_case_is_the_same_vehicle_otherwise(joints, gate_joints):
    """Only the mast changes. If anything else moved, the gate is comparing
    two different rovers and proves nothing."""
    assert set(joints) == set(gate_joints)
    for name in joints:
        if "mast" in name or "cam" in name:
            continue
        a, _ = parse_origin(joints[name])
        b, _ = parse_origin(gate_joints[name])
        assert a == pytest.approx(b, abs=1e-12), f"{name} moved between models"


def test_the_failure_case_really_does_fail_the_geometry(gate_joints):
    """Stated here so the two models cannot drift into both passing.

    At 4 km/h the vehicle needs 1.426 m to stop. The trench floor is visible
    from 1.583 m at 0.95 m of mast, and from 0.833 m at 0.50 m.
    """
    from ares_math.constants import LIMITS, TRENCH
    from ares_math.safety import negative_obstacle_range, stopping_distance

    stop = stopping_distance(
        4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    low, _ = transform_to(gate_joints, "mast_cam_optical")
    nominal_range = negative_obstacle_range(
        TRENCH.width, SENSORS.mast_height, TRENCH.depth
    )
    low_range = negative_obstacle_range(TRENCH.width, low[2], TRENCH.depth)

    assert nominal_range > stop, "the nominal model must pass the gate"
    assert low_range < stop, "the low model must fail it, or the gate is decorative"
