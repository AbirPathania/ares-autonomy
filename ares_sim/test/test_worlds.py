"""The worlds encode the gates they exist to test.

Build spec section 10. These parse the generated SDF as plain XML, so they
run on a laptop with no Gazebo, and they check the thing that actually
matters: that the geometry in each world is the geometry the analysis is
about, and that gate 3 is set up to fail at the lower mast height.

A simulation that passes everything you throw at it is not calibrated, it is
decorative. `test_gate_three_is_built_to_fail_at_the_lower_mast` is the test
that keeps this one honest.
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ares_math.constants import CAMERA, LIMITS, MOBILITY, SENSORS, TRENCH
from ares_math.optics import near_edge_range
from ares_math.safety import negative_obstacle_range, stopping_distance
from ares_math.stereo import depth_error

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "ares_sim" / "worlds"
TEXTURES = ROOT / "ares_sim" / "textures"
GENERATOR = ROOT / "tools" / "generate_worlds.py"

GATES = ["flat", "step_field", "trench_field", "snow_scree"]


@pytest.fixture(scope="module")
def worlds():
    out = {}
    for name in GATES:
        path = WORLDS / f"{name}.sdf"
        if not path.exists():
            pytest.fail(f"{path} is missing. Run tools/generate_worlds.py.")
        out[name] = ET.parse(path).getroot()
    return out


def boxes(root):
    """Every box in the world, as name to (size, pose)."""
    out = {}
    for link in root.findall(".//link"):
        box = link.find("collision/geometry/box/size")
        pose = link.find("pose")
        if box is None or pose is None:
            continue
        out[link.get("name")] = (
            [float(v) for v in box.text.split()],
            [float(v) for v in pose.text.split()],
        )
    return out


def test_all_four_gates_exist(worlds):
    assert set(worlds) == set(GATES)


@pytest.mark.parametrize("name", GATES)
def test_each_world_says_it_is_generated(name):
    text = (WORLDS / f"{name}.sdf").read_text()
    assert "GENERATED FILE" in text
    assert "constants.py" in text


@pytest.mark.parametrize("name", GATES)
def test_each_world_records_its_pass_criterion(name):
    """A world with no stated criterion is a demo, not a gate."""
    text = (WORLDS / f"{name}.sdf").read_text()
    assert "GATE" in text, f"{name} does not say what it proves"


@pytest.mark.parametrize("name", GATES)
def test_the_renderer_is_the_one_that_survives_software_rendering(worlds, name):
    """ogre, not ogre2. With LIBGL_ALWAYS_SOFTWARE the newer engine crawls."""
    sensors = worlds[name].find(".//plugin[@filename='gz-sim-sensors-system']")
    assert sensors is not None
    assert sensors.find("render_engine").text == "ogre"


@pytest.mark.parametrize("name", GATES)
def test_the_physics_step_is_fine_enough_for_a_wheel_on_an_edge(worlds, name):
    step = float(worlds[name].find(".//physics/max_step_size").text)
    assert step <= 0.001, "a coarse step lets the wheel tunnel through a corner"


# --------------------------------------------------------------------------
# Gate 1. Flat.
# --------------------------------------------------------------------------


def test_gate_one_states_the_error_bound_it_will_be_judged_against(worlds):
    from ares_math.optics import focal_px

    text = (WORLDS / "flat.sdf").read_text()
    bound = depth_error(
        Z=3.0,
        f=focal_px(CAMERA.width_px, CAMERA.hfov_deg),
        B=SENSORS.mast_baseline,
        d_disp=CAMERA.disparity_sigma_px,
    )
    assert f"{bound:.4f}" in text, "the section 03.2 bound is not written into the world"


def test_gate_one_is_actually_flat(worlds):
    named = boxes(worlds["flat"])
    assert len(named) == 1
    (size, pose) = next(iter(named.values()))
    top = pose[2] + size[2] / 2.0
    assert top == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Gate 2. The obstacle claim.
# --------------------------------------------------------------------------


def test_gate_two_brackets_the_obstacle_claim(worlds):
    """One step under the claim, one at it, one over. All three are needed:
    passing only the easy one proves nothing, and a world with no step it
    should fail cannot calibrate anything."""
    named = boxes(worlds["step_field"])
    heights = {
        name: size[2] for name, (size, _) in named.items() if name.startswith("step_")
    }
    assert len(heights) == 3
    assert heights["step_under"] < MOBILITY.max_step
    assert heights["step_at_claim"] == pytest.approx(MOBILITY.max_step, abs=1e-9)
    assert heights["step_over"] > MOBILITY.max_step


def test_gate_two_steps_stand_on_the_ground(worlds):
    named = boxes(worlds["step_field"])
    for name, (size, pose) in named.items():
        if name.startswith("step_"):
            assert pose[2] - size[2] / 2.0 == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Gate 3. The one that matters.
# --------------------------------------------------------------------------


def test_gate_three_is_the_governing_trench_from_section_02(worlds):
    named = boxes(worlds["trench_field"])
    near = named["ground_near"]
    far = named["ground_far"]
    gap_start = near[1][0] + near[0][0] / 2.0
    gap_end = far[1][0] - far[0][0] / 2.0
    assert gap_end - gap_start == pytest.approx(TRENCH.width, abs=1e-9)

    floor_size, floor_pose = named["trench_floor"]
    floor_top = floor_pose[2] + floor_size[2] / 2.0
    assert floor_top == pytest.approx(-TRENCH.depth, abs=1e-9)


def test_gate_three_is_built_to_fail_at_the_lower_mast(worlds):
    """The test that keeps the whole simulation honest.

    At 4 km/h the vehicle needs 1.426 m to stop. From 0.95 m the trench floor
    is visible at 1.583 m, which clears it. From 0.50 m it is visible at only
    0.833 m, which does not. If the simulation passes at 0.50 m, the
    simulation is wrong and not the analysis.
    """
    stop = stopping_distance(
        4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    seen_from_mast = negative_obstacle_range(
        TRENCH.width, SENSORS.mast_height, TRENCH.depth
    )
    seen_from_low = negative_obstacle_range(TRENCH.width, 0.50, TRENCH.depth)

    assert seen_from_mast > stop, "the mast height must clear the stopping distance"
    assert seen_from_low < stop, "the low mount must not, or the gate proves nothing"

    text = (WORLDS / "trench_field.sdf").read_text()
    assert "PASS" in text and "FAIL" in text
    assert f"{seen_from_mast:.3f}" in text
    assert f"{seen_from_low:.3f}" in text


def test_gate_three_forbids_the_ground_truth_shortcut(worlds):
    """Section 10's one honesty rule, written where it will be read."""
    text = (WORLDS / "trench_field.sdf").read_text()
    assert "/sim/ground_truth/depth" in text
    assert "real stereo matcher" in text
    assert "only for scoring" in text


def test_the_trench_sits_beyond_the_stopping_distance(worlds):
    """The rover must have room to be at speed before it can see the trench,
    or the gate measures the acceleration profile instead of the geometry."""
    named = boxes(worlds["trench_field"])
    near = named["ground_near"]
    trench_x = near[1][0] + near[0][0] / 2.0
    stop = stopping_distance(
        4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    assert trench_x > 3 * stop


# --------------------------------------------------------------------------
# Gate 4. The real risk.
# --------------------------------------------------------------------------


def test_gate_four_puts_textureless_ground_beyond_textured_ground(worlds):
    named = boxes(worlds["snow_scree"])
    scree = named["scree"]
    snow = named["snow"]
    scree_end = scree[1][0] + scree[0][0] / 2.0
    snow_start = snow[1][0] - snow[0][0] / 2.0
    assert scree_end == pytest.approx(snow_start, abs=1e-9)
    assert snow_start > 0, "the vehicle must start on ground it can actually see"


def test_gate_four_hides_a_real_obstacle_under_featureless_albedo(worlds):
    named = boxes(worlds["snow_scree"])
    size, pose = named["drift"]
    assert size[2] > MOBILITY.max_step, "the drift must be over the obstacle claim"
    snow = named["snow"]
    snow_start = snow[1][0] - snow[0][0] / 2.0
    assert pose[0] > snow_start, "the drift must be in the snow, not on the scree"


def test_gate_four_is_beyond_the_blind_zone(worlds):
    """The drift cannot be resolved by driving closer, so it has to be
    handled as unknown."""
    blind = near_edge_range(
        SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg
    )
    named = boxes(worlds["snow_scree"])
    assert named["drift"][1][0] > blind


def test_the_snow_texture_is_genuinely_featureless():
    """The point of gate 4 is a surface a matcher cannot work on. If the
    texture has contrast, the gate passes for the wrong reason."""
    import zlib

    def read_png_grey(path):
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        pos, idat, width, height = 8, b"", None, None
        while pos < len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            tag = data[pos + 4 : pos + 8]
            body = data[pos + 8 : pos + 8 + length]
            if tag == b"IHDR":
                width = int.from_bytes(body[0:4], "big")
                height = int.from_bytes(body[4:8], "big")
            elif tag == b"IDAT":
                idat += body
            pos += 12 + length
        raw = zlib.decompress(idat)
        stride = width + 1
        return [
            list(raw[row * stride + 1 : (row + 1) * stride]) for row in range(height)
        ]

    def spread(path):
        rows = read_png_grey(path)
        flat = [v for row in rows for v in row]
        return max(flat) - min(flat)

    snow = spread(TEXTURES / "snow.png")
    scree = spread(TEXTURES / "scree.png")
    assert snow < 8, f"the snow texture has {snow} grey levels of contrast"
    assert scree > 60, f"the scree texture only has {scree} grey levels"
    assert scree > 8 * snow


# --------------------------------------------------------------------------
# The files stay generated.
# --------------------------------------------------------------------------


def test_regenerating_reproduces_the_committed_worlds():
    before = {p: p.read_bytes() for p in sorted(WORLDS.glob("*.sdf"))}
    before |= {p: p.read_bytes() for p in sorted(TEXTURES.glob("*.png"))}
    subprocess.run(
        [sys.executable, str(GENERATOR)], check=True, capture_output=True, cwd=ROOT
    )
    for path, content in before.items():
        assert path.read_bytes() == content, (
            f"{path.name} is stale. Run tools/generate_worlds.py and commit it."
        )
