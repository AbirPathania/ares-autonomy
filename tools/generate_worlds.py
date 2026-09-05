"""Generate the four Gazebo worlds from constants.py. Build spec section 10.

    gate 1  flat.sdf          frames, calibration, the differential, drive control
    gate 2  step_field.sdf    the obstacle claim from the geometry baseline
    gate 3  trench_field.sdf  section 02, the governing case
    gate 4  snow_scree.sdf    the real risk, texture rather than geometry

Gate 3 is the one that matters, and it should *fail* at the lower mast height.
A simulation that passes everything you throw at it is not calibrated, it is
decorative. So trench_field is built to be seen from 0.95 m and missed from
0.50 m, and the phase 5 gate runs it twice to prove both.

Textures are generated here too, as PNG albedo maps. The snow world needs a
surface that is geometrically flat and visually featureless, and there is no
stock Gazebo material that is honestly textureless. Painting one is the whole
point of gate 4.

Run:  python tools/generate_worlds.py
"""

import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from xml.dom import minidom

import numpy as np

from ares_math.constants import CAMERA, CHASSIS, LIMITS, MOBILITY, SENSORS, TRENCH
from ares_math.optics import near_edge_range
from ares_math.safety import negative_obstacle_range, stopping_distance

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "ares_sim" / "worlds"
TEXTURES = ROOT / "ares_sim" / "textures"

# The world is large enough to hold the longest stopping distance the vehicle
# can produce, with room to accelerate into it and room to stop after it.
APPROACH = stopping_distance(
    LIMITS.v_transit, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
)
GROUND_HALF_WIDTH = 8.0
GROUND_THICKNESS = 0.5


def fmt(value):
    return repr(float(value))


def triple(values):
    return " ".join(fmt(v) for v in values)


def sub(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        node.text = str(text)
    return node


# --------------------------------------------------------------------------
# Textures. Written as PNG by hand so the repository needs no image library
# at generation time and the bytes are reproducible.
# --------------------------------------------------------------------------


def write_png(path, array):
    """Greyscale 8 bit PNG, no dependencies."""
    height, width = array.shape
    raw = b"".join(
        b"\x00" + array[row].astype(np.uint8).tobytes() for row in range(height)
    )

    def chunk(tag, data):
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">2I5B", width, height, 8, 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return path


def texture_field(size, contrast, base, seed):
    """Band limited noise. contrast is the peak to peak swing in grey levels."""
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size))
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    for octave in range(5):
        k = 2.0 * np.pi * (2.0**octave) * 4.0
        phase = rng.uniform(0, 2 * np.pi, 2)
        field += np.sin(k * x + phase[0]) * np.cos(k * y + phase[1]) / (2.0**octave)
    field /= np.abs(field).max()
    return np.clip(base + contrast * field, 0, 255)


def generate_textures(size=512):
    """Two surfaces that differ only in how much a matcher can find on them."""
    written = []
    # Scree. Plenty of structure, which is what passive stereo needs.
    written.append(
        write_png(TEXTURES / "scree.png", texture_field(size, 70.0, 120.0, seed=11))
    )
    # Fresh snow in full sun. Bright, near uniform, and this is the entire
    # failure mode the camera-only architecture is exposed to.
    written.append(
        write_png(TEXTURES / "snow.png", texture_field(size, 1.5, 240.0, seed=12))
    )
    # Neutral ground for the geometry worlds, where texture is not the variable.
    written.append(
        write_png(TEXTURES / "ground.png", texture_field(size, 45.0, 130.0, seed=13))
    )
    return written


# --------------------------------------------------------------------------
# World construction.
# --------------------------------------------------------------------------


def preamble(name, description):
    sdf = ET.Element("sdf", {"version": "1.10"})
    sdf.append(
        ET.Comment(
            f"\n  {description}\n\n"
            "  GENERATED FILE. Do not hand edit.\n"
            "  Source: ares_math/constants.py, via tools/generate_worlds.py.\n"
        )
    )
    world = sub(sdf, "world", name=name)

    physics = sub(world, "physics", name="default", type="dart")
    # A millisecond step. The contact between a 105 mm wheel and a step edge
    # is stiff, and a coarser step lets the wheel tunnel through the corner,
    # which would quietly pass gate 2 for the wrong reason.
    sub(physics, "max_step_size", "0.001")
    # Below one on four software rendered cores. Every gate here is a
    # geometric or logical assertion, not a timing one.
    sub(physics, "real_time_factor", "1.0")

    for plugin in (
        ("gz-sim-physics-system", "gz::sim::systems::Physics"),
        ("gz-sim-sensors-system", "gz::sim::systems::Sensors"),
        ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
        ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
        ("gz-sim-contact-system", "gz::sim::systems::Contact"),
        ("gz-sim-imu-system", "gz::sim::systems::Imu"),
    ):
        node = sub(world, "plugin", filename=plugin[0], name=plugin[1])
        if plugin[0] == "gz-sim-sensors-system":
            # ogre, not ogre2. With LIBGL_ALWAYS_SOFTWARE the newer engine
            # crawls or fails outright. Build spec section 06.2.
            sub(node, "render_engine", "ogre")

    sub(world, "gravity", "0 0 -9.8066")

    scene = sub(world, "scene")
    sub(scene, "ambient", "0.6 0.6 0.6 1")
    sub(scene, "background", "0.7 0.8 0.9 1")
    sub(scene, "shadows", "true")

    return sdf, world


def add_sun(world, elevation_deg=45.0, azimuth_deg=30.0):
    """A sun with a stated elevation, because the snow case depends on it.

    Sun at 4,500 m washes out active infrared projectors, which is one of the
    reasons the mast unit is passive stereo rather than a projector unit.
    """
    light = sub(world, "light", name="sun", type="directional")
    sub(light, "cast_shadows", "true")
    sub(light, "pose", "0 0 20 0 0 0")
    sub(light, "diffuse", "1 1 1 1")
    sub(light, "specular", "0.2 0.2 0.2 1")
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    sub(
        light,
        "direction",
        triple((-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), -np.sin(el))),
    )
    attenuation = sub(light, "attenuation")
    sub(attenuation, "range", "1000")
    sub(attenuation, "linear", "0.0")
    sub(attenuation, "constant", "1.0")
    sub(attenuation, "quadratic", "0.0")


def add_box(parent_model, name, size, pose, texture=None, static_friction=1.0):
    link = sub(parent_model, "link", name=name)
    sub(link, "pose", triple(pose) + " 0 0 0")

    collision = sub(link, "collision", name=f"{name}_collision")
    geometry = sub(collision, "geometry")
    sub(sub(geometry, "box"), "size", triple(size))
    surface = sub(collision, "surface")
    friction = sub(surface, "friction")
    ode = sub(friction, "ode")
    sub(ode, "mu", fmt(static_friction))
    sub(ode, "mu2", fmt(static_friction))

    visual = sub(link, "visual", name=f"{name}_visual")
    geometry = sub(visual, "geometry")
    sub(sub(geometry, "box"), "size", triple(size))
    material = sub(visual, "material")
    sub(material, "ambient", "0.5 0.5 0.5 1")
    sub(material, "diffuse", "0.8 0.8 0.8 1")
    if texture:
        pbr = sub(material, "pbr")
        metal = sub(pbr, "metal")
        sub(metal, "albedo_map", f"../textures/{texture}")
        sub(metal, "roughness", "0.9")
        sub(metal, "metalness", "0.0")
    return link


def ground_slab(model, name, x_start, x_end, texture, top=0.0):
    """A slab of ground whose top face sits at `top`."""
    length = x_end - x_start
    add_box(
        model,
        name,
        size=(length, 2 * GROUND_HALF_WIDTH, GROUND_THICKNESS),
        pose=(x_start + length / 2.0, 0.0, top - GROUND_THICKNESS / 2.0),
        texture=texture,
    )


def terrain_model(world, name="terrain"):
    model = sub(world, "model", name=name)
    sub(model, "static", "true")
    sub(model, "pose", "0 0 0 0 0 0")
    return model


def annotate(world, lines):
    """Record what the world is for and what it should prove, inside the file."""
    world.append(ET.Comment("\n  " + "\n  ".join(lines) + "\n  "))


# --------------------------------------------------------------------------
# The four worlds.
# --------------------------------------------------------------------------


def flat():
    """Gate 1. Frames, calibration, the differential, drive control.

    Pass criterion: stereo depth of a flat plane at 3 m agrees with ground
    truth within the section 03.2 error bound.
    """
    from ares_math.optics import focal_px
    from ares_math.stereo import depth_error

    sdf, world = preamble("flat", "Gate 1. Frames, calibration, drive control.")
    add_sun(world)
    model = terrain_model(world)
    ground_slab(model, "ground", -APPROACH, APPROACH * 2.0, "ground.png")

    f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
    bound = depth_error(
        Z=3.0, f=f, B=SENSORS.mast_baseline, d_disp=CAMERA.disparity_sigma_px
    )
    annotate(
        world,
        [
            "GATE 1",
            "Stereo depth of this plane at 3 m must agree with ground truth",
            f"to within {bound:.4f} m, which is the section 03.2 error bound",
            f"for f={f:.1f} px, B={SENSORS.mast_baseline} m and a",
            f"{CAMERA.disparity_sigma_px} px disparity sigma.",
            "Also: no TF warnings, the differential holds,",
            "and the rover drives straight when told to.",
        ],
    )
    return "flat", sdf


def step_field():
    """Gate 2. The obstacle claim from the geometry baseline.

    Pass criterion: the rover climbs the stated step height, and the belly and
    bogie stop binding where the baseline says they do.
    """
    sdf, world = preamble("step_field", "Gate 2. The obstacle claim.")
    add_sun(world)
    model = terrain_model(world)
    ground_slab(model, "ground", -APPROACH, 30.0, "ground.png")

    # A staircase that brackets the claim: comfortably under, at the claim,
    # and over it. The gate is that the vehicle clears the first two and
    # stops at the third, not that it clears everything.
    steps = [
        ("under", MOBILITY.max_step * 0.5, 4.0),
        ("at_claim", MOBILITY.max_step, 8.0),
        ("over", MOBILITY.max_step * 1.6, 12.0),
    ]
    for name, height, x in steps:
        add_box(
            model,
            f"step_{name}",
            size=(2.0, 2 * GROUND_HALF_WIDTH, height),
            pose=(x, 0.0, height / 2.0),
            texture="ground.png",
        )

    annotate(
        world,
        [
            "GATE 2",
            f"Steps at {steps[0][1]:.3f}, {steps[1][1]:.3f} and {steps[2][1]:.3f} m.",
            f"The obstacle claim is {MOBILITY.max_step} m. The rover must clear",
            "the first two and must not clear the third. The bogie must",
            "articulate on the way over: a rigid six-wheeler would fail the",
            "first step, and a model with the differential missing would",
            "climb all three because the hull cannot pitch to bind.",
            f"Belly clearance is {CHASSIS.ground_clearance} m at the V apex.",
        ],
    )
    return "step_field", sdf


def trench_field():
    """Gate 3, the one that matters. Section 02's governing case.

    Pass criterion: detects a 0.5 by 0.3 m trench and stops, at 4 km/h, with
    the mast at 0.95 m. Fails at 0.50 m, confirming the model.
    """
    sdf, world = preamble("trench_field", "Gate 3. The governing case, section 02.")
    add_sun(world)
    model = terrain_model(world)

    # Placed far enough out that the vehicle is at speed and the geometry is
    # the thing under test, not the acceleration profile.
    trench_x = 6.0
    ground_slab(model, "ground_near", -APPROACH, trench_x, "ground.png")
    ground_slab(model, "ground_far", trench_x + TRENCH.width, 30.0, "ground.png")
    # The floor of the trench, whose top face is at minus the depth.
    add_box(
        model,
        "trench_floor",
        size=(TRENCH.width, 2 * GROUND_HALF_WIDTH, GROUND_THICKNESS),
        pose=(
            trench_x + TRENCH.width / 2.0,
            0.0,
            -TRENCH.depth - GROUND_THICKNESS / 2.0,
        ),
        texture="ground.png",
    )

    r_mast = negative_obstacle_range(TRENCH.width, SENSORS.mast_height, TRENCH.depth)
    r_low = negative_obstacle_range(TRENCH.width, 0.50, TRENCH.depth)
    stop = stopping_distance(
        4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    annotate(
        world,
        [
            "GATE 3, THE ONE THAT MATTERS",
            f"A {TRENCH.width} by {TRENCH.depth} m trench at x = {trench_x} m.",
            "",
            f"Stopping distance at 4 km/h is {stop:.3f} m.",
            f"From a mast at {SENSORS.mast_height} m the floor is visible from"
            f" {r_mast:.3f} m: PASS.",
            f"From 0.50 m the floor is visible from only {r_low:.3f} m: FAIL.",
            "",
            "Run the rover at 4 km/h twice, once at each height. It must stop",
            "in time from 0.95 m and must not from 0.50 m. If it passes at",
            "0.50 m the simulation is wrong, not the analysis. Find out why",
            "before proceeding.",
            "",
            "The perception path must subscribe ONLY to the two rendered RGB",
            "topics and run a real stereo matcher. Ground truth depth goes to",
            "/sim/ground_truth/depth and is used only for scoring.",
        ],
    )
    return "trench_field", sdf


def snow_scree():
    """Gate 4. The real risk, which is texture and not geometry.

    Pass criterion: low texture ground drives confidence down and speed down,
    rather than producing confident wrong depth.
    """
    sdf, world = preamble("snow_scree", "Gate 4. Texture, not geometry.")
    # Sun high and bright, which is the case that washes out active infrared
    # projectors and leaves passive stereo with nothing but albedo.
    add_sun(world, elevation_deg=70.0)
    model = terrain_model(world)

    snow_starts = 6.0
    ground_slab(model, "scree", -APPROACH, snow_starts, "scree.png")
    ground_slab(model, "snow", snow_starts, 30.0, "snow.png")

    # A drift with real geometry hidden under featureless albedo. The vehicle
    # must not drive into it at speed, and the only signal that it is there
    # is the absence of confident depth.
    add_box(
        model,
        "drift",
        size=(1.2, 3.0, MOBILITY.max_step * 2.5),
        pose=(11.0, 0.0, MOBILITY.max_step * 1.25),
        texture="snow.png",
    )

    blind = near_edge_range(
        SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg
    )
    annotate(
        world,
        [
            "GATE 4, THE REAL RISK",
            f"Textured scree out to {snow_starts} m, then textureless snow.",
            f"A drift {MOBILITY.max_step * 2.5:.3f} m high at 11 m, which is over",
            "the obstacle claim and is hidden under featureless albedo.",
            "",
            "PASS means confidence collapses over the snow and the vehicle",
            "slows or stops. FAIL means it reports confident wrong depth and",
            "keeps its speed. A stack that treats unknown as free space will",
            "drive into this at full speed, and it will do it in the field",
            "rather than here.",
            "",
            f"The mast blind zone is {blind:.3f} m, so the drift is never",
            "resolved by driving closer. It has to be handled as unknown.",
        ],
    )
    return "snow_scree", sdf


BUILDERS = (flat, step_field, trench_field, snow_scree)


def pretty(root):
    raw = ET.tostring(root, encoding="unicode")
    text = minidom.parseString(raw).toprettyxml(indent="  ")
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


def main():
    WORLDS.mkdir(parents=True, exist_ok=True)
    textures = generate_textures()
    for path in textures:
        print(f"wrote {path.relative_to(ROOT)}")
    for builder in BUILDERS:
        name, sdf = builder()
        path = WORLDS / f"{name}.sdf"
        path.write_text(pretty(sdf))
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
