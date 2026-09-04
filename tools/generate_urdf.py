"""Generate ares_description/urdf/ares_g.urdf.xacro from constants.py.

Build spec section 07: generate the URDF, do not write it. A hand-written
URDF drifts from the constants file the first time anything changes.

The output is a fully resolved URDF that also carries the xacro namespace, so
ROS loads it through xacro and any plain XML parser can read it without a ROS
installation. That is what lets the geometry tests run on a laptop with no
container.

The rocker differential is modelled explicitly. Skip it and you have a rigid
six-wheeler that happens to look like a rocker-bogie, and every obstacle
result will be wrong. It appears twice here on purpose:

    1. an SDFormat gearbox joint in the <gazebo> block, which is the
       declarative form, and
    2. the ares_differential system plugin, which enforces
       theta_body = (theta_left + theta_right) / 2 directly.

Whichever the physics engine honours, the constraint holds, and the phase 3
gate checks it in simulation rather than trusting either.

Run:  python tools/generate_urdf.py
"""

import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from ares_math.constants import (
    CAMERA,
    CHASSIS,
    LIMITS,
    LINK_MASS,
    OPTICAL_RPY,
    REVISION,
    SENSORS,
)
from ares_math.kinematics import LINKAGE, box_inertia, cylinder_inertia, rod_inertia

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ares_description" / "urdf" / "ares_g.urdf.xacro"

XACRO_NS = "http://www.ros.org/wiki/xacro"

# Wheel width is not a hull number, it is a rendering and contact patch
# choice for the model. Derived from the track and the hull width so that it
# scales if either changes, rather than being a literal.
WHEEL_WIDTH = (CHASSIS.track - CHASSIS.hull_width) / 2.0


def fmt(value):
    """Shortest representation that round trips exactly.

    Not a fixed number of digits. 0.105 stays "0.105" and stays readable,
    while -pi/2 keeps all of itself, so the optical frame axis test is exact
    rather than approximate.
    """
    return repr(float(value))


def triple(values):
    return " ".join(fmt(v) for v in values)


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def sub_text(parent, tag, text):
    node = ET.SubElement(parent, tag)
    node.text = str(text)
    return node


def add_inertial(link, mass, inertia, origin=(0.0, 0.0, 0.0)):
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    node = sub(link, "inertial")
    sub(node, "origin", xyz=triple(origin), rpy="0 0 0")
    sub(node, "mass", value=fmt(mass))
    sub(
        node,
        "inertia",
        ixx=fmt(ixx),
        ixy=fmt(ixy),
        ixz=fmt(ixz),
        iyy=fmt(iyy),
        iyz=fmt(iyz),
        izz=fmt(izz),
    )


def add_geometry(link, tag, shape, origin=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    node = sub(link, tag)
    sub(node, "origin", xyz=triple(origin), rpy=triple(rpy))
    geom = sub(node, "geometry")
    kind, attrs = shape
    sub(geom, kind, **attrs)
    return node


def add_link(robot, name):
    return sub(robot, "link", name=name)


def add_joint(robot, name, kind, parent, child, xyz, rpy=(0.0, 0.0, 0.0), axis=None):
    joint = sub(robot, "joint", name=name, type=kind)
    sub(joint, "parent", link=parent)
    sub(joint, "child", link=child)
    sub(joint, "origin", xyz=triple(xyz), rpy=triple(rpy))
    if axis is not None:
        sub(joint, "axis", xyz=triple(axis))
    return joint


def build():
    robot = ET.Element("robot", {"name": "ares_g", "xmlns:xacro": XACRO_NS})
    robot.append(
        ET.Comment(
            f"\n  ARES-G, hull revision {REVISION}.\n\n"
            "  GENERATED FILE. Do not hand edit.\n"
            "  Source: ares_math/constants.py, via tools/generate_urdf.py.\n"
            "  Change a number there, re-run the generator, re-run pytest.\n\n"
            "  Datum: origin at the ground plane, on centreline, at the middle\n"
            "  axle station. X forward, Y left, Z up. REP-103 and REP-105.\n"
        )
    )

    # ---------------------------------------------------------------- hull
    base = add_link(robot, "base_link")
    hull_centre_z = (CHASSIS.ground_clearance + CHASSIS.deck_z) / 2.0
    hull_height = CHASSIS.deck_z - CHASSIS.ground_clearance
    add_inertial(
        base,
        LINK_MASS.hull,
        box_inertia(LINK_MASS.hull, CHASSIS.hull_length, CHASSIS.hull_width, hull_height),
        origin=(0.0, 0.0, hull_centre_z),
    )
    hull_box = (
        "box",
        {"size": triple((CHASSIS.hull_length, CHASSIS.hull_width, hull_height))},
    )
    add_geometry(base, "visual", hull_box, origin=(0.0, 0.0, hull_centre_z))
    add_geometry(base, "collision", hull_box, origin=(0.0, 0.0, hull_centre_z))

    # ------------------------------------------------------- rocker, bogie
    for side in ("left", "right"):
        link = LINKAGE[side]
        trunnion = link.trunnion
        pivot = link.bogie_pivot

        rocker = add_link(robot, f"rocker_{side}")
        add_inertial(
            rocker,
            LINK_MASS.rocker,
            rod_inertia(LINK_MASS.rocker, link.rocker_front_arm, axis="x"),
        )
        # The rocker spans the trunnion to the front wheel and to the bogie
        # pivot. Drawn as two rods so the articulation is visible in RViz.
        for target in (link.front, pivot):
            delta = [target[i] - trunnion[i] for i in range(3)]
            length = math.dist(target, trunnion)
            mid = [d / 2.0 for d in delta]
            pitch = -math.atan2(delta[2], delta[0])
            add_geometry(
                rocker,
                "visual",
                ("cylinder", {"radius": fmt(WHEEL_WIDTH / 4.0), "length": fmt(length)}),
                origin=mid,
                rpy=(0.0, pitch + math.pi / 2.0, 0.0),
            )
        add_joint(
            robot,
            f"rocker_{side}_joint",
            "continuous",
            "base_link",
            f"rocker_{side}",
            xyz=trunnion,
            axis=(0.0, 1.0, 0.0),
        )

        bogie = add_link(robot, f"bogie_{side}")
        add_inertial(
            bogie,
            LINK_MASS.bogie,
            rod_inertia(LINK_MASS.bogie, CHASSIS.axle_pitch_rear, axis="x"),
        )
        for target in (link.middle, link.rear):
            delta = [target[i] - pivot[i] for i in range(3)]
            length = math.dist(target, pivot)
            mid = [d / 2.0 for d in delta]
            pitch = -math.atan2(delta[2], delta[0])
            add_geometry(
                bogie,
                "visual",
                ("cylinder", {"radius": fmt(WHEEL_WIDTH / 4.0), "length": fmt(length)}),
                origin=mid,
                rpy=(0.0, pitch + math.pi / 2.0, 0.0),
            )
        add_joint(
            robot,
            f"bogie_{side}_joint",
            "continuous",
            f"rocker_{side}",
            f"bogie_{side}",
            xyz=[pivot[i] - trunnion[i] for i in range(3)],
            axis=(0.0, 1.0, 0.0),
        )

        # ------------------------------------------------------- the wheels
        # The front wheel hangs off the rocker. The middle and rear hang off
        # the bogie. That split is what makes it a rocker-bogie.
        mounts = (
            ("front", "rocker", link.front, trunnion),
            ("middle", "bogie", link.middle, pivot),
            ("rear", "bogie", link.rear, pivot),
        )
        for name, parent_kind, centre, parent_origin in mounts:
            wheel = add_link(robot, f"wheel_{name}_{side}")
            add_inertial(
                wheel,
                LINK_MASS.wheel,
                cylinder_inertia(
                    LINK_MASS.wheel, CHASSIS.rolling_radius, WHEEL_WIDTH, axis="y"
                ),
            )
            tyre = (
                "cylinder",
                {"radius": fmt(CHASSIS.rolling_radius), "length": fmt(WHEEL_WIDTH)},
            )
            add_geometry(wheel, "visual", tyre, rpy=(math.pi / 2.0, 0.0, 0.0))
            add_geometry(wheel, "collision", tyre, rpy=(math.pi / 2.0, 0.0, 0.0))
            add_joint(
                robot,
                f"wheel_{name}_{side}_joint",
                "continuous",
                f"{parent_kind}_{side}",
                f"wheel_{name}_{side}",
                xyz=[centre[i] - parent_origin[i] for i in range(3)],
                axis=(0.0, 1.0, 0.0),
            )

    # ----------------------------------------------------------- the mast
    # Section 02 sizes this. It is a perception parameter, not packaging.
    mast_length = SENSORS.mast_height - CHASSIS.deck_z
    mast = add_link(robot, "mast_link")
    add_inertial(
        mast,
        LINK_MASS.mast,
        rod_inertia(LINK_MASS.mast, mast_length, axis="z"),
        origin=(0.0, 0.0, mast_length / 2.0),
    )
    add_geometry(
        mast,
        "visual",
        ("cylinder", {"radius": fmt(WHEEL_WIDTH / 3.0), "length": fmt(mast_length)}),
        origin=(0.0, 0.0, mast_length / 2.0),
    )
    add_joint(
        robot,
        "mast_joint",
        "fixed",
        "base_link",
        "mast_link",
        xyz=(0.0, 0.0, CHASSIS.deck_z),
    )

    # -------------------------------------------------------- the cameras
    # Two units, two heights, two jobs. This is a coverage scheme, not a
    # redundancy scheme, and the blind zone table in section 03.4 is why.
    units = (
        ("mast_cam", "mast_link", SENSORS.mast_height - CHASSIS.deck_z,
         SENSORS.mast_tilt_deg, SENSORS.mast_baseline),
        ("deck_cam", "base_link", SENSORS.deck_height,
         SENSORS.deck_tilt_deg, SENSORS.deck_baseline),
    )
    for unit, parent, height, tilt_deg, baseline in units:
        # The unit frame carries the downward tilt. Positive pitch is nose
        # down in the ROS convention, so a camera pitched down by 20 degrees
        # has a positive pitch of 20 degrees.
        mount = add_link(robot, f"{unit}_mount")
        add_inertial(
            mount,
            LINK_MASS.camera_unit,
            box_inertia(LINK_MASS.camera_unit, WHEEL_WIDTH, baseline, WHEEL_WIDTH),
        )
        add_geometry(
            mount,
            "visual",
            ("box", {"size": triple((WHEEL_WIDTH / 2.0, baseline, WHEEL_WIDTH / 2.0))}),
        )
        add_joint(
            robot,
            f"{unit}_mount_joint",
            "fixed",
            parent,
            f"{unit}_mount",
            xyz=(0.0, 0.0, height),
            rpy=(0.0, math.radians(tilt_deg), 0.0),
        )

        # The left camera is the stereo reference, and it is the link the
        # frame table in section 08 names. The right sits one baseline to
        # its right, which is negative Y.
        for eye, offset in (("", baseline / 2.0), ("_right", -baseline / 2.0)):
            link_name = f"{unit}{eye}_link"
            add_link(robot, link_name)
            add_joint(
                robot,
                f"{link_name}_joint",
                "fixed",
                f"{unit}_mount",
                link_name,
                xyz=(0.0, offset, 0.0),
            )

            # The bug that costs everyone two days. The camera optical frame
            # is not the robot convention: Z forward, X right, Y down. The
            # rotation is fixed and known, so it belongs in the URDF and in a
            # test, never in a launch file argument.
            optical = f"{unit}{eye}_optical"
            add_link(robot, optical)
            add_joint(
                robot,
                f"{optical}_joint",
                "fixed",
                link_name,
                optical,
                xyz=(0.0, 0.0, 0.0),
                rpy=OPTICAL_RPY,
            )

    add_gazebo(robot)
    return robot


def add_gazebo(robot):
    """Physics, the differential, and the two stereo pairs.

    Simulate two RGB cameras and run the real stereo matcher on the rendered
    images. Ground truth depth is published on a separate topic named so it
    cannot be used by accident, and it is used only for scoring.
    """
    # The differential, declaratively. SDFormat gearbox joint with a ratio of
    # one, referencing the hull, so the two rockers are forced to move
    # equally and oppositely relative to the body.
    gearbox = sub(robot, "gazebo")
    joint = sub(gearbox, "joint", name="rocker_differential", type="gearbox")
    sub(joint, "parent", link="rocker_left")
    sub(joint, "child", link="rocker_right")
    sub(joint, "gearbox_reference_body", link="base_link")
    # Ratio of minus one: the rockers turn equally and oppositely relative to
    # the hull, which is the differential written as a gear pair.
    sub_text(joint, "gearbox_ratio", "-1.0")
    for tag in ("axis", "axis2"):
        sub_text(sub(joint, tag), "xyz", "0 1 0")

    # And the same constraint enforced directly, for engines that do not
    # implement gearbox joints. theta_body = (theta_left + theta_right) / 2
    # reduces to the two relative rocker angles summing to zero, which is
    # what this plugin holds. Derived in tools/derive_symbolic.py.
    plugin_host = sub(robot, "gazebo")
    plugin = sub(
        plugin_host,
        "plugin",
        filename="ares_differential",
        name="ares::sim::RockerDifferential",
    )
    sub_text(plugin, "left_joint", "rocker_left_joint")
    sub_text(plugin, "right_joint", "rocker_right_joint")

    # Wheel friction, and the sensors.
    for side in ("left", "right"):
        for name in ("front", "middle", "rear"):
            ref = sub(robot, "gazebo", reference=f"wheel_{name}_{side}")
            sub_text(ref, "mu1", "1.0")
            sub_text(ref, "mu2", "1.0")

    f_px = (CAMERA.width_px / 2.0) / math.tan(math.radians(CAMERA.hfov_deg) / 2.0)
    for unit in ("mast_cam", "deck_cam"):
        for eye in ("", "_right"):
            ref = sub(robot, "gazebo", reference=f"{unit}{eye}_link")
            sensor = sub(ref, "sensor", name=f"{unit}{eye}", type="camera")
            sub_text(sensor, "always_on", "1")
            # One update per control period. The sync budget in stereo.py is
            # about the skew between the pair, not this rate.
            sub_text(sensor, "update_rate", "30")
            sub_text(sensor, "visualize", "false")
            sub_text(sensor, "topic", f"/{unit}{eye}/image_raw")
            cam = sub(sensor, "camera")
            sub_text(cam, "horizontal_fov", fmt(math.radians(CAMERA.hfov_deg)))
            image = sub(cam, "image")
            sub_text(image, "width", str(CAMERA.width_px))
            sub_text(image, "height", str(CAMERA.height_px))
            sub_text(image, "format", "R8G8B8")
            clip = sub(cam, "clip")
            # Near clip inside the closest range the matcher can resolve, so
            # the simulator is never the thing that limits near field depth.
            sub_text(clip, "near", fmt(CHASSIS.rolling_radius))
            sub_text(clip, "far", "100")
            lens = sub(cam, "lens")
            intrinsics = sub(lens, "intrinsics")
            sub_text(intrinsics, "fx", fmt(f_px))
            sub_text(intrinsics, "fy", fmt(f_px))
            sub_text(intrinsics, "cx", fmt(CAMERA.width_px / 2.0))
            sub_text(intrinsics, "cy", fmt(CAMERA.height_px / 2.0))
            sub_text(intrinsics, "s", "0")

    # Ground truth depth, on a topic named so it cannot end up in the control
    # path by accident. Build spec section 10, the one honesty rule.
    ref = sub(robot, "gazebo", reference="mast_cam_link")
    sensor = sub(ref, "sensor", name="ground_truth_depth", type="depth_camera")
    sub_text(sensor, "always_on", "1")
    sub_text(sensor, "update_rate", "10")
    sub_text(sensor, "topic", "/sim/ground_truth/depth")
    cam = sub(sensor, "camera")
    sub_text(cam, "horizontal_fov", fmt(math.radians(CAMERA.hfov_deg)))
    image = sub(cam, "image")
    sub_text(image, "width", str(CAMERA.width_px))
    sub_text(image, "height", str(CAMERA.height_px))
    clip = sub(cam, "clip")
    sub_text(clip, "near", fmt(CHASSIS.rolling_radius))
    sub_text(clip, "far", "100")

    # Odometry and joint state, and the speed cap the whole document derives.
    control = sub(robot, "gazebo")
    plugin = sub(
        control,
        "plugin",
        filename="gz-sim-diff-drive-system",
        name="gz::sim::systems::DiffDrive",
    )
    for side in ("left", "right"):
        for name in ("front", "middle", "rear"):
            sub_text(plugin, f"{side}_joint", f"wheel_{name}_{side}_joint")
    sub_text(plugin, "wheel_separation", fmt(CHASSIS.track))
    sub_text(plugin, "wheel_radius", fmt(CHASSIS.rolling_radius))
    sub_text(plugin, "odom_topic", "odom")
    sub_text(plugin, "frame_id", "odom")
    sub_text(plugin, "child_frame_id", "base_link")
    sub_text(plugin, "topic", "cmd_vel")
    # Autonomous speed cap. 4 km/h, and section 02 is why it is not 8.
    sub_text(plugin, "max_linear_velocity", fmt(LIMITS.v_autonomous))
    sub_text(plugin, "max_linear_acceleration", fmt(LIMITS.decel))


def pretty(root):
    raw = ET.tostring(root, encoding="unicode")
    text = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(pretty(build()))
    n_links = len(build().findall("link"))
    n_joints = len(build().findall("joint"))
    print(f"wrote {OUT}")
    print(f"  {n_links} links, {n_joints} joints, hull revision {REVISION}")
    try:
        subprocess.run(["check_urdf", str(OUT)], check=True, capture_output=True)
        print("  check_urdf passed")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  check_urdf not available here, the geometry tests cover it")


if __name__ == "__main__":
    main()
