"""ARES-G geometry. SOURCE OF TRUTH: hull revision H-2, frozen.

Datum per H-2 section 01: origin at ground plane, on centerline,
at the middle axle station. X forward, Y left, Z up. All SI, metres.
Every value carries the document and revision it came from.

Rule (build spec section 11): no physical constant is written anywhere
else in this repository. ROS nodes, URDF, world files and launch files
import from here or are generated from here. A literal number outside
this file is a review rejection.
"""

import math
from dataclasses import dataclass

REVISION = "H-2"  # UPDATE, and re-run pytest, whenever the hull revises


@dataclass(frozen=True)
class Chassis:
    wheel_diameter: float = 0.190  # H-2 s01, 190 mm
    rolling_radius: float = 0.105  # H-2 s01, belt included
    axle_pitch_rear: float = 0.260  # rear to middle
    axle_pitch_front: float = 0.280  # middle to front
    track: float = 0.350  # H-2 s01
    trunnion_x: float = 0.007  # H-2 s06, frozen interface
    trunnion_z: float = 0.217  # H-2 s06, frozen interface
    trunnion_y: float = 0.135  # H-2 s06, frozen interface
    hull_length: float = 0.560
    hull_width: float = 0.260
    deck_z: float = 0.320
    ground_clearance: float = 0.115  # V apex
    stowed_height: float = 0.455  # mast retracted


@dataclass(frozen=True)
class MassBudget:
    ceiling: float = 25.0  # H-2 s10, HARD ceiling not a target
    wheeled_mine: float = 23.25  # with 12 percent margin
    tracked_mine: float = 24.91
    tracked_plus_drone: float = 26.10  # OVER by 1.10, see H-2 s10 levers


@dataclass(frozen=True)
class Sensors:
    mast_height: float = 0.95  # DERIVED here, build spec section 02, not from H-2
    mast_tilt_deg: float = 20.0
    mast_baseline: float = 0.20
    deck_height: float = 0.36
    deck_tilt_deg: float = 35.0
    deck_baseline: float = 0.12


@dataclass(frozen=True)
class Limits:
    v_autonomous: float = 1.111  # 4 km/h, justified build spec section 02
    v_transit: float = 2.222  # 8 km/h, TELEOPERATED ONLY
    decel: float = 1.0  # loose ground, conservative
    latency: float = 0.30  # sense to brake, budget in build spec section 03.3
    safety_factor: float = 1.5


@dataclass(frozen=True)
class Camera:
    """Section 03 working case. Optics, not hull geometry.

    These are the numbers every table in section 03 was computed with.
    A real unit substitutes its own datasheet values here and the tables
    regenerate. Nothing downstream hard-codes them.
    """

    width_px: int = 1280
    height_px: int = 720
    hfov_deg: float = 90.0
    vfov_deg: float = 59.0  # assumed in section 03.4
    max_disparity_px: int = 192  # matcher search range, sets minimum range

    # What the SIMULATOR renders at, which is not what the real unit sees.
    # Section 06.2: keep simulated camera resolution at 640 by 360, because
    # the renderer is software. The field of view is unchanged, so the focal
    # length simply scales with the width and every section 03 relation still
    # holds. Nodes take their intrinsics from CameraInfo, so nothing
    # downstream needs to know which of the two it is looking at.
    sim_width_px: int = 640
    sim_height_px: int = 360
    # And a rate the software renderer can actually sustain for four cameras.
    # Asking for 30 does not produce 30, it produces two eyes rendered on
    # different passes, which is a 99 ms exposure skew the synchroniser is
    # right to refuse.
    sim_update_rate: float = 5.0
    disparity_sigma_px: float = 0.25  # nominal, good matcher on textured ground
    disparity_sigma_px_pessimistic: float = 0.50  # low-contrast terrain


@dataclass(frozen=True)
class GoverningObstacle:
    """The trench that sizes the mast. Build spec section 02.

    A modest and very common off-road feature. Every speed limit and
    mast height in this repository traces back to these two numbers.
    """

    width: float = 0.5
    depth: float = 0.3


@dataclass(frozen=True)
class RockerBogie:
    """Articulation geometry.

    bogie_pivot_x is the midpoint of the middle and rear axle stations.
    bogie_pivot_z is set equal to trunnion_z, which makes the rear rocker
    arm horizontal in the neutral pose.

    MODEL ASSUMPTION, not stated in H-2. Flagged for H-3 to close.
    Everything else here is computed from Chassis.
    """

    bogie_pivot_x: float = -0.130  # -(axle_pitch_rear / 2)
    bogie_pivot_z: float = 0.217  # == trunnion_z, see docstring


@dataclass(frozen=True)
class Mobility:
    """What the vehicle can drive over. Build spec section 10, gate 2.

    max_step is the obstacle claim the step_field world exists to test. It is
    roughly 1.3 wheel diameters, which is the classic rocker-bogie figure and
    the reason the linkage is worth its mass.
    """

    max_step: float = 0.245


@dataclass(frozen=True)
class LinkMasses:
    """Per link mass for the simulation model, kilograms.

    MODEL ASSUMPTION, not stated in H-2. H-2 gives configuration totals, not
    a link breakdown, and a physics simulation needs one. These distribute
    MassBudget.wheeled_mine across the links in proportions that are
    plausible for the parts they represent, and the total is asserted equal
    to the budget by test_constants.py.

    Replace with the real breakdown when H-3 provides it. Until then, treat
    absolute dynamic results from the simulator as indicative and relative
    results, such as whether the bogie articulates over a step, as sound.
    """

    wheel: float = 0.45  # each, six of them
    rocker: float = 0.60  # each, two of them
    bogie: float = 0.45  # each, two of them
    mast: float = 0.35
    camera_unit: float = 0.15  # each, two of them
    hull: float = 17.80  # the remainder, including compute and payload


CHASSIS = Chassis()
MASS = MassBudget()
SENSORS = Sensors()
LIMITS = Limits()
CAMERA = Camera()
TRENCH = GoverningObstacle()
LINK_MASS = LinkMasses()
MOBILITY = Mobility()
BOGIE = RockerBogie()

# Axle station X positions in the base_link frame, front to rear.
AXLE_X = (CHASSIS.axle_pitch_front, 0.0, -CHASSIS.axle_pitch_rear)
# Wheel centre Y, left positive.
WHEEL_Y = CHASSIS.track / 2.0

# Camera optical frame rotation relative to its mount link.
# base_link convention is X forward, Y left, Z up.
# Optical convention is Z forward, X right, Y down.
# This is the fixed rotation between them, build spec section 08.
#
# The document prints -1.5707963268, which is a ten digit rounding of -pi/2.
# Stored exactly here instead, because the rounding leaves a 3.2 nanoradian
# residual that turns an exact axis test into an approximate one for no
# reason. test_constants.py asserts this still matches what the document
# prints, to the precision the document prints it at.
OPTICAL_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)
OPTICAL_RPY_AS_PRINTED = (-1.5707963268, 0.0, -1.5707963268)
