"""Rocker-bogie articulation, the differential, and the pose solver.

Build spec section 07.

    ROCKER DIFFERENTIAL, THE THING THAT MAKES IT A ROCKER-BOGIE
    theta_body = (theta_rocker_left + theta_rocker_right) / 2

The two rockers are coupled through a differential bar so the body pitch is
their average. Skip that and you have modelled a rigid six-wheeler that
happens to look like a rocker-bogie, and every obstacle result will be wrong.

Angle convention: rotations about +Y, ROS style, so a positive angle pitches
the nose down. Rocker and bogie angles are relative to their parent, which
is the same convention the URDF joints use.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .constants import BOGIE, CHASSIS
from .units import DEG, M, units

__all__ = [
    "body_pitch",
    "Linkage",
    "LINKAGE",
    "Terrain",
    "Plane",
    "HeightField",
    "Pose",
    "wheel_centres",
    "solve_pose",
    "pose_on_plane",
    "WHEEL_NAMES",
    "box_inertia",
    "cylinder_inertia",
    "rod_inertia",
    "rpy_matrix",
]

WHEEL_NAMES = ("front", "middle", "rear")


@units(None, (DEG, DEG))
def body_pitch(theta_rocker_left, theta_rocker_right):
    """The differential constraint, in degrees.

        theta_body = (theta_left + theta_right) / 2

    Takes and returns absolute rocker pitch angles in degrees.
    """
    return (theta_rocker_left + theta_rocker_right) / 2.0


def _ry(theta):
    """Rotation about +Y by theta radians, ROS convention, nose down positive."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rpy(roll, pitch, yaw=0.0):
    """ROS fixed axis roll pitch yaw to a rotation matrix, Rz @ Ry @ Rx."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class Linkage:
    """Neutral pose link geometry for one side, in the base_link frame.

    Every value is read from constants.py. Nothing here is a literal.
    """

    trunnion: tuple  # rocker pivot on the hull
    bogie_pivot: tuple  # bogie pivot on the rocker
    front: tuple  # front wheel centre
    middle: tuple  # middle wheel centre
    rear: tuple  # rear wheel centre
    radius: float

    @classmethod
    def from_constants(cls, side="left"):
        sign = 1.0 if side == "left" else -1.0
        y_wheel = sign * CHASSIS.track / 2.0
        y_pivot = sign * CHASSIS.trunnion_y
        r = CHASSIS.rolling_radius
        return cls(
            trunnion=(CHASSIS.trunnion_x, y_pivot, CHASSIS.trunnion_z),
            bogie_pivot=(BOGIE.bogie_pivot_x, y_pivot, BOGIE.bogie_pivot_z),
            front=(CHASSIS.axle_pitch_front, y_wheel, r),
            middle=(0.0, y_wheel, r),
            rear=(-CHASSIS.axle_pitch_rear, y_wheel, r),
            radius=r,
        )

    @property
    def rocker_front_arm(self):
        """Trunnion to front wheel, metres."""
        return float(np.linalg.norm(np.subtract(self.front, self.trunnion)))

    @property
    def rocker_rear_arm(self):
        """Trunnion to bogie pivot, metres."""
        return float(np.linalg.norm(np.subtract(self.bogie_pivot, self.trunnion)))

    @property
    def bogie_front_arm(self):
        """Bogie pivot to middle wheel, metres."""
        return float(np.linalg.norm(np.subtract(self.middle, self.bogie_pivot)))

    @property
    def bogie_rear_arm(self):
        """Bogie pivot to rear wheel, metres."""
        return float(np.linalg.norm(np.subtract(self.rear, self.bogie_pivot)))

    def points_in_body(self, theta_rocker, theta_bogie):
        """Wheel centres in the body frame for given relative joint angles.

        Returns a dict of name to (x, y, z), metres. The chain is
        hull -> rocker at the trunnion -> bogie at the bogie pivot.
        """
        t = np.array(self.trunnion)
        p0 = np.array(self.bogie_pivot)
        rr = _ry(theta_rocker)
        pivot = t + rr @ (p0 - t)
        rb = _ry(theta_rocker + theta_bogie)
        return {
            "front": t + rr @ (np.array(self.front) - t),
            "middle": pivot + rb @ (np.array(self.middle) - p0),
            "rear": pivot + rb @ (np.array(self.rear) - p0),
            "bogie_pivot": pivot,
            "trunnion": t,
        }


LINKAGE = {
    "left": Linkage.from_constants("left"),
    "right": Linkage.from_constants("right"),
}


class Terrain:
    """Ground surface. Subclasses provide height(x, y).

    support_height returns the height a wheel centre of radius r sits at when
    resting on this terrain at (x, y). The default samples the wheel envelope,
    which is what makes a wheel ride over a step corner rather than sinking
    through it, and what makes it span a trench narrower than its footprint.
    """

    samples = 401

    def height(self, x, y):
        raise NotImplementedError

    def support_height(self, x, y, radius):
        s = np.linspace(-radius, radius, self.samples)
        ground = np.array([self.height(x + si, y) for si in s])
        return float(np.max(ground + np.sqrt(np.maximum(radius**2 - s**2, 0.0))))


class Plane(Terrain):
    """A ground plane with slope and cross slope, exact support height.

    pitch_deg tilts the ground nose up along +X, roll_deg raises +Y.
    """

    def __init__(self, z0=0.0, pitch_deg=0.0, roll_deg=0.0):
        self.z0 = z0
        self.pitch = np.radians(pitch_deg)
        self.roll = np.radians(roll_deg)

    def height(self, x, y):
        return self.z0 + x * np.tan(self.pitch) + y * np.tan(self.roll)

    def support_height(self, x, y, radius):
        # Perpendicular offset, derived in tools/derive_symbolic.py.
        n = np.sqrt(1.0 + np.tan(self.pitch) ** 2 + np.tan(self.roll) ** 2)
        return self.height(x, y) + radius * n


class HeightField(Terrain):
    """Terrain from a callable, for steps, trenches and rendered scenes."""

    def __init__(self, fn):
        self.fn = fn

    def height(self, x, y):
        return float(self.fn(x, y))


@dataclass(frozen=True)
class Pose:
    """Vehicle state. Lengths in metres, angles in radians."""

    z: float
    pitch: float
    roll: float
    rocker_left: float
    rocker_right: float
    bogie_left: float
    bogie_right: float
    x: float = 0.0
    yaw: float = 0.0

    @property
    def rocker_pitch_left(self):
        """Absolute rocker pitch, the quantity the differential averages."""
        return self.pitch + self.rocker_left

    @property
    def rocker_pitch_right(self):
        return self.pitch + self.rocker_right

    @property
    def differential_residual(self):
        """Should be zero. body pitch minus the mean of the rocker pitches."""
        return self.pitch - 0.5 * (self.rocker_pitch_left + self.rocker_pitch_right)


def wheel_centres(pose):
    """Forward kinematics. Wheel centres in the world frame.

    Returns a dict keyed (side, wheel) to a length 3 numpy array, metres.
    """
    origin = np.array([pose.x, 0.0, pose.z])
    rot = _rpy(pose.roll, pose.pitch, pose.yaw)
    out = {}
    for side, theta_r, theta_b in (
        ("left", pose.rocker_left, pose.bogie_left),
        ("right", pose.rocker_right, pose.bogie_right),
    ):
        pts = LINKAGE[side].points_in_body(theta_r, theta_b)
        for name in WHEEL_NAMES:
            out[(side, name)] = origin + rot @ pts[name]
    return out


def _residuals(u, terrain, x, yaw):
    pose = Pose(
        z=u[0],
        pitch=u[1],
        roll=u[2],
        rocker_left=u[3],
        rocker_right=u[4],
        bogie_left=u[5],
        bogie_right=u[6],
        x=x,
        yaw=yaw,
    )
    centres = wheel_centres(pose)
    r = CHASSIS.rolling_radius
    res = [
        p[2] - terrain.support_height(p[0], p[1], r) for p in centres.values()
    ]
    # The differential: body pitch is the mean of the two rocker pitches,
    # which reduces to the two relative rocker angles summing to zero.
    res.append(pose.rocker_left + pose.rocker_right)
    return res


def solve_pose(terrain, x=0.0, yaw=0.0, guess=None, tol=1e-14):
    """Settle the vehicle onto terrain. Six contacts plus the differential.

    Seven residuals in seven unknowns: body height, pitch, roll, two rocker
    angles and two bogie angles. Solved numerically. Cross-checked against
    pose_on_plane, which is a closed form for planar ground.

    Returns a Pose.
    """
    if guess is None:
        z0 = terrain.support_height(x, 0.0, CHASSIS.rolling_radius)
        guess = [z0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = least_squares(
        _residuals,
        np.asarray(guess, dtype=float),
        args=(terrain, x, yaw),
        xtol=tol,
        ftol=tol,
        gtol=tol,
    )
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-9:
        raise RuntimeError(
            f"pose did not settle, max residual {np.max(np.abs(sol.fun)):.3e}"
        )
    u = sol.x
    return Pose(
        z=u[0],
        pitch=u[1],
        roll=u[2],
        rocker_left=u[3],
        rocker_right=u[4],
        bogie_left=u[5],
        bogie_right=u[6],
        x=x,
        yaw=yaw,
    )


@units(None, (DEG, DEG, M))
def pose_on_plane(ground_pitch_deg, ground_roll_deg, z0=0.0):
    """Closed form pose on a plane of given slope and cross slope.

    All six wheel centres are coplanar in the neutral pose, so on planar
    ground every wheel touches with the articulation at zero and the hull
    simply parallel to the surface. That makes the whole pose available in
    closed form, with no iteration, which is the independent implementation
    solve_pose is checked against.

    The body attitude follows from matching the hull normal to the ground
    normal:

        tan(pitch_body) = -tan(pitch_ground)
        tan(roll_body)  =  cos(pitch_body) * tan(roll_ground)

    The sign flip on pitch is not a typo. Ground pitch here is positive when
    the ground rises ahead, and ROS body pitch is positive nose down.

    Returns a Pose. Angles in radians, lengths in metres.
    """
    tan_pg = np.tan(np.radians(ground_pitch_deg))
    tan_rg = np.tan(np.radians(ground_roll_deg))
    r = CHASSIS.rolling_radius

    pitch = np.arctan(-tan_pg)
    roll = np.arctan(np.cos(pitch) * tan_rg)

    rot = _rpy(roll, pitch, 0.0)
    normal_scale = np.sqrt(1.0 + tan_pg**2 + tan_rg**2)

    # Put one known wheel on the surface and read the hull height off it.
    m_body = np.array(LINKAGE["left"].middle)
    wx, wy, wz = rot @ m_body
    z = z0 + wx * tan_pg + wy * tan_rg + r * normal_scale - wz

    return Pose(
        z=float(z),
        pitch=float(pitch),
        roll=float(roll),
        rocker_left=0.0,
        rocker_right=0.0,
        bogie_left=0.0,
        bogie_right=0.0,
    )


# --------------------------------------------------------------------------
# Rigid body inertia primitives.
#
# The URDF generator needs an inertia tensor for every link. These live here
# rather than in the generator so that the rule holds: no equation is
# implemented anywhere except ares_math. All return the six independent
# components (ixx, ixy, ixz, iyy, iyz, izz) about the body centre of mass,
# axes aligned with the link frame, in kg m^2.
# --------------------------------------------------------------------------


def box_inertia(mass, x, y, z):
    """Solid rectangular box of side lengths x, y, z."""
    k = mass / 12.0
    return (
        k * (y * y + z * z),
        0.0,
        0.0,
        k * (x * x + z * z),
        0.0,
        k * (x * x + y * y),
    )


def cylinder_inertia(mass, radius, length, axis="y"):
    """Solid cylinder. axis names the symmetry axis in the link frame."""
    along = 0.5 * mass * radius * radius
    across = mass * (3.0 * radius * radius + length * length) / 12.0
    table = {
        "x": (along, across, across),
        "y": (across, along, across),
        "z": (across, across, along),
    }
    if axis not in table:
        raise ValueError("axis must be x, y or z")
    ixx, iyy, izz = table[axis]
    return (ixx, 0.0, 0.0, iyy, 0.0, izz)


def rod_inertia(mass, length, axis="x"):
    """Thin rod of given length, used for the rocker and bogie arms."""
    across = mass * length * length / 12.0
    table = {
        "x": (0.0, across, across),
        "y": (across, 0.0, across),
        "z": (across, across, 0.0),
    }
    if axis not in table:
        raise ValueError("axis must be x, y or z")
    ixx, iyy, izz = table[axis]
    return (ixx, 0.0, 0.0, iyy, 0.0, izz)


def rpy_matrix(roll, pitch, yaw=0.0):
    """ROS fixed axis roll pitch yaw to a rotation matrix.

    The public form of the convention this module uses everywhere, exported
    so that URDF tooling composes transforms the same way the solver does
    rather than reimplementing the convention and getting it 90 degrees out.
    """
    return _rpy(roll, pitch, yaw)
