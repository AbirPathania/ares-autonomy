"""Pinhole optics. Build spec section 03.1, 03.4, 03.5 and section 08.

Everything downstream depends on focal length in pixels, and it is the most
common place to be quietly wrong, because datasheets quote field of view in
degrees and the math needs pixels.

Angles cross this boundary in degrees, which is how the build spec tables and
constants.py are written. Pass a pint Quantity in radians and it converts.
Pass a bare float and it is read as degrees.
"""

import math

from .units import DEG, PX, M, units

__all__ = [
    "focal_px",
    "hfov_from_focal",
    "vfov_from_focal",
    "project",
    "unproject",
    "pixels_on_target",
    "range_for_pixels",
    "near_edge_range",
    "far_edge_range",
]


@units(None, (PX, DEG))
def focal_px(width_px, hfov_deg):
    """Pinhole focal length in pixels from horizontal field of view.

        f_px = (W_px / 2) / tan(HFOV / 2)

    width_px: sensor width, pixels.
    hfov_deg: horizontal field of view, degrees.
    Returns pixels.
    """
    if width_px <= 0:
        raise ValueError("width_px must be positive")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must lie strictly between 0 and 180 degrees")
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


@units(None, (PX, PX))
def hfov_from_focal(width_px, f_px):
    """Inverse of focal_px. Returns horizontal field of view in degrees."""
    if f_px <= 0:
        raise ValueError("f_px must be positive")
    return math.degrees(2.0 * math.atan((width_px / 2.0) / f_px))


@units(None, (PX, PX))
def vfov_from_focal(height_px, f_px):
    """Vertical field of view in degrees, square pixels assumed."""
    if f_px <= 0:
        raise ValueError("f_px must be positive")
    return math.degrees(2.0 * math.atan((height_px / 2.0) / f_px))


@units(None, (M, M, M, PX, PX, PX, PX))
def project(x, y, z, fx, fy, cx, cy):
    """3D point in the camera optical frame to pixel coordinates.

    Optical frame per build spec section 08: Z forward, X right, Y down.
    This is not the robot convention, and that difference is the bug that
    costs everyone two days.

        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Returns (u, v) in pixels.
    """
    if z <= 0:
        raise ValueError("z must be positive, the point must be in front of the camera")
    return (fx * x / z + cx, fy * y / z + cy)


@units(None, (PX, PX, M, PX, PX, PX, PX))
def unproject(u, v, z, fx, fy, cx, cy):
    """Pixel plus depth to a 3D point in the camera optical frame.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z

    Every 2D detection becomes a 3D object through this and nothing else.
    Returns (X, Y, Z) in metres.
    """
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


@units(None, (M, PX, M))
def pixels_on_target(height, f_px, z):
    """Vertical pixels subtended by an object of height H at range Z.

        n_px = H * f / Z

    A detector needs roughly 10 to 20 pixels of object height to fire
    reliably. Returns pixels, unrounded.
    """
    if z <= 0:
        raise ValueError("z must be positive")
    return height * f_px / z


@units(None, (M, PX, PX))
def range_for_pixels(height, f_px, n_px):
    """Range at which an object of height H subtends n_px vertical pixels.

    Returns metres.
    """
    if n_px <= 0:
        raise ValueError("n_px must be positive")
    return height * f_px / n_px


@units(None, (M, DEG, DEG))
def near_edge_range(sensor_height, tilt_deg, vfov_deg):
    """Near edge of ground coverage for a camera pitched down by tilt.

        d_near = h / tan(tilt + VFOV / 2)

    This is the blind zone. The mast unit's blind zone is exactly what the
    deck unit exists to fill. Returns metres.
    """
    half_deg = tilt_deg + vfov_deg / 2.0
    if not 0.0 < half_deg < 90.0:
        raise ValueError("tilt + vfov/2 must lie strictly between 0 and 90 degrees")
    return sensor_height / math.tan(math.radians(half_deg))


@units(None, (M, DEG, DEG))
def far_edge_range(sensor_height, tilt_deg, vfov_deg):
    """Far edge of ground coverage, or infinity when the top ray never lands.

        d_far = h / tan(tilt - VFOV / 2)

    Returns metres, or math.inf when the camera sees the horizon.
    """
    half_deg = tilt_deg - vfov_deg / 2.0
    if half_deg <= 0.0:
        return math.inf
    return sensor_height / math.tan(math.radians(half_deg))
