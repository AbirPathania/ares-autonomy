"""Stopping distance, negative obstacle range, and the speed limit that
falls out of the two.

Build spec sections 02 and 03.3.

The finding this module encodes: whether you can see into a hole is a question
about where your eye is, not what kind of eye it is. No sensor type appears in
any equation here. Raising the sensor is the only thing that helps, which makes
mast height a perception parameter and not a packaging convenience.
"""

import math

from .units import DIMENSIONLESS, MPS, MPS2, M, S, units

__all__ = [
    "reaction_distance",
    "braking_distance",
    "stopping_distance",
    "negative_obstacle_range",
    "required_sensor_height",
    "max_speed_for_range",
    "max_speed_for_sensor_height",
    "detects_in_time",
]


@units(None, (MPS, S))
def reaction_distance(v, t_latency):
    """Distance covered between the world changing and the brakes acting.

        d = v * t_latency

    Latency is the term you control in software and the one most often
    ignored. At 4 km/h every 100 ms of pipeline delay costs 111 mm.
    Returns metres.
    """
    if v < 0:
        raise ValueError("v must not be negative")
    if t_latency < 0:
        raise ValueError("t_latency must not be negative")
    return v * t_latency


@units(None, (MPS, MPS2))
def braking_distance(v, a):
    """Distance covered under deceleration a from speed v.

        d = v**2 / (2 * a)

    Returns metres.
    """
    if v < 0:
        raise ValueError("v must not be negative")
    if a <= 0:
        raise ValueError("a must be positive")
    return v**2 / (2.0 * a)


@units(None, (MPS, MPS2, S, DIMENSIONLESS))
def stopping_distance(v, a, t_latency, safety_factor=1.0):
    """Reaction plus braking, times a safety factor.

        D_stop = (v * t_latency + v**2 / (2*a)) * safety_factor

    Returns metres.
    """
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    return (reaction_distance(v, t_latency) + braking_distance(v, a)) * safety_factor


@units(None, (M, M, M))
def negative_obstacle_range(w, h, d):
    """Farthest range at which the floor of a negative obstacle is visible.

        R_max = w * h / d

    w: obstacle width along the direction of travel, metres.
    h: sensor height above ground, metres.
    d: obstacle depth, metres.

    A ray leaving a sensor at height h and grazing the near edge of a hole at
    range R descends at a fixed angle. It reaches depth d at a horizontal
    distance d*R/h past that edge. If the hole is narrower than that, the
    floor is hidden and you are looking at empty space.

    No sensor type appears in this equation. A LiDAR mounted at the same
    height has exactly the same blind geometry.
    Returns metres.
    """
    if w <= 0 or h <= 0 or d <= 0:
        raise ValueError("w, h and d must be positive")
    return w * h / d


@units(None, (M, M, M))
def required_sensor_height(w, d, stop_distance):
    """Sensor height needed to see the floor of a w by d obstacle in time.

        h = d * D_stop / w

    The inverse of negative_obstacle_range, solved for h. This is the
    equation that sizes the mast. Returns metres.
    """
    if w <= 0 or d <= 0:
        raise ValueError("w and d must be positive")
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive")
    return d * stop_distance / w


@units(None, (M, MPS2, S, DIMENSIONLESS))
def max_speed_for_range(R, a, t_latency, safety_factor=1.0):
    """Fastest speed whose stopping distance still fits inside range R.

    Solves stopping_distance(v) == R for v:

        (sf/(2a)) v**2 + sf*t*v - R = 0
        v = -a*t + sqrt((a*t)**2 + 2*a*R/sf)

    This is the function that turns a trusted sensing range into a speed cap,
    and it is the same function whether the range came from mast height, from
    depth confidence collapsing on snow, or from fog. Returns metres/second.
    """
    if R <= 0:
        raise ValueError("R must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if t_latency < 0:
        raise ValueError("t_latency must not be negative")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    at = a * t_latency
    return -at + math.sqrt(at * at + 2.0 * a * R / safety_factor)


@units(None, (M, M, M, MPS2, S, DIMENSIONLESS))
def max_speed_for_sensor_height(h, w, d, a, t_latency, safety_factor=1.0):
    """Autonomous speed limit implied by sensor height against a w by d hole.

    Composition of negative_obstacle_range and max_speed_for_range. At the
    ARES numbers, a 0.95 m mast against a 0.5 m by 0.3 m trench supports
    1.18 m/s, which is 4 km/h with margin. Returns metres/second.
    """
    return max_speed_for_range(
        negative_obstacle_range(w, h, d), a, t_latency, safety_factor
    )


@units(None, (M, M, M, MPS, MPS2, S, DIMENSIONLESS))
def detects_in_time(w, h, d, v, a, t_latency, safety_factor=1.0):
    """True when the obstacle is seen before the vehicle can no longer stop.

    The PASS and FAIL column of the build spec section 02 table.
    """
    return negative_obstacle_range(w, h, d) >= stopping_distance(
        v, a, t_latency, safety_factor
    )
