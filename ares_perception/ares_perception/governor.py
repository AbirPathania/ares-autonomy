"""Confidence to speed. The wiring that makes the camera-only bet survivable.

Build spec section 09:

    Every depth pixel carries a confidence. Low confidence over a region is a
    first-class signal that means slow down or stop, exactly like an obstacle.
    Wire that from the start. A stack that treats unknown as free space will
    drive into a snowdrift at full speed, and it will do it in the field, not
    in the sim.

This module is that wiring, and it is four lines of arithmetic wrapped in a
lot of explanation, because the arithmetic is not the hard part. The hard part
is that the obvious implementation is wrong.

The obvious implementation multiplies the speed limit by a confidence score.
That is wrong twice over. It has no units, so the number it produces cannot be
checked against anything. And it degrades gracefully into a speed that is
still too fast for what the vehicle can actually see, because nothing in it
knows the stopping distance.

What this does instead: find the range out to which the perception is actually
trustworthy, then ask ares_math.safety what speed stops inside that range. The
same function that turns mast height into 4 km/h turns fog, snow, or a failed
camera into whatever speed is honest at the time. One equation, one place, and
the answer is in metres per second because it was derived rather than scaled.
"""

from dataclasses import dataclass

import numpy as np

from ares_math.constants import LIMITS
from ares_math.safety import max_speed_for_range, stopping_distance

from .traversability import OBSTACLE, UNKNOWN

__all__ = ["SpeedLimit", "trusted_range", "speed_limit"]

# How far ahead, in the vehicle's own width, a corridor has to be clear for
# the range to count as trusted. Narrower and the vehicle plans through gaps
# it cannot fit; wider and it stops for things it would pass beside.
CORRIDOR_WIDTHS = 1.5

# The share of a range band that must be decided for that band to be trusted.
# Not all of it: a few unresolved cells in an otherwise measured band is
# normal, and demanding perfection would stop the vehicle permanently.
DECIDED_FRACTION = 0.6


@dataclass(frozen=True)
class SpeedLimit:
    """A speed, and the reason for it. The reason is not decoration.

    A vehicle that slows down without saying why is indistinguishable from a
    vehicle that is broken, and the operator has to be able to tell.
    """

    speed: float
    trusted_range: float
    reason: str
    limiting_band: float

    @property
    def kmh(self):
        return self.speed * 3.6

    def as_dict(self):
        return {
            "speed": self.speed,
            "kmh": self.kmh,
            "trusted_range": self.trusted_range,
            "reason": self.reason,
            "limiting_band": self.limiting_band,
        }


def trusted_range(state, spec, corridor_half_width=None, decided_fraction=DECIDED_FRACTION):
    """How far ahead the estimator has actually decided, metres.

    Walks out from the vehicle, band by band, and stops at the first band in
    the driving corridor where too much is UNKNOWN. That distance, not the
    grid's extent and not the furthest returned point, is what the vehicle
    may plan against.

    Returns (range, fraction decided in the band that stopped it).
    """
    from ares_math.constants import CHASSIS

    if corridor_half_width is None:
        corridor_half_width = CORRIDOR_WIDTHS * CHASSIS.track / 2.0

    xs, ys = spec.centres()
    corridor = np.abs(ys) <= corridor_half_width
    if not corridor.any():
        return 0.0, 0.0

    for i, x in enumerate(xs):
        if x < 0:
            continue
        band = state[i, corridor]
        if band.size == 0:
            continue
        decided = float(np.mean(band != UNKNOWN))
        if decided < decided_fraction:
            return max(x - spec.cell, 0.0), decided

    return float(xs[-1]), 1.0


def speed_limit(
    state,
    spec,
    ceiling=None,
    decel=None,
    latency=None,
    safety_factor=None,
    corridor_half_width=None,
):
    """The speed the vehicle may drive, given what it can currently see.

    Composition of trusted_range and ares_math.safety.max_speed_for_range.
    Never returns more than the ceiling, which is the autonomous limit section
    02 derives from mast height. Returns a SpeedLimit.
    """
    ceiling = LIMITS.v_autonomous if ceiling is None else ceiling
    decel = LIMITS.decel if decel is None else decel
    latency = LIMITS.latency if latency is None else latency
    safety_factor = LIMITS.safety_factor if safety_factor is None else safety_factor

    reach, decided = trusted_range(
        state, spec, corridor_half_width=corridor_half_width
    )

    if reach <= 0.0:
        return SpeedLimit(
            speed=0.0,
            trusted_range=0.0,
            reason="nothing measured ahead, stop",
            limiting_band=0.0,
        )

    # An obstacle inside the trusted range binds before the range does.
    blocked = _nearest_obstacle(state, spec, corridor_half_width)
    limiting = min(reach, blocked) if blocked is not None else reach

    allowed = max_speed_for_range(limiting, decel, latency, safety_factor)

    if allowed >= ceiling:
        return SpeedLimit(
            speed=ceiling,
            trusted_range=reach,
            reason="clear to the autonomous limit",
            limiting_band=limiting,
        )

    if blocked is not None and blocked <= reach:
        reason = f"obstacle at {blocked:.2f} m"
    else:
        reason = (
            f"depth not trusted beyond {reach:.2f} m, "
            f"{decided * 100:.0f} per cent of that band decided"
        )
    return SpeedLimit(
        speed=allowed, trusted_range=reach, reason=reason, limiting_band=limiting
    )


def _nearest_obstacle(state, spec, corridor_half_width):
    from ares_math.constants import CHASSIS

    if corridor_half_width is None:
        corridor_half_width = CORRIDOR_WIDTHS * CHASSIS.track / 2.0
    xs, ys = spec.centres()
    corridor = np.abs(ys) <= corridor_half_width
    for i, x in enumerate(xs):
        if x < 0:
            continue
        if np.any(state[i, corridor] == OBSTACLE):
            return float(x)
    return None


def stopping_distance_at(speed):
    """What the vehicle needs to stop from this speed, with the safety factor."""
    return stopping_distance(
        speed, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
