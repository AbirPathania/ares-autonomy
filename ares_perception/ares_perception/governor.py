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

from ares_math.constants import CHASSIS, LIMITS
from ares_math.safety import max_speed_for_range, stopping_distance

from .traversability import OBSTACLE, UNKNOWN

__all__ = ["SpeedLimit", "trusted_range", "speed_limit"]

# How far ahead, in the vehicle's own width, a corridor has to be clear for
# the range to count as trusted. Narrower and the vehicle plans through gaps
# it cannot fit; wider and it stops for things it would pass beside.
CORRIDOR_WIDTHS = 1.5

# How much of the corridor has to be measured at a given range before that
# range counts as seen.
#
# This started as a fraction, 60 per cent of the cells across the corridor,
# and a fraction is the wrong shape of answer. Real stereo delivers 1 to 3
# decided cells out of 10 in the near field, not because the ground is hidden
# but because a 50 mm cell at 1 m is finer than the returns are dense, so a
# 60 per cent rule stopped the vehicle in front of open ground. Measured in
# the simulator: 0.4:20% 0.6:10% 0.8:0% 1.0:30% 1.2:0% 1.4:20% 1.6:90%.
#
# Two cells is a threshold with a meaning instead of a number. One decided
# cell is a single return and could be anything, including a speck of noise
# that survived the confidence floor. Two says the corridor was actually
# sampled at this range. It is the smallest claim worth making, and anything
# larger is a guess about density dressed up as a safety margin.
MIN_DECIDED_CELLS = 2

# How much trusted range the vehicle needs beyond its own stopping distance
# before it is planning rather than merely reacting.
#
# Section 03.2: "At 4 km/h you need trustworthy depth to about 1.5 m to stop
# safely, and to roughly 5 to 8 m to plan a decent path." This factor is the
# ratio of those two numbers, and it is the document's engineering judgement
# rather than a derivation, which is why it is named and commented here
# instead of buried in an expression.
#
# It matters because stopping distance alone gives a governor that is
# technically safe and practically useless. At 4 km/h the vehicle stops in
# 1.43 m, so a stop-only rule would happily drive at full speed into a bank
# of fog it can see 2 m into, arrive, stop, and be stuck. Requiring the
# horizon to scale with speed is what makes it slow down and keep planning.
PLANNING_HORIZON_FACTOR = 3.5

# An unknown patch smaller than the governing obstacle cannot hide the
# governing obstacle, so it does not shorten the trusted range. Derived from
# section 02's trench rather than chosen, which is also what stops a few
# scattered unresolved cells from parking the vehicle.
def _minimum_unknown_run():
    from ares_math.constants import TRENCH

    return TRENCH.width


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


def trusted_range(
    state,
    spec,
    corridor_half_width=None,
    min_decided_cells=MIN_DECIDED_CELLS,
    noise=None,
):
    """How far ahead the estimator has actually decided, metres.

    Walks out from the vehicle, band by band, and stops at the first band in
    the driving corridor where too much is UNKNOWN. That distance, not the
    grid's extent and not the furthest returned point, is what the vehicle
    may plan against.

    Measured AHEAD OF THE FRONT OF THE VEHICLE, not from the datum. The datum
    is the middle axle, and a stopping distance measured from there would let
    the front wheels reach the hazard while the arithmetic still said there
    was room. It also means an all-unknown grid returns zero, which is the
    answer that stops the vehicle rather than one that creeps.

    noise is the expected one sigma depth error per cell. It sizes the window
    each range is judged over, and without it the answer is wrong in a way
    that is easy to miss.

    Stereo depth is quantised by disparity. At 3 m on the simulator's optics
    adjacent integer disparities are about 140 mm apart, so ground returns
    land in stripes and the 50 mm cells between them receive nothing. Judged
    band by band the corridor reads 0%, 90%, 0%, 100%, 0%, and a vehicle that
    demands a full band stops in front of ground it can see perfectly well.
    Measured in the simulator: the corridor came back 0 to 30 per cent
    covered while the camera was looking at open, flat terrain.

    So each range is judged over a window as wide as the depth uncertainty
    there, which is the distance over which a single measurement actually
    constrains the surface. Same section 03.2 curve as everywhere else.

    Returns (range ahead of the bumper, fraction decided in the window that
    stopped it).
    """
    if corridor_half_width is None:
        corridor_half_width = CORRIDOR_WIDTHS * CHASSIS.track / 2.0

    xs, ys = spec.centres()
    corridor = np.abs(ys) <= corridor_half_width
    if not corridor.any():
        return 0.0, 0.0

    # Start at the front of the vehicle, not at the datum. The datum is the
    # middle axle, so the bands between it and the front bumper are the rover
    # itself. They are unmeasured by construction, and walking from zero made
    # the vehicle stop permanently because the first half metre of "unknown
    # ground ahead" was its own front wheels.
    start = CHASSIS.axle_pitch_front + CHASSIS.rolling_radius

    run_start = None
    minimum_run = _minimum_unknown_run()
    worst = 1.0

    # Half width of the window each range is judged over.
    #
    # Never less than a wheel radius, because a gap the wheel spans is not a
    # gap the vehicle can fall into, and demanding a measurement finer than
    # the contact patch asks for resolution the vehicle could not act on if
    # it had it. Near the rover this is what the window is made of: depth
    # uncertainty is small at 0.6 m, so it cannot bridge the patchiness there,
    # and the patchiness near the rover is not quantisation, it is the deck
    # unit's coverage.
    #
    # And never less than the depth uncertainty, which is what dominates
    # further out, where returns land in disparity stripes.
    floor = max(spec.cell / 2.0, CHASSIS.rolling_radius)
    if noise is None:
        half_widths = np.full(len(xs), floor)
    else:
        centre = np.asarray(noise)[:, corridor]
        half_widths = np.maximum(
            floor, np.nanmax(centre, axis=1) if centre.size else floor
        )

    for i, x in enumerate(xs):
        if x < start:
            continue
        lo = np.searchsorted(xs, x - half_widths[i])
        hi = np.searchsorted(xs, x + half_widths[i], side="right")
        band = state[lo:hi, corridor]
        if band.size == 0:
            continue
        # Per along-track row inside the window, so a single well covered
        # row does not vouch for its neighbours.
        per_row = np.sum(band != UNKNOWN, axis=1)
        decided_cells = int(np.max(per_row)) if per_row.size else 0
        decided = float(np.mean(band != UNKNOWN))

        if decided_cells >= min_decided_cells:
            run_start = None
            continue

        if run_start is None:
            run_start = x
            worst = decided
        else:
            worst = min(worst, decided)

        # One sparse band is noise. A contiguous unknown region as long as
        # the governing trench is a place the trench could be hiding.
        if x - run_start >= minimum_run:
            # Back off by the window, not by a cell. Pooling evidence over a
            # window smears a boundary by the window's length, and smearing a
            # safety range outward means claiming ground that was never
            # measured. Subtracting it puts the error on the conservative
            # side, where it belongs.
            return max(run_start - half_widths[i] - start, 0.0), worst

    return max(float(xs[-1]) - start, 0.0), 1.0


def speed_limit(
    state,
    spec,
    ceiling=None,
    decel=None,
    latency=None,
    safety_factor=None,
    corridor_half_width=None,
    noise=None,
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
        state, spec, corridor_half_width=corridor_half_width, noise=noise
    )

    if reach <= 0.0:
        return SpeedLimit(
            speed=0.0,
            trusted_range=0.0,
            reason="nothing measured ahead, stop",
            limiting_band=0.0,
        )

    # Two terms, and the vehicle obeys whichever is lower.
    #
    #   stopping   it must halt before the nearest thing it cannot pass, or
    #              before the edge of what it has measured
    #   planning   it must see far enough ahead to steer rather than only to
    #              stop, and that horizon grows with speed
    blocked = _nearest_obstacle(state, spec, corridor_half_width)
    if blocked is not None and blocked <= 0.0:
        return SpeedLimit(
            speed=0.0,
            trusted_range=reach,
            reason="obstacle against the bumper, stop",
            limiting_band=0.0,
        )
    limiting = min(reach, blocked) if blocked is not None else reach

    stop_limited = max_speed_for_range(limiting, decel, latency, safety_factor)
    plan_limited = max_speed_for_range(
        reach / PLANNING_HORIZON_FACTOR, decel, latency, safety_factor
    )
    allowed = min(ceiling, stop_limited, plan_limited)

    if allowed >= ceiling:
        return SpeedLimit(
            speed=ceiling,
            trusted_range=reach,
            reason="clear to the autonomous limit",
            limiting_band=limiting,
        )

    if stop_limited <= plan_limited and blocked is not None and blocked <= reach:
        reason = f"obstacle at {blocked:.2f} m"
    elif stop_limited <= plan_limited:
        reason = (
            f"depth not trusted beyond {reach:.2f} m, "
            f"{decided * 100:.0f} per cent of that band decided"
        )
    else:
        reason = (
            f"horizon only {reach:.2f} m, too short to plan at "
            f"{ceiling * 3.6:.1f} km/h"
        )
    return SpeedLimit(
        speed=allowed, trusted_range=reach, reason=reason, limiting_band=limiting
    )


def _nearest_obstacle(state, spec, corridor_half_width):
    if corridor_half_width is None:
        corridor_half_width = CORRIDOR_WIDTHS * CHASSIS.track / 2.0
    xs, ys = spec.centres()
    corridor = np.abs(ys) <= corridor_half_width
    start = CHASSIS.axle_pitch_front + CHASSIS.rolling_radius
    for i, x in enumerate(xs):
        if x < start:
            continue
        if np.any(state[i, corridor] == OBSTACLE):
            # Also ahead of the bumper, so it compares like for like with a
            # stopping distance.
            return float(x) - start
    return None


def stopping_distance_at(speed):
    """What the vehicle needs to stop from this speed, with the safety factor."""
    return stopping_distance(
        speed, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
