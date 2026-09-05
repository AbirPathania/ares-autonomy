"""Stereo synchronisation. Two eyes only fuse if they look at the same instant.

If the right camera exposes dt after the left while the vehicle is moving, the
pair no longer shares a rigid baseline. The matcher then solves a geometry
that never existed, and it does so confidently, because nothing in a block
matching cost function knows the two images came from different moments.

This is not a small effect and it is not one you can tune out downstream. From
ares_math.stereo, on the optical axis and exactly:

    translation    depth bias   = v * dt
    rotation       disparity    = f * tan(omega * dt)

Rotation dominates, because its disparity error enters depth multiplied by
Z**2 / (f*B) while the translation term does not scale with range at all. At
the working case, one millisecond of skew during a half radian per second turn
costs 62 mm of depth at 5 m. The same millisecond of forward motion at 4 km/h
costs 1.1 mm.

So this module does two things, and the second is the one that matters:

    1. pairs frames by timestamp
    2. REJECTS a pair whose skew exceeds the budget for the vehicle's current
       motion, rather than matching it anyway

At one radian per second, spending a tenth of the matcher's disparity sigma,
that budget is 39 microseconds. A shared hardware trigger clears it with four
times to spare. Matching on software timestamps over a network does not: a
millisecond is 25 times over budget. That is a procurement requirement, and it
is written here and asserted in test_sync.py so it survives one.

A rejected pair is not a failure to handle quietly. It is a gap in perception,
and it flows through to the same place a low confidence region does: less
trusted range, and therefore less speed.
"""

from collections import deque
from dataclasses import dataclass, field

from ares_math.constants import CAMERA
from ares_math.stereo import skew_depth_bias, skew_disparity_error, sync_skew_budget

__all__ = ["SyncStats", "StereoSynchroniser", "skew_budget_for"]

# The share of the matcher's own disparity sigma that exposure skew is allowed
# to consume. A tenth: skew should be a rounding error against the matcher's
# precision, not a comparable term. Anything larger and the section 03.2 error
# table stops describing the system.
SKEW_DISPARITY_SHARE = 0.1


def skew_budget_for(yaw_rate, focal_px, share=SKEW_DISPARITY_SHARE):
    """Largest tolerable exposure skew at this yaw rate, seconds.

    Returns math.inf when the vehicle is not rotating, which is correct and
    is why the caller must still apply a sane ceiling of its own.
    """
    budget_px = share * CAMERA.disparity_sigma_px
    return sync_skew_budget(budget_px, focal_px, yaw_rate)


@dataclass
class SyncStats:
    """What the synchroniser has seen. Published, not just logged.

    A stack that silently drops half its stereo pairs looks identical from the
    outside to one that is working, right up until it drives into something.
    """

    paired: int = 0
    rejected_skew: int = 0
    dropped_unmatched: int = 0
    worst_skew: float = 0.0
    last_skew: float = 0.0
    last_budget: float = 0.0

    @property
    def rejection_rate(self):
        total = self.paired + self.rejected_skew
        return self.rejected_skew / total if total else 0.0

    def as_dict(self):
        return {
            "paired": self.paired,
            "rejected_skew": self.rejected_skew,
            "dropped_unmatched": self.dropped_unmatched,
            "worst_skew": self.worst_skew,
            "last_skew": self.last_skew,
            "last_budget": self.last_budget,
            "rejection_rate": self.rejection_rate,
        }


@dataclass
class StereoSynchroniser:
    """Pairs left and right frames, and refuses the pairs it should refuse.

    focal_px sets the rotation sensitivity. max_skew is a hard ceiling applied
    even when the vehicle is stationary, because a pair separated by more than
    a frame interval is a bookkeeping error rather than a motion effect.

    Frames are anything with a timestamp in seconds. This class does not touch
    ROS, so it is testable in microseconds and the node is a thin wrapper.
    """

    focal_px: float
    max_skew: float = 0.005
    queue_size: int = 12
    stats: SyncStats = field(default_factory=SyncStats)
    _left: deque = field(default_factory=deque, repr=False)
    _right: deque = field(default_factory=deque, repr=False)

    def __post_init__(self):
        self._left = deque(maxlen=self.queue_size)
        self._right = deque(maxlen=self.queue_size)

    def push_left(self, stamp, payload):
        return self._push(self._left, self._right, stamp, payload, left_is_new=True)

    def push_right(self, stamp, payload):
        return self._push(self._right, self._left, stamp, payload, left_is_new=False)

    def _push(self, own, other, stamp, payload, left_is_new):
        if len(own) == own.maxlen:
            self.stats.dropped_unmatched += 1
        own.append((float(stamp), payload))
        return self._try_pair(left_is_new)

    def _try_pair(self, left_is_new):
        if not self._left or not self._right:
            return None

        # Best pairing is the closest pair of timestamps, not the newest.
        best = None
        for i, (lt, lp) in enumerate(self._left):
            for j, (rt, rp) in enumerate(self._right):
                skew = abs(lt - rt)
                if best is None or skew < best[0]:
                    best = (skew, i, j, lt, rt, lp, rp)
        skew, i, j, lt, rt, lp, rp = best

        # Only decide once both sides have had a chance to deliver a closer
        # partner. If the newest frame is at the edge of its queue, a better
        # match may still arrive.
        newest = self._left[-1][0] if left_is_new else self._right[-1][0]
        if (left_is_new and lt != newest) or (not left_is_new and rt != newest):
            return None

        budget = min(self.max_skew, self.current_budget)
        self.stats.last_skew = skew
        self.stats.last_budget = budget
        self.stats.worst_skew = max(self.stats.worst_skew, skew)

        for _ in range(i + 1):
            self._left.popleft()
        for _ in range(j + 1):
            self._right.popleft()

        if skew > budget:
            self.stats.rejected_skew += 1
            return None

        self.stats.paired += 1
        return (0.5 * (lt + rt), lp, rp, skew)

    # ---------------------------------------------------------------- motion

    current_yaw_rate: float = 0.0
    current_speed: float = 0.0

    def set_motion(self, speed, yaw_rate):
        """Tell the synchroniser how hard the vehicle is manoeuvring.

        The budget tightens as the vehicle turns. Standing still, any skew is
        harmless; turning at a radian per second, a millisecond is not.
        """
        self.current_speed = float(speed)
        self.current_yaw_rate = float(yaw_rate)

    @property
    def current_budget(self):
        return skew_budget_for(self.current_yaw_rate, self.focal_px)

    def error_if_accepted(self, skew, depth, baseline):
        """What accepting a pair with this skew would cost at range, in metres.

        Reported rather than assumed, so the rejection threshold can be argued
        about with numbers instead of adjusted by feel. Both terms, because
        returning the translation one alone understates the error by an order
        of magnitude at any useful range.
        """
        from ares_math.stereo import depth_error

        disparity = skew_disparity_error(self.focal_px, self.current_yaw_rate, skew)
        return {
            "translation_bias": skew_depth_bias(self.current_speed, skew),
            "disparity_error_px": disparity,
            "rotation_depth_error": depth_error(
                Z=depth, f=self.focal_px, B=baseline, d_disp=disparity
            )
            if disparity > 0
            else 0.0,
        }
