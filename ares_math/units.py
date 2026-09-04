"""The one unit registry for the whole stack.

Build spec section 11, mechanism 2: units carried, not assumed.

Convention for every public function in ares_math:

    Arguments are declared in the units the formula is written in.
    A pint Quantity in any compatible unit is converted on the way in,
    so passing 300 mm where metres are declared gives 0.3, and passing
    45 degrees where radians are declared gives 0.7853981634.
    A bare float is interpreted as the declared unit.
    Passing a Quantity of the wrong dimension raises DimensionalityError,
    which is the single most common class of error this catches.

    Return values are bare floats in the declared SI unit, named in the
    docstring. Wrap them with Q_ at the boundary if you need to carry them.
"""

import pint

ureg = pint.UnitRegistry()

# Pixels are counts, not a physical dimension, but giving them a dimension
# stops anyone adding a pixel count to a length by accident.
ureg.define("pixel = [image] = px")
ureg.define("disparity = pixel = disp")

Q_ = ureg.Quantity

# Shorthand used in the wraps decorators below.
#
# Angles cross the ares_math boundary in DEGREES, because every table in the
# build spec is written in degrees and constants.py stores them that way.
# Radians are an implementation detail inside the functions. A Quantity in
# radians still converts correctly on the way in.
M = "meter"
S = "second"
DEG = "degree"
RAD = "radian"
PX = "pixel"
MPS = "meter / second"
RADPS = "radian / second"
MPS2 = "meter / second ** 2"
DIMENSIONLESS = "dimensionless"


def units(ret=None, args=(), strict=False):
    """Declare the units a public function is written in.

    Thin wrapper over pint's own wraps so the intent reads at the call site
    and so strict=False, the ares convention, is not repeated everywhere.
    """
    return ureg.wraps(ret, args, strict=strict)


__all__ = [
    "ureg",
    "Q_",
    "units",
    "M",
    "S",
    "DEG",
    "RAD",
    "PX",
    "MPS",
    "RADPS",
    "MPS2",
    "DIMENSIONLESS",
]
