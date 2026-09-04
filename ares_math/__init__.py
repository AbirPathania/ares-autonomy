"""ares_math: the ARES-G geometry and perception math, with no ROS in it.

Pure Python, no ROS, no side effects, so it can be tested in milliseconds on
any machine, without a container, without a simulator, and without a robot.

Every equation in the ARES Autonomy Build Spec lives here and nowhere else.
ROS nodes import from this package. They never calculate. A magic number in a
ROS node is a review rejection, because that is precisely how a deck ends up
claiming 17 kg while the engineering says something else entirely.
"""

from . import constants, kinematics, optics, safety, stereo, units
from .constants import BOGIE, CAMERA, CHASSIS, LIMITS, MASS, REVISION, SENSORS, TRENCH
from .units import Q_, ureg

__version__ = "0.1.0"

__all__ = [
    "constants",
    "kinematics",
    "optics",
    "safety",
    "stereo",
    "units",
    "BOGIE",
    "CAMERA",
    "CHASSIS",
    "LIMITS",
    "MASS",
    "REVISION",
    "SENSORS",
    "TRENCH",
    "Q_",
    "ureg",
    "__version__",
]
