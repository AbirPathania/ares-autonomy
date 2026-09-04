"""The symbolic derivations run as part of the suite.

Build spec section 11, mechanism 5. tools/derive_symbolic.py derives every
closed form from first principles and asserts it matches what ares_math
implements. Running it here means a divergence fails the build rather than
waiting for someone to run the script by hand.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "derive_symbolic.py"


@pytest.fixture(scope="module")
def derive():
    spec = importlib.util.spec_from_file_location("derive_symbolic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["derive_symbolic"] = module
    spec.loader.exec_module(module)
    return module


def test_the_derivation_script_is_where_the_docstrings_say_it_is():
    assert SCRIPT.exists(), f"missing {SCRIPT}"


@pytest.mark.slow
def test_every_derivation_matches_its_closed_form(derive):
    derive.main()
    names = {name for name, _ in derive.RESULTS}
    for required in (
        "R_max = w * h / d",
        "dZ = Z^2 * d_disp / (f*B)",
        "f_px = (W/2)/tan(HFOV/2)",
        "d_near = h / tan(tilt + VFOV/2)",
        "differential: theta_rel_L = -theta_rel_R",
        "skew_disparity_error = f * tan(omega*dt)",
    ):
        assert required in names, f"{required} was not derived"
