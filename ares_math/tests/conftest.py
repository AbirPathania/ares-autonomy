"""Shared fixtures. The golden file is loaded once and treated as read only."""

from pathlib import Path

import pytest
import yaml

GOLDEN_PATH = Path(__file__).parent / "golden_values.yaml"

with GOLDEN_PATH.open() as _fh:
    GOLD = yaml.safe_load(_fh)

# Relative tolerance for every golden assertion. Build spec section 11.
REL = 1e-9


@pytest.fixture(scope="session")
def gold():
    return GOLD


@pytest.fixture(scope="session")
def working():
    return GOLD["working_case"]
