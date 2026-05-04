"""Determinism setup for characterization/golden-master tests.

From 35_GOLDEN_EQUITY_SCENARIO_TESTS.md §2.3.

All random sources are seeded so that test outputs are reproducible
regardless of OS, hardware, or Python hash randomisation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

DETERMINISTIC_SEED = 42


@pytest.fixture(autouse=True)
def deterministic_seeds(monkeypatch):
    """Pin all known random sources to DETERMINISTIC_SEED."""
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    monkeypatch.setenv("PYTHONHASHSEED", str(DETERMINISTIC_SEED))
    try:
        import torch

        torch.manual_seed(DETERMINISTIC_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    yield


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def approved_dir() -> Path:
    return Path(__file__).parent / "approved"
