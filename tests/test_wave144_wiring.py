"""Tests for wave-144 module wiring into trading_cycle.py.

Covers:
  Step util.2 — utils.paths (get_default_price_path)
  Step util.3 — utils.random_state (set_global_seed / seed_context)
  Step util.4 — utils.timing (timed_step / write_timings_json)
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.assembled_core.utils.paths import get_default_price_path
from src.assembled_core.utils.random_state import set_global_seed, seed_context
from src.assembled_core.utils.timing import timed_step, write_timings_json


# ---------------------------------------------------------------------------
# utils.paths (Step util.2)
# ---------------------------------------------------------------------------

def test_get_default_price_path_importable():
    assert get_default_price_path is not None


def test_get_default_price_path_1d():
    p = get_default_price_path("1d")
    assert isinstance(p, Path)
    assert "daily" in str(p).lower() or "1d" in str(p).lower() or p.suffix == ".parquet"


def test_get_default_price_path_5min():
    p = get_default_price_path("5min")
    assert isinstance(p, Path)


def test_get_default_price_path_invalid():
    with pytest.raises(ValueError):
        get_default_price_path("invalid_freq")


# ---------------------------------------------------------------------------
# utils.random_state (Step util.3)
# ---------------------------------------------------------------------------

def test_set_global_seed_importable():
    assert set_global_seed is not None


def test_set_global_seed_runs():
    set_global_seed(42)  # should not raise


def test_seed_context_importable():
    assert seed_context is not None


def test_seed_context_runs():
    with seed_context(123):
        pass  # should not raise


# ---------------------------------------------------------------------------
# utils.timing (Step util.4)
# ---------------------------------------------------------------------------

def test_timed_step_importable():
    assert timed_step is not None


def test_timed_step_records():
    timings: dict = {}
    with timed_step("test_step", timings):
        _ = 1 + 1
    assert "test_step" in timings
    assert "duration_ms" in timings["test_step"]


def test_write_timings_json_importable():
    assert write_timings_json is not None
