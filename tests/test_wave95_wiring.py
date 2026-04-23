"""Tests for wave-95 module wiring into trading_cycle.py.

Covers:
  Step 7.79 — qa.numba_kernels (compute_mark_to_market_numba)
  Step 7.80 — qa.parallel_grid (run_grid_parallel / GridPoint)
  Step 7.81 — qa.regime_aware_wf (RegimeWalkForwardResult / run_regime_aware_walk_forward)
"""

from __future__ import annotations

import pytest
import numpy as np

from src.assembled_core.qa.numba_kernels import (
    compute_mark_to_market_numba,
    compute_equity_curve_numba,
)
from src.assembled_core.qa.parallel_grid import run_grid_parallel, GridPoint
from src.assembled_core.qa.regime_aware_wf import (
    RegimeWalkForwardResult,
    run_regime_aware_walk_forward,
)


# ---------------------------------------------------------------------------
# numba_kernels (Step 7.79)
# ---------------------------------------------------------------------------

def test_compute_mark_to_market_numba_returns_float():
    prices = np.array([100.0, 105.0, 110.0])
    quantities = np.array([10.0, 5.0, -3.0])
    result = compute_mark_to_market_numba(prices, quantities)
    assert isinstance(result, float)


def test_compute_mark_to_market_numba_single_position():
    result = compute_mark_to_market_numba(np.array([100.0]), np.array([1.0]))
    assert result == 100.0


def test_compute_equity_curve_numba_importable():
    assert compute_equity_curve_numba is not None


def test_compute_mark_to_market_zero_positions():
    result = compute_mark_to_market_numba(np.array([100.0]), np.array([0.0]))
    assert result == 0.0


# ---------------------------------------------------------------------------
# parallel_grid (Step 7.80)
# ---------------------------------------------------------------------------

def test_grid_point_importable():
    assert GridPoint is not None


def test_run_grid_parallel_importable():
    assert run_grid_parallel is not None


def test_grid_point_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(GridPoint)


# ---------------------------------------------------------------------------
# regime_aware_wf (Step 7.81)
# ---------------------------------------------------------------------------

def test_regime_walk_forward_result_importable():
    assert RegimeWalkForwardResult is not None


def test_run_regime_aware_walk_forward_importable():
    assert run_regime_aware_walk_forward is not None


def test_regime_walk_forward_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(RegimeWalkForwardResult)
