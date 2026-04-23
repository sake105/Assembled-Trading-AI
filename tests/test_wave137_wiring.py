"""Tests for wave-137 module wiring into trading_cycle.py.

Covers:
  Step port.1 — portfolio.kelly_uncertainty (compute_kelly_with_uncertainty)
  Step port.2 — portfolio.turnover_penalty (TurnoverConstrainedSizer)
  Step qa.1   — qa.backtest_engine_numba (NUMBA_AVAILABLE)
"""

from __future__ import annotations

import pytest

from src.assembled_core.portfolio.kelly_uncertainty import (
    compute_kelly_with_uncertainty,
    compute_kelly_weights_with_uncertainty,
)
from src.assembled_core.portfolio.turnover_penalty import (
    TurnoverPenaltyConfig,
    TurnoverConstrainedSizer,
    apply_turnover_smoothing,
)
from src.assembled_core.qa.backtest_engine_numba import NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
# portfolio.kelly_uncertainty (Step port.1)
# ---------------------------------------------------------------------------

def test_compute_kelly_with_uncertainty_importable():
    assert compute_kelly_with_uncertainty is not None


def test_compute_kelly_basic():
    result = compute_kelly_with_uncertainty(edge=0.01, variance=0.0004)
    assert isinstance(result, float)
    assert 0.0 <= result <= 0.25


def test_compute_kelly_zero_edge():
    result = compute_kelly_with_uncertainty(edge=0.0, variance=0.0004)
    assert result == 0.0


def test_compute_kelly_weights_importable():
    assert compute_kelly_weights_with_uncertainty is not None


# ---------------------------------------------------------------------------
# portfolio.turnover_penalty (Step port.2)
# ---------------------------------------------------------------------------

def test_turnover_penalty_config_creates():
    cfg = TurnoverPenaltyConfig()
    assert cfg.enabled is True
    assert cfg.ema_alpha > 0.0


def test_turnover_constrained_sizer_creates():
    sizer = TurnoverConstrainedSizer()
    assert isinstance(sizer, TurnoverConstrainedSizer)


def test_apply_turnover_smoothing_importable():
    assert apply_turnover_smoothing is not None


# ---------------------------------------------------------------------------
# qa.backtest_engine_numba (Step qa.1)
# ---------------------------------------------------------------------------

def test_numba_available_importable():
    assert NUMBA_AVAILABLE is not None
    assert isinstance(NUMBA_AVAILABLE, bool)
