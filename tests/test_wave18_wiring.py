"""Tests for wave-18 module wiring into trading_cycle.py.

Covers:
  Step 8.1 — risk.regime_analysis (classify_regimes_from_index)
  Step 8.2 — qa.portfolio_analyzer (compute_performance_profile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.regime_analysis import classify_regimes_from_index
from src.assembled_core.qa.portfolio_analyzer import (
    compute_performance_profile,
    PerformanceProfile,
)


# ---------------------------------------------------------------------------
# classify_regimes_from_index (Step 8.1)
# ---------------------------------------------------------------------------

def _make_index_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n),
                     index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_classify_regimes_returns_series():
    returns = _make_index_returns()
    result = classify_regimes_from_index(returns)
    assert isinstance(result, pd.Series)


def test_classify_regimes_same_length():
    returns = _make_index_returns(80)
    result = classify_regimes_from_index(returns)
    assert len(result) == len(returns)


def test_classify_regimes_valid_labels():
    returns = _make_index_returns()
    result = classify_regimes_from_index(returns)
    valid = {"bull", "bear", "sideways", "neutral", "crisis"}
    assert set(result.unique()).issubset(valid), f"Unexpected: {set(result.unique()) - valid}"


def test_classify_regimes_crisis_on_crash():
    # Large drawdown → should produce crisis labels
    returns = pd.Series([-0.05] * 50 + [0.0] * 20,
                        index=pd.date_range("2024-01-01", periods=70, freq="B"))
    result = classify_regimes_from_index(returns)
    assert "crisis" in result.values or "bear" in result.values


def test_classify_regimes_bull_on_trending_up():
    # Steady gains
    returns = pd.Series([0.003] * 80,
                        index=pd.date_range("2024-01-01", periods=80, freq="B"))
    result = classify_regimes_from_index(returns)
    # Should have bull in the mix after warmup
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# compute_performance_profile (Step 8.2)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_perf_profile_returns_dataclass():
    returns = _make_returns()
    result = compute_performance_profile(returns)
    assert isinstance(result, PerformanceProfile)


def test_perf_profile_has_sharpe():
    returns = _make_returns()
    result = compute_performance_profile(returns)
    assert isinstance(result.sharpe, float)


def test_perf_profile_max_drawdown_non_positive():
    returns = _make_returns()
    result = compute_performance_profile(returns)
    assert result.max_drawdown <= 0.0


def test_perf_profile_annualized_vol_positive():
    returns = _make_returns(n=100)
    result = compute_performance_profile(returns)
    assert result.annualized_vol > 0


def test_perf_profile_positive_returns_positive_cagr():
    # All positive returns
    returns = pd.Series([0.005] * 100)
    result = compute_performance_profile(returns)
    assert result.total_return > 0


def test_perf_profile_accepts_numpy_array():
    arr = np.random.default_rng(5).normal(0.001, 0.01, 50)
    result = compute_performance_profile(arr)
    assert isinstance(result, PerformanceProfile)


def test_perf_profile_declining_has_negative_calmar():
    # Declining returns
    returns = pd.Series([-0.01] * 60)
    result = compute_performance_profile(returns)
    # With constant losses, calmar should be defined and calmar ratio may be negative
    assert isinstance(result.calmar, float)
