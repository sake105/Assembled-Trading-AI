"""Tests for wave-40 module wiring into trading_cycle.py.

Covers:
  Step 2.23 — features.incremental_updates (filter_prices_for_incremental)
  Step 8.32 — ops.dashboard_data (build_pnl_curve / compute_risk_snapshot)
  Step 8.33 — qa.ab_testing (minimum_detectable_effect / paired_ab_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.incremental_updates import (
    filter_prices_for_incremental,
    compute_only_last_session,
    compute_last_N_sessions,
)
from src.assembled_core.ops.dashboard_data import (
    build_pnl_curve,
    compute_risk_snapshot,
    build_position_table,
    compute_exposure,
)
from src.assembled_core.qa.ab_testing import (
    minimum_detectable_effect,
    paired_ab_test,
    ABTestResult,
)


# ---------------------------------------------------------------------------
# filter_prices_for_incremental (Step 2.23)
# ---------------------------------------------------------------------------

def _make_prices_df(n: int = 60, n_syms: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        for t, c in zip(ts, closes):
            rows.append({"timestamp": t, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def test_filter_returns_df():
    prices = _make_prices_df()
    result = filter_prices_for_incremental(prices, window_days=5)
    assert isinstance(result, pd.DataFrame)


def test_filter_reduces_rows():
    prices = _make_prices_df(n=60, n_syms=3)
    result = filter_prices_for_incremental(prices, window_days=5)
    assert len(result) <= len(prices)


def test_filter_empty_returns_empty():
    result = filter_prices_for_incremental(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_filter_window_days_1():
    prices = _make_prices_df(n=60, n_syms=2)
    result_1 = filter_prices_for_incremental(prices, window_days=1)
    result_5 = filter_prices_for_incremental(prices, window_days=5)
    assert len(result_1) <= len(result_5)


def test_filter_preserves_columns():
    prices = _make_prices_df()
    result = filter_prices_for_incremental(prices, window_days=5)
    for col in ["timestamp", "symbol", "close"]:
        assert col in result.columns


def test_filter_sorted_by_timestamp():
    prices = _make_prices_df()
    result = filter_prices_for_incremental(prices, window_days=5)
    if len(result) > 1:
        ts = result["timestamp"].values
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def test_compute_only_last_session_returns_df():
    prices = _make_prices_df(60, 2)
    result = compute_only_last_session(prices, builder_fn=lambda df, **kw: df)
    assert isinstance(result, pd.DataFrame)


def test_compute_last_N_sessions_returns_df():
    prices = _make_prices_df(60, 2)
    result = compute_last_N_sessions(prices, builder_fn=lambda df, **kw: df, window_days=3)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# dashboard_data (Step 8.32)
# ---------------------------------------------------------------------------

def _make_equity(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100000.0 + np.cumsum(rng.normal(50, 200, n)))


def test_build_pnl_curve_returns_dict():
    eq = _make_equity()
    result = build_pnl_curve(eq, initial_capital=100000.0)
    assert isinstance(result, dict)


def test_build_pnl_curve_length():
    eq = _make_equity(30)
    result = build_pnl_curve(eq, initial_capital=100000.0)
    assert len(result) == 30


def test_build_pnl_curve_empty_returns_empty():
    result = build_pnl_curve(pd.Series(dtype=float), initial_capital=100000.0)
    assert result == {}


def test_compute_risk_snapshot_returns_dict():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 100))
    result = compute_risk_snapshot(rets)
    assert isinstance(result, dict)


def test_compute_risk_snapshot_has_sharpe():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 100))
    result = compute_risk_snapshot(rets)
    assert "sharpe_ratio" in result


def test_build_position_table_empty_weights():
    result = build_position_table({})
    assert isinstance(result, list)
    assert len(result) == 0


def test_compute_exposure_returns_dict():
    weights = {"A": 0.3, "B": 0.2, "C": -0.1}
    result = compute_exposure(weights)
    assert isinstance(result, dict)
    assert "gross" in result


# ---------------------------------------------------------------------------
# ab_testing (Step 8.33)
# ---------------------------------------------------------------------------

def test_mde_returns_float():
    result = minimum_detectable_effect(n_days=252, baseline_vol=0.01)
    assert isinstance(result, float)
    assert result > 0.0


def test_mde_larger_n_smaller_mde():
    mde_small = minimum_detectable_effect(n_days=30, baseline_vol=0.01)
    mde_large = minimum_detectable_effect(n_days=252, baseline_vol=0.01)
    assert mde_large < mde_small


def test_mde_larger_vol_larger_mde():
    mde_low = minimum_detectable_effect(n_days=100, baseline_vol=0.005)
    mde_high = minimum_detectable_effect(n_days=100, baseline_vol=0.02)
    assert mde_high > mde_low


def test_paired_ab_test_returns_result():
    rng = np.random.default_rng(42)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    a = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    result = paired_ab_test(a, b)
    assert isinstance(result, ABTestResult)


def test_paired_ab_test_p_value_in_01():
    rng = np.random.default_rng(1)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    a = pd.Series(rng.normal(0, 0.01, n), index=idx)
    b = pd.Series(rng.normal(0, 0.01, n), index=idx)
    result = paired_ab_test(a, b)
    assert 0.0 <= result.p_value <= 1.0


def test_paired_ab_test_short_series_returns_inconclusive():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    a = pd.Series(rng.normal(0, 0.01, 5), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 5), index=idx)
    result = paired_ab_test(a, b)
    assert result.winner == "inconclusive"
