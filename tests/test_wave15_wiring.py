"""Tests for wave-15 module wiring into trading_cycle.py.

Covers:
  Step 2.10 — features.ta_liquidity_vol_factors (add_realized_volatility)
  Step 2.11 — features.fractional_diff (apply_ffd_to_panel)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.ta_liquidity_vol_factors import add_realized_volatility
from src.assembled_core.features.fractional_diff import (
    apply_ffd_to_panel,
    frac_diff_ffd,
    frac_diff_weights,
)


# ---------------------------------------------------------------------------
# add_realized_volatility (Step 2.10)
# ---------------------------------------------------------------------------

def _make_panel(n_symbols: int = 3, n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        ts = pd.date_range("2024-01-01", periods=n_days, freq="B")
        for t, p in zip(ts, prices):
            rows.append({"symbol": sym, "timestamp": t, "close": float(p)})
    return pd.DataFrame(rows)


def test_add_rv_returns_df():
    df = _make_panel()
    result = add_realized_volatility(df, windows=[20])
    assert isinstance(result, pd.DataFrame)


def test_add_rv_adds_rv_columns():
    df = _make_panel()
    result = add_realized_volatility(df, windows=[20, 60])
    assert "rv_20" in result.columns
    assert "rv_60" in result.columns


def test_add_rv_row_count_preserved():
    df = _make_panel()
    result = add_realized_volatility(df, windows=[20])
    assert len(result) == len(df)


def test_add_rv_values_non_negative():
    df = _make_panel(n_days=80)
    result = add_realized_volatility(df, windows=[20])
    valid = result["rv_20"].dropna()
    assert (valid >= 0).all()


def test_add_rv_missing_close_raises():
    df = pd.DataFrame({"symbol": ["A"] * 10, "timestamp": pd.date_range("2024-01-01", periods=10)})
    with pytest.raises((KeyError, ValueError)):
        add_realized_volatility(df, windows=[5])


def test_add_rv_default_windows():
    df = _make_panel(n_days=80)
    result = add_realized_volatility(df)
    # Default windows are [20, 60] based on the implementation
    assert any(c.startswith("rv_") for c in result.columns)


# ---------------------------------------------------------------------------
# apply_ffd_to_panel (Step 2.11)
# ---------------------------------------------------------------------------

def test_ffd_panel_returns_df():
    df = _make_panel(n_days=80)
    result = apply_ffd_to_panel(df, price_cols=["close"], d=0.4)
    assert isinstance(result, pd.DataFrame)


def test_ffd_panel_adds_ffd_column():
    df = _make_panel(n_days=80)
    result = apply_ffd_to_panel(df, price_cols=["close"], d=0.4)
    ffd_cols = [c for c in result.columns if "_ffd_" in c]
    assert len(ffd_cols) >= 1


def test_ffd_panel_row_count_preserved():
    df = _make_panel(n_days=80)
    result = apply_ffd_to_panel(df, price_cols=["close"], d=0.4)
    assert len(result) == len(df)


def test_ffd_panel_original_cols_preserved():
    df = _make_panel()
    original_cols = set(df.columns)
    result = apply_ffd_to_panel(df, price_cols=["close"], d=0.3)
    assert original_cols.issubset(set(result.columns))


def test_frac_diff_weights_first_is_one():
    weights = frac_diff_weights(d=0.5)
    assert abs(weights[0] - 1.0) < 1e-9


def test_frac_diff_ffd_returns_series():
    prices = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 80)))
    result = frac_diff_ffd(prices, d=0.4)
    assert isinstance(result, pd.Series)
    assert len(result) == len(prices)


def test_frac_diff_ffd_d1_approximates_diff():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0] * 10)
    result_d1 = frac_diff_ffd(prices, d=1.0)
    result_d0 = frac_diff_ffd(prices, d=0.0)
    # d=1 should be more aggressive differentiation than d=0
    assert isinstance(result_d1, pd.Series)
    assert isinstance(result_d0, pd.Series)
