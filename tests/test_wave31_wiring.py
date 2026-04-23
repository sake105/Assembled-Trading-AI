"""Tests for wave-31 module wiring into trading_cycle.py.

Covers:
  Step 2.20 — features.volatility_features (compute_garch_features_snapshot)
  Step 5.13 — qa.tca_arrival (compute_implementation_shortfall)
  Step 8.18 — qa.post_trade_analyzer (compute_forward_returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.volatility_features import (
    compute_garch_features_snapshot,
)
from src.assembled_core.qa.tca_arrival import (
    compute_implementation_shortfall,
    summarize_implementation_shortfall,
)
from src.assembled_core.qa.post_trade_analyzer import (
    compute_forward_returns,
    compute_signal_hit_rate,
)


# ---------------------------------------------------------------------------
# compute_garch_features_snapshot (Step 2.20)
# ---------------------------------------------------------------------------

def _make_prices(n: int = 300, n_syms: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2023-01-01", periods=n, freq="B")
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        for t, c in zip(ts, closes):
            rows.append({"timestamp": t, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def test_garch_snapshot_returns_dict():
    prices = _make_prices(300)
    result = compute_garch_features_snapshot(prices)
    # arch may not be installed → returns empty dict
    assert isinstance(result, dict)


def test_garch_snapshot_empty_df_returns_empty():
    result = compute_garch_features_snapshot(pd.DataFrame())
    assert result == {}


def test_garch_snapshot_short_series_skipped():
    prices = _make_prices(30)  # too short (< 60)
    result = compute_garch_features_snapshot(prices)
    assert isinstance(result, dict)


def test_garch_snapshot_arch_required():
    pytest.importorskip("arch", reason="arch required for GARCH")
    prices = _make_prices(300)
    result = compute_garch_features_snapshot(prices)
    if result:  # may still be empty if fit fails
        for sym, feats in result.items():
            assert "garch_vol_1d" in feats or feats == {}


def test_garch_snapshot_vol_non_negative():
    pytest.importorskip("arch", reason="arch required for GARCH")
    prices = _make_prices(300)
    result = compute_garch_features_snapshot(prices)
    for feats in result.values():
        if "garch_vol_1d" in feats:
            assert feats["garch_vol_1d"] >= 0.0


# ---------------------------------------------------------------------------
# compute_implementation_shortfall (Step 5.13)
# ---------------------------------------------------------------------------

def _make_fills(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-15", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": [f"S{i % 3}" for i in range(n)],
        "side": ["BUY"] * n,
        "qty": rng.uniform(10, 100, n),
        "fill_price": rng.uniform(50, 200, n),
    })


def _make_arrivals(fills: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    arrivals = fills[["timestamp", "symbol"]].copy()
    arrivals["arrival_price"] = fills["fill_price"] * (1.0 + rng.uniform(-0.005, 0.005, len(fills)))
    return arrivals


def test_is_returns_df():
    fills = _make_fills()
    arrivals = _make_arrivals(fills)
    result = compute_implementation_shortfall(fills, arrivals)
    assert isinstance(result, pd.DataFrame)


def test_is_has_is_bps_column():
    fills = _make_fills()
    arrivals = _make_arrivals(fills)
    result = compute_implementation_shortfall(fills, arrivals)
    assert "is_bps" in result.columns


def test_is_empty_fills_returns_empty():
    result = compute_implementation_shortfall(pd.DataFrame(), pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


def test_is_no_arrivals_returns_nan():
    fills = _make_fills()
    result = compute_implementation_shortfall(fills, pd.DataFrame(columns=["timestamp", "symbol", "arrival_price"]))
    assert "is_bps" in result.columns
    assert result["is_bps"].isna().all()


def test_is_same_prices_near_zero():
    # Fill price == arrival price → IS should be near zero
    rng = np.random.default_rng(3)
    ts = pd.date_range("2024-01-15", periods=5, freq="h")
    fills = pd.DataFrame({
        "timestamp": ts, "symbol": "S0",
        "side": "BUY", "qty": 100.0, "fill_price": 100.0,
    })
    arrivals = pd.DataFrame({
        "timestamp": ts, "symbol": "S0", "arrival_price": 100.0,
    })
    result = compute_implementation_shortfall(fills, arrivals)
    valid = result["is_bps"].dropna()
    if len(valid) > 0:
        assert (valid.abs() < 1e-6).all()


# ---------------------------------------------------------------------------
# compute_forward_returns (Step 8.18)
# ---------------------------------------------------------------------------

def test_fwd_returns_returns_df():
    prices = _make_prices(100, n_syms=2)
    result = compute_forward_returns(prices, horizon_days=5)
    assert isinstance(result, pd.DataFrame)


def test_fwd_returns_has_required_columns():
    prices = _make_prices(100, n_syms=2)
    result = compute_forward_returns(prices, horizon_days=5)
    for col in ["timestamp", "symbol", "close", "forward_return"]:
        assert col in result.columns


def test_fwd_returns_row_count_preserved():
    prices = _make_prices(60, n_syms=2)
    result = compute_forward_returns(prices, horizon_days=5)
    assert len(result) == len(prices)


def test_fwd_returns_nan_at_end():
    prices = _make_prices(60, n_syms=1)
    result = compute_forward_returns(prices, horizon_days=5)
    # Last few rows should have NaN (no future data)
    assert result["forward_return"].isna().any()


def test_fwd_returns_values_reasonable():
    prices = _make_prices(100, n_syms=1)
    result = compute_forward_returns(prices, horizon_days=5)
    valid = result["forward_return"].dropna()
    # Forward returns should be reasonable (< ±100% for synthetic data)
    assert (valid.abs() < 1.0).all()


def test_fwd_returns_missing_column_raises():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", 10), "symbol": "S"})
    with pytest.raises(ValueError):
        compute_forward_returns(df, horizon_days=5)
