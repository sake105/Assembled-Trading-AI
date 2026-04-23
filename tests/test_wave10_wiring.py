"""Tests for wave-10 module wiring into trading_cycle.py.

Covers:
  Step 3.4 — signals.mean_reversion (compute_mean_reversion_signals)
  Step 7.5 — qa.capacity (estimate_strategy_capacity)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.mean_reversion import (
    compute_mean_reversion_signals,
    compute_rsi,
)
from src.assembled_core.qa.capacity import estimate_strategy_capacity, CapacityEstimate


# ---------------------------------------------------------------------------
# compute_mean_reversion_signals (Step 3.4)
# ---------------------------------------------------------------------------

def _make_prices_panel(n_symbols: int = 3, n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        for d, p in enumerate(prices):
            rows.append({"symbol": sym, "close": float(p), "day": d})
    return pd.DataFrame(rows)


def test_mr_signals_bull_returns_df():
    df = _make_prices_panel()
    result = compute_mean_reversion_signals(df, regime="bull")
    assert isinstance(result, pd.DataFrame)
    assert "symbol" in result.columns
    assert "reversion_signal" in result.columns
    assert "reversion_type" in result.columns


def test_mr_signals_bear_returns_empty():
    df = _make_prices_panel()
    result = compute_mean_reversion_signals(df, regime="bear")
    assert result.empty


def test_mr_signals_crisis_returns_empty():
    df = _make_prices_panel()
    result = compute_mean_reversion_signals(df, regime="crisis")
    assert result.empty


def test_mr_signals_sideways_active():
    df = _make_prices_panel()
    result = compute_mean_reversion_signals(df, regime="sideways")
    assert isinstance(result, pd.DataFrame)


def test_mr_signals_values_in_range():
    df = _make_prices_panel(n_symbols=4, n_days=80)
    result = compute_mean_reversion_signals(df, regime="bull")
    if not result.empty:
        assert result["reversion_signal"].between(-2.0, 2.0).all()


def test_mr_signals_too_short_skipped():
    # Only 10 days — below the min required (~20+)
    df = _make_prices_panel(n_symbols=2, n_days=10)
    result = compute_mean_reversion_signals(df, regime="bull")
    assert isinstance(result, pd.DataFrame)
    # May be empty — no crash
    assert "symbol" in result.columns


def test_compute_rsi_returns_series():
    prices = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 50)))
    rsi = compute_rsi(prices, period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(prices)


def test_compute_rsi_values_between_0_and_100():
    prices = pd.Series(100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, 100)))
    rsi = compute_rsi(prices)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


# ---------------------------------------------------------------------------
# estimate_strategy_capacity (Step 7.5)
# ---------------------------------------------------------------------------

def _make_orders(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-02")] * n,
        "symbol": [f"S{i % 5}" for i in range(n)],
        "qty": [100.0] * n,
        "price": [50.0] * n,
        "notional": [5000.0] * n,
    })


def test_capacity_returns_dataclass():
    orders = _make_orders()
    result = estimate_strategy_capacity(orders)
    assert isinstance(result, CapacityEstimate)


def test_capacity_verdict_is_valid_string():
    orders = _make_orders()
    result = estimate_strategy_capacity(orders)
    assert result.verdict in ("ok", "warning", "exceeded")


def test_capacity_small_aum_ok():
    orders = _make_orders()
    result = estimate_strategy_capacity(orders, target_aum_usd=100_000.0)
    assert result.verdict == "ok"


def test_capacity_huge_aum_exceeded():
    orders = _make_orders()
    result = estimate_strategy_capacity(
        orders,
        target_aum_usd=1_000_000_000.0,  # 1 billion
        avg_adv_usd=500_000.0,           # small ADV
    )
    assert result.verdict == "exceeded"


def test_capacity_breakeven_aum_positive():
    orders = _make_orders()
    result = estimate_strategy_capacity(orders)
    assert result.breakeven_aum_usd > 0


def test_capacity_max_aum_positive():
    orders = _make_orders()
    result = estimate_strategy_capacity(orders)
    assert result.max_aum_usd > 0
