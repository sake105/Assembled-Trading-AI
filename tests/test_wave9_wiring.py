"""Tests for wave-9 module wiring into trading_cycle.py.

Covers:
  Step 5.8 — risk.systemic_risk (compute_return_network_centrality)
  Step 6.85 — risk.transaction_costs (estimate_per_trade_cost)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.systemic_risk import compute_return_network_centrality
from src.assembled_core.risk.transaction_costs import estimate_per_trade_cost


# ---------------------------------------------------------------------------
# compute_return_network_centrality (Step 5.8)
# ---------------------------------------------------------------------------

def _make_returns(n_symbols: int = 5, n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    syms = [f"S{i}" for i in range(n_symbols)]
    data = rng.standard_normal((n_days, n_symbols))
    return pd.DataFrame(data, columns=syms)


def test_centrality_returns_dict():
    returns = _make_returns()
    result = compute_return_network_centrality(returns)
    assert isinstance(result, dict)
    assert len(result) == 5


def test_centrality_values_between_0_and_1():
    returns = _make_returns()
    result = compute_return_network_centrality(returns)
    for sym, score in result.items():
        assert 0.0 <= score <= 1.0, f"{sym}: {score}"


def test_centrality_all_symbols_present():
    returns = _make_returns(n_symbols=4)
    result = compute_return_network_centrality(returns)
    assert set(result.keys()) == {"S0", "S1", "S2", "S3"}


def test_centrality_high_correlation_yields_high_centrality():
    n = 30
    base = np.linspace(0, 1, n)
    # All symbols perfectly correlated → max centrality
    df = pd.DataFrame({f"S{i}": base + 0.001 * i for i in range(4)})
    result = compute_return_network_centrality(df, correlation_threshold=0.5)
    for score in result.values():
        assert score == 1.0


def test_centrality_uncorrelated_yields_low_centrality():
    rng = np.random.default_rng(42)
    # Independent noise — most correlations < 0.5
    df = pd.DataFrame(rng.standard_normal((200, 4)), columns=["A", "B", "C", "D"])
    result = compute_return_network_centrality(df, correlation_threshold=0.9)
    assert isinstance(result, dict)
    assert len(result) == 4


def test_centrality_single_symbol():
    df = pd.DataFrame({"S0": [0.01, -0.02, 0.03, 0.01, -0.01]})
    result = compute_return_network_centrality(df)
    assert result == {"S0": 0.0}


# ---------------------------------------------------------------------------
# estimate_per_trade_cost (Step 6.85)
# ---------------------------------------------------------------------------

def _make_orders(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-02")] * n,
        "symbol": [f"S{i}" for i in range(n)],
        "side": ["buy"] * n,
        "qty": [100.0] * n,
        "price": [50.0 + rng.standard_normal() for _ in range(n)],
    })


def test_estimate_cost_returns_series():
    orders = _make_orders()
    result = estimate_per_trade_cost(orders)
    assert isinstance(result, pd.Series)
    assert len(result) == len(orders)


def test_estimate_cost_all_positive():
    orders = _make_orders()
    result = estimate_per_trade_cost(orders)
    assert (result > 0).all()


def test_estimate_cost_higher_bps_yields_higher_cost():
    orders = _make_orders()
    low = estimate_per_trade_cost(orders, commission_bps=0.5, slippage_bps=1.0)
    high = estimate_per_trade_cost(orders, commission_bps=5.0, slippage_bps=10.0)
    assert high.sum() > low.sum()


def test_estimate_cost_missing_column_raises():
    bad = pd.DataFrame({"symbol": ["A"], "qty": [100.0], "price": [50.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_per_trade_cost(bad)


def test_estimate_cost_index_matches_orders():
    orders = _make_orders(n=7)
    result = estimate_per_trade_cost(orders)
    assert list(result.index) == list(orders.index)


def test_estimate_cost_sell_also_has_cost():
    orders = _make_orders(n=3)
    orders["side"] = "sell"
    result = estimate_per_trade_cost(orders)
    assert (result > 0).all()
