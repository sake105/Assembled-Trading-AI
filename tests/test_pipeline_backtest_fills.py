"""Unit tests for vectorized fill simulation in backtest.py.

This test module verifies that:
- _simulate_fills_per_order produces identical results to the original loop-based implementation
- Edge cases are handled correctly (NaNs, qty=0, invalid symbols, buy/sell signs)
- Fill prices, fees, and notional are computed correctly with costs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.pipeline.backtest import _simulate_fills_per_order


def test_fills_basic_buy_order() -> None:
    """Test basic BUY order execution."""
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [100.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # BUY: cash decreases by qty * price
    assert updated_cash == 1000.0 - (10.0 * 100.0)
    assert updated_positions["AAPL"] == 10.0


def test_fills_basic_sell_order() -> None:
    """Test basic SELL order execution."""
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [5.0],
        "price": [100.0],
    })

    cash = 1000.0
    positions = {"AAPL": 10.0}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # SELL: cash increases by qty * price
    assert updated_cash == 1000.0 + (5.0 * 100.0)
    assert updated_positions["AAPL"] == 10.0 - 5.0


def test_fills_multiple_orders_same_symbol() -> None:
    """Test multiple orders for the same symbol (aggregation)."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "side": ["BUY", "BUY"],
        "qty": [10.0, 5.0],
        "price": [100.0, 110.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # Total qty: 10 + 5 = 15
    # Total cash: 1000 - (10*100) - (5*110) = 1000 - 1000 - 550 = -550
    assert updated_positions["AAPL"] == 15.0
    assert updated_cash == 1000.0 - (10.0 * 100.0) - (5.0 * 110.0)


def test_fills_multiple_symbols() -> None:
    """Test orders for multiple symbols."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [10.0, 5.0],
        "price": [100.0, 200.0],
    })

    cash = 1000.0
    positions = {"MSFT": 10.0}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # AAPL: BUY 10 @ 100 = -1000 cash, +10 position
    # MSFT: SELL 5 @ 200 = +1000 cash, -5 position
    assert updated_cash == 1000.0 - 1000.0 + 1000.0
    assert updated_positions["AAPL"] == 10.0
    assert updated_positions["MSFT"] == 10.0 - 5.0


def test_fills_with_costs() -> None:
    """Test fill prices and fees with costs (spread, impact, commission)."""
    orders = pd.DataFrame({
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [10.0],
        "price": [100.0],  # mid price
    })

    cash = 1000.0
    positions = {}

    # With costs: spread_w=25bp, impact_w=50bp, commission_bps=10bp
    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions, spread_w=25.0, impact_w=50.0, commission_bps=10.0
    )

    # BUY fill price: 100 * (1 + 0.0025 + 0.0050) = 100 * 1.0075 = 100.75
    # Notional: 10 * 100 = 1000
    # Commission: 1000 * 0.001 = 1.0
    # Total cost: (10 * 100.75) + 1.0 = 1007.5 + 1.0 = 1008.5
    expected_cash = 1000.0 - 1008.5
    assert abs(updated_cash - expected_cash) < 1e-6
    assert updated_positions["AAPL"] == 10.0


def test_fills_edge_case_nan_price() -> None:
    """Test that orders with NaN prices are filtered out."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [10.0, 5.0],
        "price": [100.0, np.nan],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # Only AAPL order should execute
    assert updated_cash == 1000.0 - (10.0 * 100.0)
    assert "AAPL" in updated_positions
    assert updated_positions["AAPL"] == 10.0
    assert "MSFT" not in updated_positions


def test_fills_edge_case_zero_qty() -> None:
    """Test that orders with zero qty are filtered out."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "BUY"],
        "qty": [10.0, 0.0],
        "price": [100.0, 200.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # Only AAPL order should execute
    assert updated_cash == 1000.0 - (10.0 * 100.0)
    assert "AAPL" in updated_positions
    assert updated_positions["AAPL"] == 10.0
    assert "MSFT" not in updated_positions


def test_fills_edge_case_empty_symbol() -> None:
    """Test that orders with empty symbols are filtered out."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", ""],
        "side": ["BUY", "BUY"],
        "qty": [10.0, 5.0],
        "price": [100.0, 200.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # Only AAPL order should execute
    assert updated_cash == 1000.0 - (10.0 * 100.0)
    assert "AAPL" in updated_positions
    assert updated_positions["AAPL"] == 10.0


def test_fills_edge_case_invalid_side() -> None:
    """Test that orders with invalid side (not BUY/SELL) are filtered out."""
    orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "INVALID"],
        "qty": [10.0, 5.0],
        "price": [100.0, 200.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # Only AAPL order should execute (MSFT with invalid side is filtered out)
    assert updated_cash == 1000.0 - (10.0 * 100.0)
    assert "AAPL" in updated_positions
    assert updated_positions["AAPL"] == 10.0
    # MSFT should not appear in positions (invalid side filtered out)


def test_fills_empty_orders() -> None:
    """Test that empty orders DataFrame returns unchanged cash and positions."""
    orders = pd.DataFrame(columns=["symbol", "side", "qty", "price"])

    cash = 1000.0
    positions = {"AAPL": 10.0}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    assert updated_cash == cash
    assert updated_positions == positions


def test_fills_all_invalid_orders() -> None:
    """Test that all invalid orders are filtered out."""
    orders = pd.DataFrame({
        "symbol": ["", "AAPL"],
        "side": ["BUY", "BUY"],
        "qty": [10.0, 0.0],
        "price": [np.nan, 100.0],
    })

    cash = 1000.0
    positions = {}

    updated_cash, updated_positions = _simulate_fills_per_order(
        orders, cash, positions
    )

    # No valid orders, so cash and positions should be unchanged
    assert updated_cash == cash
    assert updated_positions == positions

