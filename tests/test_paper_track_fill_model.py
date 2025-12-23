"""Unit tests for Paper-Track fill simulation (slippage/cost model)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.paper.paper_track import _simulate_order_fills


def test_fill_model_buy_order_with_costs():
    """Test that buy orders correctly apply spread, impact, and commission."""
    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 100.0,
                "price": 150.0,  # Mid price
            }
        ]
    )

    commission_bps = 0.5  # 0.5 bps
    spread_w = 0.25  # 0.25 bps
    impact_w = 0.5  # 0.5 bps
    current_cash = 50000.0

    filled, new_cash = _simulate_order_fills(
        orders, current_cash, commission_bps, spread_w, impact_w
    )

    # Verify fill_price (BUY pays: price * (1 + spread + impact))
    # spread_w = 0.25 bps = 0.000025, impact_w = 0.5 bps = 0.00005
    # fill_price = 150.0 * (1 + 0.000025 + 0.00005) = 150.0 * 1.000075 = 150.01125
    expected_fill_price = 150.0 * (1.0 + 0.25 * 1e-4 + 0.5 * 1e-4)
    assert filled["fill_price"].iloc[0] == pytest.approx(expected_fill_price, rel=1e-10)

    # Verify commission (on original notional: qty * price)
    notional = 100.0 * 150.0  # 15000
    expected_commission = 0.5 * 1e-4 * notional  # 0.75
    assert filled["costs"].iloc[0] == pytest.approx(expected_commission, rel=1e-10)

    # Verify cash delta (BUY: cash decreases by qty * fill_price + costs)
    expected_cash_delta = -(100.0 * expected_fill_price + expected_commission)
    assert filled["cash_delta"].iloc[0] == pytest.approx(expected_cash_delta, rel=1e-10)

    # Verify new cash
    assert new_cash == pytest.approx(current_cash + expected_cash_delta, rel=1e-10)


def test_fill_model_sell_order_with_costs():
    """Test that sell orders correctly apply spread, impact, and commission."""
    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "side": "SELL",
                "qty": 50.0,
                "price": 200.0,  # Mid price
            }
        ]
    )

    commission_bps = 1.0  # 1.0 bps
    spread_w = 0.5  # 0.5 bps
    impact_w = 1.0  # 1.0 bps
    current_cash = 30000.0

    filled, new_cash = _simulate_order_fills(
        orders, current_cash, commission_bps, spread_w, impact_w
    )

    # Verify fill_price (SELL receives: price * (1 - spread - impact))
    # spread_w = 0.5 bps = 0.00005, impact_w = 1.0 bps = 0.0001
    # fill_price = 200.0 * (1 - 0.00005 - 0.0001) = 200.0 * 0.99985 = 199.97
    expected_fill_price = 200.0 * (1.0 - 0.5 * 1e-4 - 1.0 * 1e-4)
    assert filled["fill_price"].iloc[0] == pytest.approx(expected_fill_price, rel=1e-10)

    # Verify commission (on original notional: qty * price)
    notional = 50.0 * 200.0  # 10000
    expected_commission = 1.0 * 1e-4 * notional  # 1.0
    assert filled["costs"].iloc[0] == pytest.approx(expected_commission, rel=1e-10)

    # Verify cash delta (SELL: cash increases by qty * fill_price - costs)
    expected_cash_delta = +(50.0 * expected_fill_price - expected_commission)
    assert filled["cash_delta"].iloc[0] == pytest.approx(expected_cash_delta, rel=1e-10)

    # Verify new cash
    assert new_cash == pytest.approx(current_cash + expected_cash_delta, rel=1e-10)


def test_fill_model_multiple_orders():
    """Test fill model with multiple orders (buy and sell)."""
    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 100.0,
                "price": 150.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "MSFT",
                "side": "SELL",
                "qty": 50.0,
                "price": 300.0,
            },
        ]
    )

    commission_bps = 0.5
    spread_w = 0.25
    impact_w = 0.5
    current_cash = 100000.0

    filled, new_cash = _simulate_order_fills(
        orders, current_cash, commission_bps, spread_w, impact_w
    )

    # Verify both orders processed
    assert len(filled) == 2

    # Verify buy order
    buy_order = filled[filled["side"] == "BUY"].iloc[0]
    buy_fill_price = 150.0 * (1.0 + 0.25 * 1e-4 + 0.5 * 1e-4)
    buy_commission = 0.5 * 1e-4 * (100.0 * 150.0)
    buy_cash_delta = -(100.0 * buy_fill_price + buy_commission)
    assert buy_order["fill_price"] == pytest.approx(buy_fill_price, rel=1e-10)
    assert buy_order["costs"] == pytest.approx(buy_commission, rel=1e-10)
    assert buy_order["cash_delta"] == pytest.approx(buy_cash_delta, rel=1e-10)

    # Verify sell order
    sell_order = filled[filled["side"] == "SELL"].iloc[0]
    sell_fill_price = 300.0 * (1.0 - 0.25 * 1e-4 - 0.5 * 1e-4)
    sell_commission = 0.5 * 1e-4 * (50.0 * 300.0)
    sell_cash_delta = +(50.0 * sell_fill_price - sell_commission)
    assert sell_order["fill_price"] == pytest.approx(sell_fill_price, rel=1e-10)
    assert sell_order["costs"] == pytest.approx(sell_commission, rel=1e-10)
    assert sell_order["cash_delta"] == pytest.approx(sell_cash_delta, rel=1e-10)

    # Verify total cash change
    total_cash_delta = buy_cash_delta + sell_cash_delta
    assert new_cash == pytest.approx(current_cash + total_cash_delta, rel=1e-10)


def test_fill_model_zero_costs():
    """Test fill model with zero costs (should still work)."""
    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-15", tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 100.0,
                "price": 150.0,
            }
        ]
    )

    filled, new_cash = _simulate_order_fills(orders, 50000.0, 0.0, 0.0, 0.0)

    # With zero costs, fill_price should equal original price
    assert filled["fill_price"].iloc[0] == pytest.approx(150.0, rel=1e-10)
    assert filled["costs"].iloc[0] == pytest.approx(0.0, rel=1e-10)
    assert filled["cash_delta"].iloc[0] == pytest.approx(-15000.0, rel=1e-10)
    assert new_cash == pytest.approx(35000.0, rel=1e-10)

