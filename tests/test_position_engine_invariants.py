"""Tests for position engine invariants (Sprint 13 L2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.accounting.ledger import events_from_trades
from src.assembled_core.accounting.position_engine import build_positions_from_ledger


def test_cash_invariant():
    """Test that cash_end = cash_start + sum(cash_delta)."""
    start_cash = 10000.0

    # Create trades
    trades_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 11:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 12:00:00", tz="UTC"),
            ],
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "side": ["BUY", "SELL", "SELL"],
            "qty": [100.0, 50.0, 50.0],
            "price": [150.0, 300.0, 155.0],
            "fill_qty": [100.0, 50.0, 50.0],
            "fill_price": [150.0, 300.0, 155.0],
            "status": ["filled", "filled", "filled"],
            "commission_cash": [1.5, 1.5, 1.5],
            "spread_cash": [0.5, 0.5, 0.5],
            "slippage_cash": [0.0, 0.0, 0.0],
            "total_cost_cash": [2.0, 2.0, 2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Calculate expected cash delta sum
    expected_cash_delta_sum = events_df["cash_delta"].sum()

    # Build positions
    result = build_positions_from_ledger(events_df, start_cash=start_cash)

    # Verify cash invariant
    expected_cash_end = start_cash + expected_cash_delta_sum
    assert abs(result["cash_balance"] - expected_cash_end) < 0.01


def test_partial_fills_cash_pnl_correct():
    """Test that partial fills use fill_qty for cash/pnl calculation."""
    start_cash = 10000.0

    # Order: BUY 100 @ $50, but only 30 filled
    trades_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],  # Order qty
            "price": [50.0],
            "fill_qty": [30.0],  # Partial fill
            "fill_price": [50.0],
            "status": ["partial"],
            "commission_cash": [0.45],  # Based on fill_qty
            "spread_cash": [0.15],
            "slippage_cash": [0.0],
            "total_cost_cash": [0.6],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Build positions
    result = build_positions_from_ledger(events_df, start_cash=start_cash)

    # Verify: position qty = 30 (fill_qty), not 100 (order qty)
    assert len(result["positions_df"]) == 1
    assert abs(result["positions_df"].iloc[0]["qty"] - 30.0) < 0.01

    # Verify: cash_delta based on fill_qty (30 * 50 + 0.6 = 1500.6)
    expected_cash_delta = -(30.0 * 50.0 + 0.6)
    assert abs(events_df.iloc[0]["cash_delta"] - expected_cash_delta) < 0.01


def test_flip_case_long_to_short_realized_pnl():
    """Test that flipping from long to short calculates realized PnL correctly."""
    start_cash = 10000.0

    # Buy 100 @ $50 (long position)
    trades_df1 = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [50.0],
            "fill_qty": [100.0],
            "fill_price": [50.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    # Sell 150 @ $55 (flip: close 100 long, open 50 short)
    trades_df2 = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 11:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["SELL"],
            "qty": [150.0],
            "price": [55.0],
            "fill_qty": [150.0],
            "fill_price": [55.0],
            "status": ["filled"],
            "commission_cash": [2.25],
            "spread_cash": [0.75],
            "slippage_cash": [0.0],
            "total_cost_cash": [3.0],
        }
    )

    events_df1 = events_from_trades(trades_df1, run_id="test", source="test")
    events_df2 = events_from_trades(trades_df2, run_id="test", source="test")
    events_df = pd.concat([events_df1, events_df2], ignore_index=True)

    # Build positions
    result = build_positions_from_ledger(events_df, start_cash=start_cash)

    # Verify: position is now short 50
    assert len(result["positions_df"]) == 1
    assert abs(result["positions_df"].iloc[0]["qty"] - (-50.0)) < 0.01

    # Verify: realized PnL = (55 - 50) * 100 = 500.0
    realized_pnl = result["positions_df"].iloc[0]["realized_pnl"]
    assert abs(realized_pnl - 500.0) < 0.01

    # Verify: avg_price for short = 55.0 (new position)
    avg_price = result["positions_df"].iloc[0]["avg_price"]
    assert abs(avg_price - 55.0) < 0.01


def test_deterministic_ordering():
    """Test that same inputs produce same outputs (deterministic)."""
    start_cash = 10000.0

    trades_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 11:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 12:00:00", tz="UTC"),
            ],
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "side": ["BUY", "SELL", "SELL"],
            "qty": [100.0, 50.0, 50.0],
            "price": [150.0, 300.0, 155.0],
            "fill_qty": [100.0, 50.0, 50.0],
            "fill_price": [150.0, 300.0, 155.0],
            "status": ["filled", "filled", "filled"],
            "commission_cash": [1.5, 1.5, 1.5],
            "spread_cash": [0.5, 0.5, 0.5],
            "slippage_cash": [0.0, 0.0, 0.0],
            "total_cost_cash": [2.0, 2.0, 2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Run twice
    result1 = build_positions_from_ledger(events_df, start_cash=start_cash)
    result2 = build_positions_from_ledger(events_df, start_cash=start_cash)

    # Verify: identical results
    pd.testing.assert_frame_equal(
        result1["positions_df"].sort_values("symbol").reset_index(drop=True),
        result2["positions_df"].sort_values("symbol").reset_index(drop=True),
    )
    assert abs(result1["cash_balance"] - result2["cash_balance"]) < 0.01
    assert result1["summary"] == result2["summary"]


def test_missing_prices_behavior_zero():
    """Test missing prices behavior with policy='zero'."""
    start_cash = 10000.0

    trades_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [100.0],
            "fill_price": [150.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Build positions without prices_df (missing prices)
    result = build_positions_from_ledger(
        events_df,
        start_cash=start_cash,
        prices_df=None,
        missing_price_policy="zero",
    )

    # Verify: position exists
    assert len(result["positions_df"]) == 1

    # Verify: unrealized_pnl = 0, last_price = NaN
    assert result["positions_df"].iloc[0]["unrealized_pnl"] == 0.0
    assert pd.isna(result["positions_df"].iloc[0]["last_price"])


def test_missing_prices_behavior_raise():
    """Test missing prices behavior with policy='raise'."""
    start_cash = 10000.0

    trades_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [100.0],
            "fill_price": [150.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Build positions without prices_df (missing prices) with raise policy
    with pytest.raises(ValueError, match="Missing price for symbol"):
        build_positions_from_ledger(
            events_df,
            start_cash=start_cash,
            prices_df=None,
            missing_price_policy="raise",
        )


def test_unrealized_pnl_with_prices():
    """Test unrealized PnL calculation with prices_df."""
    start_cash = 10000.0

    trades_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [100.0],
            "fill_price": [150.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Create prices_df
    prices_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15 09:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 11:00:00", tz="UTC"),  # Mark price
            ],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [145.0, 150.0, 155.0],  # Mark price = 155.0
        }
    )

    # Build positions with mark_ts = 11:00:00
    mark_ts = pd.Timestamp("2024-01-15 11:00:00", tz="UTC")
    result = build_positions_from_ledger(
        events_df,
        start_cash=start_cash,
        prices_df=prices_df,
        mark_ts=mark_ts,
    )

    # Verify: unrealized_pnl = 100 * (155 - 150) = 500.0
    assert len(result["positions_df"]) == 1
    unrealized_pnl = result["positions_df"].iloc[0]["unrealized_pnl"]
    assert abs(unrealized_pnl - 500.0) < 0.01

    # Verify: last_price = 155.0
    last_price = result["positions_df"].iloc[0]["last_price"]
    assert abs(last_price - 155.0) < 0.01


def test_short_position_unrealized_pnl():
    """Test unrealized PnL for short positions."""
    start_cash = 10000.0

    # Short 100 @ $50
    trades_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["SELL"],
            "qty": [100.0],
            "price": [50.0],
            "fill_qty": [100.0],
            "fill_price": [50.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    events_df = events_from_trades(trades_df, run_id="test", source="test")

    # Create prices_df with mark price = 45.0 (profit on short)
    prices_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 11:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "close": [45.0],
        }
    )

    mark_ts = pd.Timestamp("2024-01-15 11:00:00", tz="UTC")
    result = build_positions_from_ledger(
        events_df,
        start_cash=start_cash,
        prices_df=prices_df,
        mark_ts=mark_ts,
    )

    # Verify: unrealized_pnl = -100 * (45 - 50) = 500.0 (profit on short)
    assert len(result["positions_df"]) == 1
    unrealized_pnl = result["positions_df"].iloc[0]["unrealized_pnl"]
    assert abs(unrealized_pnl - 500.0) < 0.01

    # Verify: qty is negative (short)
    qty = result["positions_df"].iloc[0]["qty"]
    assert qty < 0.0
    assert abs(qty - (-100.0)) < 0.01
