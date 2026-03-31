"""Partial fills accounting test (Sprint 13 L5).

Tests that ledger correctly uses fill_qty from fill_model pipeline outputs.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.accounting.ledger import events_from_trades
from src.assembled_core.accounting.position_engine import build_positions_from_ledger


def test_partial_fill_uses_fill_qty():
    """Test that partial fills use fill_qty (not qty) for cash_delta calculation."""
    # Create trade with partial fill: order qty=100, fill_qty=50
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],  # Order qty
            "price": [150.0],
            "fill_qty": [50.0],  # Partial fill: only 50 filled
            "fill_price": [150.0],
            "status": ["partial"],
            "total_cost_cash": [0.75],  # Cost based on fill_qty
        }
    )

    # Generate events
    events = events_from_trades(trades_df, run_id="test_partial", source="test")

    # Should have one FILL event
    fill_events = events[events["event_type"] == "FILL"]
    assert len(fill_events) == 1

    # Event qty should be fill_qty (50), not order qty (100)
    assert abs(fill_events.iloc[0]["qty"] - 50.0) < 1e-6

    # cash_delta should be based on fill_qty * fill_price + costs
    # BUY: -(fill_qty * fill_price + total_cost_cash) = -(50 * 150 + 0.75) = -7500.75
    expected_cash_delta = -(50.0 * 150.0 + 0.75)
    assert abs(fill_events.iloc[0]["cash_delta"] - expected_cash_delta) < 1e-6


def test_rejected_fill_has_zero_cash_delta():
    """Test that rejected fills have cash_delta=0 and costs=0."""
    # Create rejected trade
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [0.0],  # Rejected: no fill
            "fill_price": [150.0],
            "status": ["rejected"],
            "total_cost_cash": [0.0],  # Costs should be 0 for rejected
        }
    )

    # Generate events
    events = events_from_trades(trades_df, run_id="test_rejected", source="test")

    # Should have one REJECT event
    reject_events = events[events["event_type"] == "REJECT"]
    assert len(reject_events) == 1

    # cash_delta should be 0
    assert abs(reject_events.iloc[0]["cash_delta"]) < 1e-6


def test_partial_fill_position_accounting():
    """Test that partial fills result in correct position quantities."""
    # Create trades with partial fills
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-15 10:00:00", "2024-01-15 11:00:00"], utc=True
            ),
            "symbol": ["AAPL", "AAPL"],
            "side": ["BUY", "BUY"],
            "qty": [100.0, 50.0],  # Order qtys
            "price": [150.0, 155.0],
            "fill_qty": [50.0, 30.0],  # Partial fills
            "fill_price": [150.0, 155.0],
            "status": ["partial", "partial"],
            "total_cost_cash": [0.75, 0.465],
        }
    )

    # Generate events
    events = events_from_trades(trades_df, run_id="test_partial_pos", source="test")

    # Build positions
    result = build_positions_from_ledger(
        events,
        prices_df=None,
        mark_ts=None,
        start_cash=10000.0,
        missing_price_policy="zero",
    )

    positions_df = result["positions_df"]
    cash_balance = result["cash_balance"]

    # Should have one position for AAPL
    aapl_pos = positions_df[positions_df["symbol"] == "AAPL"]
    assert len(aapl_pos) == 1

    # Position qty should be sum of fill_qtys: 50 + 30 = 80
    assert abs(aapl_pos.iloc[0]["qty"] - 80.0) < 1e-6

    # Cash should reflect partial fills
    # BUY 1: -(50 * 150 + 0.75) = -7500.75
    # BUY 2: -(30 * 155 + 0.465) = -4650.465
    # Total cash delta: -12151.215
    # Final cash: 10000 - 12151.215 = -2151.215 (negative cash means we're using margin/leverage)
    # But wait, we should check the actual calculation
    expected_cash_delta_1 = -(50.0 * 150.0 + 0.75)
    expected_cash_delta_2 = -(30.0 * 155.0 + 0.465)
    expected_final_cash = 10000.0 + expected_cash_delta_1 + expected_cash_delta_2
    assert abs(cash_balance - expected_final_cash) < 1e-3


def test_full_fill_vs_partial_fill_cash_delta():
    """Test that full fill and partial fill have different cash_deltas."""
    # Full fill
    full_fill_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [100.0],  # Full fill
            "fill_price": [150.0],
            "status": ["filled"],
            "total_cost_cash": [1.5],
        }
    )

    # Partial fill (50%)
    partial_fill_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],  # Same order qty
            "price": [150.0],
            "fill_qty": [50.0],  # Partial fill
            "fill_price": [150.0],
            "status": ["partial"],
            "total_cost_cash": [0.75],  # Half the cost
        }
    )

    # Generate events
    full_events = events_from_trades(full_fill_df, run_id="test_full", source="test")
    partial_events = events_from_trades(
        partial_fill_df, run_id="test_partial", source="test"
    )

    # Compare cash_deltas
    full_cash_delta = full_events[full_events["event_type"] == "FILL"].iloc[0][
        "cash_delta"
    ]
    partial_cash_delta = partial_events[partial_events["event_type"] == "FILL"].iloc[0][
        "cash_delta"
    ]

    # Partial should be half of full (approximately, accounting for costs)
    # Full: -(100 * 150 + 1.5) = -15001.5
    # Partial: -(50 * 150 + 0.75) = -7500.75
    # Partial should be approximately half
    assert (
        abs(partial_cash_delta - full_cash_delta / 2.0) < 1.0
    )  # Allow small tolerance for cost differences
