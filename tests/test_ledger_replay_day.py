"""Replay-day test for ledger system (Sprint 13 L5).

Tests that running ledger build twice from the same inputs produces identical results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.ledger import events_from_orders, events_from_trades
from src.assembled_core.accounting.ledger_store import (
    load_ledger_events_parquet,
    store_ledger_events_parquet,
)
from src.assembled_core.accounting.position_engine import build_positions_from_ledger


def test_ledger_events_identical_on_replay(tmp_path: Path):
    """Test that ledger events are identical when built twice from same inputs."""
    run_id = "replay_test_001"

    # Create test data
    orders_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-15 10:00:00", "2024-01-15 11:00:00"], utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
        }
    )

    trades_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-15 10:00:00", "2024-01-15 11:00:00"], utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
            "fill_qty": [100.0, 50.0],
            "fill_price": [150.0, 200.0],
            "status": ["filled", "filled"],
            "total_cost_cash": [1.5, 1.0],
        }
    )

    # Build ledger events first time
    order_events1 = events_from_orders(orders_df, run_id=run_id, source="test")
    trade_events1 = events_from_trades(trades_df, run_id=run_id, source="test")
    all_events1 = pd.concat([order_events1, trade_events1], ignore_index=True)

    # Store first time
    store_ledger_events_parquet(all_events1, tmp_path, run_id, mode="replace")

    # Build ledger events second time (from same inputs)
    order_events2 = events_from_orders(orders_df, run_id=run_id, source="test")
    trade_events2 = events_from_trades(trades_df, run_id=run_id, source="test")
    all_events2 = pd.concat([order_events2, trade_events2], ignore_index=True)

    # Store second time
    store_ledger_events_parquet(all_events2, tmp_path, run_id, mode="replace")

    # Load both and compare
    loaded1 = load_ledger_events_parquet(tmp_path, run_id)
    loaded2 = load_ledger_events_parquet(tmp_path, run_id)

    # Should be identical (same event_ids, same order)
    pd.testing.assert_frame_equal(
        loaded1.sort_values(
            ["event_ts", "event_type", "symbol", "event_id"]
        ).reset_index(drop=True),
        loaded2.sort_values(
            ["event_ts", "event_type", "symbol", "event_id"]
        ).reset_index(drop=True),
    )

    # Verify event_ids are stable
    assert set(loaded1["event_id"]) == set(loaded2["event_id"])


def test_positions_cash_identical_on_replay(tmp_path: Path):
    """Test that positions and cash are identical when built twice from same ledger events."""
    run_id = "replay_test_002"

    # Create test data
    orders_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-15 10:00:00", "2024-01-15 11:00:00"], utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
        }
    )

    trades_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-15 10:00:00", "2024-01-15 11:00:00"], utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
            "fill_qty": [100.0, 50.0],
            "fill_price": [150.0, 200.0],
            "status": ["filled", "filled"],
            "total_cost_cash": [1.5, 1.0],
        }
    )

    # Build ledger events
    order_events = events_from_orders(orders_df, run_id=run_id, source="test")
    trade_events = events_from_trades(trades_df, run_id=run_id, source="test")
    all_events = pd.concat([order_events, trade_events], ignore_index=True)

    # Build positions first time
    result1 = build_positions_from_ledger(
        all_events,
        prices_df=None,
        mark_ts=None,
        start_cash=10000.0,
        missing_price_policy="zero",
    )

    # Build positions second time (from same events)
    result2 = build_positions_from_ledger(
        all_events,
        prices_df=None,
        mark_ts=None,
        start_cash=10000.0,
        missing_price_policy="zero",
    )

    # Compare positions
    pd.testing.assert_frame_equal(
        result1["positions_df"].sort_values("symbol").reset_index(drop=True),
        result2["positions_df"].sort_values("symbol").reset_index(drop=True),
    )

    # Compare cash
    assert abs(result1["cash_balance"] - result2["cash_balance"]) < 1e-6


def test_reconciliation_result_identical_on_replay(tmp_path: Path):
    """Test that reconciliation result is identical when run twice from same positions."""
    from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker

    # Create test positions
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )

    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )

    # Reconcile first time
    result1 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    # Reconcile second time (same inputs)
    result2 = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    # Should be identical
    assert result1["ok"] == result2["ok"]
    assert abs(result1["cash_diff"] - result2["cash_diff"]) < 1e-6
    assert len(result1["position_diffs_df"]) == len(result2["position_diffs_df"])
    assert len(result1["missing_in_ledger"]) == len(result2["missing_in_ledger"])
    assert len(result1["missing_in_broker"]) == len(result2["missing_in_broker"])
