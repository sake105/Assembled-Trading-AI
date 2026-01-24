"""Tests for ledger storage roundtrip (Sprint 13 L1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.ledger import events_from_trades, generate_event_id
from src.assembled_core.accounting.ledger_store import (
    ledger_base_path,
    list_ledger_runs,
    load_ledger_events_parquet,
    store_ledger_events_parquet,
)


def test_store_load_same_deterministic_sort(tmp_path: Path):
    """Test that store->load preserves deterministic sort."""
    run_id = "test_run_001"

    # Create sample events
    trades_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 10:05:00", tz="UTC"),
                pd.Timestamp("2024-01-14 15:00:00", tz="UTC"),  # Earlier timestamp
            ],
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "side": ["BUY", "SELL", "BUY"],
            "qty": [100.0, 50.0, 200.0],
            "price": [150.0, 300.0, 145.0],
            "fill_qty": [100.0, 50.0, 200.0],
            "fill_price": [150.0, 300.0, 145.0],
            "status": ["filled", "filled", "filled"],
            "commission_cash": [1.5, 1.5, 2.0],
            "spread_cash": [0.5, 0.5, 0.5],
            "slippage_cash": [0.0, 0.0, 0.0],
            "total_cost_cash": [2.0, 2.0, 2.5],
        }
    )

    events_df = events_from_trades(trades_df, run_id=run_id, source="test")

    # Store
    _ = store_ledger_events_parquet(events_df, tmp_path, run_id, mode="replace")

    # Load
    loaded_df = load_ledger_events_parquet(tmp_path, run_id)

    # Verify: same number of events
    assert len(loaded_df) == len(events_df)

    # Verify: deterministic sort (event_ts, then event_id)
    assert loaded_df["event_ts"].is_monotonic_increasing
    # Check that earlier timestamp comes first
    assert loaded_df.iloc[0]["event_ts"] == pd.Timestamp("2024-01-14 15:00:00", tz="UTC")

    # Verify: all required columns present
    required_cols = ["event_ts", "event_type", "symbol", "qty", "price", "cash_delta", "run_id", "event_id"]
    for col in required_cols:
        assert col in loaded_df.columns

    # Verify: event_ids match
    assert set(loaded_df["event_id"]) == set(events_df["event_id"])


def test_append_dedupe_stable(tmp_path: Path):
    """Test that append mode deduplicates by event_id stably."""
    run_id = "test_run_002"

    # Create initial events
    trades_df1 = pd.DataFrame(
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

    events_df1 = events_from_trades(trades_df1, run_id=run_id, source="test")
    store_ledger_events_parquet(events_df1, tmp_path, run_id, mode="replace")

    # Append same events (should dedupe)
    events_df2 = events_from_trades(trades_df1, run_id=run_id, source="test")
    store_ledger_events_parquet(events_df2, tmp_path, run_id, mode="append")

    # Load
    loaded_df = load_ledger_events_parquet(tmp_path, run_id)

    # Verify: only one event (deduplicated)
    assert len(loaded_df) == 1

    # Append new event
    trades_df3 = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 11:00:00", tz="UTC")],
            "symbol": ["MSFT"],
            "side": ["SELL"],
            "qty": [50.0],
            "price": [300.0],
            "fill_qty": [50.0],
            "fill_price": [300.0],
            "status": ["filled"],
            "commission_cash": [1.5],
            "spread_cash": [0.5],
            "slippage_cash": [0.0],
            "total_cost_cash": [2.0],
        }
    )

    events_df3 = events_from_trades(trades_df3, run_id=run_id, source="test")
    store_ledger_events_parquet(events_df3, tmp_path, run_id, mode="append")

    # Load again
    loaded_df2 = load_ledger_events_parquet(tmp_path, run_id)

    # Verify: two events now
    assert len(loaded_df2) == 2


def test_no_tmp_leftovers(tmp_path: Path):
    """Test that no temp files are left after write."""
    run_id = "test_run_003"

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

    events_df = events_from_trades(trades_df, run_id=run_id, source="test")
    store_ledger_events_parquet(events_df, tmp_path, run_id, mode="replace")

    # Check ledger directory
    ledger_dir = ledger_base_path(tmp_path, run_id)

    # Verify: no .tmp.parquet files
    tmp_files = list(ledger_dir.glob("*.tmp.parquet"))
    assert len(tmp_files) == 0, f"Found temp files: {tmp_files}"

    # Verify: ledger_events.parquet exists
    ledger_path = ledger_dir / "ledger_events.parquet"
    assert ledger_path.exists()


def test_event_id_stable_across_runs():
    """Test that event_id is stable for same inputs."""
    event_ts = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    symbol = "AAPL"
    qty = 100.0
    price = 150.0

    # Generate event_id twice
    event_id1 = generate_event_id("FILL", event_ts, symbol, qty, price, row_index=0)
    event_id2 = generate_event_id("FILL", event_ts, symbol, qty, price, row_index=0)

    # Verify: same event_id
    assert event_id1 == event_id2

    # Different row_index -> different event_id
    event_id3 = generate_event_id("FILL", event_ts, symbol, qty, price, row_index=1)
    assert event_id1 != event_id3

    # Different qty -> different event_id
    event_id4 = generate_event_id("FILL", event_ts, symbol, 200.0, price, row_index=0)
    assert event_id1 != event_id4


def test_list_ledger_runs(tmp_path: Path):
    """Test listing ledger runs."""
    # Create multiple runs
    for run_id in ["run_a", "run_b", "run_c"]:
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
        events_df = events_from_trades(trades_df, run_id=run_id, source="test")
        store_ledger_events_parquet(events_df, tmp_path, run_id, mode="replace")

    # List runs
    run_ids = list_ledger_runs(tmp_path)

    # Verify: all runs listed, sorted
    assert len(run_ids) == 3
    assert run_ids == ["run_a", "run_b", "run_c"]


def test_cash_delta_calculation():
    """Test that cash_delta is calculated correctly for BUY/SELL."""
    # BUY: cash_delta should be negative
    trades_buy = pd.DataFrame(
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

    events_buy = events_from_trades(trades_buy, run_id="test", source="test")
    assert events_buy.iloc[0]["cash_delta"] < 0
    # BUY: -(100 * 150 + 2.0) = -15002.0
    assert abs(events_buy.iloc[0]["cash_delta"] - (-15002.0)) < 0.01

    # SELL: cash_delta should be positive
    trades_sell = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["SELL"],
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

    events_sell = events_from_trades(trades_sell, run_id="test", source="test")
    assert events_sell.iloc[0]["cash_delta"] > 0
    # SELL: +(100 * 150 - 2.0) = 14998.0
    assert abs(events_sell.iloc[0]["cash_delta"] - 14998.0) < 0.01

    # REJECT: cash_delta should be 0
    trades_reject = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
            "fill_qty": [0.0],
            "fill_price": [150.0],
            "status": ["rejected"],
            "commission_cash": [0.0],
            "spread_cash": [0.0],
            "slippage_cash": [0.0],
            "total_cost_cash": [0.0],
        }
    )

    events_reject = events_from_trades(trades_reject, run_id="test", source="test")
    assert events_reject.iloc[0]["event_type"] == "REJECT"
    assert events_reject.iloc[0]["cash_delta"] == 0.0


def test_append_mode_identical_writes(tmp_path: Path):
    """Test that two append writes with same events produce identical file contents."""
    run_id = "test_append_identical"

    # Create events
    trades_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-15 11:00:00", tz="UTC"),
            ],
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [100.0, 50.0],
            "price": [150.0, 200.0],
            "fill_qty": [100.0, 50.0],
            "fill_price": [150.0, 200.0],
            "status": ["filled", "filled"],
            "commission_cash": [1.5, 1.0],
            "spread_cash": [0.5, 0.5],
            "slippage_cash": [0.0, 0.0],
            "total_cost_cash": [2.0, 1.5],
        }
    )

    events_df = events_from_trades(trades_df, run_id=run_id, source="test")

    # First write (replace mode)
    path1 = store_ledger_events_parquet(events_df, tmp_path, run_id, mode="replace")

    # Second write (append mode with same events - should dedupe)
    events_df2 = events_from_trades(trades_df, run_id=run_id, source="test")
    path2 = store_ledger_events_parquet(events_df2, tmp_path, run_id, mode="append")

    # Third write (append mode again - should still be identical)
    events_df3 = events_from_trades(trades_df, run_id=run_id, source="test")
    path3 = store_ledger_events_parquet(events_df3, tmp_path, run_id, mode="append")

    # All paths should be the same
    assert path1 == path2 == path3

    # Load and compare file contents
    loaded1 = load_ledger_events_parquet(tmp_path, run_id)
    loaded2 = load_ledger_events_parquet(tmp_path, run_id)
    loaded3 = load_ledger_events_parquet(tmp_path, run_id)

    # Should have same number of events (deduplicated)
    assert len(loaded1) == len(loaded2) == len(loaded3) == len(events_df)

    # Should be identical (same event_ids, same order)
    pd.testing.assert_frame_equal(
        loaded1.sort_values(["event_ts", "event_id"], kind="mergesort").reset_index(drop=True),
        loaded2.sort_values(["event_ts", "event_id"], kind="mergesort").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        loaded2.sort_values(["event_ts", "event_id"], kind="mergesort").reset_index(drop=True),
        loaded3.sort_values(["event_ts", "event_id"], kind="mergesort").reset_index(drop=True),
    )

    # Verify deterministic sort: event_ts, then event_id
    assert loaded1["event_ts"].is_monotonic_increasing
    # Check that event_ids are also sorted within same timestamp
    for ts in loaded1["event_ts"].unique():
        ts_events = loaded1[loaded1["event_ts"] == ts]
        if len(ts_events) > 1:
            assert ts_events["event_id"].is_monotonic_increasing


def test_append_mode_no_tmp_leftovers_windows(tmp_path: Path):
    """Test that no temp files are left after append writes (Windows-safe)."""
    run_id = "test_no_tmp_append"

    # Create events
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

    events_df = events_from_trades(trades_df, run_id=run_id, source="test")

    # Multiple append writes
    for i in range(3):
        store_ledger_events_parquet(events_df, tmp_path, run_id, mode="append")

    # Check ledger directory
    ledger_dir = ledger_base_path(tmp_path, run_id)

    # Verify: no .tmp.parquet files
    tmp_files = list(ledger_dir.glob("*.tmp.parquet"))
    assert len(tmp_files) == 0, f"Found temp files after append writes: {tmp_files}"

    # Verify: ledger_events.parquet exists
    ledger_path = ledger_dir / "ledger_events.parquet"
    assert ledger_path.exists()

    # Verify: only one event (deduplicated)
    loaded_df = load_ledger_events_parquet(tmp_path, run_id)
    assert len(loaded_df) == 1


def test_canonical_float_formatting_stability():
    """Test that canonical float formatting produces stable strings for event_id."""
    from src.assembled_core.accounting.ledger import generate_event_id

    # Test with problematic floats (0.1 + 0.2, etc.)
    event_ts = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    symbol = "AAPL"

    # Test 1: Normal float
    price1 = 150.123456789
    event_id1a = generate_event_id("FILL", event_ts, symbol, 100.0, price1, row_index=0)
    event_id1b = generate_event_id("FILL", event_ts, symbol, 100.0, price1, row_index=0)
    assert event_id1a == event_id1b, "Same float should produce same event_id"

    # Test 2: Float that might have rounding issues
    price2 = 0.1 + 0.2  # Might be 0.30000000000000004 in some representations
    event_id2a = generate_event_id("FILL", event_ts, symbol, 100.0, price2, row_index=0)
    event_id2b = generate_event_id("FILL", event_ts, symbol, 100.0, 0.3, row_index=0)
    # Should be same (canonical formatting should normalize)
    assert event_id2a == event_id2b, "0.1+0.2 and 0.3 should produce same event_id"

    # Test 3: Very small float
    price3 = 1e-10
    event_id3a = generate_event_id("FILL", event_ts, symbol, 100.0, price3, row_index=0)
    event_id3b = generate_event_id("FILL", event_ts, symbol, 100.0, price3, row_index=0)
    assert event_id3a == event_id3b, "Same small float should produce same event_id"

    # Test 4: Large float
    price4 = 1e10
    event_id4a = generate_event_id("FILL", event_ts, symbol, 100.0, price4, row_index=0)
    event_id4b = generate_event_id("FILL", event_ts, symbol, 100.0, price4, row_index=0)
    assert event_id4a == event_id4b, "Same large float should produce same event_id"
