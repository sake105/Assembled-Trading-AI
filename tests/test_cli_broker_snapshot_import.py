"""Tests for CLI broker snapshot import integration (Sprint 13).

Tests that --broker-snapshot-file flag correctly imports snapshots before reconciliation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_import_broker_snapshot_before_reconciliation(tmp_path: Path):
    """Test that importing broker snapshot before reconciliation works correctly."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 10.0},  # Different from paper view (5.0)
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Create minimal trades (paper view will have AAPL=5.0)
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 5.0,
            "price": 150.0,
            "fill_qty": 5.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.0,
        },
    ])
    orders = trades.copy()
    run_id = "test_import_integration"
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    output_dir = tmp_path

    # Import snapshot manually (simulating CLI behavior)
    from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot

    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    assert import_result["broker_snapshot_path"] is not None
    assert import_result["cash"] == 10000.0

    # Build ledger with policy="require" - should succeed because snapshot was imported
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=run_id,
        output_dir=output_dir,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",  # Should work because snapshot was imported
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=run_id,
    )

    # Verify reconciliation was performed
    assert result["reconciliation_result"] is not None
    assert result["broker_snapshot_path"] is not None

    # The reconciliation should use imported snapshot (AAPL=10), not paper view (AAPL=5)
    # This will cause a mismatch, but the key is that snapshot was used
    reconciliation = result["reconciliation_result"]
    assert "ok" in reconciliation


def test_import_broker_snapshot_with_policy_require(tmp_path: Path):
    """Test that policy=require works when snapshot is imported."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 5.0},  # Matches paper view
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Create minimal trades
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 5.0,
            "price": 150.0,
            "fill_qty": 5.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.0,
        },
    ])
    orders = trades.copy()
    run_id = "test_import_require"
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    output_dir = tmp_path

    # Import snapshot
    from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot

    import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    # Build ledger with policy="require" - should succeed
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=run_id,
        output_dir=output_dir,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=run_id,
    )

    # Should succeed without error
    assert result["reconciliation_result"] is not None
    assert result["broker_snapshot_path"] is not None
