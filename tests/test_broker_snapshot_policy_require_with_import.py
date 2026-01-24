"""Tests for broker snapshot policy 'require' with import (Sprint 13).

Tests that policy='require' works correctly when snapshot is imported before reconciliation.
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

from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_policy_require_with_imported_snapshot(tmp_path: Path):
    """Test that policy='require' works when snapshot is imported."""
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
    run_id = "test_require_import"
    snapshot_run_id = "test_require_import"  # Same as run_id
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    output_dir = tmp_path

    # Step 1: Import snapshot (simulating CLI behavior)
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    assert import_result["broker_snapshot_path"] is not None
    assert import_result["cash"] == 10000.0

    # Step 2: Build ledger with policy="require" - should succeed
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
        broker_snapshot_run_id=snapshot_run_id,
    )

    # Should succeed without error
    assert result["reconciliation_result"] is not None
    assert result["broker_snapshot_path"] is not None
    assert result["reconciliation_ok"] is not None


def test_policy_require_with_imported_snapshot_different_run_id(tmp_path: Path):
    """Test that policy='require' works with imported snapshot in different run_id namespace."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 5.0},
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
    run_id = "test_ledger_run"
    snapshot_run_id = "test_snapshot_run"  # Different from run_id
    as_of_date = pd.Timestamp(base_time, tz="UTC")
    output_dir = tmp_path

    # Step 1: Import snapshot into different namespace
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    assert import_result["broker_snapshot_path"] is not None

    # Step 2: Build ledger with policy="require" and broker_snapshot_run_id pointing to imported snapshot
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
        broker_snapshot_run_id=snapshot_run_id,  # Use imported snapshot namespace
    )

    # Should succeed without error
    assert result["reconciliation_result"] is not None
    assert result["broker_snapshot_path"] is not None


def test_import_deterministic_path_layout(tmp_path: Path):
    """Test that imported snapshot uses stable path layout."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 100.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir = tmp_path / "output"
    run_id = "test_path_layout"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    # Verify path layout
    expected_dir = output_dir / f"broker_snapshot_{run_id}"
    assert expected_dir.exists()

    expected_json = expected_dir / "snapshot_2025-01-15.json"
    assert expected_json.exists()

    expected_parquet = expected_dir / "positions_2025-01-15.parquet"
    assert expected_parquet.exists()

    # Verify return path is relative
    assert import_result["broker_snapshot_path"].startswith(f"broker_snapshot_{run_id}")
    assert "/" in import_result["broker_snapshot_path"] or "\\" in import_result["broker_snapshot_path"]
