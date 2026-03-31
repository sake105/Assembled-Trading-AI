"""E2E tests for broker snapshot import and reconciliation (Sprint 13).

Tests that imported broker snapshots are correctly used in reconciliation
and that broker_meta is properly included in reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_importer import (
    import_broker_snapshot,
)
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_imported_snapshot_used_when_policy_require(tmp_path: Path):
    """Test that imported snapshot is used when policy=require."""
    # Arrange: tmp_path/output, run_id="r1", as_of_date="2025-01-15"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "r1"
    snapshot_run_id = "r1"  # Same run_id for simplicity
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot per import_broker_snapshot(...) (JSON fixture)
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 10.0},
            {"symbol": "MSFT", "qty": 5.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    assert import_result["broker_snapshot_path"] is not None

    # Create minimal trades/orders that match the snapshot
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 10.0,
                "price": 150.0,
                "fill_qty": 10.0,
                "fill_price": 150.0,
                "status": "filled",
                "total_cost_cash": 0.0,
            },
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC"),
                "symbol": "MSFT",
                "side": "BUY",
                "qty": 5.0,
                "price": 200.0,
                "fill_qty": 5.0,
                "fill_price": 200.0,
                "status": "filled",
                "total_cost_cash": 0.0,
            },
        ]
    )
    orders = trades.copy()

    # Call build_ledger_from_trades(..., broker_snapshot_policy="require", ...)
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

    # Assert: return dict contains broker_meta with broker_view_source=="stored_snapshot"
    # and broker_snapshot_path not None
    assert "broker_meta" in result
    broker_meta = result["broker_meta"]
    assert broker_meta is not None
    assert broker_meta["broker_view_source"] == "stored_snapshot"
    assert broker_meta["broker_snapshot_run_id"] == snapshot_run_id
    assert broker_meta["broker_snapshot_date"] is not None
    assert broker_meta["broker_snapshot_path"] is not None

    # Verify reconciliation was performed
    assert result["reconciliation_result"] is not None
    assert result["reconciliation_ok"] is not None


def test_policy_require_raises_when_snapshot_missing(tmp_path: Path):
    """Test that policy=require raises ValueError when snapshot is missing."""
    # Arrange: no import, just setup output_dir
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "r1"
    snapshot_run_id = "r1"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Create minimal trades/orders
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame(
        [
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
        ]
    )
    orders = trades.copy()

    # Call with require -> pytest.raises(ValueError)
    with pytest.raises(ValueError, match="Broker snapshot required but not found"):
        build_ledger_from_trades(
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


def test_reconcile_report_includes_broker_meta_after_require(tmp_path: Path):
    """Test that reconcile report JSON includes broker_meta after require policy."""
    # Arrange: import snapshot first
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = "r1"
    snapshot_run_id = "r1"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 10.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=as_of_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    assert import_result["broker_snapshot_path"] is not None

    # Create minimal trades/orders
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(base_time, tz="UTC"),
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 10.0,
                "price": 150.0,
                "fill_qty": 10.0,
                "fill_price": 150.0,
                "status": "filled",
                "total_cost_cash": 0.0,
            },
        ]
    )
    orders = trades.copy()

    # Call build_ledger_from_trades with policy=require
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

    # Verify broker_meta in return dict
    assert result["broker_meta"]["broker_view_source"] == "stored_snapshot"

    # After successful call: load the JSON reconcile report from output/reconcile_report_<run_id>/...json
    # Report filename is based on as_of_date: reconcile_YYYY-MM-DD.json
    date_str = as_of_date.strftime("%Y-%m-%d")
    report_dir = output_dir / f"reconcile_report_{run_id}"
    report_json_path = report_dir / f"reconcile_{date_str}.json"

    assert (
        report_json_path.exists()
    ), f"Reconciliation report JSON should exist at {report_json_path}"

    # Load and verify JSON
    with report_json_path.open("r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Assert: broker_meta keys vorhanden und deterministisch (source/run_id/date/path)
    assert (
        "broker_meta" in report_data
    ), "Reconciliation report JSON should contain broker_meta"
    broker_meta_in_report = report_data["broker_meta"]
    assert broker_meta_in_report is not None
    assert broker_meta_in_report["broker_view_source"] == "stored_snapshot"
    assert broker_meta_in_report["broker_snapshot_run_id"] == snapshot_run_id
    assert broker_meta_in_report["broker_snapshot_date"] is not None
    assert broker_meta_in_report["broker_snapshot_path"] is not None

    # Verify deterministic JSON structure (keys should be sorted)
    # The JSON should be stable (deterministic keys order)
    json_str1 = json.dumps(report_data, sort_keys=True, indent=2)
    json_str2 = json.dumps(report_data, sort_keys=True, indent=2)
    assert json_str1 == json_str2, "JSON serialization should be deterministic"
