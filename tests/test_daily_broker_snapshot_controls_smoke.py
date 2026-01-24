"""Smoke tests for daily broker snapshot controls (Sprint 13).

Tests build_ledger_from_trades() with imported snapshot and policy=require.
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


def test_build_ledger_with_imported_snapshot_policy_require(tmp_path: Path):
    """Test that build_ledger_from_trades() works with imported snapshot and policy=require."""
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

    # Import snapshot first
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_run_id = "test_snapshot_run"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")
    
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )
    
    assert import_result["broker_snapshot_path"] is not None

    # Create minimal trades/orders
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
    as_of_date = pd.Timestamp(base_time, tz="UTC")

    # Call build_ledger_from_trades with policy=require
    # Should NOT raise ValueError since snapshot was imported
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

    # Verify broker_meta indicates stored_snapshot was used
    assert result["broker_meta"] is not None
    assert result["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert result["broker_meta"]["broker_snapshot_run_id"] == snapshot_run_id
    assert result["broker_meta"]["broker_snapshot_path"] is not None
    
    # Verify broker_snapshot_path is set
    assert result["broker_snapshot_path"] is not None
    
    # Verify reconciliation was performed
    assert result["reconciliation_result"] is not None
    assert result["reconciliation_ok"] is not None


def test_build_ledger_with_imported_snapshot_policy_prefer(tmp_path: Path):
    """Test that build_ledger_from_trades() uses imported snapshot with policy=prefer."""
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

    # Import snapshot first
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_run_id = "test_snapshot_run"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")
    
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=snapshot_run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )
    
    assert import_result["broker_snapshot_path"] is not None

    # Create minimal trades/orders
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
    as_of_date = pd.Timestamp(base_time, tz="UTC")

    # Call build_ledger_from_trades with policy=prefer
    result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=run_id,
        output_dir=output_dir,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="prefer",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=snapshot_run_id,
    )

    # Verify broker_meta indicates stored_snapshot was used (prefer should use it if available)
    assert result["broker_meta"] is not None
    assert result["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert result["broker_meta"]["broker_snapshot_path"] is not None
