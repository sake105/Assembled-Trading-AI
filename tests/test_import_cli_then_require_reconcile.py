"""Integration test: CLI import → require reconciliation (Sprint 13).

This test verifies the complete Ops toolchain:
1. CLI import of external broker snapshot
2. Require policy reconciliation using the imported snapshot
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_cli_import_then_require_reconcile(tmp_path: Path):
    """Test that CLI import followed by require reconciliation works end-to-end."""
    # Step 1: Create minimal JSON snapshot file in tmp_path
    external_snapshot_path = tmp_path / "external_broker_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 10.0},
            {"symbol": "MSFT", "qty": 5.0},
        ],
    }
    with external_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Step 2: Run python scripts/import_broker_snapshot.py via subprocess
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    run_id = "r1"
    snapshot_date = "2025-01-15"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_snapshot_path),
            "--run-id",
            run_id,
            "--as-of-date",
            snapshot_date,
            "--output-dir",
            str(output_dir),
            "--store-parquet",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Assert exit code 0
    assert result.returncode == 0, f"CLI import failed: {result.stderr}"

    # Verify snapshot was created
    snapshot_dir = output_dir / f"broker_snapshot_{run_id}"
    assert snapshot_dir.exists(), "Snapshot directory should exist"
    snapshot_json = snapshot_dir / f"snapshot_{snapshot_date}.json"
    assert snapshot_json.exists(), "Snapshot JSON should exist"

    # Step 3: Call build_ledger_from_trades(... broker_snapshot_policy="require", ...)
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

    as_of_date = pd.Timestamp(snapshot_date, tz="UTC")

    # Call build_ledger_from_trades with policy=require
    ledger_result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id="ledger_run_001",
        output_dir=output_dir,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=run_id,
    )

    # Assert: broker_meta.broker_view_source == "stored_snapshot"
    assert "broker_meta" in ledger_result
    broker_meta = ledger_result["broker_meta"]
    assert broker_meta is not None
    assert broker_meta["broker_view_source"] == "stored_snapshot", (
        f"Expected stored_snapshot, got {broker_meta['broker_view_source']}"
    )
    assert broker_meta["broker_snapshot_run_id"] == run_id
    assert broker_meta["broker_snapshot_path"] is not None

    # Assert: reconcile report JSON exists
    date_str = snapshot_date
    report_dir = output_dir / "reconcile_report_ledger_run_001"
    report_json_path = report_dir / f"reconcile_{date_str}.json"
    assert report_json_path.exists(), (
        f"Reconciliation report JSON should exist at {report_json_path}"
    )

    # Verify report JSON contains broker_meta
    with report_json_path.open("r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert "broker_meta" in report_data, (
        "Reconciliation report JSON should contain broker_meta"
    )
    assert report_data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert report_data["broker_meta"]["broker_snapshot_run_id"] == run_id

    # Verify reconciliation was performed
    assert ledger_result["reconciliation_result"] is not None
    assert ledger_result["reconciliation_ok"] is not None
