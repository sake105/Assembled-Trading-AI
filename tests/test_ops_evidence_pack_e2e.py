"""Ops evidence pack E2E test.

This test verifies a minimal end-to-end chain:
- CLI import of external broker snapshot
- Ledger build with broker_snapshot_policy=require
- Reconciliation report with broker_meta
- Accounting report with broker_meta + links
- (Optional) Manifest path fields are POSIX + relative if a manifest exists
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


def test_ops_evidence_pack_e2e(tmp_path: Path) -> None:
    """End-to-end: CLI import -> require reconcile -> accounting report -> evidence pack."""
    # ------------------------------------------------------------------
    # Step 1: CLI import of external broker snapshot
    # ------------------------------------------------------------------
    external_snapshot_path = tmp_path / "external_broker_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 10.0},
            {"symbol": "MSFT", "qty": 5.0},
        ],
    }
    external_snapshot_path.write_text(json.dumps(snapshot_data), encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = ROOT / "scripts" / "import_broker_snapshot.py"
    snapshot_run_id = "ops_ns"
    snapshot_date_str = "2025-01-15"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(external_snapshot_path),
            "--run-id",
            snapshot_run_id,
            "--as-of-date",
            snapshot_date_str,
            "--output-dir",
            str(output_dir),
            "--store-parquet",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, f"CLI import failed: {result.stdout}\n{result.stderr}"

    # Snapshot files should exist
    snapshot_dir = output_dir / f"broker_snapshot_{snapshot_run_id}"
    assert snapshot_dir.exists(), "Snapshot directory should exist"
    snapshot_json = snapshot_dir / f"snapshot_{snapshot_date_str}.json"
    assert snapshot_json.exists(), "Snapshot JSON should exist"

    # ------------------------------------------------------------------
    # Step 2: Build ledger with broker_snapshot_policy=require
    # ------------------------------------------------------------------
    ledger_run_id = "ledger_ops_e2e"
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
    as_of_date = pd.Timestamp(snapshot_date_str, tz="UTC")

    ledger_result = build_ledger_from_trades(
        orders_df=orders,
        trades_df=trades,
        run_id=ledger_run_id,
        output_dir=output_dir,
        as_of_date=as_of_date,
        prices_df=None,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        write_paper_broker_snapshot=False,
        broker_snapshot_run_id=snapshot_run_id,
    )

    # Basic reconciliation checks
    assert ledger_result["reconciliation_result"] is not None
    assert "broker_meta" in ledger_result
    broker_meta = ledger_result["broker_meta"]
    assert broker_meta["broker_view_source"] == "stored_snapshot"
    assert broker_meta["broker_snapshot_run_id"] == snapshot_run_id

    # ------------------------------------------------------------------
    # Step 3: Evidence: reconcile JSON, accounting JSON, manifest (if any)
    # ------------------------------------------------------------------
    # Reconcile JSON
    reconcile_dir = output_dir / f"reconcile_report_{ledger_run_id}"
    reconcile_json_path = reconcile_dir / f"reconcile_{snapshot_date_str}.json"
    assert reconcile_json_path.exists(), f"Reconciliation JSON should exist at {reconcile_json_path}"

    with reconcile_json_path.open("r", encoding="utf-8") as f:
        reconcile_data = json.load(f)

    assert "broker_meta" in reconcile_data
    assert reconcile_data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert reconcile_data["broker_meta"]["broker_snapshot_run_id"] == snapshot_run_id

    # Accounting JSON
    accounting_dir = output_dir / f"accounting_report_{ledger_run_id}"
    accounting_json_path = accounting_dir / f"accounting_{snapshot_date_str}.json"
    assert accounting_json_path.exists(), f"Accounting JSON should exist at {accounting_json_path}"

    with accounting_json_path.open("r", encoding="utf-8") as f:
        accounting_data = json.load(f)

    # Broker meta carried through
    assert "broker_meta" in accounting_data
    assert accounting_data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert accounting_data["broker_meta"]["broker_snapshot_run_id"] == snapshot_run_id

    # Links are present and consistent
    assert "reconcile_report_path" in accounting_data
    # Path in accounting JSON should point to reconcile JSON relative path
    reconcile_rel = accounting_data["reconcile_report_path"]
    # It should be POSIX-ish (no Windows backslashes)
    assert "\\" not in reconcile_rel

    # Reconciliation block should include consistency flag
    if "reconciliation" in accounting_data:
        recon_block = accounting_data["reconciliation"]
        assert "cash_end_matches_reconcile_cash" in recon_block
        assert isinstance(recon_block["cash_end_matches_reconcile_cash"], bool)

    # Cash section must be cross-checkable
    assert "cash" in accounting_data
    assert accounting_data["cash"]["end"] == ledger_result["cash_balance"]

    # ------------------------------------------------------------------
    # Optional: if a manifest exists (e.g. in an EOD/Backtest run), ensure
    # its path fields are relative and use POSIX slashes.
    # ------------------------------------------------------------------
    for manifest_name in ("run_manifest_1d.json", "run_manifest_5min.json"):
        manifest_path = output_dir / manifest_name
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        for key in (
            "ledger_pack_path",
            "reconcile_report_path",
            "accounting_report_path",
            "broker_snapshot_path",
        ):
            value = manifest.get(key)
            if value is None:
                continue
            assert "\\" not in value, f"{key} should use POSIX slashes"
            # Relative paths only (no drive letters / absolute prefixes)
            assert not Path(value).is_absolute(), f"{key} should be relative: {value}"

