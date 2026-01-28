"""Smoke tests for schema_version fields on stored artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_store import (
    load_broker_snapshot_json,
    store_broker_snapshot_json,
)
from src.assembled_core.accounting.accounting_report import write_accounting_report_json
from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.accounting.reconciliation_report import write_reconcile_report_json
from src.assembled_core.pipeline.orchestrator import _write_manifest_json


def test_broker_snapshot_json_has_schema_version(tmp_path: Path) -> None:
    """Stored broker snapshot JSON should include schema_version=1."""
    output_dir = tmp_path / "output"
    run_id = "sv1"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0]})

    # Store snapshot
    snapshot_path = store_broker_snapshot_json(
        cash=10000.0,
        positions_df=positions,
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=as_of,
    )
    assert snapshot_path.exists()

    # Load and verify schema_version
    loaded = load_broker_snapshot_json(output_dir, run_id, as_of)
    assert loaded is not None
    assert loaded.get("schema_version") == 1
    assert loaded.get("as_of_date") == "2025-01-15"


def test_reconcile_report_json_has_schema_version(tmp_path: Path) -> None:
    """Reconcile report JSON should include schema_version=1."""
    output_dir = tmp_path / "output"
    run_id = "sv_reconcile"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    ledger_positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0]})
    broker_positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0]})

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )
    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    json_path = write_reconcile_report_json(
        result,
        output_dir,
        run_id,
        as_of,
        ledger_cash=10000.0,
        broker_cash=10000.0,
    )
    assert json_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == 1


def test_accounting_report_json_has_schema_version(tmp_path: Path) -> None:
    """Accounting report JSON should include schema_version=1."""
    output_dir = tmp_path / "output"
    run_id = "sv_accounting"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    positions_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [10.0],
            "avg_price": [100.0],
            "realized_pnl": [5.0],
            "unrealized_pnl": [2.0],
            "notional": [1000.0],
            "last_price": [102.0],
        }
    )
    positions_result = {
        "positions_df": positions_df,
        "cash_balance": 10007.0,
        "summary": {
            "total_realized_pnl": 5.0,
            "total_unrealized_pnl": 2.0,
            "total_pnl": 7.0,
            "n_positions": 1,
            "gross_exposure": 1000.0,
            "net_exposure": 1000.0,
        },
    }

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=output_dir,
        run_id=run_id,
        as_of=as_of,
        start_cash=10000.0,
        reconciliation_result=None,
        ledger_pack_path=None,
        reconcile_report_path=None,
        costs_breakdown=None,
        broker_meta=None,
    )
    assert json_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == 1


def test_manifest_includes_schema_version_and_is_deterministic(tmp_path: Path) -> None:
    """Manifest should include schema_version and write deterministically for same input."""
    base = tmp_path
    manifest_path = base / "run_manifest_1d.json"

    # Deterministic manifest (no runtime timestamps in this unit test)
    manifest = {
        "schema_version": 1,
        "freq": "1d",
        "start_capital": 10000.0,
        "ledger_pack_path": "ledger_run/ledger_events.parquet",
        "reconcile_report_path": "reconcile/reconcile_2025-01-01.json",
        "reconciliation_ok": True,
        "qa_report_path": None,
        "timestamps": {
            "started": "2025-01-01T00:00:00",
            "finished": "2025-01-01T00:00:01",
        },
        "failure": False,
    }

    _write_manifest_json(manifest_path, manifest)
    b1 = manifest_path.read_bytes()

    _write_manifest_json(manifest_path, manifest)
    b2 = manifest_path.read_bytes()

    assert b1 == b2

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded.get("schema_version") == 1

