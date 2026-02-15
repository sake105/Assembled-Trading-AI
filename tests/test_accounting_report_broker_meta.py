"""Tests for accounting report broker_meta and reconciliation consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.accounting_report import (
    write_accounting_report_csv,
    write_accounting_report_json,
)


def _minimal_positions_result() -> dict:
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
    summary = {
        "total_realized_pnl": 5.0,
        "total_unrealized_pnl": 2.0,
        "total_pnl": 7.0,
        "n_positions": 1,
        "gross_exposure": 1000.0,
        "net_exposure": 1000.0,
    }
    return {
        "positions_df": positions_df,
        "cash_balance": 10007.0,
        "summary": summary,
    }


def test_accounting_report_json_includes_broker_meta_and_reconcile_info(tmp_path: Path) -> None:
    """JSON report should include broker_meta, reconcile report path, and consistency flag."""
    output_dir = tmp_path / "output"
    run_id = "json_broker_meta"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    positions_result = _minimal_positions_result()

    reconciliation_result = {
        "ok": True,
        "cash_match": True,
        "cash_diff": 0.0,
    }
    reconcile_report_path = "reconcile_report_run/reconcile_2025-01-15.json"
    broker_meta = {
        "broker_view_source": "stored_snapshot",
        "broker_snapshot_run_id": "snapshot_ns",
        "broker_snapshot_date": "2025-01-15",
        "broker_snapshot_path": "broker_snapshot_snapshot_ns/snapshot_2025-01-15.json",
    }

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=output_dir,
        run_id=run_id,
        as_of=as_of,
        start_cash=10000.0,
        reconciliation_result=reconciliation_result,
        ledger_pack_path="ledger_pack_path",
        reconcile_report_path=reconcile_report_path,
        costs_breakdown=None,
        broker_meta=broker_meta,
    )

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Broker meta should be present and match
    assert "broker_meta" in data
    assert data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert data["broker_meta"]["broker_snapshot_run_id"] == "snapshot_ns"
    assert data["broker_meta"]["broker_snapshot_date"] == "2025-01-15"
    assert data["broker_meta"]["broker_snapshot_path"] == (
        "broker_snapshot_snapshot_ns/snapshot_2025-01-15.json"
    )

    # Reconciliation section should include ok, cash_match, cash_diff and consistency flag
    assert "reconciliation" in data
    reconcile = data["reconciliation"]
    assert reconcile["ok"] is True
    assert reconcile["cash_match"] is True
    assert reconcile["cash_diff"] == 0.0
    assert reconcile["cash_end_matches_reconcile_cash"] is True

    # Reconcile report path should be present at top level
    assert data["reconcile_report_path"] == reconcile_report_path

    # Cash in accounting section should be cross-checkable
    assert data["cash"]["end"] == positions_result["cash_balance"]


def test_accounting_report_csv_fixed_schema_and_broker_columns(tmp_path: Path) -> None:
    """CSV schema should be fixed and include broker source columns and consistency flag."""
    output_dir = tmp_path / "output"
    run_id = "csv_broker_meta"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    positions_result = _minimal_positions_result()

    reconciliation_result = {
        "ok": True,
        "cash_match": True,
        "cash_diff": 0.0,
    }
    reconcile_report_path = "reconcile_report_run/reconcile_2025-01-15.csv"
    broker_meta = {
        "broker_view_source": "stored_snapshot",
        "broker_snapshot_run_id": "snapshot_ns",
        "broker_snapshot_date": "2025-01-15",
        "broker_snapshot_path": "broker_snapshot_snapshot_ns/snapshot_2025-01-15.json",
    }

    csv_path = write_accounting_report_csv(
        positions_result=positions_result,
        output_dir=output_dir,
        run_id=run_id,
        as_of=as_of,
        start_cash=10000.0,
        reconciliation_result=reconciliation_result,
        ledger_pack_path="ledger_pack_path",
        reconcile_report_path=reconcile_report_path,
        costs_breakdown=None,
        broker_meta=broker_meta,
    )

    # Load CSV and inspect schema
    df = pd.read_csv(csv_path)

    expected_columns = [
        "section",
        "symbol",
        "cash_start",
        "cash_end",
        "cash_change",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "commission_cash",
        "spread_cash",
        "slippage_cash",
        "total_cost_cash",
        "reconciliation_ok",
        "cash_end_matches_reconcile_cash",
        "reconcile_report_path",
        "broker_view_source",
        "broker_snapshot_run_id",
        "broker_snapshot_date",
        "broker_snapshot_path",
        "schema_version",
    ]

    assert list(df.columns) == expected_columns

    # SUMMARY row should carry broker meta and consistency info
    summary_row = df[df["section"] == "SUMMARY"].iloc[0]
    assert summary_row["broker_view_source"] == "stored_snapshot"
    assert summary_row["broker_snapshot_run_id"] == "snapshot_ns"
    assert summary_row["broker_snapshot_date"] == "2025-01-15"
    assert summary_row["broker_snapshot_path"] == (
        "broker_snapshot_snapshot_ns/snapshot_2025-01-15.json"
    )
    assert summary_row["reconcile_report_path"] == reconcile_report_path
    # pandas will read bools as True/False
    assert summary_row["cash_end_matches_reconcile_cash"] is True

    # POSITION rows should have empty reconcile-specific fields but same columns
    position_rows = df[df["section"] == "POSITION"]
    assert not position_rows.empty
    # broker columns present; values can be empty strings for positions
    assert "broker_view_source" in position_rows.columns
    # We only require that schema is stable; values are allowed to be empty here


def test_accounting_report_consistency_flag_none_when_no_reconciliation(tmp_path: Path) -> None:
    """Consistency flag should be None/empty when no reconciliation_result is provided."""
    output_dir = tmp_path / "output"
    run_id = "no_reconcile"
    as_of = pd.Timestamp("2025-01-15", tz="UTC")

    positions_result = _minimal_positions_result()

    # CSV without reconciliation_result
    csv_path = write_accounting_report_csv(
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
    df = pd.read_csv(csv_path)
    summary_row = df[df["section"] == "SUMMARY"].iloc[0]
    # Without reconciliation, this column should be empty (read as NaN) or blank
    assert pd.isna(summary_row["cash_end_matches_reconcile_cash"]) or summary_row[
        "cash_end_matches_reconcile_cash"
    ] == ""

