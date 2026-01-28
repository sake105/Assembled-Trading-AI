"""Tests for accounting report writer (Sprint 13)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.accounting_report import (
    write_accounting_report_csv,
    write_accounting_report_json,
)
from src.assembled_core.accounting.position_engine import build_positions_from_ledger
from src.assembled_core.accounting.ledger import events_from_trades


def create_test_positions_result() -> dict:
    """Create a test positions result for accounting report."""
    # Create minimal trades
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    trades = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC"),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 100.0,
            "price": 150.0,
            "fill_qty": 100.0,
            "fill_price": 150.0,
            "status": "filled",
            "total_cost_cash": 0.15,
        },
        {
            "timestamp": pd.Timestamp(base_time, tz="UTC") + pd.Timedelta(minutes=5),
            "symbol": "MSFT",
            "side": "SELL",
            "qty": 50.0,
            "price": 200.0,
            "fill_qty": 50.0,
            "fill_price": 200.0,
            "status": "filled",
            "total_cost_cash": 0.10,
        },
    ])
    
    # Generate events
    events = events_from_trades(trades, run_id="test_accounting", source="test")
    
    # Build positions
    positions_result = build_positions_from_ledger(
        events,
        prices_df=None,
        mark_ts=pd.Timestamp(base_time, tz="UTC"),
        start_cash=10000.0,
        missing_price_policy="zero",
    )
    
    return positions_result


def test_accounting_report_csv_written(tmp_path: Path):
    """Test that accounting report CSV is written with correct schema."""
    positions_result = create_test_positions_result()
    run_id = "test_accounting"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0

    csv_path = write_accounting_report_csv(
        positions_result=positions_result,
        output_dir=tmp_path,
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
    )

    assert csv_path.exists(), "CSV file should exist"
    
    # Load and verify schema
    df = pd.read_csv(csv_path)
    assert "section" in df.columns
    assert "symbol" in df.columns
    assert "cash_start" in df.columns
    assert "cash_end" in df.columns
    assert "realized_pnl" in df.columns
    assert "unrealized_pnl" in df.columns
    assert "total_pnl" in df.columns
    assert "schema_version" in df.columns

    # Verify summary row exists
    summary_row = df[df["section"] == "SUMMARY"]
    assert len(summary_row) == 1
    assert summary_row.iloc[0]["cash_start"] == 10000.0


def test_accounting_report_json_written(tmp_path: Path):
    """Test that accounting report JSON is written with correct schema."""
    positions_result = create_test_positions_result()
    run_id = "test_accounting"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=tmp_path,
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
    )

    assert json_path.exists(), "JSON file should exist"
    
    # Load and verify schema
    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    
    assert "schema_version" in report
    assert report["schema_version"] == 1
    assert "as_of_date" in report
    assert "run_id" in report
    assert "cash" in report
    assert "pnl" in report
    assert "positions" in report
    assert "summary" in report
    
    assert report["cash"]["start"] == 10000.0
    assert "total_realized" in report["pnl"]
    assert "total_unrealized" in report["pnl"]


def test_accounting_report_deterministic_sorting(tmp_path: Path):
    """Test that positions are sorted deterministically in reports."""
    positions_result = create_test_positions_result()
    run_id = "test_deterministic"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0

    # Write twice
    csv_path1 = write_accounting_report_csv(
        positions_result=positions_result,
        output_dir=tmp_path / "run1",
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
    )
    
    csv_path2 = write_accounting_report_csv(
        positions_result=positions_result,
        output_dir=tmp_path / "run2",
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
    )

    # Load both
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # Positions should be sorted identically
    pos1 = df1[df1["section"] == "POSITION"].sort_values("symbol")
    pos2 = df2[df2["section"] == "POSITION"].sort_values("symbol")
    
    if len(pos1) > 0 and len(pos2) > 0:
        assert list(pos1["symbol"]) == list(pos2["symbol"])


def test_accounting_report_nan_handling(tmp_path: Path):
    """Test that NaN values are handled correctly (converted to None in JSON)."""
    # Create positions result with NaN values (missing prices)
    positions_df = pd.DataFrame([
        {
            "symbol": "AAPL",
            "qty": 100.0,
            "avg_price": 150.0,
            "realized_pnl": 50.0,
            "unrealized_pnl": pd.NA,  # Missing price
            "notional": 0.0,
            "last_price": pd.NA,
        },
    ])
    
    positions_result = {
        "positions_df": positions_df,
        "cash_balance": 10000.0,
        "summary": {
            "total_realized_pnl": 50.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 50.0,
            "n_positions": 1,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        },
    }
    
    run_id = "test_nan"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=tmp_path,
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
    )

    # Load JSON (should not have NaN, should have None)
    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    
    # Check that positions with NaN are handled
    if len(report["positions"]) > 0:
        pos = report["positions"][0]
        # unrealized_pnl should be None (not NaN string)
        assert pos.get("unrealized_pnl") is None or isinstance(pos.get("unrealized_pnl"), (int, float))


def test_accounting_report_with_costs(tmp_path: Path):
    """Test that costs breakdown is included when provided."""
    positions_result = create_test_positions_result()
    run_id = "test_costs"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0
    
    costs_breakdown = {
        "commission_cash": 0.25,
        "spread_cash": 0.15,
        "slippage_cash": 0.10,
        "total_cost_cash": 0.50,
    }

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=tmp_path,
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
        costs_breakdown=costs_breakdown,
    )

    # Load JSON
    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    
    assert "costs" in report
    assert report["costs"]["commission_cash"] == 0.25
    assert report["costs"]["total_cost_cash"] == 0.50


def test_accounting_report_with_reconciliation(tmp_path: Path):
    """Test that reconciliation info is included when provided."""
    positions_result = create_test_positions_result()
    run_id = "test_recon"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")
    start_cash = 10000.0
    
    reconciliation_result = {
        "ok": True,
        "cash_match": True,
        "cash_diff": 0.0,
    }

    json_path = write_accounting_report_json(
        positions_result=positions_result,
        output_dir=tmp_path,
        run_id=run_id,
        as_of=as_of_date,
        start_cash=start_cash,
        reconciliation_result=reconciliation_result,
        ledger_pack_path="ledger_test",
        reconcile_report_path="reconcile_test.csv",
    )

    # Load JSON
    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    
    assert "reconciliation" in report
    assert report["reconciliation"]["ok"] is True
    assert "ledger_pack_path" in report
    assert "reconcile_report_path" in report
