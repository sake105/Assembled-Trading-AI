"""Tests for reconciliation report broker_meta inclusion (Sprint 13).

Tests that broker_meta is correctly included in reconciliation reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.accounting.reconciliation_report import write_reconcile_report_json


def test_reconcile_report_json_includes_broker_meta_stored_snapshot(tmp_path: Path):
    """Test that JSON report includes broker_meta when stored snapshot is used."""
    run_id = "test_run_001"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Create broker_meta for stored snapshot
    broker_meta = {
        "broker_view_source": "stored_snapshot",
        "broker_snapshot_run_id": "snapshot_run_001",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": "broker_snapshot_snapshot_run_001/snapshot_2024-01-15.json",
    }

    # Write JSON report
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Verify broker_meta is present
    assert "broker_meta" in json_data
    assert json_data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert json_data["broker_meta"]["broker_snapshot_run_id"] == "snapshot_run_001"
    assert json_data["broker_meta"]["broker_snapshot_date"] == "2024-01-15T00:00:00+00:00"
    assert json_data["broker_meta"]["broker_snapshot_path"] == "broker_snapshot_snapshot_run_001/snapshot_2024-01-15.json"


def test_reconcile_report_json_includes_broker_meta_paper_view(tmp_path: Path):
    """Test that JSON report includes broker_meta when paper view is used."""
    run_id = "test_run_002"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Create broker_meta for paper view
    broker_meta = {
        "broker_view_source": "paper_view",
        "broker_snapshot_run_id": "test_run_002",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": None,
    }

    # Write JSON report
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Verify broker_meta is present
    assert "broker_meta" in json_data
    assert json_data["broker_meta"]["broker_view_source"] == "paper_view"
    assert json_data["broker_meta"]["broker_snapshot_path"] is None


def test_reconcile_report_json_stable_serialization_with_broker_meta(tmp_path: Path):
    """Test that JSON report with broker_meta has stable serialization (deterministic)."""
    run_id = "test_run_003"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [100.0],
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Create broker_meta
    broker_meta = {
        "broker_view_source": "stored_snapshot",
        "broker_snapshot_run_id": "snapshot_run_003",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": "broker_snapshot_snapshot_run_003/snapshot_2024-01-15.json",
    }

    # Write JSON report twice
    json_path1 = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Write again (should be identical)
    import shutil
    json_path2 = tmp_path / f"reconcile_report_{run_id}" / "reconcile_2024-01-15_copy.json"
    shutil.copy(json_path1, json_path2)

    # Read both
    with json_path1.open("r", encoding="utf-8") as f:
        json_data1 = json.load(f)

    with json_path2.open("r", encoding="utf-8") as f:
        json_data2 = json.load(f)

    # Verify broker_meta is present in both
    assert "broker_meta" in json_data1
    assert "broker_meta" in json_data2

    # Verify broker_meta keys are sorted (deterministic)
    broker_meta1 = json_data1["broker_meta"]
    broker_meta2 = json_data2["broker_meta"]
    
    assert list(broker_meta1.keys()) == sorted(broker_meta1.keys())
    assert list(broker_meta2.keys()) == sorted(broker_meta2.keys())
    
    # Verify values are identical
    assert broker_meta1 == broker_meta2
