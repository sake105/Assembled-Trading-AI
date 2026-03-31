"""Tests for CSV reconciliation report broker_meta columns (Sprint 13)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.accounting.reconciliation_report import (
    write_reconcile_report_csv,
)


def test_csv_includes_broker_meta_columns_stored_snapshot(tmp_path: Path):
    """Test that CSV includes broker_meta columns when stored snapshot is used."""
    run_id = "test_run_csv_meta_001"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )

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
        "broker_snapshot_run_id": "snapshot_run_csv_001",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": "broker_snapshot_snapshot_run_csv_001/snapshot_2024-01-15.json",
    }

    # Write CSV
    csv_path = write_reconcile_report_csv(
        result,
        tmp_path,
        run_id,
        as_of,
        ledger_cash=10000.0,
        broker_cash=10000.0,
        broker_meta=broker_meta,
    )

    # Read back
    df = pd.read_csv(csv_path)

    # Verify broker_meta columns are present (including schema_version)
    expected_cols = [
        "type",
        "symbol",
        "ledger_value",
        "broker_value",
        "diff",
        "match",
        "broker_view_source",
        "broker_snapshot_run_id",
        "broker_snapshot_date",
        "broker_snapshot_path",
        "schema_version",
    ]
    assert list(df.columns) == expected_cols

    # Verify broker_meta values are constant across all rows
    assert all(df["broker_view_source"] == "stored_snapshot")
    assert all(df["broker_snapshot_run_id"] == "snapshot_run_csv_001")
    assert all(df["broker_snapshot_date"] == "2024-01-15T00:00:00+00:00")
    assert all(
        df["broker_snapshot_path"]
        == "broker_snapshot_snapshot_run_csv_001/snapshot_2024-01-15.json"
    )

    # Verify deterministic sorting is preserved (cash row first)
    assert df.iloc[0]["type"] == "cash"


def test_csv_includes_broker_meta_columns_paper_view(tmp_path: Path):
    """Test that CSV includes broker_meta columns when paper view is used."""
    run_id = "test_run_csv_meta_002"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )

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
        "broker_snapshot_run_id": "test_run_csv_meta_002",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": None,
    }

    # Write CSV
    csv_path = write_reconcile_report_csv(
        result,
        tmp_path,
        run_id,
        as_of,
        ledger_cash=10000.0,
        broker_cash=10000.0,
        broker_meta=broker_meta,
    )

    # Read back
    df = pd.read_csv(csv_path)

    # Verify broker_meta columns are present
    assert "broker_view_source" in df.columns
    assert "broker_snapshot_run_id" in df.columns
    assert "broker_snapshot_date" in df.columns
    assert "broker_snapshot_path" in df.columns

    # Verify broker_meta values (None should be represented as empty string in CSV)
    assert all(df["broker_view_source"] == "paper_view")
    assert all(df["broker_snapshot_run_id"] == "test_run_csv_meta_002")
    # snapshot_path is None -> empty string in CSV (pandas may read empty strings as NaN, so check both)
    assert all(df["broker_snapshot_path"].fillna("") == "")


def test_csv_includes_broker_meta_columns_when_none(tmp_path: Path):
    """Test that CSV includes broker_meta columns even when broker_meta=None (fixed schema)."""
    run_id = "test_run_csv_meta_none"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],
        }
    )

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Write CSV without broker_meta (None)
    csv_path = write_reconcile_report_csv(
        result,
        tmp_path,
        run_id,
        as_of,
        ledger_cash=10000.0,
        broker_cash=10000.0,
        broker_meta=None,
    )

    # Read back
    df = pd.read_csv(csv_path)

    # Verify broker_meta columns are present (fixed schema, including schema_version)
    expected_cols = [
        "type",
        "symbol",
        "ledger_value",
        "broker_value",
        "diff",
        "match",
        "broker_view_source",
        "broker_snapshot_run_id",
        "broker_snapshot_date",
        "broker_snapshot_path",
        "schema_version",
    ]
    assert list(df.columns) == expected_cols

    # Verify broker_meta columns are empty (pandas may read empty strings as NaN, so check both)
    assert all(df["broker_view_source"].fillna("") == "")
    assert all(df["broker_snapshot_run_id"].fillna("") == "")
    assert all(df["broker_snapshot_date"].fillna("") == "")
    assert all(df["broker_snapshot_path"].fillna("") == "")


def test_csv_deterministic_sorting_preserved_with_broker_meta(tmp_path: Path):
    """Test that deterministic sorting is preserved when broker_meta columns are added."""
    run_id = "test_run_csv_meta_003"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result with multiple position diffs
    ledger_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 50.0, 200.0],
        }
    )
    broker_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [105.0, 50.0, 190.0],  # AAPL: +5, GOOGL: -10 (larger diff)
        }
    )

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
        "broker_snapshot_run_id": "snapshot_run_csv_003",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": "broker_snapshot_snapshot_run_csv_003/snapshot_2024-01-15.json",
    }

    # Write CSV
    csv_path = write_reconcile_report_csv(
        result,
        tmp_path,
        run_id,
        as_of,
        ledger_cash=10000.0,
        broker_cash=10000.0,
        broker_meta=broker_meta,
    )

    # Read back
    df = pd.read_csv(csv_path)

    # Verify: cash row is first
    assert df.iloc[0]["type"] == "cash"

    # Verify: position rows are sorted by abs(diff) desc, then symbol asc
    position_rows = df[df["type"] == "position"].copy()
    if len(position_rows) > 1:
        # Check sorting: abs(diff) desc
        diffs = position_rows["diff"].abs()
        assert diffs.is_monotonic_decreasing or all(
            diffs.iloc[i] >= diffs.iloc[i + 1] for i in range(len(diffs) - 1)
        )

    # Verify broker_meta columns don't affect sorting
    assert "broker_view_source" in df.columns
    assert all(df["broker_view_source"] == "stored_snapshot")
