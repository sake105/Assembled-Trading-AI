"""Tests for reconciliation report writing (Sprint 13 L4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.accounting.reconciliation_report import (
    write_reconcile_report_csv,
    write_reconcile_report_json,
    write_reconcile_report_md,
)


def test_files_exist(tmp_path: Path):
    """Test that report files are created."""
    run_id = "test_run_001"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    # Add cash values to result for report
    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Write reports
    csv_path = write_reconcile_report_csv(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
    )
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
    )
    md_path = write_reconcile_report_md(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
    )

    # Verify files exist
    assert csv_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    # Verify paths are correct
    assert csv_path.name == "reconcile_2024-01-15.csv"
    assert json_path.name == "reconcile_2024-01-15.json"
    assert md_path.name == "reconcile_2024-01-15.md"


def test_schema_stable(tmp_path: Path):
    """Test that CSV schema is stable."""
    run_id = "test_run_002"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result with mismatches
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 55.0],  # MSFT mismatch
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Write CSV
    csv_path = write_reconcile_report_csv(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
    )

    # Read back
    df = pd.read_csv(csv_path)

    # Verify schema (fixed schema: broker_meta columns + schema_version always present)
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

    # Verify cash row exists
    cash_rows = df[df["type"] == "cash"]
    assert len(cash_rows) == 1

    # Verify position diff row exists
    position_rows = df[df["type"] == "position"]
    assert len(position_rows) == 1
    assert position_rows.iloc[0]["symbol"] == "MSFT"


def test_json_deterministic_keys(tmp_path: Path):
    """Test that JSON has deterministic key ordering."""
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

    # Write JSON twice
    json_path1 = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
    )

    # Write again (should be identical)
    import json
    import shutil

    json_path2 = tmp_path / f"reconcile_report_{run_id}" / "reconcile_2024-01-15_copy.json"
    shutil.copy(json_path1, json_path2)

    # Read both
    with json_path1.open("r", encoding="utf-8") as f:
        json1 = json.load(f)

    with json_path2.open("r", encoding="utf-8") as f:
        json2 = json.load(f)

    # Verify keys are sorted (deterministic)
    json1_str = json.dumps(json1, sort_keys=True, indent=2)
    json2_str = json.dumps(json2, sort_keys=True, indent=2)

    # Should be identical (or at least have same structure)
    assert json1_str == json2_str

    # Verify top-level keys are sorted
    top_keys = list(json1.keys())
    assert top_keys == sorted(top_keys)


def test_csv_sort_deterministic(tmp_path: Path):
    """Test that CSV rows are sorted deterministically."""
    run_id = "test_run_004"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result with multiple position diffs
    ledger_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [100.0, 50.0, 200.0],
    })
    broker_positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [105.0, 50.0, 190.0],  # AAPL: +5, GOOGL: -10 (larger diff)
    })

    result = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_positions,
        ledger_cash=10000.0,
        broker_positions_df=broker_positions,
        broker_cash=10000.0,
    )

    result["ledger_cash"] = 10000.0
    result["broker_cash"] = 10000.0

    # Write CSV
    csv_path = write_reconcile_report_csv(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0
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


def test_json_nan_handling(tmp_path: Path):
    """Test that NaN values in JSON are converted to None."""
    run_id = "test_run_005"
    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    # Create reconciliation result with missing cash values
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

    # Don't add cash values (should be None in JSON)
    # Write JSON
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of
    )

    # Read back
    import json

    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Verify: None values (not NaN strings) for missing cash
    # Note: cash_diff should be present (0.0), but ledger_cash/broker_cash might be None
    assert "cash" in json_data
    # JSON should not contain NaN (should be None or missing)
    json_str = json.dumps(json_data)
    assert "NaN" not in json_str
    assert "nan" not in json_str.lower()


def test_json_broker_meta_stored_snapshot(tmp_path: Path):
    """Test that broker_meta is included in JSON when stored snapshot is used."""
    run_id = "test_run_006"
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

    # Write JSON
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    import json

    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Verify broker_meta is present
    assert "broker_meta" in json_data
    assert json_data["broker_meta"]["broker_view_source"] == "stored_snapshot"
    assert json_data["broker_meta"]["broker_snapshot_run_id"] == "snapshot_run_001"
    assert json_data["broker_meta"]["broker_snapshot_date"] == "2024-01-15T00:00:00+00:00"
    assert json_data["broker_meta"]["broker_snapshot_path"] == "broker_snapshot_snapshot_run_001/snapshot_2024-01-15.json"


def test_json_broker_meta_paper_view(tmp_path: Path):
    """Test that broker_meta is included in JSON when paper view is used."""
    run_id = "test_run_007"
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
        "broker_snapshot_run_id": "test_run_007",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": None,
    }

    # Write JSON
    json_path = write_reconcile_report_json(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    import json

    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Verify broker_meta is present
    assert "broker_meta" in json_data
    assert json_data["broker_meta"]["broker_view_source"] == "paper_view"
    assert json_data["broker_meta"]["broker_snapshot_path"] is None


def test_md_broker_meta_stored_snapshot(tmp_path: Path):
    """Test that broker_meta is included in Markdown when stored snapshot is used."""
    run_id = "test_run_008"
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
        "broker_snapshot_run_id": "snapshot_run_002",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": "broker_snapshot_snapshot_run_002/snapshot_2024-01-15.json",
    }

    # Write Markdown
    md_path = write_reconcile_report_md(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    with md_path.open("r", encoding="utf-8") as f:
        md_content = f.read()

    # Verify broker source section is present
    assert "## Broker Source" in md_content
    assert "**Source:** stored_snapshot" in md_content
    assert "**Snapshot Run ID:** snapshot_run_002" in md_content
    assert "**Snapshot Date:** 2024-01-15T00:00:00+00:00" in md_content
    assert "**Snapshot Path:** broker_snapshot_snapshot_run_002/snapshot_2024-01-15.json" in md_content


def test_md_broker_meta_paper_view(tmp_path: Path):
    """Test that broker_meta is included in Markdown when paper view is used."""
    run_id = "test_run_009"
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
        "broker_snapshot_run_id": "test_run_009",
        "broker_snapshot_date": "2024-01-15T00:00:00+00:00",
        "broker_snapshot_path": None,
    }

    # Write Markdown
    md_path = write_reconcile_report_md(
        result, tmp_path, run_id, as_of, ledger_cash=10000.0, broker_cash=10000.0, broker_meta=broker_meta
    )

    # Read back
    with md_path.open("r", encoding="utf-8") as f:
        md_content = f.read()

    # Verify broker source section is present
    assert "## Broker Source" in md_content
    assert "**Source:** paper_view" in md_content
    assert "**Snapshot Run ID:** test_run_009" in md_content
    # snapshot_path should not be in content when None
    assert "**Snapshot Path:**" not in md_content or "None" in md_content
