"""Hardening tests for broker snapshot importer.

Focus on:
- Robust CSV/JSON parsing for messy real-world inputs
- Quantity parsing with thousands separators and whitespace
- Cash parsing from strings
- Duplicate symbol aggregation
- Deterministic output bytes for same input
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot import normalize_broker_snapshot
from src.assembled_core.accounting.broker_snapshot_importer import (
    import_broker_snapshot,
    load_external_broker_snapshot,
)
from src.assembled_core.accounting.broker_snapshot_store import load_broker_snapshot_json


def test_csv_qty_parsing_thousands_and_whitespace(tmp_path: Path) -> None:
    """CSV qty parsing: '1,000' and ' 2.5 ' are parsed correctly."""
    csv_path = tmp_path / "snapshot.csv"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": ["1,000", " 2.5 "],
        }
    )
    df.to_csv(csv_path, index=False)

    cash, positions_df = load_external_broker_snapshot(csv_path)

    assert cash is None
    assert len(positions_df) == 2
    assert positions_df.loc[0, "qty"] == 1000.0
    assert positions_df.loc[1, "qty"] == 2.5


def test_csv_qty_parsing_zero_and_empty_error(tmp_path: Path) -> None:
    """CSV qty parsing: '0' is allowed, empty string should raise."""
    csv_path = tmp_path / "snapshot.csv"
    # Second row has empty qty -> should raise ValueError
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": ["0", ""],
        }
    )
    df.to_csv(csv_path, index=False)

    try:
        _ = load_external_broker_snapshot(csv_path)
    except ValueError as exc:
        msg = str(exc)
        assert "Missing qty value in CSV file" in msg
        assert "snapshot.csv" in msg
    else:
        raise AssertionError("Expected ValueError for empty qty in CSV")


def test_json_cash_string_parsing(tmp_path: Path) -> None:
    """JSON cash can be string like '1000.00' and is parsed as float."""
    json_path = tmp_path / "snapshot.json"
    data = {
        "cash": "1000.00",
        "positions": [
            {"symbol": "AAPL", "qty": 10},
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")

    cash, positions_df = load_external_broker_snapshot(json_path)

    assert cash == 1000.0
    assert len(positions_df) == 1


def test_json_positions_ignore_unknown_keys_and_normalize_whitespace(tmp_path: Path) -> None:
    """JSON positions ignore unknown keys and normalize symbol whitespace."""
    json_path = tmp_path / "snapshot.json"
    data = {
        "cash": 5000.0,
        "positions": [
            {"symbol": "  AAPL   US  ", "qty": "1,000", "foo": "bar"},
            {"symbol": "MSFT", "qty": 5, "extra": 123},
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")

    cash, positions_df = load_external_broker_snapshot(json_path)

    assert cash == 5000.0
    assert len(positions_df) == 2
    symbols = sorted(positions_df["symbol"].tolist())
    assert "AAPL US" in symbols
    assert "MSFT" in symbols


def test_normalize_broker_snapshot_aggregates_duplicate_symbols(tmp_path: Path) -> None:
    """Duplicate symbols are aggregated by summing qty."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "qty": [10.0, 5.0, -2.0],
        }
    )
    cash = 1000.0

    result = normalize_broker_snapshot(cash=cash, positions_df=positions, qty_tol=1e-8)
    normalized = result["positions_df"]

    assert len(normalized) == 2
    # After aggregation, AAPL should have qty 15, MSFT -2
    aapl_row = normalized[normalized["symbol"] == "AAPL"].iloc[0]
    msft_row = normalized[normalized["symbol"] == "MSFT"].iloc[0]
    assert aapl_row["qty"] == 15.0
    assert msft_row["qty"] == -2.0


def test_normalize_broker_snapshot_duplicate_symbols_with_tiny_residual(tmp_path: Path) -> None:
    """Duplicate symbols with tiny residual should be filtered by qty_tol after aggregation."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "qty": [1e-9, -1e-9],
        }
    )
    cash = 1000.0

    result = normalize_broker_snapshot(cash=cash, positions_df=positions, qty_tol=1e-8)
    normalized = result["positions_df"]

    # Sum is 0, should be removed by qty_tol threshold
    assert normalized.empty


def test_import_csv_with_messy_inputs_stable_bytes(tmp_path: Path) -> None:
    """Importing same messy CSV twice produces identical JSON bytes."""
    csv_path = tmp_path / "snapshot.csv"
    df = pd.DataFrame(
        {
            "symbol": ["  AAPL  ", "MSFT", "AAPL"],
            "qty": ["1,000", " 2.5 ", "0"],
            "cash": ["1000.00", None, None],
        }
    )
    df.to_csv(csv_path, index=False)

    output_dir1 = tmp_path / "out1"
    output_dir2 = tmp_path / "out2"
    run_id = "hardening"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    result1 = import_broker_snapshot(
        snapshot_path=csv_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir1,
        qty_tol=1e-8,
        store_parquet=False,
    )
    result2 = import_broker_snapshot(
        snapshot_path=csv_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir2,
        qty_tol=1e-8,
        store_parquet=False,
    )

    path1 = output_dir1 / result1["broker_snapshot_path"]
    path2 = output_dir2 / result2["broker_snapshot_path"]

    with path1.open("rb") as f1, path2.open("rb") as f2:
        assert f1.read() == f2.read(), "Stored snapshots should be byte-identical for same input"


def test_csv_cash_column_string_parsing_first_non_null(tmp_path: Path) -> None:
    """CSV cash column: first non-null string value is parsed."""
    csv_path = tmp_path / "snapshot.csv"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": ["1.0", "2.0"],
            "cash": ["", "1,000.50"],
        }
    )
    df.to_csv(csv_path, index=False)

    cash, positions_df = load_external_broker_snapshot(csv_path)

    assert cash == 1000.50
    assert len(positions_df) == 2


def test_json_invalid_qty_reports_index_and_path(tmp_path: Path) -> None:
    """Invalid JSON qty should raise ValueError that includes path and index context."""
    json_path = tmp_path / "snapshot.json"
    data = {
        "cash": 1000.0,
        "positions": [
            {"symbol": "AAPL", "qty": "not-a-number"},
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")

    try:
        _ = load_external_broker_snapshot(json_path)
    except ValueError as exc:
        msg = str(exc)
        # We at least expect file path and that qty is invalid
        assert "snapshot.json" in msg
        assert "qty" in msg or "Invalid qty" in msg
    else:
        raise AssertionError("Expected ValueError for invalid JSON qty")


def test_import_json_with_messy_whitespace_and_duplicates(tmp_path: Path) -> None:
    """End-to-end: messy JSON with whitespace and duplicates is normalized deterministically."""
    json_path = tmp_path / "snapshot.json"
    data = {
        "cash": "10,000",
        "positions": [
            {"symbol": "  AAPL  ", "qty": "1,000"},
            {"symbol": "AAPL", "qty": " 500 "},
            {"symbol": "MSFT", "qty": "0"},
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")

    output_dir = tmp_path / "output"
    run_id = "json_hardening"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    result = import_broker_snapshot(
        snapshot_path=json_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=False,
    )

    stored = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    assert stored is not None
    assert stored["cash"] == 10000.0
    # Duplicates for AAPL should be aggregated
    assert len(stored["positions"]) == 1
    assert stored["positions"][0]["symbol"] == "AAPL"
    assert stored["positions"][0]["qty"] == 1500.0

