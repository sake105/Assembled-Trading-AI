"""Tests for broker snapshot importer (Sprint 13)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_importer import (
    import_broker_snapshot,
    load_external_broker_snapshot,
)


def test_load_external_broker_snapshot_json_basic(tmp_path: Path):
    """Test loading broker snapshot from JSON file."""
    # Create test JSON file
    json_path = tmp_path / "snapshot.json"
    snapshot_data = {
        "as_of": "2025-01-15",
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 100.0},
            {"symbol": "MSFT", "qty": 50.0},
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Load snapshot
    cash, positions_df = load_external_broker_snapshot(json_path)

    assert cash == 10000.0
    assert len(positions_df) == 2
    assert "symbol" in positions_df.columns
    assert "qty" in positions_df.columns
    assert "AAPL" in positions_df["symbol"].values
    assert "MSFT" in positions_df["symbol"].values


def test_load_external_broker_snapshot_json_cash_optional(tmp_path: Path):
    """Test that cash is optional in JSON snapshot."""
    json_path = tmp_path / "snapshot_no_cash.json"
    snapshot_data = {
        "positions": [
            {"symbol": "AAPL", "qty": 100.0},
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    cash, positions_df = load_external_broker_snapshot(json_path)

    assert cash is None
    assert len(positions_df) == 1


def test_load_external_broker_snapshot_json_missing_positions(tmp_path: Path):
    """Test that missing positions field raises ValueError."""
    json_path = tmp_path / "snapshot_invalid.json"
    snapshot_data = {
        "cash": 10000.0,
        # Missing "positions"
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    with pytest.raises(ValueError, match="Missing required field 'positions'"):
        load_external_broker_snapshot(json_path)


def test_load_external_broker_snapshot_csv_basic(tmp_path: Path):
    """Test loading broker snapshot from CSV file."""
    csv_path = tmp_path / "snapshot.csv"
    positions_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    positions_df.to_csv(csv_path, index=False)

    cash, loaded_df = load_external_broker_snapshot(csv_path)

    assert cash is None  # CSV doesn't have cash column
    assert len(loaded_df) == 2
    assert "symbol" in loaded_df.columns
    assert "qty" in loaded_df.columns


def test_load_external_broker_snapshot_csv_with_cash(tmp_path: Path):
    """Test loading CSV with cash column."""
    csv_path = tmp_path / "snapshot_with_cash.csv"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
            "cash": [10000.0, 10000.0],  # Same cash for all rows
        }
    )
    df.to_csv(csv_path, index=False)

    cash, loaded_df = load_external_broker_snapshot(csv_path)

    assert cash == 10000.0
    assert len(loaded_df) == 2


def test_load_external_broker_snapshot_csv_missing_columns(tmp_path: Path):
    """Test that missing required columns raises ValueError."""
    csv_path = tmp_path / "snapshot_invalid.csv"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            # Missing "qty"
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_external_broker_snapshot(csv_path)


def test_import_broker_snapshot_json_roundtrip(tmp_path: Path):
    """Test importing JSON snapshot and verifying stored output."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "  AAPL  ", "qty": 100.0},  # Will be trimmed
            {"symbol": "MSFT", "qty": 1e-10},  # Will be filtered (tiny residual)
            {"symbol": "GOOGL", "qty": 50.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    # Import snapshot
    output_dir = tmp_path / "output"
    result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id="test_import",
        snapshot_date=pd.Timestamp("2025-01-15", tz="UTC"),
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    # Verify return dict
    assert "broker_snapshot_path" in result
    assert "broker_positions_path" in result
    assert "cash" in result
    assert result["cash"] == 10000.0

    # Verify stored JSON exists
    stored_json_path = output_dir / result["broker_snapshot_path"]
    assert stored_json_path.exists()

    # Load and verify stored snapshot
    with stored_json_path.open("r", encoding="utf-8") as f:
        stored_data = json.load(f)

    assert stored_data["cash"] == 10000.0
    assert len(stored_data["positions"]) == 2  # MSFT filtered out, AAPL trimmed
    symbols = [pos["symbol"] for pos in stored_data["positions"]]
    assert "AAPL" in symbols  # Trimmed
    assert "MSFT" not in symbols  # Filtered (tiny residual)
    assert "GOOGL" in symbols

    # Verify positions are sorted (deterministic)
    assert symbols == sorted(symbols)


def test_import_broker_snapshot_csv_roundtrip(tmp_path: Path):
    """Test importing CSV snapshot and verifying stored output."""
    # Create external CSV snapshot
    external_path = tmp_path / "external_snapshot.csv"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    df.to_csv(external_path, index=False)

    # Import snapshot (with cash override)
    output_dir = tmp_path / "output"
    result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id="test_import_csv",
        snapshot_date=pd.Timestamp("2025-01-15", tz="UTC"),
        output_dir=output_dir,
        cash_override=10000.0,  # Provide cash since CSV doesn't have it
        store_parquet=True,
    )

    # Verify return dict
    assert result["cash"] == 10000.0
    assert result["broker_snapshot_path"] is not None
    assert result["broker_positions_path"] is not None


def test_import_broker_snapshot_deterministic_output(tmp_path: Path):
    """Test that importing same snapshot twice produces identical outputs."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "MSFT", "qty": 50.0},
            {"symbol": "AAPL", "qty": 100.0},  # Unsorted order
            {"symbol": "GOOGL", "qty": 25.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir1 = tmp_path / "output1"
    output_dir2 = tmp_path / "output2"

    # Import twice
    result1 = import_broker_snapshot(
        snapshot_path=external_path,
        run_id="test_deterministic",
        snapshot_date=pd.Timestamp("2025-01-15", tz="UTC"),
        output_dir=output_dir1,
        store_parquet=False,  # Only JSON for byte-comparison
    )

    result2 = import_broker_snapshot(
        snapshot_path=external_path,
        run_id="test_deterministic",
        snapshot_date=pd.Timestamp("2025-01-15", tz="UTC"),
        output_dir=output_dir2,
        store_parquet=False,
    )

    # Verify outputs are byte-identical
    path1 = output_dir1 / result1["broker_snapshot_path"]
    path2 = output_dir2 / result2["broker_snapshot_path"]

    with path1.open("rb") as f1, path2.open("rb") as f2:
        assert f1.read() == f2.read(), "Stored snapshots should be byte-identical"


def test_import_broker_snapshot_empty_positions(tmp_path: Path):
    """Test importing snapshot with empty positions list."""
    json_path = tmp_path / "snapshot_empty.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir = tmp_path / "output"
    result = import_broker_snapshot(
        snapshot_path=json_path,
        run_id="test_empty",
        snapshot_date=pd.Timestamp("2025-01-15", tz="UTC"),
        output_dir=output_dir,
        store_parquet=True,  # Should return None for empty positions
    )

    assert result["cash"] == 10000.0
    assert result["broker_positions_path"] is None  # Empty positions -> no Parquet


def test_import_broker_snapshot_file_not_found(tmp_path: Path):
    """Test that FileNotFoundError is raised for missing file."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_external_broker_snapshot(missing_path)


def test_import_broker_snapshot_unsupported_format(tmp_path: Path):
    """Test that unsupported file format raises ValueError."""
    txt_path = tmp_path / "snapshot.txt"
    txt_path.write_text("not a snapshot")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_external_broker_snapshot(txt_path)
