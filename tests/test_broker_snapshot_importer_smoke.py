"""Smoke tests for broker snapshot importer (Sprint 13).

Tests JSON/CSV import, normalization, and deterministic output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot
from src.assembled_core.accounting.broker_snapshot_store import (
    load_broker_snapshot_json,
)


def test_import_json_roundtrip_store_load(tmp_path: Path):
    """Test JSON import -> store -> load roundtrip."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "AAPL", "qty": 100.0},
            {"symbol": "MSFT", "qty": 50.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir = tmp_path / "output"
    run_id = "test_roundtrip"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
    )

    # Verify return dict
    assert import_result["broker_snapshot_path"] is not None
    assert import_result["cash"] == 10000.0

    # Load stored snapshot
    stored_json_path = output_dir / import_result["broker_snapshot_path"]
    assert stored_json_path.exists()

    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    assert loaded is not None
    assert loaded["cash"] == 10000.0
    assert len(loaded["positions"]) == 2


def test_import_json_trimming_and_filtering(tmp_path: Path):
    """Test that imported snapshot normalizes (trim, filter tiny residuals)."""
    # Create external JSON with whitespace and tiny residual
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

    output_dir = tmp_path / "output"
    run_id = "test_normalize"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot
    _ = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=False,  # Only JSON for this test
    )

    # Load stored snapshot
    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    assert loaded is not None
    assert len(loaded["positions"]) == 2  # MSFT filtered out

    symbols = [pos["symbol"] for pos in loaded["positions"]]
    assert "AAPL" in symbols  # Trimmed
    assert "MSFT" not in symbols  # Filtered (tiny residual)
    assert "GOOGL" in symbols

    # Verify symbols are trimmed (no whitespace)
    for pos in loaded["positions"]:
        assert pos["symbol"] == pos["symbol"].strip()


def test_import_json_deterministic_sorting(tmp_path: Path):
    """Test that imported snapshot positions are sorted deterministically."""
    # Create external JSON with unsorted positions
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

    output_dir = tmp_path / "output"
    run_id = "test_sorting"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot
    _ = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=False,
    )

    # Load stored snapshot
    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    assert loaded is not None
    symbols = [pos["symbol"] for pos in loaded["positions"]]

    # Verify sorted alphabetically (deterministic)
    assert symbols == ["AAPL", "GOOGL", "MSFT"]


def test_import_json_stable_bytes(tmp_path: Path):
    """Test that importing same file twice produces identical JSON bytes."""
    # Create external JSON snapshot
    external_path = tmp_path / "external_snapshot.json"
    snapshot_data = {
        "cash": 10000.0,
        "positions": [
            {"symbol": "MSFT", "qty": 50.0},
            {"symbol": "AAPL", "qty": 100.0},  # Unsorted in input
            {"symbol": "GOOGL", "qty": 25.0},
        ],
    }
    with external_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    output_dir1 = tmp_path / "output1"
    output_dir2 = tmp_path / "output2"
    run_id = "test_stable"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import twice
    result1 = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir1,
        qty_tol=1e-8,
        store_parquet=False,
    )

    result2 = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir2,
        qty_tol=1e-8,
        store_parquet=False,
    )

    # Load both stored snapshots (using results)
    path1 = output_dir1 / result1["broker_snapshot_path"]
    path2 = output_dir2 / result2["broker_snapshot_path"]

    # Verify byte-identical
    with path1.open("rb") as f1, path2.open("rb") as f2:
        assert f1.read() == f2.read(), "Stored snapshots should be byte-identical"


def test_import_csv_basic(tmp_path: Path):
    """Test CSV import with cash override."""
    # Create external CSV snapshot
    external_path = tmp_path / "external_snapshot.csv"
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
    })
    df.to_csv(external_path, index=False)

    output_dir = tmp_path / "output"
    run_id = "test_csv"
    snapshot_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Import snapshot with cash override
    import_result = import_broker_snapshot(
        snapshot_path=external_path,
        run_id=run_id,
        snapshot_date=snapshot_date,
        output_dir=output_dir,
        qty_tol=1e-8,
        store_parquet=True,
        cash_override=10000.0,  # CSV doesn't have cash
    )

    # Verify return dict
    assert import_result["cash"] == 10000.0
    assert import_result["broker_snapshot_path"] is not None

    # Load stored snapshot
    loaded = load_broker_snapshot_json(
        output_dir=output_dir,
        run_id=run_id,
        as_of_date=snapshot_date,
    )

    assert loaded is not None
    assert loaded["cash"] == 10000.0
    assert len(loaded["positions"]) == 2
