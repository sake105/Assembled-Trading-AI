"""Smoke tests for broker snapshot normalization and storage (Sprint 13)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.broker_snapshot import normalize_broker_snapshot
from src.assembled_core.accounting.broker_snapshot_store import (
    load_broker_snapshot_json,
    load_broker_snapshot_parquet,
    store_broker_snapshot_json,
    store_broker_snapshot_parquet,
)


def test_normalize_broker_snapshot_basic():
    """Test basic normalization of broker snapshot."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 50.0, 25.0],
        }
    )
    cash = 10000.0

    result = normalize_broker_snapshot(cash, positions)

    assert result["cash"] == 10000.0
    assert isinstance(result["positions_df"], pd.DataFrame)
    assert len(result["positions_df"]) == 3
    assert list(result["positions_df"]["symbol"]) == ["AAPL", "GOOGL", "MSFT"]  # Sorted


def test_normalize_broker_snapshot_trimming():
    """Test that symbol strings are trimmed."""
    positions = pd.DataFrame(
        {
            "symbol": ["  AAPL  ", " MSFT ", "GOOGL"],
            "qty": [100.0, 50.0, 25.0],
        }
    )
    cash = 10000.0

    result = normalize_broker_snapshot(cash, positions)

    # Symbols should be trimmed
    assert all(
        result["positions_df"]["symbol"].str.strip() == result["positions_df"]["symbol"]
    )
    assert "AAPL" in result["positions_df"]["symbol"].values
    assert "MSFT" in result["positions_df"]["symbol"].values


def test_normalize_broker_snapshot_tiny_residuals_ignored():
    """Test that tiny residual quantities are filtered out."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [100.0, 1e-10, 25.0],  # MSFT has tiny residual
        }
    )
    cash = 10000.0

    result = normalize_broker_snapshot(cash, positions, qty_tol=1e-8)

    # MSFT should be filtered out
    assert len(result["positions_df"]) == 2
    assert "MSFT" not in result["positions_df"]["symbol"].values
    assert "AAPL" in result["positions_df"]["symbol"].values
    assert "GOOGL" in result["positions_df"]["symbol"].values


def test_normalize_broker_snapshot_deterministic_sorting():
    """Test that positions are sorted deterministically."""
    positions = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "GOOGL", "TSLA"],
            "qty": [100.0, 50.0, 25.0, 10.0],
        }
    )
    cash = 10000.0

    result1 = normalize_broker_snapshot(cash, positions)
    result2 = normalize_broker_snapshot(cash, positions)

    # Should be sorted alphabetically
    assert list(result1["positions_df"]["symbol"]) == ["AAPL", "GOOGL", "MSFT", "TSLA"]
    assert list(result1["positions_df"]["symbol"]) == list(
        result2["positions_df"]["symbol"]
    )


def test_store_and_load_broker_snapshot_json(tmp_path: Path):
    """Test roundtrip: store and load broker snapshot JSON."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    cash = 10000.0
    run_id = "test_snapshot"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Store
    stored_path = store_broker_snapshot_json(
        cash, positions, tmp_path, run_id, as_of_date
    )
    assert stored_path.exists()

    # Load
    loaded = load_broker_snapshot_json(tmp_path, run_id, as_of_date)
    assert loaded is not None
    assert loaded["cash"] == 10000.0
    assert len(loaded["positions"]) == 2
    assert loaded["positions"][0]["symbol"] == "AAPL"
    assert loaded["positions"][0]["qty"] == 100.0


def test_store_and_load_broker_snapshot_parquet(tmp_path: Path):
    """Test roundtrip: store and load broker snapshot Parquet."""
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [100.0, 50.0],
        }
    )
    run_id = "test_snapshot"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Store
    stored_path = store_broker_snapshot_parquet(positions, tmp_path, run_id, as_of_date)
    assert stored_path is not None
    assert stored_path.exists()

    # Load
    loaded = load_broker_snapshot_parquet(tmp_path, run_id, as_of_date)
    assert loaded is not None
    assert len(loaded) == 2
    assert "symbol" in loaded.columns
    assert "qty" in loaded.columns
    assert "as_of_date" in loaded.columns
    assert list(loaded["symbol"]) == ["AAPL", "MSFT"]


def test_store_broker_snapshot_empty_positions(tmp_path: Path):
    """Test that empty positions DataFrame is handled correctly."""
    positions = pd.DataFrame(columns=["symbol", "qty"])
    cash = 10000.0
    run_id = "test_empty"
    as_of_date = pd.Timestamp("2025-01-15", tz="UTC")

    # Store JSON (should work)
    stored_path = store_broker_snapshot_json(
        cash, positions, tmp_path, run_id, as_of_date
    )
    assert stored_path.exists()

    # Store Parquet (should return None for empty)
    stored_parquet = store_broker_snapshot_parquet(
        positions, tmp_path, run_id, as_of_date
    )
    assert stored_parquet is None

    # Load JSON
    loaded = load_broker_snapshot_json(tmp_path, run_id, as_of_date)
    assert loaded is not None
    assert loaded["cash"] == 10000.0
    assert len(loaded["positions"]) == 0
