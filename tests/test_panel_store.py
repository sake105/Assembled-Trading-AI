# tests/test_panel_store.py
"""Tests for Panel Store Module (Sprint 3 / D3).

This test suite verifies:
1. Roundtrip parquet (write -> read -> same rows/cols)
2. Deterministic sorting (symbol, timestamp) after load
3. UTC normalization
4. Atomic write (no partial files)
5. Path computation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.panel_store import (
    load_price_panel_parquet,
    panel_exists,
    panel_path,
    store_price_panel_parquet,
)


def test_panel_path_computation() -> None:
    """Test that panel_path computes deterministic paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Test with universe
        path1 = panel_path(freq="1d", universe="ai_tech", root=root)
        assert path1 == root / "panels" / "1d" / "ai_tech" / "panel.parquet"

        # Test without universe (default)
        path2 = panel_path(freq="1d", universe=None, root=root)
        assert path2 == root / "panels" / "1d" / "default" / "panel.parquet"

        # Test 5min
        path3 = panel_path(freq="5min", universe="ai_tech", root=root)
        assert path3 == root / "panels" / "5min" / "ai_tech" / "panel.parquet"


def test_store_and_load_roundtrip() -> None:
    """Test roundtrip: write -> read -> same rows/cols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create test data
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [150.0 + i * 0.5 for i in range(10)],
            "volume": [1000.0] * 10,  # Optional column
        })

        # Store
        stored_path = store_price_panel_parquet(
            df=prices,
            freq="1d",
            universe="test",
            root=root,
        )

        assert stored_path.exists(), "Panel file should exist after store"

        # Load
        loaded_prices = load_price_panel_parquet(
            freq="1d",
            universe="test",
            root=root,
        )

        # Verify same rows/cols
        assert len(loaded_prices) == len(prices), "Should have same number of rows"
        assert set(loaded_prices.columns) == set(prices.columns), "Should have same columns"

        # Verify data (after sorting, should be identical)
        prices_sorted = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        assert loaded_prices.equals(prices_sorted), "Loaded data should match stored data (after sort)"


def test_deterministic_sorting_after_load() -> None:
    """Test that loaded panel is sorted by (symbol, timestamp)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create unsorted data
        prices = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2024-01-03", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            ],
            "symbol": ["MSFT", "AAPL", "GOOGL"],
            "close": [200.0, 150.0, 100.0],
        })

        # Store
        store_price_panel_parquet(df=prices, freq="1d", universe="test", root=root)

        # Load
        loaded_prices = load_price_panel_parquet(freq="1d", universe="test", root=root)

        # Verify sorted by (symbol, timestamp)
        assert loaded_prices["symbol"].is_monotonic_increasing, "Should be sorted by symbol"
        # Within each symbol, should be sorted by timestamp
        for symbol in loaded_prices["symbol"].unique():
            symbol_data = loaded_prices[loaded_prices["symbol"] == symbol]
            assert symbol_data["timestamp"].is_monotonic_increasing, f"Should be sorted by timestamp for {symbol}"


def test_utc_normalization() -> None:
    """Test that timestamps are normalized to UTC."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create data with naive timestamps (will be assumed UTC)
        prices_naive = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d"),  # Naive
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })

        # Store
        store_price_panel_parquet(df=prices_naive, freq="1d", universe="test", root=root)

        # Load
        loaded_prices = load_price_panel_parquet(freq="1d", universe="test", root=root)

        # Verify UTC-normalized
        assert loaded_prices["timestamp"].dt.tz is not None, "Timestamps should be timezone-aware"
        assert loaded_prices["timestamp"].dt.tz.zone == "UTC", "Timestamps should be UTC"


def test_panel_exists() -> None:
    """Test panel_exists function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Should not exist initially
        assert not panel_exists(freq="1d", universe="test", root=root), "Panel should not exist initially"

        # Create panel
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })
        store_price_panel_parquet(df=prices, freq="1d", universe="test", root=root)

        # Should exist now
        assert panel_exists(freq="1d", universe="test", root=root), "Panel should exist after store"


def test_load_missing_panel_raises_error() -> None:
    """Test that loading missing panel raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match="Panel file not found"):
            load_price_panel_parquet(freq="1d", universe="nonexistent", root=root)


def test_store_missing_required_columns_raises_error() -> None:
    """Test that storing panel with missing required columns raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Missing "close" column
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            # Missing "close"
        })

        with pytest.raises(ValueError, match="missing required columns"):
            store_price_panel_parquet(df=prices, freq="1d", universe="test", root=root)


def test_atomic_write_no_partial_files() -> None:
    """Test that atomic write prevents partial files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })

        # Store (should be atomic)
        panel_file = store_price_panel_parquet(df=prices, freq="1d", universe="test", root=root)

        # Verify no temp files remain
        temp_files = list(panel_file.parent.glob("*.tmp"))
        assert len(temp_files) == 0, "No temp files should remain after atomic write"

        # Verify panel file exists and is valid
        assert panel_file.exists(), "Panel file should exist"
        loaded = load_price_panel_parquet(freq="1d", universe="test", root=root)
        assert len(loaded) == 5, "Panel should be valid and loadable"


def test_append_mode_loads_existing_and_merges() -> None:
    """Test that append mode loads existing panel and merges with new data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Initial panel (D1..D10)
        prices_initial = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [150.0 + i * 0.5 for i in range(10)],
        })

        # Store initial panel
        store_price_panel_parquet(
            df=prices_initial, freq="1d", universe="test", root=root, mode="replace"
        )

        # New day (D11)
        prices_new = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-11", tz="UTC")],
            "symbol": ["AAPL"],
            "close": [155.0],
        })

        # Append new day
        store_price_panel_parquet(
            df=prices_new, freq="1d", universe="test", root=root, mode="append"
        )

        # Load and verify
        loaded = load_price_panel_parquet(freq="1d", universe="test", root=root)
        assert len(loaded) == 11, "Should have 11 rows (D1..D11)"
        assert loaded["timestamp"].min() == pd.Timestamp("2024-01-01", tz="UTC")
        assert loaded["timestamp"].max() == pd.Timestamp("2024-01-11", tz="UTC")


def test_append_mode_equals_recompute() -> None:
    """Test that append-day equals recompute-last-day (D3 requirement)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Initial panel (D1..D10)
        prices_initial = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [150.0 + i * 0.5 for i in range(10)],
        })

        # Store initial panel
        store_price_panel_parquet(
            df=prices_initial, freq="1d", universe="test", root=root, mode="replace"
        )

        # New day (D11)
        prices_new = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-11", tz="UTC")],
            "symbol": ["AAPL"],
            "close": [155.0],
        })

        # Append new day
        store_price_panel_parquet(
            df=prices_new, freq="1d", universe="test", root=root, mode="append"
        )

        # Full recompute (D1..D11)
        prices_full = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=11, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 11,
            "close": [150.0 + i * 0.5 for i in range(11)],
        })

        # Store full recompute in different universe for comparison
        store_price_panel_parquet(
            df=prices_full, freq="1d", universe="test_full", root=root, mode="replace"
        )

        # Compare: append result vs full recompute
        loaded_append = load_price_panel_parquet(freq="1d", universe="test", root=root)
        loaded_full = load_price_panel_parquet(freq="1d", universe="test_full", root=root)

        # Should be identical after sorting (both are sorted by symbol, timestamp)
        assert len(loaded_append) == len(loaded_full), "Should have same number of rows"
        assert loaded_append.equals(loaded_full), "Append result should equal full recompute"


def test_append_mode_dedupe_handles_overlaps() -> None:
    """Test that append mode correctly dedupes overlapping (symbol, timestamp) pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Initial panel (D1..D10)
        prices_initial = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [150.0 + i * 0.5 for i in range(10)],
        })

        # Store initial panel
        store_price_panel_parquet(
            df=prices_initial, freq="1d", universe="test", root=root, mode="replace"
        )

        # New data with overlap (D10 updated, D11 new)
        prices_overlap = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2024-01-10", tz="UTC"),  # Overlap (should be updated)
                pd.Timestamp("2024-01-11", tz="UTC"),  # New
            ],
            "symbol": ["AAPL", "AAPL"],
            "close": [155.0, 156.0],  # D10: 154.5 -> 155.0 (updated), D11: new
        })

        # Append with overlap
        store_price_panel_parquet(
            df=prices_overlap, freq="1d", universe="test", root=root, mode="append"
        )

        # Load and verify
        loaded = load_price_panel_parquet(freq="1d", universe="test", root=root)
        assert len(loaded) == 11, "Should have 11 rows (D1..D11, D10 deduped)"
        
        # D10 should have updated value (keep="last")
        d10_row = loaded[loaded["timestamp"] == pd.Timestamp("2024-01-10", tz="UTC")]
        assert len(d10_row) == 1, "Should have exactly one D10 row"
        assert d10_row.iloc[0]["close"] == 155.0, "D10 should have updated value (keep='last')"


def test_append_mode_creates_new_if_not_exists() -> None:
    """Test that append mode creates new panel if none exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Append to non-existent panel (should create new)
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })

        store_price_panel_parquet(
            df=prices, freq="1d", universe="test", root=root, mode="append"
        )

        # Verify panel exists and is valid
        assert panel_exists(freq="1d", universe="test", root=root), "Panel should exist"
        loaded = load_price_panel_parquet(freq="1d", universe="test", root=root)
        assert len(loaded) == 5, "Panel should contain all rows"
