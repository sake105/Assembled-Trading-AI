# tests/test_universe_as_of.py
"""Tests for Universe Management (Sprint 4 / C2).

This test suite verifies:
1. Universe membership is resolved correctly at as_of timestamps
2. end_date is EXCLUSIVE (symbol not in universe on end_date)
3. Active symbols (end_date=None) remain in universe indefinitely
4. Universe change day: membership changes correctly between dates
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.universe import (
    get_universe_members,
    load_universe_history,
    store_universe_history,
)


def test_universe_change_day_membership_switches() -> None:
    """Test that universe membership switches correctly between two dates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history:
        # AAPL: added 2024-01-01, removed 2024-06-30 (exclusive)
        # MSFT: added 2024-01-01, still active (end_date=None)
        # GOOGL: added 2024-07-01, still active (end_date=None)
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-07-01", tz="UTC"),
            ],
            "end_date": [
                pd.Timestamp("2024-06-30", tz="UTC"),  # Removed on 2024-06-30 (exclusive)
                None,  # Still active
                None,  # Still active
            ],
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: as_of = 2024-06-29 (before AAPL removal)
        as_of_before = pd.Timestamp("2024-06-29", tz="UTC")
        members_before = get_universe_members(as_of_before, universe_name="test", root=root)
        assert set(members_before) == {"AAPL", "MSFT"}, "AAPL and MSFT should be in universe before removal"
        assert "GOOGL" not in members_before, "GOOGL should not be in universe yet (added on 2024-07-01)"
        
        # Test: as_of = 2024-06-30 (AAPL removal day, EXCLUSIVE)
        as_of_removal = pd.Timestamp("2024-06-30", tz="UTC")
        members_removal = get_universe_members(as_of_removal, universe_name="test", root=root)
        assert set(members_removal) == {"MSFT"}, "Only MSFT should be in universe on removal day (AAPL excluded)"
        assert "AAPL" not in members_removal, "AAPL should not be in universe on removal day (end_date exclusive)"
        assert "GOOGL" not in members_removal, "GOOGL should not be in universe yet"
        
        # Test: as_of = 2024-07-01 (GOOGL addition day)
        as_of_after = pd.Timestamp("2024-07-01", tz="UTC")
        members_after = get_universe_members(as_of_after, universe_name="test", root=root)
        assert set(members_after) == {"MSFT", "GOOGL"}, "MSFT and GOOGL should be in universe after addition"
        assert "AAPL" not in members_after, "AAPL should not be in universe after removal"


def test_end_date_exclusive() -> None:
    """Test that end_date is EXCLUSIVE (symbol not in universe on end_date)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history:
        # AAPL: added 2024-01-01, removed 2024-06-30 (exclusive)
        history = pd.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2024-01-01", tz="UTC")],
            "end_date": [pd.Timestamp("2024-06-30", tz="UTC")],  # Exclusive
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: as_of = 2024-06-29 (day before removal)
        as_of_before = pd.Timestamp("2024-06-29", tz="UTC")
        members_before = get_universe_members(as_of_before, universe_name="test", root=root)
        assert "AAPL" in members_before, "AAPL should be in universe before end_date"
        
        # Test: as_of = 2024-06-30 (removal day, EXCLUSIVE)
        as_of_removal = pd.Timestamp("2024-06-30", tz="UTC")
        members_removal = get_universe_members(as_of_removal, universe_name="test", root=root)
        assert "AAPL" not in members_removal, "AAPL should NOT be in universe on end_date (exclusive)"
        
        # Test: as_of = 2024-07-01 (day after removal)
        as_of_after = pd.Timestamp("2024-07-01", tz="UTC")
        members_after = get_universe_members(as_of_after, universe_name="test", root=root)
        assert "AAPL" not in members_after, "AAPL should not be in universe after end_date"


def test_active_symbols_end_date_none() -> None:
    """Test that active symbols (end_date=None) remain in universe indefinitely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history:
        # MSFT: added 2024-01-01, still active (end_date=None)
        history = pd.DataFrame({
            "symbol": ["MSFT"],
            "start_date": [pd.Timestamp("2024-01-01", tz="UTC")],
            "end_date": [None],  # Still active
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: as_of = 2024-01-01 (addition day)
        as_of_start = pd.Timestamp("2024-01-01", tz="UTC")
        members_start = get_universe_members(as_of_start, universe_name="test", root=root)
        assert "MSFT" in members_start, "MSFT should be in universe on start_date"
        
        # Test: as_of = 2024-12-31 (later date)
        as_of_later = pd.Timestamp("2024-12-31", tz="UTC")
        members_later = get_universe_members(as_of_later, universe_name="test", root=root)
        assert "MSFT" in members_later, "MSFT should still be in universe (end_date=None)"
        
        # Test: as_of = 2025-12-31 (even later date)
        as_of_future = pd.Timestamp("2025-12-31", tz="UTC")
        members_future = get_universe_members(as_of_future, universe_name="test", root=root)
        assert "MSFT" in members_future, "MSFT should still be in universe (end_date=None, indefinite)"


def test_get_universe_members_deterministic() -> None:
    """Test that get_universe_members is deterministic (same input -> same output)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-07-01", tz="UTC"),
            ],
            "end_date": [None, None, None],
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: Call multiple times with same as_of
        as_of = pd.Timestamp("2024-06-15", tz="UTC")
        members1 = get_universe_members(as_of, universe_name="test", root=root)
        members2 = get_universe_members(as_of, universe_name="test", root=root)
        members3 = get_universe_members(as_of, universe_name="test", root=root)
        
        # Verify: Same result each time
        assert members1 == members2 == members3, "Results should be identical (deterministic)"
        
        # Verify: Sorted (alphabetically, uppercase)
        assert members1 == sorted(members1), "Members should be sorted"
        assert all(s.isupper() for s in members1), "Members should be uppercase"


def test_get_universe_members_utc_normalized() -> None:
    """Test that timestamps are UTC-normalized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history
        history = pd.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2024-01-01", tz="UTC")],
            "end_date": [None],
        })
        
        # Store universe history
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: Naive timestamp (should be interpreted as UTC)
        as_of_naive = pd.Timestamp("2024-06-15")  # No timezone
        members_naive = get_universe_members(as_of_naive, universe_name="test", root=root)
        
        # Test: UTC-aware timestamp
        as_of_utc = pd.Timestamp("2024-06-15", tz="UTC")
        members_utc = get_universe_members(as_of_utc, universe_name="test", root=root)
        
        # Verify: Same result (naive interpreted as UTC)
        assert members_naive == members_utc, "Naive timestamps should be interpreted as UTC"


def test_load_universe_history_formats() -> None:
    """Test that load_universe_history supports multiple formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
            ],
            "end_date": [None, None],
        })
        
        # Test: Parquet format
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        loaded_parquet = load_universe_history(universe_name="test", root=root)
        assert len(loaded_parquet) == 2, "Should load 2 records from parquet"
        assert set(loaded_parquet["symbol"]) == {"AAPL", "MSFT"}, "Should load correct symbols"
        
        # Test: CSV format
        store_universe_history(history, universe_name="test_csv", root=root, format="csv")
        loaded_csv = load_universe_history(universe_name="test_csv", root=root)
        assert len(loaded_csv) == 2, "Should load 2 records from CSV"
        assert set(loaded_csv["symbol"]) == {"AAPL", "MSFT"}, "Should load correct symbols"


def test_load_universe_history_missing_file() -> None:
    """Test that load_universe_history returns empty DataFrame if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Try to load non-existent universe
        loaded = load_universe_history(universe_name="nonexistent", root=root)
        
        # Verify: Empty DataFrame with correct columns
        assert loaded.empty, "Should return empty DataFrame if file doesn't exist"
        assert set(loaded.columns) == {"symbol", "start_date", "end_date"}, "Should have correct columns"


def test_store_universe_history_roundtrip() -> None:
    """Test that store/load roundtrip preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create universe history
        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "start_date": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
            ],
            "end_date": [
                pd.Timestamp("2024-06-30", tz="UTC"),
                None,
            ],
        })
        
        # Store and load
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        loaded = load_universe_history(universe_name="test", root=root)
        
        # Verify: Data preserved
        assert len(loaded) == len(history), "Should preserve number of records"
        assert set(loaded["symbol"]) == set(history["symbol"]), "Should preserve symbols"
        
        # Verify: Timestamps preserved (within tolerance)
        # Ensure loaded timestamps are datetime
        assert pd.api.types.is_datetime64_any_dtype(loaded["start_date"]), "start_date should be datetime"
        
        for symbol in history["symbol"]:
            orig = history[history["symbol"] == symbol].iloc[0]
            loaded_row = loaded[loaded["symbol"] == symbol].iloc[0]
            
            # Compare start_date (normalize to UTC if needed)
            orig_start = pd.to_datetime(orig["start_date"], utc=True)
            loaded_start = pd.to_datetime(loaded_row["start_date"], utc=True)
            assert loaded_start == orig_start, "Should preserve start_date"
            
            # Compare end_date
            if orig["end_date"] is None:
                assert pd.isna(loaded_row["end_date"]) or loaded_row["end_date"] is None, "Should preserve None end_date"
            else:
                orig_end = pd.to_datetime(orig["end_date"], utc=True)
                loaded_end = pd.to_datetime(loaded_row["end_date"], utc=True)
                # Handle NaT comparison (use pd.isna for NaT)
                if pd.isna(orig_end) and pd.isna(loaded_end):
                    pass  # Both are NaT, which is correct
                else:
                    assert loaded_end == orig_end, "Should preserve end_date"


def test_get_universe_members_empty_universe() -> None:
    """Test that get_universe_members returns empty list for empty universe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create empty universe history (with proper dtypes)
        history = pd.DataFrame({
            "symbol": pd.Series([], dtype="string"),
            "start_date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "end_date": pd.Series([], dtype="object"),
        })
        store_universe_history(history, universe_name="test", root=root, format="parquet")
        
        # Test: Get members
        as_of = pd.Timestamp("2024-06-15", tz="UTC")
        members = get_universe_members(as_of, universe_name="test", root=root)
        
        # Verify: Empty list
        assert members == [], "Should return empty list for empty universe"
