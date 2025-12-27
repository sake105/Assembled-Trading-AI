"""Tests for precomputed index snapshot extraction."""

from __future__ import annotations


import numpy as np
import pandas as pd

from src.assembled_core.pipeline.precomputed_index import (
    build_panel_index,
    snapshot_as_of,
)


def test_build_panel_index_basic():
    """Test building panel index from a simple DataFrame."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT", "GOOGL"],
        "timestamp": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-01",
            "2020-01-02",
            "2020-01-01",
        ], utc=True),
        "close": [100.0, 101.0, 200.0, 201.0, 1500.0],
    })
    
    index = build_panel_index(df)
    
    assert len(index.symbols) == 3
    assert "AAPL" in index.symbols
    assert "MSFT" in index.symbols
    assert "GOOGL" in index.symbols
    
    assert "AAPL" in index.timestamps_by_symbol
    assert len(index.timestamps_by_symbol["AAPL"]) == 2
    assert "MSFT" in index.timestamps_by_symbol
    assert len(index.timestamps_by_symbol["MSFT"]) == 2
    assert "GOOGL" in index.timestamps_by_symbol
    assert len(index.timestamps_by_symbol["GOOGL"]) == 1


def test_snapshot_as_of_basic():
    """Test snapshot extraction using index."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "timestamp": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-01",
            "2020-01-02",
        ], utc=True),
        "close": [100.0, 101.0, 200.0, 201.0],
    })
    
    index = build_panel_index(df)
    
    # Snapshot at 2020-01-01 should return first row for each symbol
    as_of_1 = pd.Timestamp("2020-01-01", tz="UTC")
    snapshot_1 = snapshot_as_of(df, index, as_of_1)
    
    assert len(snapshot_1) == 2
    assert set(snapshot_1["symbol"].values) == {"AAPL", "MSFT"}
    assert snapshot_1[snapshot_1["symbol"] == "AAPL"]["close"].values[0] == 100.0
    assert snapshot_1[snapshot_1["symbol"] == "MSFT"]["close"].values[0] == 200.0
    
    # Snapshot at 2020-01-02 should return last row for each symbol
    as_of_2 = pd.Timestamp("2020-01-02", tz="UTC")
    snapshot_2 = snapshot_as_of(df, index, as_of_2)
    
    assert len(snapshot_2) == 2
    assert snapshot_2[snapshot_2["symbol"] == "AAPL"]["close"].values[0] == 101.0
    assert snapshot_2[snapshot_2["symbol"] == "MSFT"]["close"].values[0] == 201.0


def test_snapshot_as_of_empty():
    """Test snapshot extraction with empty DataFrame."""
    df = pd.DataFrame(columns=["symbol", "timestamp", "close"])
    
    index = build_panel_index(df)
    
    as_of = pd.Timestamp("2020-01-01", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    assert snapshot.empty
    assert list(snapshot.columns) == ["symbol", "timestamp", "close"]


def test_snapshot_as_of_no_rows_before():
    """Test snapshot extraction when as_of is before all timestamps."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "close": [100.0, 101.0],
    })
    
    index = build_panel_index(df)
    
    as_of = pd.Timestamp("2019-12-31", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    # Should return empty (no rows <= as_of)
    assert snapshot.empty


def test_snapshot_as_of_monotonic_optimization():
    """Test that monotonic optimization works for sequential queries."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        "timestamp": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-01",
            "2020-01-02",
        ], utc=True),
        "close": [100.0, 101.0, 102.0, 200.0, 201.0],
    })
    
    index = build_panel_index(df)
    
    # Sequential queries (monotonically increasing as_of)
    snapshot_1 = snapshot_as_of(df, index, pd.Timestamp("2020-01-01", tz="UTC"))
    snapshot_2 = snapshot_as_of(df, index, pd.Timestamp("2020-01-02", tz="UTC"))
    snapshot_3 = snapshot_as_of(df, index, pd.Timestamp("2020-01-03", tz="UTC"))
    
    # Check results are correct
    assert len(snapshot_1) == 2
    assert len(snapshot_2) == 2
    assert len(snapshot_3) == 2
    
    assert snapshot_1[snapshot_1["symbol"] == "AAPL"]["close"].values[0] == 100.0
    assert snapshot_2[snapshot_2["symbol"] == "AAPL"]["close"].values[0] == 101.0
    assert snapshot_3[snapshot_3["symbol"] == "AAPL"]["close"].values[0] == 102.0


def test_snapshot_as_of_equivalence_with_groupby():
    """Test that index-based snapshot produces same results as groupby approach."""
    np.random.seed(42)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
    
    rows = []
    for symbol in symbols:
        for date in dates:
            rows.append({
                "symbol": symbol,
                "timestamp": date,
                "close": np.random.rand() * 200 + 100,
            })
    
    df = pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    index = build_panel_index(df)
    
    # Test multiple as_of timestamps
    test_dates = [
        pd.Timestamp("2020-01-15", tz="UTC"),
        pd.Timestamp("2020-06-15", tz="UTC"),
        pd.Timestamp("2020-12-15", tz="UTC"),
    ]
    
    for as_of in test_dates:
        # Index-based snapshot
        snapshot_index = snapshot_as_of(df, index, as_of, use_monotonic_optimization=False)
        
        # Groupby-based snapshot (reference)
        df_filtered = df[df["timestamp"] <= as_of].copy()
        snapshot_groupby = (
            df_filtered.groupby("symbol", group_keys=False, dropna=False)
            .last()
            .reset_index()
            .sort_values("symbol")
            .reset_index(drop=True)
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            snapshot_index[["symbol", "timestamp", "close"]].sort_values("symbol").reset_index(drop=True),
            snapshot_groupby[["symbol", "timestamp", "close"]].sort_values("symbol").reset_index(drop=True),
            check_dtype=False,  # Allow minor dtype differences
        )


def test_build_panel_index_preserves_row_order():
    """Test that row indices in index match original DataFrame."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
        "close": [100.0, 101.0, 200.0],
    })
    
    index = build_panel_index(df)
    
    # Check that row indices are correct
    assert "AAPL" in index.row_idx_by_symbol
    assert len(index.row_idx_by_symbol["AAPL"]) == 2
    assert index.row_idx_by_symbol["AAPL"][0] == 0  # First AAPL row
    assert index.row_idx_by_symbol["AAPL"][1] == 1  # Second AAPL row
    
    assert "MSFT" in index.row_idx_by_symbol
    assert len(index.row_idx_by_symbol["MSFT"]) == 1
    assert index.row_idx_by_symbol["MSFT"][0] == 2  # MSFT row


def test_snapshot_with_missing_days():
    """Test snapshot extraction with gaps in data (missing days for some symbols)."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],  # AAPL has gap, MSFT continuous
        "timestamp": pd.to_datetime([
            "2020-01-01",
            "2020-01-03",  # Gap: 2020-01-02 missing for AAPL
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ], utc=True),
        "close": [100.0, 102.0, 200.0, 201.0, 202.0],
    })
    
    index = build_panel_index(df)
    
    # Snapshot at 2020-01-02: AAPL should get 2020-01-01 (last available), MSFT should get 2020-01-02
    as_of = pd.Timestamp("2020-01-02", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    assert len(snapshot) == 2
    assert set(snapshot["symbol"].values) == {"AAPL", "MSFT"}
    
    # AAPL: should have 2020-01-01 (gap at 2020-01-02, so last available before as_of)
    aapl_row = snapshot[snapshot["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["timestamp"] == pd.Timestamp("2020-01-01", tz="UTC")
    assert aapl_row["close"] == 100.0
    
    # MSFT: should have 2020-01-02 (exact match)
    msft_row = snapshot[snapshot["symbol"] == "MSFT"].iloc[0]
    assert msft_row["timestamp"] == pd.Timestamp("2020-01-02", tz="UTC")
    assert msft_row["close"] == 201.0


def test_snapshot_as_of_before_first_timestamp():
    """Test snapshot when as_of is before first timestamp for all symbols."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "timestamp": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"], utc=True),
        "close": [100.0, 101.0, 200.0],
    })
    
    index = build_panel_index(df)
    
    # as_of before all timestamps -> should return empty snapshot
    # (no rows <= as_of means no snapshot available)
    as_of = pd.Timestamp("2020-01-01", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    # Semantics: if as_of is before first timestamp for a symbol, that symbol is excluded
    assert snapshot.empty
    assert list(snapshot.columns) == list(df.columns)


def test_snapshot_as_of_exact_timestamp_match():
    """Test snapshot when as_of exactly matches a timestamp."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
        "close": [100.0, 101.0, 200.0],
    })
    
    index = build_panel_index(df)
    
    # as_of exactly on 2020-01-02 -> should include that row (<= comparison)
    as_of = pd.Timestamp("2020-01-02", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    assert len(snapshot) == 2
    assert set(snapshot["symbol"].values) == {"AAPL", "MSFT"}
    
    # AAPL: should have 2020-01-02 (exact match, included because <= as_of)
    aapl_row = snapshot[snapshot["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["timestamp"] == pd.Timestamp("2020-01-02", tz="UTC")
    assert aapl_row["close"] == 101.0
    
    # MSFT: should have 2020-01-01 (latest <= as_of)
    msft_row = snapshot[snapshot["symbol"] == "MSFT"].iloc[0]
    assert msft_row["timestamp"] == pd.Timestamp("2020-01-01", tz="UTC")
    assert msft_row["close"] == 200.0


def test_snapshot_timezone_aware_handling():
    """Test that timezone-aware timestamps are normalized to UTC."""
    # Create DataFrame with timezone-aware timestamps (EST)
    df_est = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "timestamp": pd.to_datetime(["2020-01-01 09:00", "2020-01-02 09:00"], tz="America/New_York"),
        "close": [100.0, 101.0],
    })
    
    index = build_panel_index(df_est)
    
    # Query with UTC timestamp (should match correctly)
    as_of_utc = pd.Timestamp("2020-01-02 14:00", tz="UTC")  # Same as EST 09:00
    snapshot = snapshot_as_of(df_est, index, as_of_utc)
    
    assert len(snapshot) == 1
    assert snapshot.iloc[0]["symbol"] == "AAPL"
    # Should get the 2020-01-02 row (UTC normalization should work)
    assert snapshot.iloc[0]["timestamp"] == pd.Timestamp("2020-01-02 09:00", tz="America/New_York")
    assert snapshot.iloc[0]["close"] == 101.0


def test_snapshot_naive_timestamps_normalized_to_utc():
    """Test that naive timestamps are normalized to UTC."""
    # Create DataFrame with naive timestamps
    df_naive = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),  # Naive (no timezone)
        "close": [100.0, 101.0],
    })
    
    index = build_panel_index(df_naive)
    
    # Query with UTC timestamp (should work after normalization)
    as_of_utc = pd.Timestamp("2020-01-02", tz="UTC")
    snapshot = snapshot_as_of(df_naive, index, as_of_utc)
    
    assert len(snapshot) == 1
    assert snapshot.iloc[0]["symbol"] == "AAPL"
    # Timestamp should be normalized to UTC in the result
    result_timestamp = snapshot.iloc[0]["timestamp"]
    assert pd.Timestamp(result_timestamp, tz="UTC") == pd.Timestamp("2020-01-02", tz="UTC")
    assert snapshot.iloc[0]["close"] == 101.0


def test_snapshot_missing_symbols():
    """Test snapshot when some symbols have no data at all."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],  # GOOGL missing entirely
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"], utc=True),
        "close": [100.0, 101.0, 200.0],
    })
    
    index = build_panel_index(df)
    
    # Snapshot should only include symbols that exist in the DataFrame
    as_of = pd.Timestamp("2020-01-02", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    assert len(snapshot) == 2
    assert set(snapshot["symbol"].values) == {"AAPL", "MSFT"}
    # GOOGL should not appear (was never in the DataFrame)


def test_snapshot_partial_missing_symbols():
    """Test snapshot when one symbol has data, another has none for the date range."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],  # MSFT only has 2020-01-01
        "timestamp": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-01",  # MSFT only has this date
        ], utc=True),
        "close": [100.0, 101.0, 200.0],
    })
    
    index = build_panel_index(df)
    
    # Snapshot at 2020-01-03: AAPL should get 2020-01-02, MSFT should get 2020-01-01
    as_of = pd.Timestamp("2020-01-03", tz="UTC")
    snapshot = snapshot_as_of(df, index, as_of)
    
    assert len(snapshot) == 2
    assert set(snapshot["symbol"].values) == {"AAPL", "MSFT"}
    
    # AAPL: latest available (2020-01-02)
    aapl_row = snapshot[snapshot["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["timestamp"] == pd.Timestamp("2020-01-02", tz="UTC")
    
    # MSFT: latest available (2020-01-01, only one row)
    msft_row = snapshot[snapshot["symbol"] == "MSFT"].iloc[0]
    assert msft_row["timestamp"] == pd.Timestamp("2020-01-01", tz="UTC")

