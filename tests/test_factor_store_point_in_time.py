"""Point-in-Time (PIT) safety tests for factor store.

Tests ensure that loading factors with as_of parameter correctly filters out
future data (no look-ahead bias).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    load_factors,
    store_factors,
)

pytestmark = pytest.mark.advanced


def test_factor_store_point_in_time_basic(tmp_path: Path) -> None:
    """Test basic PIT filtering: store t0..t9, load with as_of=t4 → only t0..t4."""
    symbols = ["AAPL"]
    # Create data for days t0..t9 (10 days)
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": range(len(dates)),  # Values 0..9 for easy verification
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store all days (t0..t9)
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load with as_of=t4 (2024-01-05, which is the 5th day, index 4)
    as_of_date = pd.Timestamp("2024-01-05", tz="UTC")
    
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        as_of=as_of_date,
        factors_root=tmp_path / "factors",
    )
    
    # Verify loaded data is not None
    assert loaded_df is not None, "load_factors should return DataFrame"
    assert not loaded_df.empty, "Loaded DataFrame should not be empty"
    
    # Verify only t0..t4 (5 days: 2024-01-01 to 2024-01-05) are returned
    assert len(loaded_df) == 5, f"Expected 5 rows, got {len(loaded_df)}"
    
    # Verify max timestamp is <= as_of
    assert loaded_df["timestamp"].max() <= as_of_date, \
        f"Max timestamp {loaded_df['timestamp'].max()} should be <= as_of {as_of_date}"
    
    # Verify all timestamps are <= as_of (strict PIT check)
    assert (loaded_df["timestamp"] <= as_of_date).all(), \
        "All timestamps must be <= as_of (no future data)"
    
    # Verify exact dates returned
    expected_dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    assert set(loaded_df["timestamp"].dt.date) == set(expected_dates.date), \
        "Expected dates 2024-01-01 to 2024-01-05"
    
    # Verify values match original (for returned dates)
    original_filtered = original_df[original_df["timestamp"] <= as_of_date].sort_values("timestamp")
    loaded_sorted = loaded_df.sort_values("timestamp")
    
    pd.testing.assert_series_equal(
        original_filtered["ta_ma_20"].reset_index(drop=True),
        loaded_sorted["ta_ma_20"].reset_index(drop=True),
        check_names=False,
    )


def test_factor_store_point_in_time_with_start_date(tmp_path: Path) -> None:
    """Test PIT filtering combined with start_date: store t0..t9, load start=t2, as_of=t4 → only t2..t4."""
    symbols = ["AAPL"]
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": range(len(dates)),
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store all days (t0..t9)
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load with start_date=t2 (2024-01-03) and as_of=t4 (2024-01-05)
    start_date = pd.Timestamp("2024-01-03", tz="UTC")
    as_of_date = pd.Timestamp("2024-01-05", tz="UTC")
    
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date=start_date,
        as_of=as_of_date,
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == 3, "Expected 3 rows (t2, t3, t4)"
    
    # Verify all timestamps are in range [start_date, as_of]
    assert (loaded_df["timestamp"] >= start_date).all(), "All timestamps must be >= start_date"
    assert (loaded_df["timestamp"] <= as_of_date).all(), "All timestamps must be <= as_of"
    
    # Verify exact dates
    expected_dates = pd.date_range("2024-01-03", "2024-01-05", freq="D", tz="UTC")
    assert set(loaded_df["timestamp"].dt.date) == set(expected_dates.date)


def test_factor_store_point_in_time_as_of_takes_precedence(tmp_path: Path) -> None:
    """Test that as_of takes precedence over end_date."""
    symbols = ["AAPL"]
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": range(len(dates)),
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load with end_date=t9 but as_of=t4 → should only return t0..t4
    end_date = pd.Timestamp("2024-01-10", tz="UTC")  # Would include all 10 days
    as_of_date = pd.Timestamp("2024-01-05", tz="UTC")  # Should cut at 5 days
    
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        end_date=end_date,
        as_of=as_of_date,
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == 5, "as_of should take precedence, returning only 5 days (t0..t4)"
    assert loaded_df["timestamp"].max() <= as_of_date, "Max timestamp should be <= as_of"


def test_factor_store_point_in_time_multiple_symbols(tmp_path: Path) -> None:
    """Test PIT filtering with multiple symbols."""
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    rows = []
    for symbol in symbols:
        for date in dates:
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "ta_ma_20": 100.0 + hash(symbol) % 10,
            })
    
    original_df = pd.DataFrame(rows)
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store all days for all symbols (t0..t9)
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load with as_of=t4
    as_of_date = pd.Timestamp("2024-01-05", tz="UTC")
    
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        as_of=as_of_date,
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == 10, "Expected 10 rows (2 symbols × 5 days)"
    
    # Verify all timestamps are <= as_of
    assert (loaded_df["timestamp"] <= as_of_date).all(), "No future data allowed"
    
    # Verify both symbols are present
    assert set(loaded_df["symbol"].unique()) == {"AAPL", "MSFT"}
    
    # Verify each symbol has exactly 5 days
    for symbol in symbols:
        symbol_df = loaded_df[loaded_df["symbol"] == symbol]
        assert len(symbol_df) == 5, f"Symbol {symbol} should have 5 days"
        assert symbol_df["timestamp"].max() <= as_of_date


def test_factor_store_point_in_time_no_future_data(tmp_path: Path) -> None:
    """Test that no future data is ever returned, even if stored."""
    symbols = ["AAPL"]
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": range(len(dates)),
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store all days (t0..t9)
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load with as_of in the middle of the stored range
    as_of_date = pd.Timestamp("2024-01-05", tz="UTC")
    
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        as_of=as_of_date,
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    
    # Verify no data after as_of is returned
    future_data = loaded_df[loaded_df["timestamp"] > as_of_date]
    assert len(future_data) == 0, f"Found {len(future_data)} rows with timestamp > as_of (future data leak)"
    
    # Verify all data is <= as_of
    assert (loaded_df["timestamp"] <= as_of_date).all(), "All data must be <= as_of"

