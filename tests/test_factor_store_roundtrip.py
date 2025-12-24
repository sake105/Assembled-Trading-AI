"""Roundtrip tests for factor store (store → load → verify)."""

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


def test_factor_store_roundtrip_basic(tmp_path: Path) -> None:
    """Test basic roundtrip: store → load → verify.

    Creates a mini factor panel (2 symbols × 5 days) and verifies that
    storing and loading preserves data integrity.
    """
    # Create mini factor panel
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    
    rows = []
    for symbol in symbols:
        for date in dates:
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "ta_ma_20": 100.0 + hash(symbol) % 10,  # Different values per symbol
                "ta_rsi_14": 50.0 + hash(symbol) % 20,
            })
    
    original_df = pd.DataFrame(rows)
    
    # Verify original structure
    assert len(original_df) == 10  # 2 symbols × 5 days
    assert set(original_df["symbol"].unique()) == {"AAPL", "MSFT"}
    assert len(original_df["timestamp"].unique()) == 5
    
    # Store factors
    universe_key = compute_universe_key(symbols=symbols)
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
        write_manifest=True,
    )
    
    # Load factors
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date="2024-01-01",
        end_date="2024-01-05",
        factors_root=tmp_path / "factors",
    )
    
    # Verify loaded data is not None
    assert loaded_df is not None, "load_factors should return DataFrame, not None"
    assert not loaded_df.empty, "Loaded DataFrame should not be empty"
    
    # Verify row count
    assert len(loaded_df) == len(original_df), f"Row count mismatch: {len(loaded_df)} vs {len(original_df)}"
    
    # Verify columns
    # Note: load_factors may add 'date' column if not present in original
    original_cols = set(original_df.columns)
    
    # Required columns must be present
    assert "timestamp" in loaded_df.columns, "timestamp column must be present"
    assert "symbol" in loaded_df.columns, "symbol column must be present"
    
    # Factor columns must be present
    factor_cols = original_cols - {"timestamp", "symbol"}
    for col in factor_cols:
        assert col in loaded_df.columns, f"Factor column {col} must be present"
    
    # Sort both DataFrames by timestamp, symbol for comparison
    original_sorted = original_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    loaded_sorted = loaded_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Verify timestamp values match (with timezone awareness)
    pd.testing.assert_series_equal(
        original_sorted["timestamp"],
        loaded_sorted["timestamp"],
        check_names=False,
    )
    
    # Verify symbol values match
    pd.testing.assert_series_equal(
        original_sorted["symbol"],
        loaded_sorted["symbol"],
        check_names=False,
    )
    
    # Verify factor columns (with tolerance for float comparisons)
    for col in factor_cols:
        pd.testing.assert_series_equal(
            original_sorted[col],
            loaded_sorted[col],
            check_names=False,
            rtol=1e-9,  # Relative tolerance for float comparisons
            atol=1e-9,  # Absolute tolerance
        )
    
    # Verify sorting: loaded data should be sorted by timestamp, symbol
    assert loaded_df.equals(loaded_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)), \
        "Loaded DataFrame should be sorted by timestamp, symbol"


def test_factor_store_roundtrip_multiple_years(tmp_path: Path) -> None:
    """Test roundtrip with data spanning multiple years."""
    symbols = ["AAPL"]
    dates = pd.date_range("2023-12-30", "2024-01-02", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": [100.0] * len(dates),
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load full range
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date="2023-12-30",
        end_date="2024-01-02",
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == len(original_df)
    
    # Verify all dates are present
    assert set(loaded_df["timestamp"].dt.date) == set(original_df["timestamp"].dt.date)


def test_factor_store_roundtrip_date_filtering(tmp_path: Path) -> None:
    """Test roundtrip with date filtering (partial load)."""
    symbols = ["AAPL"]
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    
    original_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "ta_ma_20": range(len(dates)),
    })
    
    universe_key = compute_universe_key(symbols=symbols)
    
    # Store full range
    store_factors(
        df=original_df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )
    
    # Load partial range
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date="2024-01-03",
        end_date="2024-01-07",
        factors_root=tmp_path / "factors",
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == 5  # 5 days: 2024-01-03 to 2024-01-07
    
    # Verify date filtering
    assert loaded_df["timestamp"].min() >= pd.Timestamp("2024-01-03", tz="UTC")
    assert loaded_df["timestamp"].max() <= pd.Timestamp("2024-01-07", tz="UTC")
    
    # Verify values match original (for filtered range)
    original_filtered = original_df[
        (original_df["timestamp"] >= "2024-01-03") & 
        (original_df["timestamp"] <= "2024-01-07")
    ].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    loaded_sorted = loaded_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    pd.testing.assert_series_equal(
        original_filtered["ta_ma_20"],
        loaded_sorted["ta_ma_20"],
        check_names=False,
    )

