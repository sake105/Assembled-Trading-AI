# tests/test_factor_store_roundtrip.py
"""Roundtrip tests for factor store (store/load equals, partitions ok)."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    load_factors,
    store_factors,
    list_available_panels,
)


@pytest.fixture
def temp_factor_store(tmp_path, monkeypatch):
    """Set up temporary factor store root."""
    temp_store = tmp_path / "factors"
    temp_store.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch get_factor_store_root to return temp store
    from src.assembled_core import data
    original_get_root = data.factor_store.get_factor_store_root
    
    def mock_get_root(settings=None):
        return temp_store
    
    monkeypatch.setattr(data.factor_store, "get_factor_store_root", mock_get_root)
    
    yield temp_store
    
    # Cleanup (optional, tmp_path will be cleaned up anyway)
    # monkeypatch.undo()


@pytest.fixture
def sample_factors_df():
    """Create sample factors DataFrame for testing."""
    base_date = datetime(2023, 1, 1, tzinfo=pd.Timestamp.utcnow().tz)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for symbol in symbols:
        for i in range(100):  # 100 days per symbol
            ts = base_date + timedelta(days=i)
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "ta_ma_20": 100.0 + i * 0.1,
                "ta_ma_50": 100.0 + i * 0.05,
                "ta_rsi_14": 50.0 + (i % 20) * 0.5,
            })
    
    df = pd.DataFrame(data)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


def test_store_and_load_roundtrip(temp_factor_store, sample_factors_df):
    """Test that storing and loading factors produces identical DataFrame."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT", "GOOGL"])
    
    # Store factors
    store_path = store_factors(
        df=sample_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    assert store_path.exists()
    
    # Load factors
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Compare DataFrames (ignore order)
    loaded_df = loaded_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    expected_df = sample_factors_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Check columns
    assert set(loaded_df.columns) == set(expected_df.columns)
    
    # Check shape
    assert len(loaded_df) == len(expected_df)
    
    # Check values (with tolerance for floating point)
    pd.testing.assert_frame_equal(
        loaded_df[["timestamp", "symbol", "ta_ma_20", "ta_ma_50", "ta_rsi_14"]],
        expected_df[["timestamp", "symbol", "ta_ma_20", "ta_ma_50", "ta_rsi_14"]],
        check_dtype=False,  # Parquet may change dtypes slightly
        atol=1e-6,
    )


def test_store_partitions_by_year(temp_factor_store, sample_factors_df):
    """Test that factors are partitioned by year correctly."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT", "GOOGL"])
    
    # Store factors (sample_factors_df spans 2023)
    store_path = store_factors(
        df=sample_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Check that year partition files exist
    year_2023_file = store_path / "year=2023.parquet"
    assert year_2023_file.exists(), "Year 2023 partition should exist"
    
    # Create data spanning multiple years
    multi_year_data = []
    base_date = datetime(2023, 12, 1, tzinfo=pd.Timestamp.utcnow().tz)
    symbols = ["AAPL", "MSFT"]
    
    for symbol in symbols:
        # Add dates in 2023
        for i in range(31):  # December 2023
            ts = base_date + timedelta(days=i)
            multi_year_data.append({
                "timestamp": ts,
                "symbol": symbol,
                "ta_ma_20": 100.0,
                "ta_ma_50": 95.0,
            })
        # Add dates in 2024
        for i in range(31):  # January 2024
            ts = base_date + timedelta(days=31 + i)
            multi_year_data.append({
                "timestamp": ts,
                "symbol": symbol,
                "ta_ma_20": 105.0,
                "ta_ma_50": 100.0,
            })
    
    multi_year_df = pd.DataFrame(multi_year_data)
    multi_year_df["date"] = multi_year_df["timestamp"].dt.date.astype(str)
    multi_year_df = multi_year_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Store multi-year data
    universe_key_2 = compute_universe_key(symbols=["AAPL", "MSFT"])
    store_path_2 = store_factors(
        df=multi_year_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key_2,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Check that both year partitions exist
    year_2023_file_2 = store_path_2 / "year=2023.parquet"
    year_2024_file_2 = store_path_2 / "year=2024.parquet"
    
    assert year_2023_file_2.exists(), "Year 2023 partition should exist"
    assert year_2024_file_2.exists(), "Year 2024 partition should exist"
    
    # Load and verify all data is present
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key_2,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert len(loaded_df) == len(multi_year_df)
    
    # Check that years are correct
    years_in_loaded = sorted(loaded_df["timestamp"].dt.year.unique())
    assert years_in_loaded == [2023, 2024]


def test_store_append_mode(temp_factor_store, sample_factors_df):
    """Test that append mode merges with existing data correctly."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT", "GOOGL"])
    
    # Store initial data
    store_path = store_factors(
        df=sample_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Create additional data (overlapping dates should be updated, new dates added)
    additional_data = []
    base_date = datetime(2023, 12, 1, tzinfo=pd.Timestamp.utcnow().tz)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        # Add one overlapping date (should be updated)
        ts_overlap = base_date + timedelta(days=30)  # Overlap with existing data
        additional_data.append({
            "timestamp": ts_overlap,
            "symbol": symbol,
            "ta_ma_20": 999.0,  # Different value to verify update
            "ta_ma_50": 888.0,
            "ta_rsi_14": 77.0,
        })
        # Add one new date (should be added)
        ts_new = base_date + timedelta(days=101)  # New date
        additional_data.append({
            "timestamp": ts_new,
            "symbol": symbol,
            "ta_ma_20": 200.0,
            "ta_ma_50": 190.0,
            "ta_rsi_14": 60.0,
        })
    
    additional_df = pd.DataFrame(additional_data)
    additional_df["date"] = additional_df["timestamp"].dt.date.astype(str)
    additional_df = additional_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Append additional data
    store_factors(
        df=additional_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="append",
        factors_root=temp_factor_store,
    )
    
    # Load and verify
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    
    # Check that overlapping row was updated (take last occurrence)
    overlap_row = loaded_df[
        (loaded_df["timestamp"] == additional_df["timestamp"].iloc[0]) &
        (loaded_df["symbol"] == additional_df["symbol"].iloc[0])
    ]
    assert not overlap_row.empty
    assert overlap_row["ta_ma_20"].iloc[0] == pytest.approx(999.0, abs=1e-6)
    
    # Check that new row was added
    new_row = loaded_df[
        (loaded_df["timestamp"] == additional_df["timestamp"].iloc[3]) &
        (loaded_df["symbol"] == additional_df["symbol"].iloc[3])
    ]
    assert not new_row.empty
    assert new_row["ta_ma_20"].iloc[0] == pytest.approx(200.0, abs=1e-6)
    
    # Check total rows: original 300 rows - 3 overlapping (replaced) + 3 new = 300 total
    # Append mode deduplicates by (timestamp, symbol), keeping last occurrence
    # So we have: original rows - overlapping rows + new rows
    expected_rows = len(sample_factors_df) - len(symbols) + len(additional_df)
    assert len(loaded_df) == expected_rows, f"Expected {expected_rows} rows, got {len(loaded_df)}"


def test_list_available_panels(temp_factor_store, sample_factors_df):
    """Test that list_available_panels returns correct metadata."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT", "GOOGL"])
    
    # Store factors
    store_factors(
        df=sample_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # List panels
    panels = list_available_panels(
        factor_group=factor_group,
        freq=freq,
        factors_root=temp_factor_store,
    )
    
    assert len(panels) >= 1
    
    # Find our panel
    our_panel = next(
        (p for p in panels if p["universe_key"] == universe_key),
        None
    )
    assert our_panel is not None
    
    # Check metadata
    assert our_panel["factor_group"] == factor_group
    assert our_panel["freq"] == freq
    assert 2023 in our_panel["years"]
    assert "factor_columns" in our_panel or "factor_columns" in our_panel.get("schema", {})
