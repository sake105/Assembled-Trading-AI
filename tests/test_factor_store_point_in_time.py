# tests/test_factor_store_point_in_time.py
"""Point-in-time (PIT) safety tests for factor store (no rows > end/as_of)."""

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


@pytest.fixture
def multi_year_factors_df():
    """Create factors DataFrame spanning multiple years for PIT testing."""
    base_date = datetime(2023, 6, 1, tzinfo=pd.Timestamp.utcnow().tz)
    symbols = ["AAPL", "MSFT"]
    
    data = []
    # Create data from 2023-06-01 to 2024-06-01 (1 year span)
    for symbol in symbols:
        for i in range(366):  # 366 days
            ts = base_date + timedelta(days=i)
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "ta_ma_20": 100.0 + i * 0.1,
                "ta_ma_50": 95.0 + i * 0.05,
            })
    
    df = pd.DataFrame(data)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


def test_load_factors_with_end_date_filter(temp_factor_store, multi_year_factors_df):
    """Test that end_date filter excludes rows after end_date."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with end_date filter (cut off at 2023-12-31)
    end_date = pd.Timestamp("2023-12-31", tz="UTC")
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        end_date=end_date,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Verify no rows after end_date
    assert (loaded_df["timestamp"] <= end_date).all(), "All rows should be <= end_date"
    
    # Verify max timestamp is <= end_date
    max_timestamp = loaded_df["timestamp"].max()
    assert max_timestamp <= end_date, f"Max timestamp {max_timestamp} should be <= end_date {end_date}"
    
    # Verify we got some data (not empty)
    assert len(loaded_df) > 0, "Should have some rows before end_date"


def test_load_factors_with_start_date_filter(temp_factor_store, multi_year_factors_df):
    """Test that start_date filter excludes rows before start_date."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with start_date filter (start at 2024-01-01)
    start_date = pd.Timestamp("2024-01-01", tz="UTC")
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        start_date=start_date,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Verify no rows before start_date
    assert (loaded_df["timestamp"] >= start_date).all(), "All rows should be >= start_date"
    
    # Verify min timestamp is >= start_date
    min_timestamp = loaded_df["timestamp"].min()
    assert min_timestamp >= start_date, f"Min timestamp {min_timestamp} should be >= start_date {start_date}"
    
    # Verify we got some data (not empty)
    assert len(loaded_df) > 0, "Should have some rows after start_date"


def test_load_factors_with_date_range_filter(temp_factor_store, multi_year_factors_df):
    """Test that start_date and end_date together filter correctly."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with both start_date and end_date (2023-09-01 to 2023-12-31)
    start_date = pd.Timestamp("2023-09-01", tz="UTC")
    end_date = pd.Timestamp("2023-12-31", tz="UTC")
    
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        start_date=start_date,
        end_date=end_date,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Verify all rows are within date range
    assert (loaded_df["timestamp"] >= start_date).all(), "All rows should be >= start_date"
    assert (loaded_df["timestamp"] <= end_date).all(), "All rows should be <= end_date"
    
    # Verify min and max timestamps
    min_timestamp = loaded_df["timestamp"].min()
    max_timestamp = loaded_df["timestamp"].max()
    
    assert min_timestamp >= start_date, f"Min timestamp {min_timestamp} should be >= start_date {start_date}"
    assert max_timestamp <= end_date, f"Max timestamp {max_timestamp} should be <= end_date {end_date}"


def test_load_factors_with_as_of_pit_filter(temp_factor_store, multi_year_factors_df):
    """Test that as_of (PIT) filter excludes rows after as_of (strict PIT safety)."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with as_of filter (cut off at 2024-01-15)
    as_of = pd.Timestamp("2024-01-15", tz="UTC")
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        as_of=as_of,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Verify no rows after as_of (strict PIT safety: <= as_of)
    assert (loaded_df["timestamp"] <= as_of).all(), "All rows should be <= as_of (PIT safety)"
    
    # Verify max timestamp is <= as_of
    max_timestamp = loaded_df["timestamp"].max()
    assert max_timestamp <= as_of, f"Max timestamp {max_timestamp} should be <= as_of {as_of}"
    
    # Verify we got some data
    assert len(loaded_df) > 0, "Should have some rows <= as_of"


def test_load_factors_as_of_takes_precedence_over_end_date(temp_factor_store, multi_year_factors_df):
    """Test that as_of takes precedence over end_date (as_of is stricter)."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with both end_date (later) and as_of (earlier)
    # as_of should take precedence
    end_date = pd.Timestamp("2024-06-01", tz="UTC")  # Later
    as_of = pd.Timestamp("2024-01-15", tz="UTC")  # Earlier (should take precedence)
    
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        end_date=end_date,
        as_of=as_of,
        factors_root=temp_factor_store,
    )
    
    assert loaded_df is not None
    assert not loaded_df.empty
    
    # Verify all rows are <= as_of (not end_date)
    assert (loaded_df["timestamp"] <= as_of).all(), "All rows should be <= as_of (takes precedence)"
    
    max_timestamp = loaded_df["timestamp"].max()
    assert max_timestamp <= as_of, f"Max timestamp {max_timestamp} should be <= as_of {as_of} (not end_date {end_date})"


def test_load_factors_empty_result_when_all_data_after_start(temp_factor_store, multi_year_factors_df):
    """Test that loading with start_date after all data returns None or empty DataFrame."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors (data from 2023-06-01 to 2024-06-01)
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with start_date after all data (2025-01-01)
    start_date = pd.Timestamp("2025-01-01", tz="UTC")
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        start_date=start_date,
        factors_root=temp_factor_store,
    )
    
    # Should return None or empty DataFrame
    if loaded_df is not None:
        assert loaded_df.empty, "Should return empty DataFrame when all data is before start_date"


def test_load_factors_empty_result_when_all_data_before_end(temp_factor_store, multi_year_factors_df):
    """Test that loading with end_date before all data returns None or empty DataFrame."""
    factor_group = "core_ta"
    freq = "1d"
    universe_key = compute_universe_key(symbols=["AAPL", "MSFT"])
    
    # Store factors (data from 2023-06-01 to 2024-06-01)
    store_factors(
        df=multi_year_factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite",
        factors_root=temp_factor_store,
    )
    
    # Load with end_date before all data (2023-01-01)
    end_date = pd.Timestamp("2023-01-01", tz="UTC")
    loaded_df = load_factors(
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        end_date=end_date,
        factors_root=temp_factor_store,
    )
    
    # Should return None or empty DataFrame
    if loaded_df is not None:
        assert loaded_df.empty, "Should return empty DataFrame when all data is after end_date"
