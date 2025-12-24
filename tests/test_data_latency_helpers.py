"""Tests for event latency and PIT-safe filtering helpers (B2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)


def test_ensure_event_schema_required_columns() -> None:
    """Test ensure_event_schema validates required columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    
    # Should pass with default required columns
    result = ensure_event_schema(df, required_cols=["timestamp", "symbol"], strict=True)
    assert len(result) == 5
    
    # Should fail if required column missing (strict=True)
    df_missing = df.drop(columns=["symbol"])
    with pytest.raises(ValueError, match="Missing required columns"):
        ensure_event_schema(df_missing, required_cols=["timestamp", "symbol"], strict=True)


def test_ensure_event_schema_non_strict() -> None:
    """Test ensure_event_schema creates missing columns if strict=False."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "value": [1.0] * 5,
    })
    
    # Should create missing "symbol" column
    result = ensure_event_schema(df, required_cols=["timestamp", "symbol"], strict=False)
    assert "symbol" in result.columns
    assert result["symbol"].isna().all()  # Default: None/NaN
    
    # Should create disclosure_date from timestamp if needed
    result2 = ensure_event_schema(
        df, required_cols=["timestamp", "disclosure_date"], strict=False
    )
    assert "disclosure_date" in result2.columns
    assert not result2["disclosure_date"].isna().any()


def test_filter_events_as_of_basic() -> None:
    """Test filter_events_as_of filters by disclosure_date."""
    events = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "disclosure_date": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
        "value": range(10),
    })
    
    as_of = pd.Timestamp("2025-01-05", tz="UTC")
    
    filtered = filter_events_as_of(events, as_of, disclosure_col="disclosure_date")
    
    # Should include events up to and including 2025-01-05
    assert len(filtered) == 5
    assert filtered["disclosure_date"].max() <= as_of.normalize()
    assert (filtered["disclosure_date"] <= as_of.normalize()).all()


def test_filter_events_as_of_fallback_to_event_date() -> None:
    """Test filter_events_as_of with fallback to event_date."""
    events = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "event_date": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
        "value": range(10),
    })
    
    as_of = pd.Timestamp("2025-01-05", tz="UTC")
    
    filtered = filter_events_as_of(
        events,
        as_of,
        disclosure_col="disclosure_date",
        event_date_col="event_date",
        fallback_to_event_date=True,
    )
    
    # Should include events up to and including 2025-01-05
    assert len(filtered) == 5
    assert filtered["event_date"].max() <= as_of.normalize()


def test_filter_events_as_of_missing_disclosure() -> None:
    """Test filter_events_as_of raises error if disclosure column missing."""
    events = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    as_of = pd.Timestamp("2025-01-05", tz="UTC")
    
    # Should raise error if disclosure_col missing and no fallback
    with pytest.raises(ValueError, match="Cannot filter events"):
        filter_events_as_of(
            events, as_of, disclosure_col="disclosure_date", fallback_to_event_date=False
        )


def test_apply_source_latency_basic() -> None:
    """Test apply_source_latency derives disclosure_date from event_date."""
    events = pd.DataFrame({
        "event_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "value": range(5),
    })
    
    # Apply 2-day latency: disclosure_date = event_date + 2 days
    result = apply_source_latency(events, days=2, event_date_col="event_date")
    
    assert "disclosure_date" in result.columns
    assert (result["disclosure_date"] == result["event_date"] + pd.Timedelta(days=2)).all()


def test_apply_source_latency_fallback_to_timestamp() -> None:
    """Test apply_source_latency uses timestamp if event_date missing."""
    events = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    result = apply_source_latency(
        events, days=3, event_date_col="event_date", timestamp_col="timestamp"
    )
    
    assert "disclosure_date" in result.columns
    assert (result["disclosure_date"] == result["timestamp"] + pd.Timedelta(days=3)).all()


def test_apply_source_latency_shift_mode() -> None:
    """Test apply_source_latency shift mode."""
    events = pd.DataFrame({
        "event_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "disclosure_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    # Shift existing disclosure_date by 5 days
    result = apply_source_latency(
        events, days=5, event_date_col="event_date", mode="shift"
    )
    
    assert (result["disclosure_date"] == events["disclosure_date"] + pd.Timedelta(days=5)).all()


def test_filter_events_as_of_pit_safety() -> None:
    """Test that filter_events_as_of enforces PIT safety (no future data)."""
    # Event happens on 2025-01-05, but disclosed on 2025-01-08 (3-day delay)
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Filter for date 2025-01-07 (before disclosure)
    filtered_before = filter_events_as_of(
        events, pd.Timestamp("2025-01-07", tz="UTC"), disclosure_col="disclosure_date"
    )
    assert len(filtered_before) == 0  # Should be excluded (not yet disclosed)
    
    # Filter for date 2025-01-08 (on disclosure)
    filtered_on = filter_events_as_of(
        events, pd.Timestamp("2025-01-08", tz="UTC"), disclosure_col="disclosure_date"
    )
    assert len(filtered_on) == 1  # Should be included (disclosed)
    
    # Filter for date 2025-01-10 (after disclosure)
    filtered_after = filter_events_as_of(
        events, pd.Timestamp("2025-01-10", tz="UTC"), disclosure_col="disclosure_date"
    )
    assert len(filtered_after) == 1  # Should be included (already disclosed)


def test_apply_source_latency_zero_latency() -> None:
    """Test apply_source_latency with zero latency (disclosure = event)."""
    events = pd.DataFrame({
        "event_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    result = apply_source_latency(events, days=0, event_date_col="event_date")
    
    assert (result["disclosure_date"] == result["event_date"]).all()


def test_apply_source_latency_negative_latency() -> None:
    """Test apply_source_latency with negative latency (rare case)."""
    events = pd.DataFrame({
        "event_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    # Negative latency: disclosure_date = event_date - 1 day (disclosed before event)
    result = apply_source_latency(events, days=-1, event_date_col="event_date")
    
    assert (result["disclosure_date"] == result["event_date"] - pd.Timedelta(days=1)).all()


def test_ensure_event_schema_empty_dataframe() -> None:
    """Test ensure_event_schema handles empty DataFrame."""
    df_empty = pd.DataFrame()
    
    result = ensure_event_schema(df_empty, required_cols=["timestamp", "symbol"], strict=False)
    assert result.empty
    assert set(result.columns) == {"timestamp", "symbol"}


def test_filter_events_as_of_timezone_handling() -> None:
    """Test filter_events_as_of handles timezone-aware timestamps correctly."""
    events = pd.DataFrame({
        "disclosure_date": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
    })
    
    # as_of without timezone (should be localized to UTC)
    as_of_naive = pd.Timestamp("2025-01-03")
    filtered = filter_events_as_of(events, as_of_naive, disclosure_col="disclosure_date")
    
    assert len(filtered) == 3  # Should include 2025-01-01, 2025-01-02, 2025-01-03


def test_apply_source_latency_normalizes_dates() -> None:
    """Test that apply_source_latency normalizes dates to end-of-day."""
    events = pd.DataFrame({
        "event_date": pd.to_datetime(["2025-01-01 10:30:00", "2025-01-02 15:45:00"], utc=True),
        "symbol": ["AAPL", "MSFT"],
    })
    
    result = apply_source_latency(events, days=2, event_date_col="event_date")
    
    # disclosure_date should be normalized (end-of-day, no time component)
    assert all(result["disclosure_date"].dt.hour == 0)
    assert all(result["disclosure_date"].dt.minute == 0)

