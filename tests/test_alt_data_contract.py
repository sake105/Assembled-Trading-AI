"""Tests for Alt-Data Event Contract (Sprint 10.A).

Tests verify:
1. UTC normalization (naive + tz-aware -> same normalized timestamps)
2. Validation (effective_date fallback, constraints)
3. Deterministic sorting
4. PIT filtering (future disclosure filtered out)
5. Deduplication (deterministic)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.altdata.contract import (
    REQUIRED_COLUMNS,
    filter_events_pit,
    normalize_alt_events,
)


def test_utc_normalization_naive_and_tz_aware() -> None:
    """Test that naive and tz-aware timestamps normalize to same UTC timestamps."""
    # Create events with naive timestamps
    events_naive = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16"]),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"]),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18"]),
    })

    # Create events with tz-aware timestamps (EST -> UTC)
    events_tz = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
    })

    # Normalize both
    normalized_naive = normalize_alt_events(events_naive)
    normalized_tz = normalize_alt_events(events_tz)

    # Verify: same normalized timestamps
    pd.testing.assert_frame_equal(
        normalized_naive[REQUIRED_COLUMNS],
        normalized_tz[REQUIRED_COLUMNS],
        check_dtype=False,
    )

    # Verify: all timestamps are UTC-aware
    for col in ["event_date", "disclosure_date", "effective_date"]:
        assert normalized_naive[col].dt.tz is not None, f"{col} should be timezone-aware"
        assert str(normalized_naive[col].dt.tz) == "UTC", f"{col} should be UTC"


def test_effective_date_fallback() -> None:
    """Test that missing effective_date falls back to disclosure_date."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        # effective_date missing
    })

    normalized = normalize_alt_events(events)

    # Verify: effective_date = disclosure_date
    assert "effective_date" in normalized.columns
    pd.testing.assert_series_equal(
        normalized["effective_date"],
        normalized["disclosure_date"],
        check_names=False,
    )


def test_effective_date_fallback_partial_nan() -> None:
    """Test that NaN effective_date values are filled with disclosure_date."""
    events = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "effective_date": [pd.NaT, pd.to_datetime("2024-01-18", utc=True)],
    })

    normalized = normalize_alt_events(events)

    # Verify: NaN filled with disclosure_date
    assert normalized.loc[0, "effective_date"] == normalized.loc[0, "disclosure_date"]
    # Second row unchanged (not NaN)
    assert normalized.loc[1, "effective_date"] == pd.to_datetime("2024-01-18", utc=True).normalize()


def test_constraint_validation_effective_ge_disclosure() -> None:
    """Test that effective_date >= disclosure_date constraint is enforced."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-16"], utc=True),  # < disclosure_date
    })

    with pytest.raises(ValueError, match="effective_date < disclosure_date"):
        normalize_alt_events(events)


def test_constraint_validation_disclosure_ge_event() -> None:
    """Test that disclosure_date >= event_date constraint is enforced."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-17"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),  # < event_date
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
    })

    with pytest.raises(ValueError, match="disclosure_date < event_date"):
        normalize_alt_events(events)


def test_deterministic_sorting() -> None:
    """Test that output is sorted deterministically."""
    events = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "AAPL"],
        "event_date": pd.to_datetime(["2024-01-16", "2024-01-15", "2024-01-14"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-18", "2024-01-17", "2024-01-16"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-18", "2024-01-17", "2024-01-16"], utc=True),
    })

    normalized = normalize_alt_events(events)

    # Verify: sorted by symbol, event_date, disclosure_date, effective_date
    assert normalized["symbol"].is_monotonic_increasing
    # AAPL rows should be sorted by event_date (ascending)
    aapl_rows = normalized[normalized["symbol"] == "AAPL"]
    assert len(aapl_rows) == 2
    assert aapl_rows["event_date"].is_monotonic_increasing


def test_filter_events_pit_future_disclosure_filtered() -> None:
    """Test that events with disclosure_date > as_of are filtered out."""
    events = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18", "2024-01-19"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18", "2024-01-19"], utc=True),
    })

    # Normalize first
    normalized = normalize_alt_events(events)

    # Filter at as_of = 2024-01-17
    as_of = pd.Timestamp("2024-01-17", tz="UTC")
    filtered = filter_events_pit(normalized, as_of)

    # Verify: Only AAPL (disclosure_date 2024-01-17 <= as_of)
    assert len(filtered) == 1
    assert filtered.iloc[0]["symbol"] == "AAPL"
    assert filtered.iloc[0]["disclosure_date"] == pd.Timestamp("2024-01-17", tz="UTC").normalize()


def test_filter_events_pit_inclusive_boundary() -> None:
    """Test that disclosure_date == as_of is included (inclusive boundary)."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
    })

    normalized = normalize_alt_events(events)

    # Filter at as_of = 2024-01-17 (same as disclosure_date)
    as_of = pd.Timestamp("2024-01-17", tz="UTC")
    filtered = filter_events_pit(normalized, as_of)

    # Verify: Event included (disclosure_date == as_of)
    assert len(filtered) == 1
    assert filtered.iloc[0]["symbol"] == "AAPL"


def test_deduplication_deterministic() -> None:
    """Test that duplicates are removed deterministically."""
    events = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-17", "2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-17", "2024-01-17"], utc=True),
        "value": [1000.0, 2000.0, 3000.0],  # Different values, but same dates
    })

    normalized = normalize_alt_events(events)

    # Verify: Only one row (duplicates removed)
    assert len(normalized) == 1

    # Verify: First occurrence kept (deterministic)
    assert normalized.iloc[0]["value"] == 1000.0


def test_deduplication_different_dates_not_deduped() -> None:
    """Test that events with different dates are not deduplicated."""
    events = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),  # Different
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),  # Different
    })

    normalized = normalize_alt_events(events)

    # Verify: Both rows kept (different disclosure_date/effective_date)
    assert len(normalized) == 2


def test_string_trimming() -> None:
    """Test that string columns are trimmed."""
    events = pd.DataFrame({
        "symbol": ["  AAPL  ", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "event_type": ["  BUY  ", "SELL"],
        "source": ["  SEC_FORM4  ", "SEC_FORM4"],
    })

    normalized = normalize_alt_events(events)

    # Verify: Strings trimmed
    assert normalized.iloc[0]["symbol"] == "AAPL"
    assert normalized.iloc[0]["event_type"] == "BUY"
    assert normalized.iloc[0]["source"] == "SEC_FORM4"


def test_missing_required_columns_raises_error() -> None:
    """Test that missing required columns raise ValueError."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        # disclosure_date missing
        # effective_date missing
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_alt_events(events)


def test_empty_dataframe_handling() -> None:
    """Test that empty DataFrame is handled gracefully."""
    events = pd.DataFrame()

    normalized = normalize_alt_events(events)

    # Verify: Empty DataFrame with required columns
    assert normalized.empty
    # is_public is now in OPTIONAL_COLUMNS
    assert set(normalized.columns) == set(REQUIRED_COLUMNS + ["event_type", "source", "value", "is_public"])


def test_filter_events_pit_empty_returns_empty() -> None:
    """Test that filtering empty DataFrame returns empty."""
    events = pd.DataFrame(columns=REQUIRED_COLUMNS)

    as_of = pd.Timestamp("2024-01-17", tz="UTC")
    filtered = filter_events_pit(events, as_of)

    assert filtered.empty


def test_filter_events_pit_missing_disclosure_date_raises_error() -> None:
    """Test that filtering without disclosure_date raises ValueError."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        # disclosure_date missing
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
    })

    as_of = pd.Timestamp("2024-01-17", tz="UTC")

    with pytest.raises(ValueError, match="Missing required column 'disclosure_date'"):
        filter_events_pit(events, as_of)


def test_optional_columns_preserved() -> None:
    """Test that optional columns are preserved during normalization."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
        "event_type": ["BUY"],
        "source": ["SEC_FORM4"],
        "value": [1000.0],
    })

    normalized = normalize_alt_events(events)

    # Verify: Optional columns preserved
    assert "event_type" in normalized.columns
    assert "source" in normalized.columns
    assert "value" in normalized.columns
    assert normalized.iloc[0]["event_type"] == "BUY"
    assert normalized.iloc[0]["source"] == "SEC_FORM4"
    assert normalized.iloc[0]["value"] == 1000.0


def test_is_public_false_raises_error() -> None:
    """Test that is_public=False raises ValueError (Public Disclosures Only policy)."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
        "is_public": [False],  # Non-public data
    })

    with pytest.raises(ValueError, match="Public Disclosures Only policy violated"):
        normalize_alt_events(events)


def test_is_public_true_allowed() -> None:
    """Test that is_public=True is allowed."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17"], utc=True),
        "is_public": [True],  # Public data
    })

    normalized = normalize_alt_events(events)

    # Verify: Normalization succeeds
    assert len(normalized) == 1
    # Verify: is_public column preserved
    assert "is_public" in normalized.columns
    assert normalized.iloc[0]["is_public"] == True  # noqa: E712


def test_is_public_mixed_raises_error() -> None:
    """Test that mixed is_public values (some False) raise ValueError."""
    events = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
        "is_public": [True, False],  # One non-public
    })

    with pytest.raises(ValueError, match="Public Disclosures Only policy violated"):
        normalize_alt_events(events)


def test_missing_disclosure_date_raises_error_with_policy_message() -> None:
    """Test that missing disclosure_date raises ValueError with policy message."""
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-15"], utc=True),
        # disclosure_date missing
    })

    with pytest.raises(ValueError, match="disclosure_date is mandatory for PIT-safe filtering"):
        normalize_alt_events(events)
