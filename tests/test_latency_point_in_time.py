"""Point-in-time (PIT) tests for event latency filtering (B2).

These tests guarantee that features are "blind" to events that have not yet
been disclosed. This prevents look-ahead bias in backtests and ensures
point-in-time correctness.

Core rule: For backtest date T, only events with disclosure_date <= T
may be used in feature computation.

Boundary semantics: "disclosure_date <= as_of" means that events disclosed
on the same day as as_of are included (inclusive).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.latency import filter_events_as_of


def test_pit_filtering_excludes_future_disclosure() -> None:
    """Test that events with disclosure_date > as_of are excluded (feature is blind)."""
    # Event happens on 2025-01-05, disclosed on 2025-01-08 (3-day delay)
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],  # This value should NOT be visible before disclosure
    })
    
    # Filter for date 2025-01-07 (before disclosure)
    as_of_before = pd.Timestamp("2025-01-07", tz="UTC")
    filtered_before = filter_events_as_of(
        events, as_of_before, disclosure_col="disclosure_date"
    )
    
    # Feature must be blind: event should be excluded
    assert len(filtered_before) == 0, (
        "Feature must be blind to events not yet disclosed. "
        f"Event disclosed on {events['disclosure_date'].iloc[0]}, "
        f"but as_of={as_of_before} should exclude it."
    )
    
    # Verify no value leakage
    assert filtered_before.empty, "Filtered DataFrame must be empty before disclosure"
    
    # Filter for date 2025-01-08 (on disclosure day - inclusive)
    as_of_on = pd.Timestamp("2025-01-08", tz="UTC")
    filtered_on = filter_events_as_of(
        events, as_of_on, disclosure_col="disclosure_date"
    )
    
    # Event should now be visible (disclosure_date <= as_of, inclusive)
    assert len(filtered_on) == 1, (
        "Event should be included on disclosure_date (inclusive boundary)"
    )
    assert filtered_on["value"].iloc[0] == 100.0, "Event value should be accessible after disclosure"
    
    # Filter for date 2025-01-10 (after disclosure)
    as_of_after = pd.Timestamp("2025-01-10", tz="UTC")
    filtered_after = filter_events_as_of(
        events, as_of_after, disclosure_col="disclosure_date"
    )
    
    # Event should still be visible
    assert len(filtered_after) == 1, "Event should remain visible after disclosure"
    assert filtered_after["value"].iloc[0] == 100.0


def test_pit_filtering_boundary_inclusive() -> None:
    """Test that boundary condition (as_of == disclosure_date) is inclusive."""
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Boundary: as_of exactly equals disclosure_date (inclusive)
    as_of_boundary = pd.Timestamp("2025-01-08", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of_boundary, disclosure_col="disclosure_date"
    )
    
    # Must include event (disclosure_date <= as_of, with <= being inclusive)
    assert len(filtered) == 1, (
        "Boundary condition must be inclusive: disclosure_date == as_of should include event"
    )
    assert filtered["disclosure_date"].iloc[0] == as_of_boundary.normalize()


def test_pit_filtering_excludes_one_before_boundary() -> None:
    """Test that as_of one day before disclosure_date excludes event."""
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # One day before disclosure
    as_of_before = pd.Timestamp("2025-01-07", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of_before, disclosure_col="disclosure_date"
    )
    
    # Must exclude (disclosure_date > as_of)
    assert len(filtered) == 0, (
        "Event disclosed on 2025-01-08 must be excluded when as_of=2025-01-07 "
        "(one day before disclosure)"
    )


def test_pit_filtering_multi_events_partial_disclosure() -> None:
    """Test PIT filtering with multiple events at different disclosure dates."""
    events = pd.DataFrame({
        "event_date": [
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-03", tz="UTC"),
            pd.Timestamp("2025-01-05", tz="UTC"),
        ],
        "disclosure_date": [
            pd.Timestamp("2025-01-02", tz="UTC"),  # Disclosed early
            pd.Timestamp("2025-01-06", tz="UTC"),  # Disclosed later
            pd.Timestamp("2025-01-10", tz="UTC"),  # Disclosed much later
        ],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "value": [10.0, 20.0, 30.0],
    })
    
    # Filter for date 2025-01-05 (only first event disclosed)
    as_of_early = pd.Timestamp("2025-01-05", tz="UTC")
    filtered_early = filter_events_as_of(
        events, as_of_early, disclosure_col="disclosure_date"
    )
    
    # Only first event should be visible (disclosed on 2025-01-02 <= 2025-01-05)
    assert len(filtered_early) == 1, (
        "Only events with disclosure_date <= as_of should be visible"
    )
    assert filtered_early["value"].iloc[0] == 10.0
    assert filtered_early["disclosure_date"].iloc[0] == pd.Timestamp("2025-01-02", tz="UTC").normalize()
    
    # Filter for date 2025-01-07 (first two events disclosed)
    as_of_mid = pd.Timestamp("2025-01-07", tz="UTC")
    filtered_mid = filter_events_as_of(
        events, as_of_mid, disclosure_col="disclosure_date"
    )
    
    # First two events should be visible
    assert len(filtered_mid) == 2, (
        "Events with disclosure_date <= as_of should be visible"
    )
    assert set(filtered_mid["value"].tolist()) == {10.0, 20.0}
    
    # Third event must still be excluded (disclosed on 2025-01-10 > 2025-01-07)
    assert 30.0 not in filtered_mid["value"].tolist(), (
        "Event with disclosure_date > as_of must be excluded (feature blind)"
    )
    
    # Filter for date 2025-01-10 (all events disclosed)
    as_of_late = pd.Timestamp("2025-01-10", tz="UTC")
    filtered_late = filter_events_as_of(
        events, as_of_late, disclosure_col="disclosure_date"
    )
    
    # All events should be visible
    assert len(filtered_late) == 3, "All events should be visible after final disclosure"
    assert set(filtered_late["value"].tolist()) == {10.0, 20.0, 30.0}


def test_pit_filtering_multi_symbol_independent() -> None:
    """Test that PIT filtering works independently per symbol."""
    events = pd.DataFrame({
        "event_date": [
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-03", tz="UTC"),
            pd.Timestamp("2025-01-05", tz="UTC"),
        ],
        "disclosure_date": [
            pd.Timestamp("2025-01-02", tz="UTC"),  # AAPL disclosed early
            pd.Timestamp("2025-01-08", tz="UTC"),  # MSFT disclosed later
            pd.Timestamp("2025-01-02", tz="UTC"),  # GOOGL disclosed early
        ],
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "value": [100.0, 200.0, 300.0],
    })
    
    # Filter for date 2025-01-05 (AAPL and GOOGL disclosed, MSFT not)
    as_of = pd.Timestamp("2025-01-05", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of, disclosure_col="disclosure_date"
    )
    
    # AAPL and GOOGL should be visible (disclosed <= 2025-01-05)
    # MSFT should be excluded (disclosed on 2025-01-08 > 2025-01-05)
    assert len(filtered) == 2, (
        "Only symbols with disclosure_date <= as_of should be visible"
    )
    assert set(filtered["symbol"].tolist()) == {"AAPL", "GOOGL"}
    assert set(filtered["value"].tolist()) == {100.0, 300.0}
    
    # MSFT must be excluded (future disclosure)
    assert "MSFT" not in filtered["symbol"].tolist(), (
        "MSFT event with disclosure_date=2025-01-08 must be excluded when as_of=2025-01-05"
    )
    assert 200.0 not in filtered["value"].tolist(), (
        "MSFT value must not leak before disclosure (feature blind)"
    )


def test_pit_filtering_same_day_event_and_disclosure() -> None:
    """Test PIT filtering when event_date == disclosure_date (no latency)."""
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-05", tz="UTC")],  # Same day
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Filter for same day (event and disclosure on 2025-01-05)
    as_of = pd.Timestamp("2025-01-05", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of, disclosure_col="disclosure_date"
    )
    
    # Should be included (disclosure_date <= as_of, inclusive)
    assert len(filtered) == 1, (
        "Event with disclosure_date == as_of should be included (inclusive boundary)"
    )


def test_pit_filtering_excludes_before_same_day() -> None:
    """Test that as_of before same-day disclosure excludes event."""
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-05", tz="UTC")],  # Same day
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Filter for day before (2025-01-04 < 2025-01-05)
    as_of_before = pd.Timestamp("2025-01-04", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of_before, disclosure_col="disclosure_date"
    )
    
    # Must exclude (disclosure_date > as_of)
    assert len(filtered) == 0, (
        "Event disclosed on 2025-01-05 must be excluded when as_of=2025-01-04 "
        "(even if event_date is in the past)"
    )


def test_pit_filtering_time_of_day_ignored() -> None:
    """Test that time-of-day is ignored (normalized to end-of-day)."""
    # Event disclosed at 10:00 AM on 2025-01-08
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05 10:00:00", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08 10:00:00", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Filter for same date but earlier time (2025-01-08 08:00 AM)
    # After normalization, both become 2025-01-08 00:00:00, so should be included
    as_of_early_time = pd.Timestamp("2025-01-08 08:00:00", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of_early_time, disclosure_col="disclosure_date"
    )
    
    # Should be included (normalized dates are equal: 2025-01-08 <= 2025-01-08)
    assert len(filtered) == 1, (
        "Time-of-day should be ignored: disclosure_date normalized to 2025-01-08 "
        "should match as_of normalized to 2025-01-08 (inclusive)"
    )


def test_pit_filtering_strict_less_equal_semantics() -> None:
    """Test that 'disclosure_date <= as_of' semantics are strictly enforced.
    
    This test documents and verifies the strict semantics:
    - disclosure_date < as_of: INCLUDED
    - disclosure_date == as_of: INCLUDED (inclusive)
    - disclosure_date > as_of: EXCLUDED (strict)
    """
    base_date = pd.Timestamp("2025-01-10", tz="UTC")
    
    events = pd.DataFrame({
        "event_date": [base_date] * 3,
        "disclosure_date": [
            base_date - pd.Timedelta(days=1),  # 1 day before
            base_date,                          # exactly on
            base_date + pd.Timedelta(days=1),  # 1 day after
        ],
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "value": [10.0, 20.0, 30.0],
    })
    
    # Filter for base_date
    filtered = filter_events_as_of(
        events, base_date, disclosure_col="disclosure_date"
    )
    
    # Only first two should be included (disclosure_date <= base_date)
    assert len(filtered) == 2, (
        "Strict <= semantics: only events with disclosure_date <= as_of included"
    )
    assert set(filtered["symbol"].tolist()) == {"AAPL", "MSFT"}
    assert set(filtered["value"].tolist()) == {10.0, 20.0}
    
    # GOOGL must be excluded (disclosure_date > as_of, strict exclusion)
    assert "GOOGL" not in filtered["symbol"].tolist(), (
        "Event with disclosure_date > as_of must be strictly excluded "
        "(no future disclosure leakage)"
    )
    assert 30.0 not in filtered["value"].tolist(), (
        "GOOGL value must not leak before disclosure (feature blind)"
    )


def test_pit_filtering_empty_result_no_leakage() -> None:
    """Test that empty filtered result does not leak any information."""
    events = pd.DataFrame({
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })
    
    # Filter before disclosure
    as_of = pd.Timestamp("2025-01-07", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of, disclosure_col="disclosure_date"
    )
    
    # Must be completely empty (no metadata, no shape hints)
    assert filtered.empty, "Filtered result must be empty before disclosure"
    assert len(filtered) == 0, "Length must be 0"
    assert list(filtered.columns) == list(events.columns), (
        "Empty result should preserve column structure"
    )
    
    # Verify no accidental value access
    if len(filtered) == 0:
        # This is expected: accessing .iloc[0] on empty DataFrame should raise IndexError
        with pytest.raises(IndexError):
            _ = filtered["value"].iloc[0]


def test_pit_filtering_mixed_disclosure_dates_strict_order() -> None:
    """Test PIT filtering maintains strict temporal order (no future leakage)."""
    # Create events with mixed disclosure dates
    events = pd.DataFrame({
        "event_date": [
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
            pd.Timestamp("2025-01-03", tz="UTC"),
            pd.Timestamp("2025-01-04", tz="UTC"),
        ],
        "disclosure_date": [
            pd.Timestamp("2025-01-10", tz="UTC"),  # Disclosed late
            pd.Timestamp("2025-01-03", tz="UTC"),  # Disclosed early
            pd.Timestamp("2025-01-08", tz="UTC"),  # Disclosed mid
            pd.Timestamp("2025-01-05", tz="UTC"),  # Disclosed mid
        ],
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    
    # Filter for 2025-01-06 (only MSFT and TSLA disclosed)
    as_of = pd.Timestamp("2025-01-06", tz="UTC")
    filtered = filter_events_as_of(
        events, as_of, disclosure_col="disclosure_date"
    )
    
    # Only MSFT (2025-01-03) and TSLA (2025-01-05) should be visible
    assert len(filtered) == 2, "Only events with disclosure_date <= as_of visible"
    assert set(filtered["symbol"].tolist()) == {"MSFT", "TSLA"}
    assert set(filtered["value"].tolist()) == {20.0, 40.0}
    
    # AAPL and GOOGL must be excluded (future disclosures)
    excluded_symbols = {"AAPL", "GOOGL"}
    assert excluded_symbols.isdisjoint(set(filtered["symbol"].tolist())), (
        f"Symbols {excluded_symbols} with future disclosure_dates must be excluded"
    )
    excluded_values = {10.0, 30.0}
    assert excluded_values.isdisjoint(set(filtered["value"].tolist())), (
        f"Values {excluded_values} from future disclosures must not leak (feature blind)"
    )

