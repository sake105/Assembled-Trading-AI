# tests/test_time_and_calendar_nyse.py
"""Tests for NYSE calendar utilities (DST + Holidays).

This test suite verifies:
1. Weekend is not a trading day
2. Holidays are not trading days
3. DST handling (Winter: 21:00 UTC, Summer: 20:00 UTC)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.calendar import (
    get_nyse_calendar,
    is_trading_day,
    normalize_as_of_to_session_close,
    session_close_utc,
    trading_sessions,
)


@pytest.fixture
def nyse_cal():
    """Get NYSE calendar (skip test if exchange_calendars not installed)."""
    try:
        return get_nyse_calendar()
    except ImportError:
        pytest.skip("exchange_calendars not installed")


def test_weekend_not_trading_day(nyse_cal):
    """Test that weekend is not a trading day."""
    # Saturday
    assert not is_trading_day(date(2024, 1, 6)), "Saturday should not be a trading day"
    # Sunday
    assert not is_trading_day(date(2024, 1, 7)), "Sunday should not be a trading day"


def test_holiday_not_trading_day(nyse_cal):
    """Test that holidays are not trading days.

    Using stable holidays that are unlikely to change:
    - 2024-01-01 (New Year's Day)
    - 2024-12-25 (Christmas)
    """
    # New Year's Day
    assert not is_trading_day(date(2024, 1, 1)), "New Year's Day should not be a trading day"
    # Christmas
    assert not is_trading_day(date(2024, 12, 25)), "Christmas should not be a trading day"


def test_regular_trading_day(nyse_cal):
    """Test that regular weekdays are trading days."""
    # Tuesday
    assert is_trading_day(date(2024, 1, 2)), "Regular weekday should be a trading day"
    # Wednesday
    assert is_trading_day(date(2024, 1, 3)), "Regular weekday should be a trading day"


def test_session_close_utc_winter(nyse_cal):
    """Test session close in UTC during winter (EST, UTC-5).

    Winter: 16:00 ET = 21:00 UTC
    Using February 2024 (Standard Time).
    """
    # February 1, 2024 (Thursday, trading day)
    close = session_close_utc(date(2024, 2, 1))

    # Verify: 21:00 UTC (16:00 ET in winter)
    assert close.hour == 21, f"Winter session close should be 21:00 UTC, got {close.hour}"
    assert close.minute == 0, f"Session close should be on the hour, got {close.minute}"
    assert close.tz.zone == "UTC", f"Session close should be UTC, got {close.tz}"
    assert close.date() == date(2024, 2, 1), f"Session close date should match, got {close.date()}"


def test_session_close_utc_summer(nyse_cal):
    """Test session close in UTC during summer (EDT, UTC-4).

    Summer: 16:00 ET = 20:00 UTC
    Using June 2024 (Daylight Time).
    """
    # June 1, 2024 (Saturday, but we'll use June 3, 2024 which is Monday)
    # Actually, let's use a trading day in June
    # June 3, 2024 is Monday (trading day)
    close = session_close_utc(date(2024, 6, 3))

    # Verify: 20:00 UTC (16:00 ET in summer)
    assert close.hour == 20, f"Summer session close should be 20:00 UTC, got {close.hour}"
    assert close.minute == 0, f"Session close should be on the hour, got {close.minute}"
    assert close.tz.zone == "UTC", f"Session close should be UTC, got {close.tz}"
    assert close.date() == date(2024, 6, 3), f"Session close date should match, got {close.date()}"


def test_session_close_utc_holiday_raises(nyse_cal):
    """Test that session_close_utc raises ValueError for holidays."""
    # New Year's Day (holiday)
    with pytest.raises(ValueError, match="not a NYSE trading day"):
        session_close_utc(date(2024, 1, 1))


def test_normalize_as_of_to_session_close_date_string(nyse_cal):
    """Test normalize_as_of_to_session_close with date string."""
    # Winter date
    as_of = normalize_as_of_to_session_close("2024-02-01")
    assert as_of.hour == 21, f"Winter session close should be 21:00 UTC, got {as_of.hour}"
    assert as_of.tz.zone == "UTC", f"Should be UTC, got {as_of.tz}"

    # Summer date
    as_of = normalize_as_of_to_session_close("2024-06-03")
    assert as_of.hour == 20, f"Summer session close should be 20:00 UTC, got {as_of.hour}"
    assert as_of.tz.zone == "UTC", f"Should be UTC, got {as_of.tz}"


def test_normalize_as_of_to_session_close_timestamp(nyse_cal):
    """Test normalize_as_of_to_session_close with timestamp (extracts date, normalizes to close)."""
    # Winter: arbitrary timestamp -> normalized to session close
    input_ts = pd.Timestamp("2024-02-01 12:00:00", tz="UTC")
    as_of = normalize_as_of_to_session_close(input_ts)

    # Should be normalized to session close (21:00 UTC in winter)
    assert as_of.hour == 21, f"Should normalize to 21:00 UTC, got {as_of.hour}"
    assert as_of.date() == date(2024, 2, 1), f"Date should match, got {as_of.date()}"
    assert as_of.tz.zone == "UTC", f"Should be UTC, got {as_of.tz}"

    # Summer: arbitrary timestamp -> normalized to session close
    input_ts = pd.Timestamp("2024-06-03 12:00:00", tz="UTC")
    as_of = normalize_as_of_to_session_close(input_ts)

    # Should be normalized to session close (20:00 UTC in summer)
    assert as_of.hour == 20, f"Should normalize to 20:00 UTC, got {as_of.hour}"
    assert as_of.date() == date(2024, 6, 3), f"Date should match, got {as_of.date()}"


def test_normalize_as_of_to_session_close_date_object(nyse_cal):
    """Test normalize_as_of_to_session_close with date object."""
    # Winter
    as_of = normalize_as_of_to_session_close(date(2024, 2, 1))
    assert as_of.hour == 21, f"Winter session close should be 21:00 UTC, got {as_of.hour}"

    # Summer
    as_of = normalize_as_of_to_session_close(date(2024, 6, 3))
    assert as_of.hour == 20, f"Summer session close should be 20:00 UTC, got {as_of.hour}"


def test_normalize_as_of_to_session_close_holiday_raises(nyse_cal):
    """Test that normalize_as_of_to_session_close raises ValueError for holidays."""
    # New Year's Day (holiday)
    with pytest.raises(ValueError, match="not a NYSE trading day"):
        normalize_as_of_to_session_close("2024-01-01")


def test_trading_sessions_range(nyse_cal):
    """Test trading_sessions returns correct range of trading days."""
    # January 2024: 1-31 (excluding weekends and holidays)
    sessions = trading_sessions(date(2024, 1, 1), date(2024, 1, 31))

    # Should be a DatetimeIndex
    assert isinstance(sessions, pd.DatetimeIndex), f"Should return DatetimeIndex, got {type(sessions)}"

    # Should exclude weekends and holidays
    # January 1, 2024 is New Year's Day (holiday) -> excluded
    # January 6-7, 2024 is weekend -> excluded
    # etc.

    # Verify all returned dates are trading days
    for session_date in sessions:
        assert is_trading_day(session_date.date()), f"{session_date.date()} should be a trading day"

    # Verify no weekends
    for session_date in sessions:
        weekday = session_date.weekday()  # 0=Monday, 6=Sunday
        assert weekday < 5, f"{session_date.date()} should not be weekend (weekday={weekday})"


def test_dst_transition_winter_to_summer(nyse_cal):
    """Test DST transition: Winter (EST) to Summer (EDT).

    DST typically starts in March and ends in November.
    Test dates before and after DST transition.
    """
    # Before DST (February, EST, UTC-5)
    close_feb = session_close_utc(date(2024, 2, 1))
    assert close_feb.hour == 21, "February should be 21:00 UTC (EST)"

    # After DST (June, EDT, UTC-4)
    close_jun = session_close_utc(date(2024, 6, 3))
    assert close_jun.hour == 20, "June should be 20:00 UTC (EDT)"


def test_dst_transition_summer_to_winter(nyse_cal):
    """Test DST transition: Summer (EDT) to Winter (EST).

    DST typically ends in November.
    Test dates before and after DST end.
    """
    # Before DST end (October, EDT, UTC-4)
    close_oct = session_close_utc(date(2024, 10, 1))
    assert close_oct.hour == 20, "October should be 20:00 UTC (EDT)"

    # After DST end (December, EST, UTC-5)
    close_dec = session_close_utc(date(2024, 12, 2))
    assert close_dec.hour == 21, "December should be 21:00 UTC (EST)"
