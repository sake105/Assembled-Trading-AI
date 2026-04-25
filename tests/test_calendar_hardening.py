"""Tests for M7-T01: Calendar hardening — filter + fallback.

Covers:
- is_weekday: weekdays, weekends
- calendar_mode: returns valid string
- is_trading_day_safe: works with and without exchange_calendars
- filter_prices_to_trading_days: removes weekends, handles missing column, empty input
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.data.calendar import (
    calendar_mode,
    filter_prices_to_trading_days,
    is_trading_day_safe,
    is_weekday,
)


# ---------------------------------------------------------------------------
# is_weekday
# ---------------------------------------------------------------------------


class TestIsWeekday:
    def test_monday_is_weekday(self):
        assert is_weekday(pd.Timestamp("2024-01-08")) is True  # Monday

    def test_friday_is_weekday(self):
        assert is_weekday(pd.Timestamp("2024-01-12")) is True  # Friday

    def test_saturday_is_not_weekday(self):
        assert is_weekday(pd.Timestamp("2024-01-13")) is False  # Saturday

    def test_sunday_is_not_weekday(self):
        assert is_weekday(pd.Timestamp("2024-01-14")) is False  # Sunday

    def test_accepts_date_object(self):
        import datetime

        assert is_weekday(datetime.date(2024, 1, 8)) is True  # Monday


# ---------------------------------------------------------------------------
# calendar_mode
# ---------------------------------------------------------------------------


class TestCalendarMode:
    def test_returns_valid_mode(self):
        mode = calendar_mode()
        assert mode in ("nyse", "fallback")


# ---------------------------------------------------------------------------
# is_trading_day_safe
# ---------------------------------------------------------------------------


class TestIsTradingDaySafe:
    def test_monday_is_trading_day(self):
        # 2024-01-08 is a regular Monday
        result = is_trading_day_safe(pd.Timestamp("2024-01-08"))
        assert result is True

    def test_saturday_is_not_trading_day(self):
        result = is_trading_day_safe(pd.Timestamp("2024-01-13"))
        assert result is False

    def test_sunday_is_not_trading_day(self):
        result = is_trading_day_safe(pd.Timestamp("2024-01-14"))
        assert result is False

    def test_returns_bool(self):
        result = is_trading_day_safe(pd.Timestamp("2024-01-08"))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# filter_prices_to_trading_days
# ---------------------------------------------------------------------------


def _make_prices_with_weekends(n_days: int = 10) -> pd.DataFrame:
    """Build a price DataFrame spanning weekdays and weekends."""
    dates = pd.date_range("2024-01-08", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * n_days,
            "close": [100.0 + i for i in range(n_days)],
        }
    )


class TestFilterPricesToTradingDays:
    def test_removes_saturday_and_sunday(self):
        prices = _make_prices_with_weekends(10)
        filtered = filter_prices_to_trading_days(prices)
        # No Saturday (weekday=5) or Sunday (weekday=6) in result
        weekdays = filtered["timestamp"].dt.weekday
        assert (weekdays < 5).all()

    def test_weekdays_preserved(self):
        prices = _make_prices_with_weekends(10)
        filtered = filter_prices_to_trading_days(prices)
        assert len(filtered) > 0
        assert len(filtered) < len(prices)  # weekends removed

    def test_empty_input_returns_empty(self):
        result = filter_prices_to_trading_days(pd.DataFrame())
        assert len(result) == 0

    def test_missing_ts_col_returns_unchanged(self):
        df = pd.DataFrame({"date": ["2024-01-08"], "close": [100.0]})
        result = filter_prices_to_trading_days(df, ts_col="timestamp")
        # Should return unchanged (column missing)
        assert len(result) == len(df)

    def test_result_is_copy(self):
        prices = _make_prices_with_weekends(10)
        filtered = filter_prices_to_trading_days(prices)
        # Modifying filtered should not affect original
        if len(filtered) > 0:
            filtered.iloc[0, filtered.columns.get_loc("close")] = -999.0
            assert prices["close"].iloc[0] != -999.0

    def test_all_weekdays_input_unchanged_length(self):
        # Input with only weekdays → nothing filtered out
        dates = pd.date_range(
            "2024-01-08", periods=5, freq="B", tz="UTC"
        )  # business days
        df = pd.DataFrame({"timestamp": dates, "close": [100.0] * 5})
        result = filter_prices_to_trading_days(df)
        assert len(result) == 5

    def test_custom_ts_col(self):
        dates = pd.date_range("2024-01-08", periods=7, freq="D", tz="UTC")
        df = pd.DataFrame({"ts": dates, "close": [1.0] * 7})
        result = filter_prices_to_trading_days(df, ts_col="ts")
        # Weekends removed
        assert len(result) == 5

    def test_none_input_returns_empty_df(self):
        result = filter_prices_to_trading_days(None)  # type: ignore[arg-type]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
