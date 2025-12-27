"""Tests for Paper Track risk report triggers."""

from __future__ import annotations


import pandas as pd

from scripts.run_paper_track import (
    _is_friday,
    _is_month_end,
    _should_generate_risk_report,
)


def test_is_friday() -> None:
    """Test _is_friday helper function."""
    # 2025-01-17 is a Friday
    assert _is_friday(pd.Timestamp("2025-01-17", tz="UTC")) is True
    # 2025-01-13 is a Monday
    assert _is_friday(pd.Timestamp("2025-01-13", tz="UTC")) is False
    # 2025-01-14 is a Tuesday
    assert _is_friday(pd.Timestamp("2025-01-14", tz="UTC")) is False


def test_is_month_end() -> None:
    """Test _is_month_end helper function."""
    # 2025-01-31 is month-end
    assert _is_month_end(pd.Timestamp("2025-01-31", tz="UTC")) is True
    # 2025-01-15 is mid-month
    assert _is_month_end(pd.Timestamp("2025-01-15", tz="UTC")) is False
    # 2025-02-28 is month-end (2025 is not a leap year)
    assert _is_month_end(pd.Timestamp("2025-02-28", tz="UTC")) is True
    # 2024-02-29 is month-end (2024 is a leap year)
    assert _is_month_end(pd.Timestamp("2024-02-29", tz="UTC")) is True


def test_should_generate_risk_report_daily() -> None:
    """Test risk report trigger for daily frequency."""
    # Daily should always return True
    date = pd.Timestamp("2025-01-13", tz="UTC")  # Monday
    assert _should_generate_risk_report(date, "daily") is True

    date = pd.Timestamp("2025-01-17", tz="UTC")  # Friday
    assert _should_generate_risk_report(date, "daily") is True


def test_should_generate_risk_report_weekly() -> None:
    """Test risk report trigger for weekly frequency."""
    # Friday should trigger
    date = pd.Timestamp("2025-01-17", tz="UTC")  # Friday
    assert _should_generate_risk_report(date, "weekly") is True

    # Monday should not trigger
    date = pd.Timestamp("2025-01-13", tz="UTC")  # Monday
    assert _should_generate_risk_report(date, "weekly") is False

    # Saturday with is_last_day_in_range should trigger
    date = pd.Timestamp("2025-01-18", tz="UTC")  # Saturday
    assert _should_generate_risk_report(date, "weekly", is_last_day_in_range=True) is True

    # Friday with is_last_day_in_range should trigger
    date = pd.Timestamp("2025-01-17", tz="UTC")  # Friday
    assert _should_generate_risk_report(date, "weekly", is_last_day_in_range=True) is True


def test_should_generate_risk_report_monthly() -> None:
    """Test risk report trigger for monthly frequency."""
    # Month-end should trigger
    date = pd.Timestamp("2025-01-31", tz="UTC")  # Month-end
    assert _should_generate_risk_report(date, "monthly") is True

    # Mid-month should not trigger
    date = pd.Timestamp("2025-01-15", tz="UTC")  # Mid-month
    assert _should_generate_risk_report(date, "monthly") is False

    # Any date with is_last_day_in_range should trigger
    date = pd.Timestamp("2025-01-15", tz="UTC")  # Mid-month
    assert _should_generate_risk_report(date, "monthly", is_last_day_in_range=True) is True

