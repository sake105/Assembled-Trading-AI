"""Centralized date/time format constants.

All strftime/strptime calls should reference these constants for consistency.
"""

from __future__ import annotations

DATE_FMT = "%Y-%m-%d"  # ISO date (2026-05-07)
DATETIME_FMT = "%Y-%m-%dT%H:%M:%SZ"  # ISO datetime UTC (2026-05-07T14:30:00Z)
DATETIME_LOCAL_FMT = "%Y-%m-%dT%H:%M:%S%z"  # ISO datetime with tz offset
COMPACT_DATE_FMT = "%Y%m%d"  # compact date (20260507) for file names
LOG_FMT = "%Y-%m-%d %H:%M:%S"  # human-readable log timestamp

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
