"""US equity market calendar utilities (Pre-Flight Check 5).

Provides is_trading_day() without requiring pandas_market_calendars.
Covers NYSE holidays through 2027.
"""

from __future__ import annotations

import pandas as pd

# NYSE observed holidays (static list, updated through 2027)
# Format: "YYYY-MM-DD"
_NYSE_HOLIDAYS: frozenset[str] = frozenset(
    [
        # 2020
        "2020-01-01",
        "2020-01-20",
        "2020-02-17",
        "2020-04-10",
        "2020-05-25",
        "2020-07-03",
        "2020-09-07",
        "2020-11-26",
        "2020-12-25",
        # 2021
        "2021-01-01",
        "2021-01-18",
        "2021-02-15",
        "2021-04-02",
        "2021-05-31",
        "2021-07-05",
        "2021-09-06",
        "2021-11-25",
        "2021-12-24",
        # 2022
        "2022-01-17",
        "2022-02-21",
        "2022-04-15",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-12-26",
        # 2023
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-04-07",
        "2023-05-29",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2023-12-25",
        # 2024
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-03-29",
        "2024-05-27",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2024-12-25",
        # 2025
        "2025-01-01",
        "2025-01-09",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
        # 2026
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
        "2026-05-25",
        "2026-07-03",
        "2026-09-07",
        "2026-11-26",
        "2026-12-25",
        # 2027
        "2027-01-01",
        "2027-01-18",
        "2027-02-15",
        "2027-03-26",
        "2027-05-31",
        "2027-07-05",
        "2027-09-06",
        "2027-11-25",
        "2027-12-24",
    ]
)

# NYSE early-close days (1:00 PM ET close = 18:00 UTC)
_NYSE_EARLY_CLOSE: frozenset[str] = frozenset(
    [
        "2024-07-03",
        "2024-11-29",
        "2024-12-24",
        "2025-07-03",
        "2025-11-28",
        "2025-12-24",
        "2026-11-27",
        "2026-12-24",
    ]
)


def is_trading_day(date: pd.Timestamp | str) -> bool:
    """Return True if NYSE is open on *date* (ignores time-of-day).

    Args:
        date: Any timestamp or date string parseable by pandas.

    Returns:
        True if date is a weekday and not a NYSE holiday.
    """
    ts = pd.Timestamp(date)
    if ts.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return ts.strftime("%Y-%m-%d") not in _NYSE_HOLIDAYS


def is_early_close(date: pd.Timestamp | str) -> bool:
    """Return True if NYSE closes at 1 PM ET on *date*."""
    ts = pd.Timestamp(date)
    return ts.strftime("%Y-%m-%d") in _NYSE_EARLY_CLOSE


def next_trading_day(date: pd.Timestamp | str) -> pd.Timestamp:
    """Return the next NYSE trading day after *date*."""
    ts = pd.Timestamp(date) + pd.Timedelta(days=1)
    while not is_trading_day(ts):
        ts += pd.Timedelta(days=1)
    return ts


def trading_days_between(start: pd.Timestamp | str, end: pd.Timestamp | str) -> int:
    """Count NYSE trading days in [start, end] inclusive."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    count = 0
    cur = s
    while cur <= e:
        if is_trading_day(cur):
            count += 1
        cur += pd.Timedelta(days=1)
    return count
