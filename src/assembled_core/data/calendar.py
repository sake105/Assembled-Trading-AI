"""Trading calendar utilities (NYSE-based).

Wraps exchange_calendars for session lookups.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import exchange_calendars as xcals
    _NYSE = xcals.get_calendar("XNYS")
except Exception:
    _NYSE = None
    logger.warning("exchange_calendars not available – calendar helpers degraded")


def get_nyse_calendar():
    """Return the cached NYSE calendar instance."""
    if _NYSE is None:
        raise RuntimeError("exchange_calendars not installed")
    return _NYSE


def is_trading_day(dt: date | pd.Timestamp) -> bool:
    """Check whether *dt* is a valid NYSE trading session."""
    cal = get_nyse_calendar()
    ts = pd.Timestamp(dt)
    return cal.is_session(ts)


def session_close_utc(dt: date | pd.Timestamp) -> pd.Timestamp:
    """Return the UTC close time for the NYSE session on *dt*."""
    cal = get_nyse_calendar()
    ts = pd.Timestamp(dt)
    return cal.session_close(ts).tz_convert("UTC")


def trading_sessions(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return NYSE sessions between *start* and *end* (inclusive)."""
    cal = get_nyse_calendar()
    return cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def normalize_as_of_to_session_close(as_of: pd.Timestamp) -> pd.Timestamp:
    """Snap *as_of* to the previous session close if it falls outside hours."""
    cal = get_nyse_calendar()
    sess = cal.previous_close(as_of)
    return sess.tz_convert("UTC") if sess.tzinfo is None else sess
