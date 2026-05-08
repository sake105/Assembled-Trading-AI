"""Trading calendar utilities (NYSE-based).

Wraps exchange_calendars for session lookups.
Falls back to a pure-Python weekday approximation when exchange_calendars
is not installed or unavailable (M7 hardening).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import exchange_calendars as xcals

    _NYSE = xcals.get_calendar("XNYS")
    _CALENDAR_MODE = "nyse"
except Exception:
    _NYSE = None
    _CALENDAR_MODE = "fallback"
    logger.warning("exchange_calendars not available – using weekday fallback")


def get_nyse_calendar():
    """Return the cached NYSE calendar instance."""
    if _NYSE is None:
        raise RuntimeError("exchange_calendars not installed")
    return _NYSE


def _to_naive(dt: date | pd.Timestamp) -> pd.Timestamp:
    """Convert to timezone-naive date-only timestamp for exchange_calendars."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def is_trading_day(dt: date | pd.Timestamp) -> bool:
    """Check whether *dt* is a valid NYSE trading session."""
    cal = get_nyse_calendar()
    return cal.is_session(_to_naive(dt))


def session_close_utc(dt: date | pd.Timestamp) -> pd.Timestamp:
    """Return the UTC close time for the NYSE session on *dt*.

    Raises:
        ValueError: If *dt* is not a NYSE trading day.
    """
    cal = get_nyse_calendar()
    naive = _to_naive(dt)
    if not cal.is_session(naive):
        raise ValueError(f"not a NYSE trading day: {dt}")
    return cal.session_close(naive)


def trading_sessions(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return NYSE sessions between *start* and *end* (inclusive)."""
    cal = get_nyse_calendar()
    return cal.sessions_in_range(_to_naive(start), _to_naive(end))


def normalize_as_of_to_session_close(as_of: pd.Timestamp) -> pd.Timestamp:
    """Return the session close for the trading day of *as_of*.

    Raises:
        ValueError: If the date of *as_of* is not a NYSE trading day.
    """
    return session_close_utc(as_of)


# ---------------------------------------------------------------------------
# Fallback calendar (pure-Python weekday check, no exchange_calendars required)
# ---------------------------------------------------------------------------


def is_weekday(dt: date | pd.Timestamp) -> bool:
    """Return True if *dt* is a weekday (Mon–Fri), ignoring holidays.

    Used as a fallback when exchange_calendars is not installed.
    Less precise than is_trading_day — does not know about market holidays.
    """
    return pd.Timestamp(dt).weekday() < 5  # 0=Mon … 4=Fri


def calendar_mode() -> str:
    """Return the active calendar mode: 'nyse' or 'fallback'."""
    return _CALENDAR_MODE


def is_market_open_now(as_of: datetime | None = None) -> bool:
    """Return True if NYSE is currently in a regular trading session.

    DST-safe: uses 'America/New_York' zoneinfo, so spring-forward / fall-back
    transitions are handled automatically regardless of the server's local timezone.

    Covers regular session only: 09:30–16:00 ET on a trading day.
    Early-close days (e.g. day before Thanksgiving) are handled via the
    exchange_calendars session data when available.

    Args:
        as_of: UTC datetime to check (defaults to now). Passing a non-UTC
            tz-aware datetime is supported — it is converted to UTC first.

    Returns:
        True during regular NYSE session, False otherwise.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
        except ImportError:
            ZoneInfo = None  # type: ignore[misc, assignment]

    if as_of is None:
        as_of = datetime.now(tz=timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # Convert to NYSE local time (handles US DST correctly)
    if ZoneInfo is not None:
        nyse_tz = ZoneInfo("America/New_York")
        local_dt = as_of.astimezone(nyse_tz)
    else:
        # ZoneInfo unavailable — rough UTC-4/UTC-5 offset; less precise during DST transitions
        logger.warning(
            "[calendar] zoneinfo not available — DST handling is approximate (±1h risk)"
        )
        utc_offset_hours = -4  # assume EDT (DST active); will be wrong for EST period
        from datetime import timedelta

        local_dt = as_of + timedelta(hours=utc_offset_hours)

    session_open = time(9, 30)
    session_close = time(16, 0)
    local_time = local_dt.time()

    if not (session_open <= local_time < session_close):
        return False

    # Check it's a trading day (not weekend or holiday)
    return is_trading_day_safe(local_dt.date())


def minutes_to_market_open(as_of: datetime | None = None) -> int | None:
    """Return minutes until next NYSE open, or None if market is open now.

    Useful for scheduling: sleep until NYSE opens.
    Returns 0 if market is currently open.
    """
    if as_of is None:
        as_of = datetime.now(tz=timezone.utc)

    if is_market_open_now(as_of):
        return 0

    # Find next session open
    if _NYSE is not None:
        try:
            ts = pd.Timestamp(as_of, tz="UTC")
            # Get next session after current time
            next_open = _NYSE.next_open(ts)
            delta = next_open - ts
            return max(0, int(delta.total_seconds() / 60))
        except Exception as exc:
            logger.debug("[calendar] minutes_to_market_open fallback: %s", exc)

    return None


def is_trading_day_safe(dt: date | pd.Timestamp) -> bool:
    """Check whether *dt* is a trading day, with automatic fallback.

    Uses the full NYSE calendar when exchange_calendars is available,
    otherwise falls back to weekday check (no holiday awareness).

    Args:
        dt: Date or timestamp to check.

    Returns:
        True if the date is a trading day (or probable trading day in fallback mode).
    """
    if _NYSE is not None:
        return is_trading_day(dt)
    return is_weekday(dt)


# ---------------------------------------------------------------------------
# Price DataFrame filtering
# ---------------------------------------------------------------------------


def filter_prices_to_trading_days(
    prices: pd.DataFrame,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Filter a price DataFrame to rows that fall on trading days.

    Uses the full NYSE calendar when exchange_calendars is available,
    otherwise falls back to weekday filtering.

    Args:
        prices: DataFrame with a timestamp column.
        ts_col: Name of the timestamp column (default: "timestamp").

    Returns:
        Filtered DataFrame (copy). Preserves original index.
        Returns *prices* unchanged if ts_col is missing or prices is empty.
    """
    if prices is None or prices.empty:
        return prices if prices is not None else pd.DataFrame()
    if ts_col not in prices.columns:
        logger.warning(
            "filter_prices_to_trading_days: column '%s' not found — returning unchanged",
            ts_col,
        )
        return prices

    ts = pd.to_datetime(prices[ts_col], utc=True, errors="coerce")

    if _CALENDAR_MODE == "nyse" and _NYSE is not None:
        # Vectorized path: build a set of valid NYSE session dates and do one
        # membership test instead of one Python function call per row.
        valid_ts = ts.dropna()
        if valid_ts.empty:
            return prices[pd.Series(False, index=prices.index)].copy()
        min_date = _to_naive(valid_ts.min())
        max_date = _to_naive(valid_ts.max())
        try:
            sessions = _NYSE.sessions_in_range(min_date, max_date)
            valid_dates = set(sessions.normalize().tz_localize(None).date)
            mask = (
                ts.dt.tz_convert(None)
                .dt.normalize()
                .dt.date.map(lambda d: d in valid_dates if d is not None else False)
            )
        except Exception as _cal_exc:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "[calendar] vectorized trading-day filter failed (%s) — falling back to per-row O(n) check",
                _cal_exc,
            )
            mask = ts.apply(
                lambda t: is_trading_day_safe(t) if not pd.isna(t) else False
            )
    else:
        # Fallback: weekday filter (Mon–Fri)
        mask = ts.dt.dayofweek < 5

    return prices[mask].copy()


# ---------------------------------------------------------------------------
# Calendar Validation (Plan 10.3)
# ---------------------------------------------------------------------------


def validate_dates_against_calendar(
    dates: pd.DatetimeIndex | list,
    tolerance_missing_pct: float = 5.0,
) -> dict:
    """Validate a set of dates against the NYSE trading calendar.

    Checks for:
    - Non-trading-day entries (weekends, holidays present in data)
    - Missing trading days (gaps in expected sessions)

    Args:
        dates: Dates to validate.
        tolerance_missing_pct: Alert threshold for missing session %.

    Returns:
        Dict with n_dates, n_non_trading, non_trading_dates,
        n_missing_sessions, missing_pct, valid.
    """
    dates = pd.DatetimeIndex(dates)
    if dates.empty:
        return {
            "n_dates": 0,
            "valid": True,
            "n_non_trading": 0,
            "n_missing_sessions": 0,
        }

    date_set = set(dates.normalize().date)

    # Check for non-trading days in data
    non_trading = []
    for d in sorted(date_set):
        if not is_trading_day_safe(d):
            non_trading.append(str(d))

    # Check for missing trading sessions
    min_d = min(date_set)
    max_d = max(date_set)
    expected = trading_sessions(str(min_d), str(max_d))
    expected_set = set(expected)
    missing = expected_set - date_set
    missing_pct = len(missing) / len(expected_set) * 100 if expected_set else 0.0

    return {
        "n_dates": len(date_set),
        "n_non_trading": len(non_trading),
        "non_trading_dates": non_trading[:10],  # sample
        "n_expected_sessions": len(expected_set),
        "n_missing_sessions": len(missing),
        "missing_pct": round(missing_pct, 2),
        "valid": len(non_trading) == 0 and missing_pct <= tolerance_missing_pct,
    }


__all__ = [
    "is_trading_day",
    "is_trading_day_safe",
    "is_weekday",
    "is_market_open_now",
    "minutes_to_market_open",
    "session_close_utc",
    "trading_sessions",
    "calendar_mode",
    "filter_prices_to_trading_days",
    "normalize_as_of_to_session_close",
    "get_nyse_calendar",
]
