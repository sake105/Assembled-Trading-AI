"""Missing trading-day detection. From 37_DATA_QUALITY_GATE.md §3.2."""
from __future__ import annotations

import pandas as pd


def detect_missing_trading_days(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    expected_market_calendar: str = "NYSE",
    max_missing_frac: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame of (ticker, missing_date) pairs.

    Falls back to simple BDay detection if pandas_market_calendars unavailable.
    Only reports tickers whose missing fraction exceeds max_missing_frac to
    avoid noise from newly listed or delisted tickers.
    """
    try:
        import pandas_market_calendars as mcal  # optional dep
        _use_mcal = True
    except ImportError:
        _use_mcal = False

    missing_events: list[dict] = []

    for ticker, group in df.groupby(ticker_col, sort=False):
        group = group.sort_values(timestamp_col)
        start = group[timestamp_col].min()
        end = group[timestamp_col].max()

        if _use_mcal:
            cal = mcal.get_calendar(expected_market_calendar)
            schedule = cal.schedule(
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            expected_days = set(schedule.index.normalize().date)
        else:
            idx = pd.bdate_range(start=start, end=end)
            expected_days = set(d.date() for d in idx)

        actual_days: set = set()
        ts_col = group[timestamp_col]
        if hasattr(ts_col.dt, "date"):
            actual_days = set(ts_col.dt.date)
        else:
            actual_days = set(pd.to_datetime(ts_col).dt.date)

        missing = expected_days - actual_days
        if not missing:
            continue

        missing_frac = len(missing) / max(len(expected_days), 1)
        if missing_frac > max_missing_frac:
            for d in sorted(missing):
                missing_events.append({
                    "ticker": ticker,
                    "missing_date": d,
                    "reason": "missing_trading_day",
                    "missing_frac": round(missing_frac, 4),
                })

    return pd.DataFrame(missing_events)
