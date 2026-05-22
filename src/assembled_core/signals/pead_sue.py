"""Post-Earnings-Announcement-Drift (PEAD) via Standardized Unexpected Earnings (SUE).

From 13_FREE_MODULE.md §13.6.
Bernard-Thomas 1989, rezent 5-8% annualized. Decile long-short on SUE.

Datenquelle: Finnhub /stock/earnings (free tier).
SUE = (actual - estimate) / σ(historical_surprises, trailing 8 quarters)

Pre-trade safety: block or reduce size if days_to_earnings <= 2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sue(
    ticker: str,
    finnhub_client: Any,
    lookback_quarters: int = 8,
) -> float:
    """Compute Standardized Unexpected Earnings for a ticker via live Finnhub fetch.

    Args:
        ticker: Stock ticker symbol
        finnhub_client: finnhub.Client instance
        lookback_quarters: Number of past quarters to compute surprise std

    Returns:
        SUE score. Positive = positive surprise, Negative = miss.
        Returns NaN if insufficient data.

    Note:
        This function fetches live data from the Finnhub API.
        For offline/research-grade SUE with an explicit expected-EPS model
        (seasonal RW, Foster, or external IBES consensus), use
        ``features.pead_sue.compute_sue`` or ``features.pead_sue.compute_sue_from_expected``.
    """
    try:
        earnings = finnhub_client.company_earnings(ticker, limit=lookback_quarters + 1)
    except Exception as exc:
        logger.debug("Finnhub company_earnings failed for %s: %s", ticker, exc)
        return float("nan")

    if not earnings or len(earnings) < 2:
        return float("nan")

    surprises = []
    for e in earnings:
        actual = e.get("actual")
        estimate = e.get("estimate")
        if actual is not None and estimate is not None:
            surprises.append(actual - estimate)

    if len(surprises) < 2:
        return float("nan")

    latest = surprises[0]
    historical_std = float(np.std(surprises[1:]))
    if historical_std < 1e-9:
        return float("nan")

    return float(latest / historical_std)


def batch_sue(
    tickers: list[str],
    finnhub_client: Any,
    lookback_quarters: int = 8,
) -> pd.Series:
    """Compute SUE for a batch of tickers.

    Returns Series indexed by ticker.
    """
    scores = {}
    for ticker in tickers:
        scores[ticker] = compute_sue(ticker, finnhub_client, lookback_quarters)
    return pd.Series(scores, name="sue")


def pre_trade_earnings_check(
    ticker: str,
    finnhub_client: Any,
    days_threshold: int = 2,
    today: "datetime.date | None" = None,  # noqa: F821 — forward ref
) -> bool:
    """Return True if earnings within days_threshold — reduce or block position.

    F-B-11 MAJOR fix: added explicit `today` parameter so backtest callers can
    pass `today=as_of.date()` for PIT-safety. Previously used `datetime.date.today()`
    unconditionally, which leaks future earnings calendar data into backtests.
    Live/paper callers can omit `today` and the function defaults to UTC today.
    """
    import datetime

    try:
        cal = finnhub_client.earnings_calendar(
            _from="", to="", symbol=ticker, international=False
        )
        if not cal or "earningsCalendar" not in cal:
            return False
        items = cal["earningsCalendar"]
        if not items:
            return False

        if today is None:
            today = datetime.datetime.now(tz=datetime.timezone.utc).date()
        for item in items[:3]:
            date_str = item.get("date", "")
            if not date_str:
                continue
            try:
                event_date = datetime.date.fromisoformat(date_str)
                delta = (event_date - today).days
                if 0 <= delta <= days_threshold:
                    return True
            except ValueError:
                continue
    except Exception as exc:
        logger.debug("Earnings check failed for %s: %s", ticker, exc)
    return False


__all__ = [
    "compute_sue",
    "batch_sue",
    "pre_trade_earnings_check",
]
