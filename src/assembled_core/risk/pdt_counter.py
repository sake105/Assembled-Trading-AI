"""Pattern Day Trader (PDT) rule enforcement.

US FINRA rule: account < $25k equity cannot make 4+ round-trip day trades
within any rolling 5-trading-day window. Violation = account frozen 90 days.

Usage: Call record_day_trade() after each same-day round-trip fill.
       Check is_pdt_at_risk() before submitting new day-trade orders.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Deque

try:
    from zoneinfo import ZoneInfo

    _NY_TZ: ZoneInfo | None = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _NY_TZ = None


def _us_market_today() -> date:
    """Return today in America/New_York (or UTC fallback).

    F-C2-R2-2 Round-2 fix: sister to F-C-4 (compliance/pdt.py). PDT is a US
    market rule; date.today() local-tz produces wrong rolling-window dates on
    a CET box between 18:00 CET and midnight.
    """
    if _NY_TZ is not None:
        return datetime.now(tz=_NY_TZ).date()
    return datetime.now(tz=timezone.utc).date()


log = logging.getLogger(__name__)

PDT_EQUITY_THRESHOLD = 25_000.0  # USD — below this the rule applies
PDT_MAX_TRADES = 3  # 4th same-day round-trip triggers PDT
PDT_WINDOW_DAYS = 7  # rolling calendar days (covers 5 trading days incl. weekend)


class PDTCounter:
    """Thread-safe rolling PDT trade counter.

    Records day-trade events and enforces the 3-in-5-days limit
    when account equity is below PDT_EQUITY_THRESHOLD.
    """

    def __init__(self, equity_threshold: float = PDT_EQUITY_THRESHOLD) -> None:
        self._threshold = equity_threshold
        self._trades: Deque[date] = deque()

    def _evict_old(self, today: date) -> None:
        cutoff = today - timedelta(days=PDT_WINDOW_DAYS - 1)
        while self._trades and self._trades[0] < cutoff:
            self._trades.popleft()

    def record_day_trade(self, trade_date: date | None = None) -> None:
        """Record one day-trade round-trip."""
        d = trade_date or _us_market_today()
        self._evict_old(d)
        self._trades.append(d)
        log.info(
            "[pdt] recorded day-trade on %s (window total: %d)", d, len(self._trades)
        )

    def count_in_window(self, today: date | None = None) -> int:
        d = today or _us_market_today()
        self._evict_old(d)
        return len(self._trades)

    def is_pdt_at_risk(self, account_equity: float, today: date | None = None) -> bool:
        """Return True if next day-trade would trigger PDT violation."""
        if account_equity >= self._threshold:
            return False
        count = self.count_in_window(today)
        at_risk = count >= PDT_MAX_TRADES
        if at_risk:
            log.warning(
                "[pdt] RISK: %d day-trades in window, equity $%.0f < $%.0f threshold — "
                "next round-trip would trigger PDT violation",
                count,
                account_equity,
                self._threshold,
            )
        return at_risk

    def reset(self) -> None:
        self._trades.clear()
