"""Pattern Day Trader (PDT) rule tracking and pre-order check.

From 41_PDT_REGEL_INTRADAY_MARGIN.md.

Timeline:
  - Until 2026-06-03 (Effective Date of FINRA 26-10): old PDT rule
  - After broker migration (Alpaca: TBD, no later than 2027-10-20): new intraday margin
  - Feature flag PDT_RULE_ACTIVE controls which path is active

PDT definition (FINRA Rule 4210, until cutover):
  Account flagged as PDT when ≥4 day-trades within 5 business days
  AND those day-trades > 6 % of total trades in the period.
  A "day-trade" = open AND close of the same symbol on the same calendar day.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Feature flag: set PDT_RULE_ACTIVE=false after Alpaca confirms migration
_PDT_ACTIVE = os.environ.get("PDT_RULE_ACTIVE", "true").lower() not in (
    "false",
    "0",
    "no",
)

# Equity threshold below which PDT rules apply
_PDT_EQUITY_THRESHOLD = 25_000.0


class DayTradeRecord(NamedTuple):
    symbol: str
    trade_date: date
    open_side: str  # 'buy' or 'sell_short'


class PDTCounter:
    """Tracks day-trades in a rolling 5-business-day window.

    A day-trade is counted when a position opened and closed on the
    same calendar day.  The counter uses a deque capped at recent records,
    not a database — for production, wire ``add_day_trade`` to persist
    fills and reconstruct on restart.
    """

    _WINDOW_DAYS = 5  # rolling business-day window
    _MAX_DAY_TRADES = 3  # 4th trade would trigger PDT flag → block at 3

    def __init__(self) -> None:
        self._records: deque[DayTradeRecord] = deque(maxlen=500)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_day_trade(
        self, symbol: str, trade_date: date | None = None, open_side: str = "buy"
    ) -> None:
        """Record a completed day-trade.

        Args:
            symbol: Ticker that was round-tripped intraday.
            trade_date: Date of the day-trade (defaults to today UTC).
            open_side: 'buy' (long day-trade) or 'sell_short' (short day-trade).
        """
        d = trade_date or datetime.now(tz=timezone.utc).date()
        self._records.append(
            DayTradeRecord(symbol=symbol, trade_date=d, open_side=open_side)
        )
        logger.debug("Day-trade recorded: %s %s %s", symbol, d, open_side)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def day_trades_in_window(self, as_of: date | None = None) -> list[DayTradeRecord]:
        """Return day-trade records within the last 5 business days."""
        today = as_of or datetime.now(tz=timezone.utc).date()
        cutoff = self._business_days_ago(today, self._WINDOW_DAYS)
        return [r for r in self._records if r.trade_date >= cutoff]

    def count_in_window(self, as_of: date | None = None) -> int:
        return len(self.day_trades_in_window(as_of))

    def would_trigger_pdt(self, as_of: date | None = None) -> bool:
        """Return True if placing one more day-trade today would trigger PDT."""
        return self.count_in_window(as_of) >= self._MAX_DAY_TRADES

    # ------------------------------------------------------------------
    # Pre-order check
    # ------------------------------------------------------------------

    def pre_order_check(
        self,
        symbol: str,
        side: str,
        would_be_day_trade: bool,
        account_equity: float,
        as_of: date | None = None,
    ) -> tuple[bool, str | None]:
        """Check whether submitting this order is safe under PDT rules.

        Args:
            symbol: Ticker symbol.
            side: 'buy', 'sell', 'sell_short', 'buy_to_cover'.
            would_be_day_trade: True if opening+closing same symbol same day.
            account_equity: Current account equity in USD.
            as_of: Reference date (defaults to today UTC).

        Returns:
            (allowed: bool, reason: str | None).  reason is None if allowed.
        """
        if not _PDT_ACTIVE:
            return (
                True,
                None,
            )  # post-migration: intraday margin checks handled by broker

        if account_equity >= _PDT_EQUITY_THRESHOLD:
            return True, None  # above threshold, no restriction

        if not would_be_day_trade:
            return True, None  # overnight hold, not a day-trade

        if self.would_trigger_pdt(as_of):
            logger.warning(
                "PDT block: %s %s — %d day-trades already in window",
                symbol,
                side,
                self.count_in_window(as_of),
            )
            return False, "pdt_risk"

        return True, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _business_days_ago(ref: date, n: int) -> date:
        """Return the date that is n business days before ref (approx — no holiday calendar)."""
        d = ref
        count = 0
        while count < n:
            d -= timedelta(days=1)
            if d.weekday() < 5:  # Mon-Fri
                count += 1
        return d

    def reset(self) -> None:
        """Clear all recorded day-trades (useful in tests)."""
        self._records.clear()

    def summary(self, as_of: date | None = None) -> dict[str, int | bool]:
        count = self.count_in_window(as_of)
        return {
            "day_trades_in_window": count,
            "would_trigger_pdt": count >= self._MAX_DAY_TRADES,
            "pdt_rule_active": _PDT_ACTIVE,
        }


# ---------------------------------------------------------------------------
# Module-level singleton for typical single-process usage
# ---------------------------------------------------------------------------

_counter: PDTCounter | None = None


def get_pdt_counter() -> PDTCounter:
    global _counter
    if _counter is None:
        _counter = PDTCounter()
    return _counter


__all__ = ["PDTCounter", "DayTradeRecord", "get_pdt_counter"]
