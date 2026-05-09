"""PDT (Pattern Day Trader) tracking.

From 41_PDT_REGEL_INTRADAY_MARGIN.md §3.1.

Tracks day-trades over a rolling 5-business-day window per FINRA Rule 4210.
Set enabled=False after Alpaca migrates to intraday-margin (expected 4 June 2026+).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List

import pandas as pd


@dataclass
class DayTrade:
    """One completed round-trip transaction on the same calendar day."""

    ticker: str
    open_timestamp: datetime
    close_timestamp: datetime
    side: str  # "long" or "short"
    quantity: int
    entry_price: float
    exit_price: float

    @property
    def trade_date(self) -> date:
        return self.open_timestamp.date()

    @property
    def pnl(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.quantity
        return (self.entry_price - self.exit_price) * self.quantity


class PDTTracker:
    """Tracks day-trades in the rolling 5-business-day window.

    Args:
        account_equity: Current account equity in USD.
        enabled: Set to False after broker migrates away from PDT rules.
    """

    def __init__(self, account_equity: float, enabled: bool = True) -> None:
        self.account_equity = account_equity
        self.enabled = enabled
        self.day_trades: List[DayTrade] = []

    def record_day_trade(self, trade: DayTrade) -> None:
        self.day_trades.append(trade)

    def count_recent_day_trades(self, reference_date: date | None = None) -> int:
        if reference_date is None:
            reference_date = datetime.now(timezone.utc).date()
        cutoff = self._business_days_ago(reference_date, 5)
        return sum(1 for t in self.day_trades if t.trade_date > cutoff)

    def would_violate_pdt(self, reference_date: date | None = None) -> bool:
        if not self.enabled:
            return False
        if self.account_equity >= 25_000:
            return False
        current = self.count_recent_day_trades(reference_date)
        return current >= 3  # 3 existing + 1 new = 4 = PDT threshold

    @staticmethod
    def _business_days_ago(reference: date, n: int) -> date:
        from src.assembled_core.utils.market_calendar import is_trading_day

        cur = pd.Timestamp(reference)
        counted = 0
        while counted < n:
            cur -= pd.Timedelta(days=1)
            if is_trading_day(cur):
                counted += 1
        return cur.date()

    def days_until_pdt_reset(self, reference_date: date | None = None) -> int:
        from src.assembled_core.utils.market_calendar import trading_days_between

        if reference_date is None:
            reference_date = datetime.now(timezone.utc).date()
        recent = [
            t
            for t in self.day_trades
            if t.trade_date > self._business_days_ago(reference_date, 5)
        ]
        if not recent:
            return 0
        oldest = min(t.trade_date for t in recent)
        days_since_oldest = trading_days_between(oldest, reference_date)
        return max(0, 5 - days_since_oldest)


__all__ = ["DayTrade", "PDTTracker"]
