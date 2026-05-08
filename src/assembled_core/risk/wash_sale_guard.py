"""Proactive US/DE wash-sale rule guard.

US rule: cannot deduct loss if you buy the same (or substantially identical)
security within 30 days before or after the sale at a loss.

DE rule (§20 EStG): different from US but broadly: realized losses in one
calendar year can only offset gains in the same year, and certain rapid
round-trips may be scrutinized.

This module provides a pre-trade check (not post-reject reactive) so the
system avoids sending orders that would trigger the rule.

Usage:
    guard = WashSaleGuard()
    # After a loss realization:
    guard.record_loss_realization("AAPL", date(2026, 5, 1), loss_amount=500.0)
    # Before re-entering:
    if guard.is_wash_sale_risk("AAPL", date(2026, 5, 15)):
        skip_order()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

log = logging.getLogger(__name__)

US_WASH_SALE_WINDOW_DAYS = 30
DE_WASH_SALE_WINDOW_DAYS = 30  # conservative approximation


class WashSaleGuard:
    """In-memory proactive wash-sale rule guard.

    Thread-safe for single-process use (no Lock needed: GIL covers dict ops
    for CPython; add Lock if multi-threaded access is needed).
    """

    def __init__(self, window_days: int = US_WASH_SALE_WINDOW_DAYS) -> None:
        self._window = window_days
        # symbol → list of (loss_date, loss_amount)
        self._loss_events: dict[str, list[tuple[date, float]]] = defaultdict(list)

    def record_loss_realization(
        self,
        symbol: str,
        loss_date: date,
        loss_amount: float,
    ) -> None:
        """Record a realized loss for a symbol."""
        if loss_amount <= 0:
            return  # only track actual losses
        self._loss_events[symbol].append((loss_date, loss_amount))
        log.info(
            "[wash-sale] Loss recorded: %s on %s, amount=%.2f",
            symbol,
            loss_date,
            loss_amount,
        )

    def is_wash_sale_risk(
        self,
        symbol: str,
        trade_date: date,
        jurisdiction: str = "US",
    ) -> bool:
        """Return True if buying this symbol risks triggering a wash-sale disallowance.

        A wash-sale risk exists if there was a loss on this symbol within
        `window_days` before `trade_date`.
        """
        window = self._window
        cutoff = trade_date - timedelta(days=window)
        for loss_date, _ in self._loss_events.get(symbol, []):
            if loss_date >= cutoff:
                log.warning(
                    "[wash-sale] RISK: %s had loss on %s, within %d-day window of %s (%s rule)",
                    symbol,
                    loss_date,
                    window,
                    trade_date,
                    jurisdiction,
                )
                return True
        return False

    def clear_old_records(self, as_of: date) -> int:
        """Remove loss events older than the wash-sale window. Returns count removed."""
        cutoff = as_of - timedelta(days=self._window + 1)
        removed = 0
        for sym in list(self._loss_events):
            before = len(self._loss_events[sym])
            self._loss_events[sym] = [
                (d, a) for d, a in self._loss_events[sym] if d >= cutoff
            ]
            removed += before - len(self._loss_events[sym])
            if not self._loss_events[sym]:
                del self._loss_events[sym]
        return removed

    def active_symbols(self) -> list[str]:
        """Return symbols currently under wash-sale monitoring."""
        return sorted(self._loss_events.keys())
