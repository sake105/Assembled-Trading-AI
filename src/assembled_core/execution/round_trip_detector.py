"""Round-trip (day-trade) detector.

From 41_PDT_REGEL_INTRADAY_MARGIN.md §3.2.

Detects when a position opened and closed on the same calendar day,
records it in the PDTTracker. Handles partial fills correctly per FINRA
examples: multiple partial closes of one original position = one day-trade.

Known limitation: FINRA Example D (mixed add-then-close) may record differently
than FINRA's exact count. Acceptable for simple buy-then-sell strategies.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from assembled_core.execution.pdt_tracker import DayTrade, PDTTracker


class RoundTripDetector:
    """Detects day-trades on every order fill."""

    def __init__(self, tracker: PDTTracker) -> None:
        self.tracker = tracker
        # ticker → (open_ts, qty, avg_price, side)
        self.open_positions: dict[str, tuple[datetime, int, float, str]] = {}
        # tickers where day-trade already recorded (to avoid double-count on partial closes)
        self._day_trade_recorded: set[str] = set()

    def on_fill(self, fill_event) -> Optional[DayTrade]:
        """Call on every broker fill event.

        fill_event must have: .ticker, .side ('buy'/'sell'), .quantity, .price, .timestamp
        Returns DayTrade if a round-trip was detected, else None.
        """
        ticker = fill_event.ticker
        side = fill_event.side
        qty = fill_event.quantity
        price = fill_event.price
        ts: datetime = fill_event.timestamp

        if ticker not in self.open_positions:
            position_side = "long" if side == "buy" else "short"
            self.open_positions[ticker] = (ts, qty, price, position_side)
            return None

        open_ts, open_qty, open_price, open_side = self.open_positions[ticker]
        is_closing = (
            (open_side == "long" and side == "sell")
            or (open_side == "short" and side == "buy")
        )

        if not is_closing:
            new_qty = open_qty + qty
            if abs(new_qty) < 1e-10:
                del self.open_positions[ticker]
            else:
                new_price = (open_qty * open_price + qty * price) / new_qty
                self.open_positions[ticker] = (open_ts, new_qty, new_price, open_side)
            return None

        if open_ts.date() == ts.date():
            trade = DayTrade(
                ticker=ticker,
                open_timestamp=open_ts,
                close_timestamp=ts,
                side=open_side,
                quantity=min(open_qty, qty),
                entry_price=open_price,
                exit_price=price,
            )
            # Record only the first close as a day-trade (FINRA Example C: partial closes count as 1)
            if ticker not in self._day_trade_recorded:
                self.tracker.record_day_trade(trade)
                self._day_trade_recorded.add(ticker)

            if qty >= open_qty:
                del self.open_positions[ticker]
                self._day_trade_recorded.discard(ticker)
            else:
                self.open_positions[ticker] = (open_ts, open_qty - qty, open_price, open_side)
            return trade

        # Closing at a different day → swing trade
        if qty >= open_qty:
            del self.open_positions[ticker]
            self._day_trade_recorded.discard(ticker)
        else:
            self.open_positions[ticker] = (open_ts, open_qty - qty, open_price, open_side)
        return None


__all__ = ["RoundTripDetector"]
