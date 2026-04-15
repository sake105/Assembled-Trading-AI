"""Minute Bar Aggregator (M23 Task 23.4).

Aggregates streaming tick/trade data into 1-minute OHLCV bars.
Maintains an in-memory panel for the active universe.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AggregatedBar:
    """A single aggregated minute bar."""
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int


class MinuteBarAggregator:
    """Aggregates streaming trades into minute bars.

    Usage:
        agg = MinuteBarAggregator()
        agg.on_trade("AAPL", 150.25, 100, timestamp)
        bars = agg.flush_completed_bars()
    """

    def __init__(self, max_history_minutes: int = 390):
        """Initialize aggregator.

        Args:
            max_history_minutes: Maximum minute bars to keep per symbol.
        """
        self.max_history = max_history_minutes
        self._current_bars: dict[str, dict] = {}
        self._completed: list[AggregatedBar] = []
        self._history: dict[str, list[AggregatedBar]] = defaultdict(list)

    def on_trade(
        self,
        symbol: str,
        price: float,
        size: int,
        timestamp: float,
    ) -> AggregatedBar | None:
        """Process a trade tick.

        Args:
            symbol: Symbol.
            price: Trade price.
            size: Trade size (shares).
            timestamp: Unix timestamp.

        Returns:
            AggregatedBar if a bar was completed, else None.
        """
        ts = pd.Timestamp(timestamp, unit="s")
        minute_key = ts.floor("min")

        if symbol not in self._current_bars:
            self._current_bars[symbol] = {
                "minute": minute_key,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": size,
                "trades": 1,
            }
            return None

        bar = self._current_bars[symbol]

        # New minute → complete old bar
        if minute_key > bar["minute"]:
            completed = AggregatedBar(
                symbol=symbol,
                timestamp=bar["minute"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                trade_count=bar["trades"],
            )
            self._completed.append(completed)
            self._history[symbol].append(completed)

            # Trim history
            if len(self._history[symbol]) > self.max_history:
                self._history[symbol] = self._history[symbol][-self.max_history:]

            # Start new bar
            self._current_bars[symbol] = {
                "minute": minute_key,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": size,
                "trades": 1,
            }
            return completed

        # Update current bar
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += size
        bar["trades"] += 1
        return None

    def flush_completed_bars(self) -> list[AggregatedBar]:
        """Get and clear completed bars.

        Returns:
            List of completed AggregatedBar since last flush.
        """
        bars = self._completed.copy()
        self._completed.clear()
        return bars

    def get_history(self, symbol: str) -> pd.DataFrame:
        """Get minute bar history for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            DataFrame with OHLCV columns.
        """
        bars = self._history.get(symbol, [])
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return pd.DataFrame([
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "trade_count": b.trade_count,
            }
            for b in bars
        ]).set_index("timestamp")

    def get_all_latest(self) -> dict[str, float]:
        """Get latest price for all tracked symbols.

        Returns:
            {symbol: last_price} dict.
        """
        return {
            sym: bar["close"]
            for sym, bar in self._current_bars.items()
        }

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        return {
            "tracked_symbols": len(self._current_bars),
            "pending_completed": len(self._completed),
            "history_bars": sum(len(v) for v in self._history.values()),
        }


__all__ = [
    "AggregatedBar",
    "MinuteBarAggregator",
]
