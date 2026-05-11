"""In-memory L2-Limit-Order-Book.

Datenstruktur
-------------
Zwei sorted-dicts (price → size):
- bids: descending price
- asks: ascending price

Updates
-------
- ``add_order(side, price, size)``  : add to level
- ``cancel(side, price, size)``     : reduce level
- ``trade(side, price, size)``      : execution → reduce + record

Standard-Methoden
-----------------
- ``best_bid_ask()``    : (best_bid, best_ask)
- ``mid_price()``       : midpoint
- ``imbalance(depth)``  : sum-bid-volume / (sum-bid + sum-ask) im top-K levels
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class LOBState:
    """Limit-Order-Book State."""

    bids: OrderedDict[float, float] = field(default_factory=OrderedDict)
    asks: OrderedDict[float, float] = field(default_factory=OrderedDict)
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    last_trade_side: str | None = None

    def _sort_levels(self) -> None:
        self.bids = OrderedDict(sorted(self.bids.items(), key=lambda kv: -kv[0]))
        self.asks = OrderedDict(sorted(self.asks.items(), key=lambda kv: kv[0]))

    def add_order(self, side: str, price: float, size: float) -> None:
        book = self.bids if side == "buy" else self.asks
        book[price] = book.get(price, 0) + size
        self._sort_levels()

    def cancel(self, side: str, price: float, size: float) -> None:
        book = self.bids if side == "buy" else self.asks
        if price in book:
            book[price] = max(book[price] - size, 0)
            if book[price] <= 0:
                del book[price]

    def trade(self, side: str, price: float, size: float) -> None:
        """Execution event: side = 'buy' means buyer hits ask (so consume ask level)."""
        book = self.asks if side == "buy" else self.bids
        if price in book:
            book[price] = max(book[price] - size, 0)
            if book[price] <= 0:
                del book[price]
        self.last_trade_price = price
        self.last_trade_size = size
        self.last_trade_side = side

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        p = next(iter(self.bids))
        return p, self.bids[p]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        p = next(iter(self.asks))
        return p, self.asks[p]

    def best_bid_ask(self) -> tuple[float | None, float | None]:
        bb = self.best_bid()
        ba = self.best_ask()
        return (bb[0] if bb else None, ba[0] if ba else None)

    def mid_price(self) -> float | None:
        bb, ba = self.best_bid_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def spread(self) -> float | None:
        bb, ba = self.best_bid_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def imbalance(self, depth: int = 1) -> float | None:
        """Volume-imbalance top-``depth`` levels.

        Returns:
            (bid_vol − ask_vol) / (bid_vol + ask_vol) ∈ [-1, 1].
            Positive = mehr bids = buying-pressure.
        """
        bid_vol = sum(list(self.bids.values())[:depth])
        ask_vol = sum(list(self.asks.values())[:depth])
        total = bid_vol + ask_vol
        if total <= 0:
            return None
        return (bid_vol - ask_vol) / total

    def total_volume(self, side: str, depth: int = 5) -> float:
        book = self.bids if side == "buy" else self.asks
        return float(sum(list(book.values())[:depth]))

    def snapshot(self) -> dict:
        bb, ba = self.best_bid_ask()
        return {
            "best_bid": bb,
            "best_ask": ba,
            "mid": self.mid_price(),
            "spread": self.spread(),
            "imbalance_l1": self.imbalance(1),
            "imbalance_l5": self.imbalance(5),
            "bid_vol_l5": self.total_volume("buy", 5),
            "ask_vol_l5": self.total_volume("sell", 5),
        }


__all__ = ["LOBState"]
