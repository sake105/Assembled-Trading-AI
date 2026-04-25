"""Event schemas for the Event-Replay-System.

From 42_EVENT_REPLAY_SYSTEM.md §1.

All events that touch the live system are modeled here.  During live
operation they are appended to the EventStore; during replay the same
sequence is fed back to the decision logic to verify determinism.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventSource(str, Enum):
    MARKET_DATA = "market_data"
    NEWS = "news"
    ORDER = "order"
    CLOCK = "clock"
    SYSTEM = "system"


@dataclass
class BaseEvent:
    event_type: str
    source: EventSource
    session_id: str
    sequence: int
    occurred_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["occurred_at"] = self.occurred_at.isoformat()
        d["source"] = self.source.value
        return json.dumps(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseEvent":
        d = dict(d)
        d["occurred_at"] = datetime.fromisoformat(d["occurred_at"])
        d["source"] = EventSource(d["source"])
        return cls(**d)


def make_market_tick(session_id: str, sequence: int,
                     symbol: str, bid: float, ask: float,
                     last: float, volume: int) -> BaseEvent:
    return BaseEvent(
        event_type="market_tick_received",
        source=EventSource.MARKET_DATA,
        session_id=session_id,
        sequence=sequence,
        payload={"symbol": symbol, "bid": bid, "ask": ask,
                 "last": last, "volume": volume},
    )


def make_news_event(session_id: str, sequence: int,
                    headline: str, ticker: str | None,
                    sentiment: float) -> BaseEvent:
    return BaseEvent(
        event_type="news_received",
        source=EventSource.NEWS,
        session_id=session_id,
        sequence=sequence,
        payload={"headline": headline, "ticker": ticker,
                 "sentiment": sentiment},
    )


def make_order_filled(session_id: str, sequence: int,
                      symbol: str, side: str, qty: float,
                      fill_price: float, client_order_id: str) -> BaseEvent:
    return BaseEvent(
        event_type="order_filled",
        source=EventSource.ORDER,
        session_id=session_id,
        sequence=sequence,
        payload={"symbol": symbol, "side": side, "qty": qty,
                 "fill_price": fill_price,
                 "client_order_id": client_order_id},
    )


def make_order_rejected(session_id: str, sequence: int,
                        symbol: str, reason: str,
                        client_order_id: str) -> BaseEvent:
    return BaseEvent(
        event_type="order_rejected",
        source=EventSource.ORDER,
        session_id=session_id,
        sequence=sequence,
        payload={"symbol": symbol, "reason": reason,
                 "client_order_id": client_order_id},
    )


def make_clock_tick(session_id: str, sequence: int, ts: datetime) -> BaseEvent:
    return BaseEvent(
        event_type="clock_tick",
        source=EventSource.CLOCK,
        session_id=session_id,
        sequence=sequence,
        payload={"ts": ts.isoformat()},
    )


# Keep aliases for backward compatibility
MarketTickReceived = BaseEvent
NewsReceived = BaseEvent
OrderFilled = BaseEvent
OrderRejected = BaseEvent
ClockTick = BaseEvent

__all__ = [
    "EventSource",
    "BaseEvent",
    "make_market_tick",
    "make_news_event",
    "make_order_filled",
    "make_order_rejected",
    "make_clock_tick",
    # aliases
    "MarketTickReceived",
    "NewsReceived",
    "OrderFilled",
    "OrderRejected",
    "ClockTick",
]
