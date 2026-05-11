"""IEX-DEEP-Format Parser-Skeleton.

Quelle
------
IEX (Investor's Exchange) bietet **kostenlose** L1+L2-Market-Data via DEEP-Feed:
- Top-of-Book + Depth-Snapshots + Trade-Reports
- HIST-Datendownload: https://www.iexexchange.io/products/market-data-connectivity

Format
------
Binary-Format mit message-types:
- 'Q' / 'q': Quote-Update
- 'T' / '8': Trade-Report
- 'D' / 'D': Depth-Update (price-level changes)

Hier ist ein **Skeleton** für message-parsing — für realen Pcap/HIST-Download
braucht es mehr Spezialfunktionen + große Daten. Wir liefern die Datenstruktur,
sodass eigene Parser-Logik darüber sitzen kann.

Production-Hinweis
------------------
Für echte IEX-DEEP-Production-Daten siehe ``iexcloud-sdk`` oder ``pyiex`` —
beide haben tiefere Format-Unterstützung.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from erweiterung.orderbook.lob_state import LOBState


@dataclass(frozen=True)
class DEEPMessage:
    """Generic IEX-DEEP-Message-Wrapper."""

    msg_type: str  # 'Q' | 'T' | 'D' | 'S' (system event)
    timestamp_ns: int
    symbol: str
    payload: dict


def parse_quote(msg: DEEPMessage) -> dict:
    """Parse a quote-update message.

    Returns:
        dict with bid_price, bid_size, ask_price, ask_size.
    """
    p = msg.payload
    return {
        "bid_price": float(p.get("bid_price", 0)),
        "bid_size": float(p.get("bid_size", 0)),
        "ask_price": float(p.get("ask_price", 0)),
        "ask_size": float(p.get("ask_size", 0)),
        "timestamp_ns": msg.timestamp_ns,
        "symbol": msg.symbol,
    }


def parse_trade(msg: DEEPMessage) -> dict:
    """Parse a trade-report message."""
    p = msg.payload
    return {
        "price": float(p.get("price", 0)),
        "size": float(p.get("size", 0)),
        "side": p.get("side", "unknown"),  # may be unavailable
        "timestamp_ns": msg.timestamp_ns,
        "symbol": msg.symbol,
    }


def parse_depth_update(msg: DEEPMessage) -> dict:
    """Parse depth-update."""
    p = msg.payload
    return {
        "side": p.get("side"),  # 'buy' or 'sell'
        "price": float(p.get("price", 0)),
        "size": float(p.get("size", 0)),  # new size at level (0 = level cleared)
        "timestamp_ns": msg.timestamp_ns,
        "symbol": msg.symbol,
    }


def replay_messages_to_lob(
    messages: Iterator[DEEPMessage], symbol_filter: str | None = None
) -> tuple[LOBState, list[dict]]:
    """Replay stream of DEEP-Messages → final LOB-state + trade-log.

    Args:
        messages: iterable of DEEPMessage.
        symbol_filter: optional symbol to filter.

    Returns:
        (final_state, trade_log).
    """
    state = LOBState()
    trade_log: list[dict] = []
    for msg in messages:
        if symbol_filter and msg.symbol != symbol_filter:
            continue
        if msg.msg_type == "Q":
            q = parse_quote(msg)
            # In a real impl we'd reset top-of-book here; simplified:
            if q["bid_price"] > 0:
                state.bids = type(state.bids)(
                    {q["bid_price"]: q["bid_size"], **state.bids}
                )
            if q["ask_price"] > 0:
                state.asks = type(state.asks)(
                    {q["ask_price"]: q["ask_size"], **state.asks}
                )
            state._sort_levels()
        elif msg.msg_type == "T":
            t = parse_trade(msg)
            state.trade(t["side"] or "buy", t["price"], t["size"])
            trade_log.append(t)
        elif msg.msg_type == "D":
            d = parse_depth_update(msg)
            side = d["side"] or "buy"
            book = state.bids if side == "buy" else state.asks
            if d["size"] <= 0:
                book.pop(d["price"], None)
            else:
                book[d["price"]] = d["size"]
            state._sort_levels()
    return state, trade_log


__all__ = [
    "DEEPMessage",
    "parse_quote",
    "parse_trade",
    "parse_depth_update",
    "replay_messages_to_lob",
]
