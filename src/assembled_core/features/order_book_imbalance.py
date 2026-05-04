"""Order-book imbalance features for intraday alpha generation.

Computes multi-level bid/ask imbalance signals from L2 order book snapshots.
All functions are pure (no I/O) and accept plain Python structures so they can
be called from both streaming and backtesting paths.

Imbalance convention
--------------------
    imbalance = (bid_size - ask_size) / (bid_size + ask_size)

Value in [-1, +1]:
    +1  → all size on the bid  → bullish pressure
    -1  → all size on the ask  → bearish pressure
     0  → balanced book
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BookLevel:
    """Single price level in an order book."""

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """L2 snapshot for a single instrument at a point in time."""

    symbol: str
    timestamp: float  # Unix epoch seconds
    bids: list[BookLevel] = field(default_factory=list)  # sorted best-first
    asks: list[BookLevel] = field(default_factory=list)  # sorted best-first


@dataclass
class ImbalanceFeatures:
    """All imbalance features derived from one snapshot."""

    symbol: str
    timestamp: float
    # Level 1
    l1_imbalance: float  # top-of-book imbalance
    # Multi-level (first N levels)
    l5_imbalance: float  # 5-level imbalance
    l10_imbalance: float  # 10-level imbalance
    # Volume-weighted
    vw_imbalance: float  # price-weighted imbalance across all levels
    # Spread
    spread: float  # best ask - best bid
    spread_bps: float  # spread in basis points relative to mid
    # Pressure ratio
    bid_depth: float  # total bid size (top 10)
    ask_depth: float  # total ask size (top 10)
    mid_price: float


def _level_imbalance(bids: list[BookLevel], asks: list[BookLevel], n: int) -> float:
    """Imbalance using first n levels on each side."""
    bid_sz = sum(lv.size for lv in bids[:n])
    ask_sz = sum(lv.size for lv in asks[:n])
    total = bid_sz + ask_sz
    if total < 1e-9:
        return 0.0
    return (bid_sz - ask_sz) / total


def _volume_weighted_imbalance(bids: list[BookLevel], asks: list[BookLevel]) -> float:
    """Price-weighted imbalance: levels closer to mid get more weight.

    Weight = 1 / level_index (rank-based, so best bid/ask have weight 1).
    """

    def weighted_size(levels: list[BookLevel]) -> float:
        return sum(lv.size / (i + 1) for i, lv in enumerate(levels))

    bid_w = weighted_size(bids)
    ask_w = weighted_size(asks)
    total = bid_w + ask_w
    if total < 1e-9:
        return 0.0
    return (bid_w - ask_w) / total


def compute_imbalance_features(snap: OrderBookSnapshot) -> ImbalanceFeatures:
    """Derive all imbalance features from a single order book snapshot.

    Args:
        snap: OrderBookSnapshot with sorted bids/asks.

    Returns:
        ImbalanceFeatures with computed metrics.
    """
    bids = snap.bids
    asks = snap.asks

    # Mid price and spread
    best_bid = bids[0].price if bids else 0.0
    best_ask = asks[0].price if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
    spread = best_ask - best_bid if (best_bid > 0 and best_ask > 0) else 0.0
    spread_bps = (spread / mid * 10_000) if mid > 1e-9 else 0.0

    bid_depth = sum(lv.size for lv in bids[:10])
    ask_depth = sum(lv.size for lv in asks[:10])

    return ImbalanceFeatures(
        symbol=snap.symbol,
        timestamp=snap.timestamp,
        l1_imbalance=_level_imbalance(bids, asks, 1),
        l5_imbalance=_level_imbalance(bids, asks, 5),
        l10_imbalance=_level_imbalance(bids, asks, 10),
        vw_imbalance=_volume_weighted_imbalance(bids, asks),
        spread=spread,
        spread_bps=spread_bps,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        mid_price=mid,
    )


def imbalance_from_dict(snap_dict: dict[str, Any]) -> ImbalanceFeatures:
    """Construct ImbalanceFeatures from a plain dict representation.

    Dict format::

        {
          "symbol": "AAPL",
          "timestamp": 1700000000.0,
          "bids": [{"price": 180.00, "size": 100}, ...],
          "asks": [{"price": 180.05, "size": 80}, ...],
        }
    """

    def parse_levels(raw: list[dict]) -> list[BookLevel]:
        return [BookLevel(price=float(d["price"]), size=float(d["size"])) for d in raw]

    snap = OrderBookSnapshot(
        symbol=snap_dict.get("symbol", ""),
        timestamp=float(snap_dict.get("timestamp", 0.0)),
        bids=parse_levels(snap_dict.get("bids", [])),
        asks=parse_levels(snap_dict.get("asks", [])),
    )
    return compute_imbalance_features(snap)


def rolling_imbalance_signal(
    snapshots: list[dict[str, Any]],
    lookback: int = 10,
) -> list[float]:
    """Compute a rolling average of L5 imbalance over a sequence of snapshots.

    Useful for smoothing microstructure noise in a streaming context.

    Args:
        snapshots: Ordered list of snapshot dicts (oldest first).
        lookback: Rolling window length.

    Returns:
        List of smoothed imbalance values, same length as snapshots.
        Earlier entries where window is not full use all available data.
    """
    feats = [imbalance_from_dict(s) for s in snapshots]
    signals: list[float] = []
    for i, f in enumerate(feats):
        window = feats[max(0, i - lookback + 1) : i + 1]
        avg = sum(w.l5_imbalance for w in window) / len(window)
        signals.append(round(avg, 6))
    return signals
