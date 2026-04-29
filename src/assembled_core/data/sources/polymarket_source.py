"""Polymarket prediction market data source (read-only, no auth required).

Uses the Gamma Markets API (market metadata) and CLOB REST API (order-book snapshots).
Both are public endpoints — no API key required for read operations.

Public API reference:
  Gamma: https://gamma-api.polymarket.com
  CLOB:  https://clob.polymarket.com

Returned structures are plain dicts so the caller can decide how to persist or
route the data without pulling in pandas at import time.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"

# Topics that map well to geo-risk signals
GEO_KEYWORDS = frozenset({
    "election", "war", "conflict", "sanctions", "tariff", "trade",
    "invasion", "ceasefire", "coup", "nato", "military", "nuclear",
    "strait", "oil", "energy", "opec", "taiwan", "ukraine", "russia",
    "china", "iran", "north korea",
})


def _get(url: str, params: dict[str, Any] | None = None, timeout: int = 10) -> Any:
    """HTTP GET with httpx; returns parsed JSON or None on any error."""
    try:
        import httpx  # type: ignore[import]
        r = httpx.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("[WARN] polymarket: GET %s failed: %s", url, exc)
        return None


def fetch_active_markets(limit: int = 200, geo_filter: bool = True) -> list[dict[str, Any]]:
    """Fetch active Polymarket markets from Gamma API.

    Args:
        limit: Maximum number of markets to return.
        geo_filter: If True, keep only markets with geo/macro keywords in the question.

    Returns:
        List of market dicts with keys: id, question, end_date, volume, liquidity,
        last_trade_price, outcomes.
    """
    data = _get(
        f"{_GAMMA_BASE}/markets",
        params={"active": "true", "limit": str(limit), "order": "volume", "ascending": "false"},
    )
    if not data:
        return []

    markets: list[dict[str, Any]] = []
    for item in (data if isinstance(data, list) else data.get("markets", [])):
        if geo_filter:
            question_lower = str(item.get("question", "")).lower()
            if not any(kw in question_lower for kw in GEO_KEYWORDS):
                continue
        markets.append({
            "id":              item.get("id", ""),
            "question":        item.get("question", ""),
            "end_date":        item.get("endDate", ""),
            "volume":          float(item.get("volume", 0) or 0),
            "liquidity":       float(item.get("liquidity", 0) or 0),
            "last_trade_price": float(item.get("lastTradePrice", 0.5) or 0.5),
            "outcomes":        item.get("outcomes", []),
        })

    return markets[:limit]


def fetch_clob_midprice(condition_id: str) -> float | None:
    """Fetch best-bid / best-ask midpoint from the CLOB for a single market.

    Returns mid-price in [0, 1] (Polymarket YES token price = market-implied probability),
    or None if the order book is unavailable.
    """
    data = _get(f"{_CLOB_BASE}/book", params={"token_id": condition_id})
    if not data:
        return None
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    if not bids or not asks:
        return None
    try:
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        return (best_bid + best_ask) / 2.0
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def get_market_implied_geo_signal(
    policy: dict[str, Any] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Compute an aggregate geo-risk signal from Polymarket market prices.

    Aggregates market-implied probabilities for geo/macro risk events into
    a scalar signal suitable for use in the georisk_overlay.

    Args:
        policy: Optional policy dict (unused, kept for interface parity with overlay callers).
        limit: Number of active markets to consider.

    Returns:
        Dict with:
          signal:        float in [0, 1] — higher = more market-implied geo-risk
          n_markets:     int — number of geo-risk markets found
          avg_prob:      float — average YES probability across markets
          volume_weighted_prob: float — volume-weighted probability
          source:        "polymarket"
    """
    markets = fetch_active_markets(limit=limit, geo_filter=True)
    if not markets:
        return {
            "signal": 0.0,
            "n_markets": 0,
            "avg_prob": 0.0,
            "volume_weighted_prob": 0.0,
            "source": "polymarket",
        }

    probs = [m["last_trade_price"] for m in markets]
    vols = [m["volume"] for m in markets]

    avg_prob = sum(probs) / len(probs)
    total_vol = sum(vols)
    if total_vol > 0:
        vol_weighted = sum(p * v for p, v in zip(probs, vols)) / total_vol
    else:
        vol_weighted = avg_prob

    # Signal: rescale average prob from ~[0.3, 0.7] to [0, 1]
    # Most binary markets hover near 0.5; elevation above 0.55 is notable
    signal = min(1.0, max(0.0, (avg_prob - 0.40) / 0.40))

    return {
        "signal":               round(signal, 4),
        "n_markets":            len(markets),
        "avg_prob":             round(avg_prob, 4),
        "volume_weighted_prob": round(vol_weighted, 4),
        "source":               "polymarket",
    }
