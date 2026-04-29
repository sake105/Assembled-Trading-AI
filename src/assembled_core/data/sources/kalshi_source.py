"""Kalshi prediction market data source (CFTC-regulated, read-only public endpoints).

Kalshi is a CFTC-designated contract market. Market metadata and settlement prices
are publicly accessible without authentication.

Public REST API base: https://trading-api.kalshi.com/trade-api/v2

Complements Polymarket — both cover similar macro/geo event categories but with
independent liquidity, allowing cross-venue probability comparison.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

GEO_SERIES_TAGS = frozenset({
    "GEOPOLITICS", "ECONOMICS", "ELECTIONS", "ENERGY", "FED", "RATES",
    "TRADE", "DEFENSE", "SANCTIONS", "MACRO",
})


def _get(url: str, params: dict[str, Any] | None = None, timeout: int = 10) -> Any:
    try:
        import httpx  # type: ignore[import]
        headers = {"Accept": "application/json"}
        r = httpx.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("[WARN] kalshi: GET %s failed: %s", url, exc)
        return None


def fetch_active_markets(
    limit: int = 100,
    series_tags: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch open Kalshi markets.

    Args:
        limit: Maximum markets to return.
        series_tags: Filter to these series tags (geo/macro by default).

    Returns:
        List of dicts: ticker, title, yes_bid, yes_ask, volume, open_interest, close_time.
    """
    tags = series_tags or GEO_SERIES_TAGS
    data = _get(
        f"{_KALSHI_BASE}/markets",
        params={"limit": str(min(limit, 200)), "status": "open"},
    )
    if not data:
        return []

    raw_markets = data.get("markets") or (data if isinstance(data, list) else [])
    markets: list[dict[str, Any]] = []

    for m in raw_markets:
        # Filter by series tag when available
        series = str(m.get("series_ticker", "")).upper()
        if series_tags is not None:
            # Only keep if any tag token appears in the series ticker
            if not any(tag in series for tag in tags):
                continue

        yes_bid = float(m.get("yes_bid", 0) or 0) / 100.0
        yes_ask = float(m.get("yes_ask", 100) or 100) / 100.0
        mid = (yes_bid + yes_ask) / 2.0

        markets.append({
            "ticker":         m.get("ticker", ""),
            "title":          m.get("title", ""),
            "yes_bid":        round(yes_bid, 4),
            "yes_ask":        round(yes_ask, 4),
            "mid":            round(mid, 4),
            "volume":         int(m.get("volume", 0) or 0),
            "open_interest":  int(m.get("open_interest", 0) or 0),
            "close_time":     m.get("close_time", ""),
            "series_ticker":  m.get("series_ticker", ""),
        })

        if len(markets) >= limit:
            break

    return markets


def get_market_implied_geo_signal(
    series_tags: set[str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Aggregate Kalshi market probabilities into a geo-risk signal.

    Returns:
        Dict with signal [0,1], n_markets, avg_mid, volume_weighted_mid, source.
    """
    markets = fetch_active_markets(limit=limit, series_tags=series_tags)
    if not markets:
        return {
            "signal": 0.0,
            "n_markets": 0,
            "avg_mid": 0.0,
            "volume_weighted_mid": 0.0,
            "source": "kalshi",
        }

    mids = [m["mid"] for m in markets]
    vols = [m["volume"] for m in markets]

    avg_mid = sum(mids) / len(mids)
    total_vol = sum(vols)
    vol_weighted = (
        sum(mid * vol for mid, vol in zip(mids, vols)) / total_vol
        if total_vol > 0
        else avg_mid
    )

    signal = min(1.0, max(0.0, (avg_mid - 0.40) / 0.40))

    return {
        "signal":               round(signal, 4),
        "n_markets":            len(markets),
        "avg_mid":              round(avg_mid, 4),
        "volume_weighted_mid":  round(vol_weighted, 4),
        "source":               "kalshi",
    }


def fetch_combined_prediction_signal(
    poly_signal: dict[str, Any] | None = None,
    kalshi_signal: dict[str, Any] | None = None,
    poly_weight: float = 0.6,
) -> dict[str, Any]:
    """Blend Polymarket and Kalshi signals into a single prediction-market geo indicator.

    Args:
        poly_signal: Result from polymarket_source.get_market_implied_geo_signal().
        kalshi_signal: Result from kalshi_source.get_market_implied_geo_signal().
        poly_weight: Weight for Polymarket (Kalshi gets 1 - poly_weight).

    Returns:
        Combined signal dict.
    """
    if poly_signal is None and kalshi_signal is None:
        return {"signal": 0.0, "source": "prediction_markets_combined", "n_sources": 0}

    poly_s = float((poly_signal or {}).get("signal", 0.0))
    kals_s = float((kalshi_signal or {}).get("signal", 0.0))

    if poly_signal is not None and kalshi_signal is not None:
        combined = poly_weight * poly_s + (1.0 - poly_weight) * kals_s
    elif poly_signal is not None:
        combined = poly_s
    else:
        combined = kals_s

    return {
        "signal":      round(combined, 4),
        "poly_signal": poly_s,
        "kals_signal": kals_s,
        "source":      "prediction_markets_combined",
        "n_sources":   (1 if poly_signal else 0) + (1 if kalshi_signal else 0),
    }
