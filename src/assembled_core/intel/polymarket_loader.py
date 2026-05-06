"""Polymarket prediction market loader.

Fetches implied probabilities for financially-relevant markets.
Free read-only API — no auth required.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com/markets"

RELEVANT_MARKET_KEYWORDS: list[str] = [
    "iran", "hormuz", "russia", "ukraine", "taiwan", "china",
    "tariff", "fed", "federal reserve", "recession", "inflation",
    "oil", "opec", "sanction", "war", "conflict", "escalation",
    "rate", "gdp", "default", "debt ceiling", "election",
    "trade", "semiconductor", "chips act", "nato", "israel",
]


def fetch_polymarket_markets(
    keywords: list[str] | None = None,
    limit: int = 50,
    timeout: int = 15,
) -> list[dict]:
    """Fetch open Polymarket prediction markets filtered by keyword relevance.

    Parameters
    ----------
    keywords : keyword filter list (defaults to RELEVANT_MARKET_KEYWORDS)
    limit : max markets to return (sorted by 24h volume)
    timeout : HTTP timeout in seconds

    Returns
    -------
    List of dicts with: question, slug, implied_probability, volume_24h,
                        end_date, fetched_at
    """
    try:
        import urllib.request
        import json
    except ImportError as exc:
        logger.warning("[SKIP] polymarket_loader: %s", exc)
        return []

    kw_set = {k.lower() for k in (keywords or RELEVANT_MARKET_KEYWORDS)}

    params = "?closed=false&order=volume24hr&ascending=false&limit=200"
    url = POLYMARKET_API + params

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AssembledTradingAI/1.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw: list[dict] = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("[WARN] polymarket_loader: fetch failed: %s", exc)
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    for market in raw:
        question: str = (market.get("question") or "").lower()
        if not any(kw in question for kw in kw_set):
            continue

        # Extract implied probability from best-ask on YES outcome
        outcomes = market.get("outcomes") or []
        yes_prob: float | None = None
        for outcome in outcomes:
            if isinstance(outcome, dict) and outcome.get("title", "").upper() == "YES":
                try:
                    yes_prob = float(outcome.get("price", 0.0))
                except (TypeError, ValueError):
                    pass
                break

        if yes_prob is None:
            # Fallback: use outcomePrices first element
            prices_raw = market.get("outcomePrices")
            if prices_raw:
                try:
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    yes_prob = float(prices[0])
                except Exception:
                    yes_prob = 0.0

        results.append(
            {
                "question": market.get("question", ""),
                "slug": market.get("slug", ""),
                "implied_probability": round(yes_prob or 0.0, 4),
                "volume_24h": float(market.get("volume24hr") or 0.0),
                "end_date": market.get("endDate", ""),
                "fetched_at": fetched_at,
            }
        )

        if len(results) >= limit:
            break

    logger.info("[OK] polymarket_loader: fetched %d relevant markets", len(results))
    return results


def polymarket_to_dataframe(markets: list[dict]) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Convert fetch_polymarket_markets() output to a DataFrame."""
    import pandas as pd

    if not markets:
        return pd.DataFrame()
    return pd.DataFrame(markets)
