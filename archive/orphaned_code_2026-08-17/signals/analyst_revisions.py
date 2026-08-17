"""Analyst Revisions Signal via Finnhub free tier.

From 13_FREE_MODULE.md §13.5.
Blitz/Hanauer/Honarvar 2023: IC 0.02–0.05. Cheap and effective.

Endpoint: Finnhub /stock/recommendation
Measures delta in (buy+2*strongBuy) − (sell+2*strongSell) between current and prior period.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def analyst_revision_score(
    ticker: str,
    finnhub_client: Any,
) -> float:
    """Compute normalized analyst revision score for a ticker.

    Positive = bullish upgrade momentum, Negative = downgrade momentum.
    Range roughly [-1, +1].

    Args:
        ticker: Stock ticker symbol
        finnhub_client: finnhub.Client instance (from finnhub-python)

    Returns:
        Normalized revision delta. Returns 0.0 if insufficient data.
    """
    try:
        recs = finnhub_client.recommendation_trends(ticker)
    except Exception as exc:
        logger.debug("Finnhub recommendation_trends failed for %s: %s", ticker, exc)
        return 0.0

    if not recs or len(recs) < 2:
        return 0.0

    def _score(r: dict) -> float:
        return float(
            (r.get("buy", 0) + 2 * r.get("strongBuy", 0))
            - (r.get("sell", 0) + 2 * r.get("strongSell", 0))
        )

    current_score = _score(recs[0])
    prior_score = _score(recs[1])

    delta = current_score - prior_score
    total = abs(current_score) + abs(prior_score) + 1e-6
    return float(delta / total)


def batch_analyst_revisions(
    tickers: list[str],
    finnhub_client: Any,
) -> pd.Series:
    """Compute analyst revision scores for a list of tickers.

    Returns Series indexed by ticker, values in [-1, +1].
    """
    scores = {}
    for ticker in tickers:
        scores[ticker] = analyst_revision_score(ticker, finnhub_client)
    return pd.Series(scores, name="analyst_revision_score")


__all__ = [
    "analyst_revision_score",
    "batch_analyst_revisions",
]
