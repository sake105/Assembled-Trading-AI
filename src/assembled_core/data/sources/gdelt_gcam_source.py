"""GDELT 2.1 GCAM deep features — sentiment dimensions beyond basic tone.

From 10_FREE_DATEN.md §10.5.
GCAM = Global Content Analysis Measures: 2200+ emotion/theme scores per article.
Key GCAM dimensions for finance:
  c6.8  = Negative economic outlook
  c6.5  = Positive economic outlook
  c17.1 = Economic uncertainty
  c12.1 = Financial markets
  c18.3 = Employment / jobs

Also implements:
  - NumMentions × AvgTone: amplified sentiment feature
  - Mentions-DB tracking for article amplification velocity

Install: pip install gdeltdoc
BigQuery alternative: GCP Free-Tier 1 TB/month queries

Caution from plan: GCAM accuracy ~55%, redundancy ~20% (MDPI 2025).
Dedup + correction layer required before production use.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Key GCAM dimension IDs for financial trading signals
GCAM_FINANCE_DIMS = {
    "c6.8": "neg_economic_outlook",
    "c6.5": "pos_economic_outlook",
    "c17.1": "economic_uncertainty",
    "c12.1": "financial_markets",
    "c18.3": "employment_labor",
    "c7.1": "financial_crisis",
    "c14.2": "monetary_policy",
    "c10.1": "stock_market",
}


def _try_gdeltdoc():
    try:
        from gdeltdoc import GdeltDoc, Filters
        return GdeltDoc, Filters
    except ImportError:
        logger.warning("gdeltdoc not installed — pip install gdeltdoc")
        return None, None


def fetch_gcam_sentiment(
    keyword: str,
    hours: int = 24,
    timespan: str | None = None,
) -> dict[str, float]:
    """Fetch GCAM-weighted sentiment features for a keyword/ticker.

    Args:
        keyword: Search term (company name, ticker, topic)
        hours: Lookback in hours (default 24)
        timespan: Override timespan string (e.g. '3months') if hours > 168

    Returns:
        Dict mapping GCAM dimension name → mean score.
        Empty dict if gdeltdoc unavailable or query fails.
    """
    GdeltDoc, Filters = _try_gdeltdoc()
    if GdeltDoc is None:
        return {}

    try:
        f = Filters(keyword=keyword, timespan=timespan or f"{min(hours, 168)}hours")
        gd = GdeltDoc()
        articles = gd.article_search(f)

        if articles.empty:
            return {}

        result: dict[str, float] = {}

        # NumMentions × AvgTone amplified sentiment
        if "V2Tone" in articles.columns and "NumMentions" in articles.columns:
            tones = pd.to_numeric(articles["V2Tone"].str.split(",").str[0], errors="coerce")
            mentions = pd.to_numeric(articles["NumMentions"], errors="coerce")
            if not tones.empty and not mentions.empty:
                weighted_tone = float((tones * mentions).sum() / mentions.sum().clip(lower=1))
                result["weighted_tone"] = weighted_tone

        # GCAM dimensions (if present in result columns)
        for gcam_id, dim_name in GCAM_FINANCE_DIMS.items():
            col_candidates = [c for c in articles.columns if gcam_id in c]
            if col_candidates:
                vals = pd.to_numeric(articles[col_candidates[0]], errors="coerce").dropna()
                if not vals.empty:
                    result[f"gcam_{dim_name}"] = float(vals.mean())

        return result

    except Exception as exc:
        logger.debug("GDELT GCAM fetch failed for '%s': %s", keyword, exc)
        return {}


def fetch_mentions_velocity(
    keyword: str,
    hours_recent: int = 4,
    hours_baseline: int = 48,
) -> float:
    """Compute article-amplification velocity via GDELT Mentions DB proxy.

    Velocity = (recent_mentions_per_hour) / (baseline_mentions_per_hour)
    > 2.0 = significant acceleration (potential news event)

    Args:
        keyword: Search term
        hours_recent: Recent window for velocity (default 4h)
        hours_baseline: Baseline window (default 48h)

    Returns:
        Velocity ratio. Returns 1.0 on failure (neutral).
    """
    GdeltDoc, Filters = _try_gdeltdoc()
    if GdeltDoc is None:
        return 1.0

    try:
        gd = GdeltDoc()

        f_recent = Filters(keyword=keyword, timespan=f"{hours_recent}hours")
        recent = gd.article_search(f_recent)
        recent_count = len(recent)

        f_baseline = Filters(keyword=keyword, timespan=f"{hours_baseline}hours")
        baseline = gd.article_search(f_baseline)
        baseline_count = len(baseline)

        if baseline_count == 0:
            return 1.0

        recent_rate = recent_count / max(hours_recent, 1)
        baseline_rate = baseline_count / max(hours_baseline, 1)
        if baseline_rate < 0.01:
            return 1.0

        return float(recent_rate / baseline_rate)

    except Exception as exc:
        logger.debug("GDELT mentions velocity failed for '%s': %s", keyword, exc)
        return 1.0


def gdelt_composite_signal(
    keyword: str,
    hours: int = 24,
) -> dict[str, float]:
    """Combined GDELT signal: GCAM dimensions + mentions velocity.

    Returns dict with:
      weighted_tone: NumMentions-weighted AvgTone (negative = bearish)
      mentions_velocity: Recent vs baseline amplification ratio
      neg_economic_outlook: GCAM c6.8 mean
      economic_uncertainty: GCAM c17.1 mean
    """
    result = fetch_gcam_sentiment(keyword, hours=hours)
    result["mentions_velocity"] = fetch_mentions_velocity(keyword)
    return result


__all__ = [
    "GCAM_FINANCE_DIMS",
    "fetch_gcam_sentiment",
    "fetch_mentions_velocity",
    "gdelt_composite_signal",
]
