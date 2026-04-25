"""Web Scraping Feature Extraction (M38d).

Extracts structured features from web-scraped data sources:
- Job posting counts (hiring momentum)
- Product review sentiment
- App store ratings/downloads
- Website traffic estimates

All features are PIT-safe: only data available by scrape_date is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WebScrapingConfig:
    """Configuration for web scraping feature extraction."""

    min_data_points: int = 3
    trend_window_days: int = 30
    processing_lag_days: int = 1


def compute_job_posting_features(
    postings: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: WebScrapingConfig | None = None,
    symbol_col: str = "symbol",
    date_col: str = "scrape_date",
    count_col: str = "job_count",
) -> pd.DataFrame:
    """Compute hiring momentum features from job posting data.

    Args:
        postings: DataFrame with job posting counts per symbol/date.
        as_of: Reference date (PIT cutoff).
        config: WebScrapingConfig.
        symbol_col: Symbol column.
        date_col: Scrape date column.
        count_col: Job posting count column.

    Returns:
        DataFrame with job_posting_count, job_posting_growth_30d per symbol.
    """
    cfg = config or WebScrapingConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if postings.empty:
        return pd.DataFrame(columns=[symbol_col, "job_posting_count", "job_posting_growth_30d"])

    df = postings.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    if df.empty:
        return pd.DataFrame(columns=[symbol_col, "job_posting_count", "job_posting_growth_30d"])

    trend_cutoff = pit_cutoff - pd.Timedelta(days=cfg.trend_window_days)
    rows = []

    for sym, grp in df.groupby(symbol_col):
        sorted_grp = grp.sort_values(date_col)
        if len(sorted_grp) < cfg.min_data_points:
            continue

        latest_count = int(sorted_grp[count_col].iloc[-1])

        # 30-day growth
        recent = sorted_grp[sorted_grp[date_col] > trend_cutoff]
        if len(recent) >= 2:
            first = float(recent[count_col].iloc[0])
            last = float(recent[count_col].iloc[-1])
            growth = (last - first) / max(first, 1.0)
        else:
            growth = 0.0

        rows.append({
            symbol_col: sym,
            "job_posting_count": latest_count,
            "job_posting_growth_30d": round(float(growth), 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[WebScraping] Computed job posting features for %d symbols", len(result))
    return result


def compute_app_rating_features(
    ratings: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: WebScrapingConfig | None = None,
    symbol_col: str = "symbol",
    date_col: str = "scrape_date",
    rating_col: str = "avg_rating",
    review_count_col: str = "review_count",
) -> pd.DataFrame:
    """Compute app store rating features.

    Args:
        ratings: DataFrame with app store ratings per symbol/date.
        as_of: Reference date.
        config: WebScrapingConfig.

    Returns:
        DataFrame with app_rating, app_rating_trend, app_review_velocity per symbol.
    """
    cfg = config or WebScrapingConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if ratings.empty:
        return pd.DataFrame(columns=[
            symbol_col, "app_rating", "app_rating_trend", "app_review_velocity",
        ])

    df = ratings.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    if df.empty:
        return pd.DataFrame(columns=[
            symbol_col, "app_rating", "app_rating_trend", "app_review_velocity",
        ])

    trend_cutoff = pit_cutoff - pd.Timedelta(days=cfg.trend_window_days)
    rows = []

    for sym, grp in df.groupby(symbol_col):
        sorted_grp = grp.sort_values(date_col)
        if len(sorted_grp) < cfg.min_data_points:
            continue

        latest_rating = float(sorted_grp[rating_col].iloc[-1])

        # Rating trend
        recent = sorted_grp[sorted_grp[date_col] > trend_cutoff]
        if len(recent) >= 2:
            trend = float(recent[rating_col].iloc[-1] - recent[rating_col].iloc[0])
        else:
            trend = 0.0

        # Review velocity (reviews per day in recent window)
        if review_count_col in sorted_grp.columns and len(recent) >= 2:
            days_span = max((recent[date_col].iloc[-1] - recent[date_col].iloc[0]).days, 1)
            total_reviews = float(recent[review_count_col].sum())
            velocity = total_reviews / days_span
        else:
            velocity = 0.0

        rows.append({
            symbol_col: sym,
            "app_rating": round(latest_rating, 2),
            "app_rating_trend": round(trend, 4),
            "app_review_velocity": round(velocity, 2),
        })

    result = pd.DataFrame(rows)
    logger.info("[WebScraping] Computed app rating features for %d symbols", len(result))
    return result


def compute_website_traffic_features(
    traffic: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: WebScrapingConfig | None = None,
    symbol_col: str = "symbol",
    date_col: str = "scrape_date",
    visits_col: str = "estimated_visits",
) -> pd.DataFrame:
    """Compute website traffic features from estimated visit data.

    Args:
        traffic: DataFrame with website traffic estimates.
        as_of: Reference date.
        config: WebScrapingConfig.

    Returns:
        DataFrame with web_traffic_index, web_traffic_trend per symbol.
    """
    cfg = config or WebScrapingConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if traffic.empty:
        return pd.DataFrame(columns=[symbol_col, "web_traffic_index", "web_traffic_trend"])

    df = traffic.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    if df.empty:
        return pd.DataFrame(columns=[symbol_col, "web_traffic_index", "web_traffic_trend"])

    trend_cutoff = pit_cutoff - pd.Timedelta(days=cfg.trend_window_days)
    rows = []

    for sym, grp in df.groupby(symbol_col):
        sorted_grp = grp.sort_values(date_col)
        if len(sorted_grp) < cfg.min_data_points:
            continue

        mean_visits = float(sorted_grp[visits_col].mean())
        if mean_visits < 1.0:
            continue

        latest = float(sorted_grp[visits_col].iloc[-1])
        traffic_index = latest / mean_visits

        # Trend
        recent = sorted_grp[sorted_grp[date_col] > trend_cutoff]
        if len(recent) >= 2:
            first = float(recent[visits_col].iloc[0])
            last = float(recent[visits_col].iloc[-1])
            trend = (last - first) / max(first, 1.0)
        else:
            trend = 0.0

        rows.append({
            symbol_col: sym,
            "web_traffic_index": round(float(traffic_index), 4),
            "web_traffic_trend": round(float(trend), 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[WebScraping] Computed web traffic features for %d symbols", len(result))
    return result


__all__ = [
    "WebScrapingConfig",
    "compute_job_posting_features",
    "compute_app_rating_features",
    "compute_website_traffic_features",
]
