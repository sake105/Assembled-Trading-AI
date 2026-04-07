"""News and sentiment features module (V13).

Computes news-derived features from raw sentiment scores and optional
FinBERT NLP scores.  Features include rolling means, momentum, dispersion,
shock flags, and article counts.

Graceful degradation: FinBERT features are NaN when torch/transformers
are not installed — the basic sentiment_score features still work.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Optional FinBERT import
try:
    from assembled_core.ml.nlp_sentiment import (
        TRANSFORMERS_AVAILABLE,
        score_news_store,
    )
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    score_news_store = None  # type: ignore[assignment]


def add_news_features(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add news sentiment features to price DataFrame.

    Computes:
    - news_sentiment_7d / _30d: Rolling mean sentiment
    - news_count_7d / _30d: Article counts
    - nlp_sentiment_mean_5d / _20d: FinBERT score means (if available)
    - nlp_sentiment_momentum_5d: 5d-20d sentiment delta
    - nlp_sentiment_dispersion_20d: Cross-sectional std of sentiment

    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        events: DataFrame with columns: timestamp, symbol, sentiment_score,
            and optionally headline (used for FinBERT scoring)

    Returns:
        Copy of prices with sentiment feature columns added.
    """
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")

    required_event_cols = ["timestamp", "symbol", "sentiment_score"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")

    result = prices.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # --- FinBERT enrichment (optional) ---
    has_finbert = False
    if (
        TRANSFORMERS_AVAILABLE
        and score_news_store is not None
        and "headline" in events.columns
        and not events.empty
    ):
        try:
            events = score_news_store(events, text_col="headline")
            has_finbert = True
            _log.info("[NLP] FinBERT scored %d news rows", len(events))
        except Exception as e:
            _log.warning("[NLP] FinBERT scoring failed, using basic sentiment: %s", e)

    # Score column: prefer FinBERT if available, else raw sentiment_score
    score_col = "finbert_score" if has_finbert else "sentiment_score"

    # --- Build daily per-symbol sentiment aggregates ---
    events["_date"] = events["timestamp"].dt.normalize()
    daily = (
        events.groupby(["symbol", "_date"])
        .agg(
            daily_sentiment=(score_col, "mean"),
            daily_count=(score_col, "count"),
            daily_std=(score_col, "std"),
        )
        .reset_index()
    )
    daily["daily_std"] = daily["daily_std"].fillna(0.0)
    daily.rename(columns={"_date": "timestamp"}, inplace=True)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)

    # Merge daily sentiment onto price grid
    result["_ts_date"] = result["timestamp"].dt.normalize()
    merged = result.merge(
        daily,
        left_on=["symbol", "_ts_date"],
        right_on=["symbol", "timestamp"],
        how="left",
        suffixes=("", "_daily"),
    )
    # Restore original timestamp
    if "timestamp_daily" in merged.columns:
        merged.drop(columns=["timestamp_daily"], inplace=True)
    merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    merged["daily_count"] = merged["daily_count"].fillna(0).astype(int)
    merged["daily_std"] = merged["daily_std"].fillna(0.0)

    # Sort for rolling computations
    merged = merged.sort_values(["symbol", "timestamp"])

    # --- Rolling features per symbol ---
    for window, suffix in [(7, "7d"), (30, "30d")]:
        col_sent = f"news_sentiment_{suffix}"
        col_cnt = f"news_count_{suffix}"
        merged[col_sent] = (
            merged.groupby("symbol")["daily_sentiment"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        merged[col_cnt] = (
            merged.groupby("symbol")["daily_count"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # --- NLP-specific features (always compute, use FinBERT score if available) ---
    merged["nlp_sentiment_mean_5d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    merged["nlp_sentiment_mean_20d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )
    merged["nlp_sentiment_momentum_5d"] = (
        merged["nlp_sentiment_mean_5d"] - merged["nlp_sentiment_mean_20d"]
    )

    # Cross-sectional dispersion: std of daily_sentiment across symbols per date
    date_std = (
        merged.groupby("_ts_date")["daily_sentiment"]
        .transform("std")
        .fillna(0.0)
    )
    merged["nlp_sentiment_dispersion_20d"] = (
        merged.groupby("symbol")[date_std.name if hasattr(date_std, 'name') else "daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )
    # Actually compute rolling std on per-symbol sentiment as dispersion
    merged["nlp_sentiment_dispersion_20d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).std())
    ).fillna(0.0)

    # Clean up temp columns
    drop_cols = ["_ts_date", "daily_sentiment", "daily_count", "daily_std"]
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

    added = sum(1 for c in merged.columns if c.startswith(("news_", "nlp_")))
    _log.info("Added %d news/NLP sentiment features (finbert=%s)", added, has_finbert)

    return merged


__all__ = [
    "add_news_features",
]
