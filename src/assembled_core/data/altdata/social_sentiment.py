"""Social Sentiment Aggregation (M38a).

Aggregates sentiment signals from public social media / forum sources.
Designed to consume pre-scraped or API-fetched sentiment data (e.g. Reddit,
StockTwits, Twitter/X) and produce symbol-level daily sentiment features.

Features produced:
    sentiment_score       — daily aggregated sentiment [-1, 1]
    sentiment_volume      — number of mentions / posts
    sentiment_momentum_5d — 5-day change in sentiment score
    bullish_ratio         — fraction of bullish mentions
    sentiment_dispersion  — std of individual mention scores (disagreement)

All features are PIT-safe: only data available by end-of-day is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuration for sentiment aggregation."""

    min_mentions: int = 5
    momentum_window: int = 5
    decay_halflife: int = 3
    max_age_days: int = 2


@dataclass
class SymbolSentiment:
    """Aggregated sentiment for one symbol on one date."""

    symbol: str
    date: str
    sentiment_score: float
    sentiment_volume: int
    bullish_ratio: float
    sentiment_dispersion: float


def aggregate_daily_sentiment(
    mentions: pd.DataFrame,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    score_col: str = "sentiment",
    config: SentimentConfig | None = None,
) -> pd.DataFrame:
    """Aggregate raw mention-level sentiment to daily symbol-level features.

    Args:
        mentions: DataFrame with columns [symbol, timestamp, sentiment].
            sentiment: float in [-1, 1] (negative=bearish, positive=bullish).
        symbol_col: Symbol column name.
        timestamp_col: Timestamp column name.
        score_col: Sentiment score column name.
        config: SentimentConfig (default: standard settings).

    Returns:
        DataFrame with columns [symbol, date, sentiment_score, sentiment_volume,
        bullish_ratio, sentiment_dispersion].
    """
    cfg = config or SentimentConfig()

    if mentions.empty:
        return pd.DataFrame(columns=[
            symbol_col, "date", "sentiment_score", "sentiment_volume",
            "bullish_ratio", "sentiment_dispersion",
        ])

    df = mentions.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["date"] = df[timestamp_col].dt.date

    groups = df.groupby([symbol_col, "date"])
    rows = []

    for (sym, date), grp in groups:
        scores = grp[score_col].values
        n = len(scores)

        if n < cfg.min_mentions:
            continue

        # Exponential decay weighting (most recent mentions weight more)
        if len(grp) > 1:
            # Sort by timestamp within day
            sorted_scores = grp.sort_values(timestamp_col)[score_col].values
            weights = np.array([
                0.5 ** (i / max(cfg.decay_halflife, 1))
                for i in range(len(sorted_scores) - 1, -1, -1)
            ])
            weights /= weights.sum()
            weighted_score = float(np.dot(weights, sorted_scores))
        else:
            weighted_score = float(scores[0])

        bullish = float(np.mean(scores > 0))
        dispersion = float(np.std(scores)) if n > 1 else 0.0

        rows.append({
            symbol_col: sym,
            "date": date,
            "sentiment_score": np.clip(weighted_score, -1.0, 1.0),
            "sentiment_volume": n,
            "bullish_ratio": bullish,
            "sentiment_dispersion": dispersion,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # Sort for downstream processing
    result = result.sort_values([symbol_col, "date"]).reset_index(drop=True)

    logger.info(
        "[Sentiment] Aggregated %d symbol-day rows from %d raw mentions",
        len(result), len(mentions),
    )
    return result


def add_sentiment_momentum(
    daily_sentiment: pd.DataFrame,
    symbol_col: str = "symbol",
    window: int = 5,
) -> pd.DataFrame:
    """Add sentiment momentum (rolling change) to daily sentiment DataFrame.

    Args:
        daily_sentiment: Output of aggregate_daily_sentiment().
        symbol_col: Symbol column.
        window: Lookback window for momentum.

    Returns:
        DataFrame with sentiment_momentum_{window}d column added.
    """
    if daily_sentiment.empty:
        return daily_sentiment.copy()

    df = daily_sentiment.copy()
    col_name = f"sentiment_momentum_{window}d"
    df[col_name] = df.groupby(symbol_col)["sentiment_score"].diff(window)
    return df


def compute_crowd_consensus(
    daily_sentiment: pd.DataFrame,
    threshold_bullish: float = 0.7,
    threshold_bearish: float = 0.3,
) -> pd.DataFrame:
    """Detect crowd consensus extremes (potential contrarian signals).

    Args:
        daily_sentiment: Output of aggregate_daily_sentiment().
        threshold_bullish: Bullish ratio above this = crowd bullish.
        threshold_bearish: Bullish ratio below this = crowd bearish.

    Returns:
        DataFrame with crowd_signal column: 1=crowd bullish, -1=crowd bearish, 0=mixed.
    """
    if daily_sentiment.empty:
        return daily_sentiment.copy()

    df = daily_sentiment.copy()
    df["crowd_signal"] = np.where(
        df["bullish_ratio"] > threshold_bullish, 1.0,
        np.where(df["bullish_ratio"] < threshold_bearish, -1.0, 0.0),
    )
    return df


__all__ = [
    "SentimentConfig",
    "SymbolSentiment",
    "aggregate_daily_sentiment",
    "add_sentiment_momentum",
    "compute_crowd_consensus",
]
