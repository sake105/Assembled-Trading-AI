"""News and sentiment features module (Phase 6 Skeleton).

This module provides functions to compute features from news and sentiment data.
Currently provides skeleton implementations with simple aggregation logic.

Zukünftige Integration:
- Sentiment-weighted features (recent news more important)
- Source credibility weighting
- Topic extraction and classification
- Sentiment momentum indicators
"""

from __future__ import annotations

import pandas as pd


def add_news_features(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add news sentiment features to price DataFrame.

    Computes features like:
    - news_sentiment_7d: Average sentiment score in last 7 days
    - news_sentiment_30d: Average sentiment score in last 30 days
    - news_count_7d: Number of news articles in last 7 days
    - news_count_30d: Number of news articles in last 30 days

    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from news_ingest.load_news_sample() with columns:
            timestamp, symbol, headline, sentiment_score, source

    Returns:
        Copy of prices DataFrame with additional columns:
        - news_sentiment_7d: Average sentiment in last 7 days
        - news_sentiment_30d: Average sentiment in last 30 days
        - news_count_7d: Article count in last 7 days
        - news_count_30d: Article count in last 30 days

    Raises:
        KeyError: If required columns are missing in prices or events
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")

    required_event_cols = ["timestamp", "symbol", "sentiment_score"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")

    result = prices.copy()

    # Ensure timestamps are datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Initialize feature columns
    result["news_sentiment_7d"] = pd.NA
    result["news_sentiment_30d"] = pd.NA
    result["news_count_7d"] = 0
    result["news_count_30d"] = 0

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, compute features based on events in rolling windows
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]

            # 7-day window
            window_7d = symbol_events[
                (symbol_events["timestamp"] <= price_time)
                & (symbol_events["timestamp"] > price_time - pd.Timedelta(days=7))
            ]
            if not window_7d.empty:
                result.loc[idx, "news_sentiment_7d"] = window_7d[
                    "sentiment_score"
                ].mean()
                result.loc[idx, "news_count_7d"] = len(window_7d)

            # 30-day window
            window_30d = symbol_events[
                (symbol_events["timestamp"] <= price_time)
                & (symbol_events["timestamp"] > price_time - pd.Timedelta(days=30))
            ]
            if not window_30d.empty:
                result.loc[idx, "news_sentiment_30d"] = window_30d[
                    "sentiment_score"
                ].mean()
                result.loc[idx, "news_count_30d"] = len(window_30d)

    return result
