"""News data ingestion module (Phase 6 Skeleton).

This module provides functions to load news and sentiment data.
Currently provides skeleton implementations with dummy data.

Zukünftige Integration:
- Echte Datenquellen: NewsAPI, Alpha Vantage News, FinBERT sentiment analysis, etc.
- Normalisierung auf Standardformat: timestamp, symbol, headline, sentiment_score, source
- Validierung und Datenqualitätsprüfungen
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


def load_news_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Load news sample data.
    
    If path is provided, loads from CSV/Parquet file.
    If path is None, generates a small dummy DataFrame with sample news events.
    
    Args:
        path: Optional path to news data file (CSV or Parquet). If None, generates dummy data.
    
    Returns:
        DataFrame with columns:
        - timestamp: UTC timestamp of the news article
        - symbol: Stock symbol
        - headline: News headline text
        - sentiment_score: Sentiment score (-1 to 1, negative = bearish, positive = bullish)
        - source: News source (e.g., "Reuters", "Bloomberg")
    
    Examples:
        >>> # Generate dummy data
        >>> df = load_news_sample()
        >>> # Load from file
        >>> df = load_news_sample(path="data/news/news_articles.parquet")
    """
    if path is not None:
        path = Path(path)
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except (IOError, OSError) as exc:
                raise IOError(f"Failed to read news data file {path}") from exc
        elif path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
            except (IOError, OSError) as exc:
                raise IOError(f"Failed to read news data file {path}") from exc
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")
    
    # Generate dummy data
    now = datetime.now(timezone.utc)
    symbols = ["AAPL", "MSFT", "TSLA"]
    headlines = [
        "Tech giant reports strong earnings",
        "Company announces new product launch",
        "Stock surges on positive analyst ratings",
        "Market volatility impacts tech sector",
        "Regulatory concerns weigh on shares",
        "Partnership deal boosts investor confidence"
    ]
    sources = ["Reuters", "Bloomberg", "WSJ"]
    
    data = []
    for i, symbol in enumerate(symbols):
        for j in range(2):
            sentiment = 0.3 + (i * 0.2) - (j * 0.1)  # Varying sentiment
            data.append({
                "timestamp": now - timedelta(days=i*2 + j),
                "symbol": symbol,
                "headline": headlines[(i*2 + j) % len(headlines)],
                "sentiment_score": sentiment,
                "source": sources[i % len(sources)]
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize news DataFrame to standard format.
    
    Ensures columns: timestamp (UTC), symbol, headline, sentiment_score, source.
    
    Args:
        df: Raw news DataFrame
    
    Returns:
        Normalized DataFrame with standard columns
    
    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()
    
    # Ensure timestamp is UTC-aware
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise KeyError("timestamp column is required")
    
    # Ensure symbol exists
    if "symbol" not in df.columns:
        raise KeyError("symbol column is required")
    
    # Ensure headline
    if "headline" not in df.columns:
        df["headline"] = ""
    
    # Ensure sentiment_score is numeric and in range [-1, 1]
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0).astype("float64")
        df["sentiment_score"] = df["sentiment_score"].clip(-1.0, 1.0)
    else:
        df["sentiment_score"] = 0.0
    
    # Ensure source
    if "source" not in df.columns:
        df["source"] = "Unknown"
    
    # Sort by symbol, then timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    return df

