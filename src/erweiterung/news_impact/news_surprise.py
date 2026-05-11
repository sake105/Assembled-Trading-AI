"""News-Surprise — Erwartetes vs realisiertes Sentiment.

Theorie
-------
**Roher Sentiment** ist verzerrt: einige Firmen haben strukturell positiveres
Sentiment (z.B. AAPL hat fast immer Tone > 0). Was relevant für Trading ist:
**Surprise = realized − expected**.

Erwartungsbildung:
1. **Rolling-Mean-Baseline**: μ_t = mean(sentiment_{t-60..t-1})
2. **Symbol-Level-Mean**: für jedes Symbol seine eigene Baseline.
3. **Topic-conditional**: bedingt auf Topic (M&A, Earnings, Litigation, ...)
4. **Time-of-day-bedingt**: morgenliche Press Releases vs. Intraday.

Reference
---------
Tetlock, P. (2007). Giving Content to Investor Sentiment.
Tetlock, P., Saar-Tsechansky, M. & Macskassy, S. (2008). More Than Words.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sentiment_baseline(
    news_df: pd.DataFrame,
    window: int = 60,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Compute symbol-level rolling-mean sentiment baseline.

    Args:
        news_df: DataFrame [date, symbol, sentiment].
        window: rolling window in days for baseline.

    Returns:
        DataFrame with added ``baseline`` column.
    """
    df = news_df.copy().sort_values([symbol_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    # Group by symbol, compute time-aware rolling mean (in calendar days)
    df["baseline"] = df.groupby(symbol_col)[sentiment_col].transform(
        lambda s: s.shift(1)
        .rolling(window=window, min_periods=max(5, window // 4))
        .mean()
    )
    return df


def compute_surprise(
    news_df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    baseline_col: str = "baseline",
) -> pd.DataFrame:
    """Add ``surprise = sentiment − baseline`` column.

    Surprise > 0 = unexpectedly positive news.
    """
    df = news_df.copy()
    df["surprise"] = df[sentiment_col] - df[baseline_col].fillna(0)
    return df


def standardized_surprise(
    news_df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    by: str = "symbol",
    window: int = 60,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add ``surprise_z = (sentiment − μ) / σ`` (Symbol-level rolling).

    Args:
        news_df: DataFrame [date, symbol, sentiment].
        by: grouping column.
        window: rolling-window for μ and σ.
    """
    df = news_df.copy().sort_values([by, date_col])
    g = df.groupby(by)[sentiment_col]
    mu = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 4)).mean()
    )
    sd = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 4)).std()
    )
    df["surprise_z"] = (df[sentiment_col] - mu) / sd.replace(0, np.nan)
    return df


def cross_section_surprise_rank(
    news_df: pd.DataFrame,
    date_col: str = "date",
    surprise_col: str = "surprise",
) -> pd.DataFrame:
    """Cross-sectional surprise ranking per day.

    Useful for "buy top-decile, short bottom-decile"-strategies.
    """
    df = news_df.copy()
    df["surprise_rank_pct"] = df.groupby(date_col)[surprise_col].rank(pct=True)
    return df


def surprise_to_signal(
    df: pd.DataFrame,
    surprise_z_col: str = "surprise_z",
    threshold: float = 1.5,
) -> pd.Series:
    """Convert surprise_z to long/flat/short signal.

    +1 if surprise_z > threshold, -1 if < -threshold, 0 else.

    Note: Tetlock (2007) found that **extreme negative** surprise
    overreaction reverses within ~3-5 days. Use signal with decay-model.
    """
    s = df[surprise_z_col]
    sig = pd.Series(0.0, index=s.index)
    sig[s > threshold] = +1.0
    sig[s < -threshold] = -1.0
    return sig


__all__ = [
    "rolling_sentiment_baseline",
    "compute_surprise",
    "standardized_surprise",
    "cross_section_surprise_rank",
    "surprise_to_signal",
]
