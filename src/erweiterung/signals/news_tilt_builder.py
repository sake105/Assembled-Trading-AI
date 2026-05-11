"""Build per-symbol news-tilt z-scores for LiveDecisionEngine.

Konzept
-------
Aus sparse News-Sentiment (typisch <1 event/symbol/day) wird ein Rolling-Mean
pro Symbol gebildet, dann täglich cross-section z-normalisiert. Diese z-Scores
können via ``engine.attach_news_tilt_scores()`` als additives Tilt-Signal in
die EqTopN-Selection eingespeist werden.

Output
------
DataFrame indexed by daily UTC timestamps, columns = symbols, values = z-scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_news_sentiment(
    cache_path: str = "output/news_sentiment_fused.parquet",
) -> pd.DataFrame:
    """Lade Fused-News-Sentiment-Parquet."""
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(f"news sentiment cache not found at {cache_path}")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df = df.rename(columns={"sentiment_score": "sentiment"})
    return df


def build_daily_news_tilt(
    news_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    rolling_days: int = 14,
    decay_halflife_days: int = 3,
) -> pd.DataFrame:
    """Daily symbol-level news-tilt z-scores.

    Steps:
    1. Daily aggregate sentiment per symbol (mean over events that day).
    2. Exponentially-weighted rolling mean per symbol (recent events weighted higher).
    3. Cross-section z-normalize across symbols per day.

    Args:
        news_df: long-format with columns [date, symbol, sentiment].
        daily_index: target daily index (UTC).
        rolling_days: window for symbol-level EWM (half-life-controlled).
        decay_halflife_days: EWM half-life in days (3d = recent dominates).

    Returns:
        DataFrame indexed by daily_index, columns = symbols, z-scores.
        NaN where insufficient data.
    """
    if news_df.empty:
        return pd.DataFrame(index=daily_index)

    if news_df["date"].dt.tz is None:
        news_df = news_df.copy()
        news_df["date"] = news_df["date"].dt.tz_localize("UTC")

    # Daily mean sentiment per symbol
    daily = (
        news_df.groupby(["date", "symbol"], as_index=False)["sentiment"]
        .mean()
        .pivot(index="date", columns="symbol", values="sentiment")
    )

    # Reindex to full daily index
    if daily_index.tz is None:
        daily_index = daily_index.tz_localize("UTC")
    daily = daily.reindex(daily_index)

    # Compute "days since last news event" mask per symbol
    n = len(daily)
    positions = pd.DataFrame(
        np.tile(np.arange(n)[:, None], (1, len(daily.columns))),
        index=daily.index,
        columns=daily.columns,
    )
    last_obs_pos = positions.where(daily.notna()).ffill()
    days_since = positions - last_obs_pos
    stale_mask = days_since.gt(
        rolling_days
    )  # NaN > x is False, so unobserved cols safe

    # EWM-smooth per symbol after ffill (limited)
    daily_filled = daily.ffill(limit=rolling_days)
    smoothed = daily_filled.ewm(halflife=decay_halflife_days, min_periods=1).mean()

    # Force stale entries to NaN
    smoothed = smoothed.where(~stale_mask)

    # Cross-section z-score per day
    mu = smoothed.mean(axis=1)
    sd = smoothed.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = smoothed.sub(mu, axis=0).div(sd, axis=0)
    return z


def news_tilt_for_date(
    z_panel: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series:
    """Lookup z-scores for a specific date with ffill fallback."""
    if z_panel.empty:
        return pd.Series(dtype=float)
    if date not in z_panel.index:
        z_panel = z_panel.reindex(z_panel.index.union([date])).sort_index().ffill()
    return z_panel.loc[date].dropna()
