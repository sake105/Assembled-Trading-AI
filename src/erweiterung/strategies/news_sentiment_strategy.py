"""News-Sentiment-Cross-Section-Strategie.

Idee
----
Tagesweise Aggregation des News-Sentiment pro Symbol. Cross-Section:
long Top-Quintile (highest sentiment), short Bottom-Quintile oder
Long-Only (Top-Quintile, equal-weight).

Voraussetzung: News-Daten mit ``timestamp``, ``symbol``, ``sentiment_score``,
``sentiment_volume`` Spalten — z. B. ``output/news_sentiment_fused.parquet``.

Daten-Caveat
------------
Die im Repo vorhandenen News-Daten umfassen nur **2025-12 → 2026-05**
(423 Rows, 91 Symbole). Das ist **zu sparse** für statistisch belastbare
Backtests. Dieses Modul ist die Skeleton-Implementation; bei verfügbarem
Multi-Jahr-News-Feed wird dieselbe Code-Pfad genutzt.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class NewsSentimentConfig:
    aggregation_window_days: int = 5
    quantile_long: float = 0.2  # top 20 %
    quantile_short: float = 0.2  # bottom 20 %
    long_only: bool = True
    min_observations_per_symbol: int = 3
    min_symbols_for_cross_section: int = 5
    smoothing_window: int = 3


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    config: NewsSentimentConfig | None = None,
) -> pd.DataFrame:
    """Aggregiere News auf (date × symbol)-Panel.

    Args:
        news_df: DataFrame mit Spalten timestamp, symbol, sentiment_score,
            sentiment_volume (oder count).

    Returns:
        Long DataFrame [date, symbol, sentiment_mean, volume_sum, sentiment_smoothed].
    """
    cfg = config or NewsSentimentConfig()
    if news_df.empty:
        return pd.DataFrame()

    df = news_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()

    vol_col = (
        "sentiment_volume"
        if "sentiment_volume" in df.columns
        else "count" if "count" in df.columns else None
    )

    agg = df.groupby(["date", "symbol"]).agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_count=("sentiment_score", "count"),
    )
    if vol_col:
        agg["volume_sum"] = df.groupby(["date", "symbol"])[vol_col].sum()
    agg = agg.reset_index()

    # Smoothing pro Symbol
    agg = agg.sort_values(["symbol", "date"]).reset_index(drop=True)
    agg["sentiment_smoothed"] = agg.groupby("symbol")["sentiment_mean"].transform(
        lambda s: s.rolling(cfg.smoothing_window, min_periods=1).mean()
    )
    return agg


def cross_section_signal(
    sentiment_panel: pd.DataFrame,
    config: NewsSentimentConfig | None = None,
) -> pd.DataFrame:
    """Cross-Section-Signal: long top-quintile, short bottom-quintile (oder long-only).

    Args:
        sentiment_panel: Output von aggregate_daily_sentiment().

    Returns:
        DataFrame [date, symbol, position] mit position ∈ {-1, 0, +1}.
    """
    cfg = config or NewsSentimentConfig()
    if sentiment_panel.empty:
        return pd.DataFrame()
    out = sentiment_panel.copy()
    # t-1-shift: today's position based on yesterday's news
    out["sig_lag"] = out.groupby("symbol")["sentiment_smoothed"].shift(1)
    # Cross-section quantiles per date
    out["sig_pct"] = out.groupby("date")["sig_lag"].rank(pct=True)
    out["position"] = 0
    out.loc[out["sig_pct"] >= 1 - cfg.quantile_long, "position"] = 1
    if not cfg.long_only:
        out.loc[out["sig_pct"] <= cfg.quantile_short, "position"] = -1

    # Validate min cross-section size
    n_today = out.groupby("date")["sig_lag"].count()
    valid_dates = n_today[n_today >= cfg.min_symbols_for_cross_section].index
    out = out[out["date"].isin(valid_dates)]
    return out[["date", "symbol", "sentiment_smoothed", "sig_pct", "position"]]


def backtest_news_signal(
    sentiment_panel: pd.DataFrame,
    price_returns: pd.DataFrame,
    config: NewsSentimentConfig | None = None,
) -> pd.Series:
    """Backtest des News-Cross-Section-Signals.

    Args:
        sentiment_panel: Output von aggregate_daily_sentiment().
        price_returns: DataFrame [date, symbol, return] mit Tagesreturns.
        config: NewsSentimentConfig.

    Returns:
        Series mit Tagesreturns des Portfolios.
    """
    cfg = config or NewsSentimentConfig()
    signal = cross_section_signal(sentiment_panel, cfg)
    if signal.empty:
        return pd.Series(dtype=float)

    if not isinstance(price_returns, pd.DataFrame):
        raise TypeError("price_returns must be DataFrame")
    if {"date", "symbol", "return"} - set(price_returns.columns):
        raise ValueError("price_returns needs date, symbol, return columns")

    pr = price_returns.copy()
    pr["date"] = pd.to_datetime(pr["date"], utc=True).dt.normalize()
    signal["date"] = pd.to_datetime(signal["date"], utc=True).dt.normalize()

    merged = signal.merge(pr, on=["date", "symbol"], how="inner")
    if merged.empty:
        return pd.Series(dtype=float)

    # Equal-weight innerhalb der Long-Seite
    daily_long = merged[merged["position"] == 1].groupby("date")["return"].mean()
    daily_short = (
        merged[merged["position"] == -1].groupby("date")["return"].mean()
        if not cfg.long_only
        else None
    )

    if cfg.long_only or daily_short is None:
        port = daily_long.fillna(0)
    else:
        port = (daily_long - daily_short).fillna(0)
    return port


__all__ = [
    "NewsSentimentConfig",
    "aggregate_daily_sentiment",
    "cross_section_signal",
    "backtest_news_signal",
]
