"""Sentiment-Divergence: Traditional-News vs Social-Media-Sentiment.

Theorie
-------
Wenn klassische News positiv, Social-Media negativ (oder umgekehrt), entsteht
**Informations-Asymmetrie**:
- Social leads on retail-driven moves (meme stocks, GME-style).
- Traditional leads on institutional flows.

Divergenz = (z(sent_news) − z(sent_social)). Hohe |Divergenz| = Vorhersage-
relevant. Akademisch: Cookson/Niessner (2020) zeigen Divergenz prognostiziert
mid-term volatility-spikes.

Anwendung
---------
- Bei extremer Divergenz: erhöhte Vola erwartet → Position-Size reduzieren.
- Direction-Bias: hohes Social-Sentiment + niedriges News = retail-frenzy → short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_divergence_panel(
    news_sentiment: pd.DataFrame,
    social_sentiment: pd.DataFrame,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Compute z-normalized sentiment-divergence per (date, symbol).

    Args:
        news_sentiment: DataFrame [date, symbol, sentiment] from traditional news.
        social_sentiment: dito from Reddit/Twitter.
        rolling_window: for z-score-normalization.

    Returns:
        DataFrame [date, symbol, news_z, social_z, divergence_z, abs_divergence].
    """

    def _z(df, label):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.sort_values([symbol_col, date_col])
        g = df.groupby(symbol_col)[sentiment_col]
        mu = g.transform(
            lambda s: s.shift(1)
            .rolling(rolling_window, min_periods=max(5, rolling_window // 4))
            .mean()
        )
        sd = g.transform(
            lambda s: s.shift(1)
            .rolling(rolling_window, min_periods=max(5, rolling_window // 4))
            .std()
        )
        df[f"{label}_z"] = (df[sentiment_col] - mu) / sd.replace(0, np.nan)
        return df[[date_col, symbol_col, f"{label}_z"]]

    news_z = _z(news_sentiment, "news")
    social_z = _z(social_sentiment, "social")
    merged = news_z.merge(social_z, on=[date_col, symbol_col], how="outer")
    merged["divergence_z"] = merged["news_z"].fillna(0) - merged["social_z"].fillna(0)
    merged["abs_divergence"] = merged["divergence_z"].abs()
    return merged


def detect_extreme_divergence(
    divergence_panel: pd.DataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Filter to (date, symbol) pairs with |divergence_z| > threshold.

    Returns:
        Filtered DataFrame, sorted by abs_divergence descending.
    """
    df = divergence_panel.copy()
    extreme = df[df["abs_divergence"] > threshold].sort_values(
        "abs_divergence", ascending=False
    )
    return extreme.reset_index(drop=True)


def divergence_implied_vol_spike(
    divergence_z: pd.Series, forward_returns: pd.Series, lookback: int = 60
) -> dict:
    """Test: does extreme divergence predict next-period |return|?

    Returns dict with correlation + p-value.
    """
    df = pd.concat(
        [divergence_z.rename("div"), forward_returns.abs().rename("abs_ret")], axis=1
    ).dropna()
    if len(df) < 30:
        return {"error": "too few obs"}
    corr = float(df["div"].abs().corr(df["abs_ret"]))
    # Spearman alternative
    rank_corr = float(df["div"].abs().rank().corr(df["abs_ret"].rank()))
    return {
        "pearson_corr": corr,
        "spearman_corr": rank_corr,
        "n_obs": int(len(df)),
        "mean_abs_ret_when_high_divergence": (
            float(df.loc[df["div"].abs() > 2.0, "abs_ret"].mean())
            if (df["div"].abs() > 2.0).any()
            else float("nan")
        ),
    }


__all__ = [
    "compute_divergence_panel",
    "detect_extreme_divergence",
    "divergence_implied_vol_spike",
]
