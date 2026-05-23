"""Cross-sectional News Sentiment and Macro Regime factors (B2.4).

Companion to :mod:`earnings_insider_wrapper` (B2.3). Provides a PIT-safe
single-snapshot interface for the multifactor dispatch — one row per symbol
at a given ``as_of_date``.

Factors (plan section 4.1, factors 21-24):

- ``news_sentiment_7d_z``: 7-day rolling mean of daily sentiment, PIT-gated
  on ``timestamp <= as_of_date``, cross-sectionally z-scored and clipped +-3.
- ``news_volume_spike_z``: news volume relative to 30-day average, z-scored.
- ``macro_growth_momentum_z``: growth regime indicator from macro series,
  z-scored across symbols (all symbols share the same macro value on a date,
  so z-scoring is degenerate — kept for interface consistency, returns 0.0
  when all identical).
- ``macro_inflation_surprise_z``: inflation regime indicator, same pattern.

PIT-safety: all timestamps are gated with ``timestamp <= as_of_date``.
No price or forward-looking information is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_REQUIRED_COLS = ("symbol", "timestamp", "sentiment_score")
NEWS_VOLUME_COL = "sentiment_volume"  # optional
MACRO_REQUIRED_COLS = ("timestamp", "macro_code", "value", "country")

SENTIMENT_LOOKBACK_DAYS = 7
VOLUME_BASELINE_DAYS = 30
CLIP_BOUND = 3.0
SAFE_DIVIDE_EPS = 1e-6


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. Got: {list(df.columns)}"
        )


def _zscore_clip(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score, clipped to +/- CLIP_BOUND."""
    valid = values.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    mean = valid.mean()
    std = valid.std(ddof=0)
    if std < SAFE_DIVIDE_EPS:
        out = pd.Series(np.nan, index=values.index, dtype=float)
        out.loc[valid.index] = 0.0
        return out
    z = (values - mean) / std
    return z.clip(lower=-CLIP_BOUND, upper=CLIP_BOUND)


# ---------------------------------------------------------------------------
# Raw helpers
# ---------------------------------------------------------------------------


def _news_sentiment_raw(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    news_df: pd.DataFrame,
) -> pd.Series:
    """7-day rolling mean sentiment per symbol, PIT-gated."""
    out = pd.Series(np.nan, index=symbols, dtype=float, name="news_sentiment_raw")
    if news_df.empty:
        return out

    # Normalize both sides to tz-naive so comparisons never raise TypeError.
    # altdata_loader strips tz via .dt.tz_localize(None); callers may pass either.
    as_of_naive = as_of_date.tz_localize(None) if as_of_date.tzinfo else as_of_date
    df = news_df.copy()
    _ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = _ts.dt.tz_localize(None) if _ts.dt.tz is not None else _ts
    # PIT gate
    df = df[df["timestamp"] <= as_of_naive]
    if df.empty:
        return out

    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return out

    window_start = as_of_naive - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS)
    df = df[df["timestamp"] > window_start]
    if df.empty:
        return out

    means = df.groupby("symbol")["sentiment_score"].mean()
    out.loc[means.index] = means
    return out


def _news_volume_spike_raw(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    news_df: pd.DataFrame,
) -> pd.Series:
    """News volume in last 7d relative to 30-day baseline, per symbol."""
    out = pd.Series(np.nan, index=symbols, dtype=float, name="news_volume_spike_raw")
    if news_df.empty or NEWS_VOLUME_COL not in news_df.columns:
        return out

    as_of_naive = as_of_date.tz_localize(None) if as_of_date.tzinfo else as_of_date
    df = news_df.copy()
    _ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = _ts.dt.tz_localize(None) if _ts.dt.tz is not None else _ts
    df = df[df["timestamp"] <= as_of_naive]
    if df.empty:
        return out

    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return out

    baseline_start = as_of_naive - pd.Timedelta(days=VOLUME_BASELINE_DAYS)
    recent_start = as_of_naive - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS)

    # Non-overlapping windows: baseline = (30d, 7d] preceding period; recent = last 7d.
    # Including the recent window in the baseline would dilute the spike ratio.
    baseline = df[
        (df["timestamp"] > baseline_start) & (df["timestamp"] <= recent_start)
    ]
    recent = df[df["timestamp"] > recent_start]

    if baseline.empty:
        return out

    baseline_mean = baseline.groupby("symbol")[NEWS_VOLUME_COL].mean()
    recent_mean = recent.groupby("symbol")[NEWS_VOLUME_COL].mean()

    for sym in symbols:
        b = baseline_mean.get(sym, np.nan)
        r = recent_mean.get(sym, np.nan)
        if pd.notna(b) and pd.notna(r) and b > SAFE_DIVIDE_EPS:
            out.loc[sym] = r / b  # ratio: >1 = spike
    return out


def _macro_regime_raw(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    macro_df: pd.DataFrame,
    code: str,
    country: str = "US",
) -> pd.Series:
    """Latest macro regime value for a given code, broadcast to all symbols."""
    out = pd.Series(np.nan, index=symbols, dtype=float, name=f"macro_{code}_raw")
    if macro_df.empty:
        return out

    as_of_naive = as_of_date.tz_localize(None) if as_of_date.tzinfo else as_of_date
    df = macro_df.copy()
    _ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = _ts.dt.tz_localize(None) if _ts.dt.tz is not None else _ts
    df = df[df["timestamp"] <= as_of_naive]
    if df.empty:
        return out

    df = df[df["country"] == country]
    df = df[df["macro_code"] == code]
    if df.empty:
        return out

    latest = df.sort_values("timestamp").iloc[-1]
    val = latest["value"]
    if pd.notna(val):
        out[:] = val  # market-wide: same for all symbols
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_news_macro_factors(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    news_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    country: str = "US",
) -> pd.DataFrame:
    """Compute cross-sectional news sentiment and macro regime factors.

    Args:
        as_of_date: Point-in-time cutoff. Must be a pandas Timestamp.
        symbols: Universe for the cross-section.
        news_df: DataFrame with columns ``symbol, timestamp, sentiment_score``
            and optionally ``sentiment_volume``.
        macro_df: DataFrame with columns ``timestamp, macro_code, value, country``.
        country: Country filter for macro indicators (default ``"US"``).

    Returns:
        DataFrame indexed by ``symbols`` with columns
        ``["news_sentiment_7d_z", "news_volume_spike_z",
          "macro_growth_momentum_z", "macro_inflation_surprise_z"]``.
    """
    if not isinstance(as_of_date, pd.Timestamp):
        raise ValueError(
            f"as_of_date must be a pandas Timestamp, got {type(as_of_date).__name__}"
        )

    _validate_columns(news_df, NEWS_REQUIRED_COLS, "news_df")
    _validate_columns(macro_df, MACRO_REQUIRED_COLS, "macro_df")

    symbols = list(symbols)

    # News sentiment
    raw_sentiment = _news_sentiment_raw(as_of_date, symbols, news_df)
    sentiment_z = _zscore_clip(raw_sentiment)
    sentiment_z.name = "news_sentiment_7d_z"

    # News volume spike
    raw_volume = _news_volume_spike_raw(as_of_date, symbols, news_df)
    volume_z = _zscore_clip(raw_volume)
    volume_z.name = "news_volume_spike_z"

    # Macro growth — yield_curve_spread is a standard growth-regime proxy.
    # Note: cross-sectional z-score of a market-wide broadcast value is degenerate
    # (all symbols share the same value → std=0 → all 0.0). Kept for interface
    # consistency. A time-series normalization approach is a tracked improvement.
    raw_growth = _macro_regime_raw(
        as_of_date, symbols, macro_df, "yield_curve_spread", country
    )
    growth_z = _zscore_clip(raw_growth)
    growth_z.name = "macro_growth_momentum_z"

    # Macro inflation — cpi_yoy matches FRED macro.parquet column names.
    raw_inflation = _macro_regime_raw(as_of_date, symbols, macro_df, "cpi_yoy", country)
    inflation_z = _zscore_clip(raw_inflation)
    inflation_z.name = "macro_inflation_surprise_z"

    return pd.DataFrame(
        {
            "news_sentiment_7d_z": sentiment_z,
            "news_volume_spike_z": volume_z,
            "macro_growth_momentum_z": growth_z,
            "macro_inflation_surprise_z": inflation_z,
        },
        index=symbols,
    )


__all__ = [
    "compute_news_macro_factors",
    "_news_sentiment_raw",
    "_news_volume_spike_raw",
    "_macro_regime_raw",
]
