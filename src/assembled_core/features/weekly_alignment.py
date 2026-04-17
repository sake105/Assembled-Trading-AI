"""F3 — Multi-timeframe (weekly) alignment filter (Plan v3 Part F3).

Daily trend signals in choppy sideways markets produce whipsaw losses. The
plan's fix is simple::

    long_ok  = daily_trend > 0  AND  weekly_ema_slope > 0
    short_ok = daily_trend < 0  AND  weekly_ema_slope < 0

This module is intentionally narrow — one function, one filter, tested
against a deterministic fixture. The caller decides whether to apply the
mask to target-positions, to the signal column, or post-scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_WEEKLY_EMA_SPAN = 10  # ~10 weekly bars ≈ 2 months


@dataclass(frozen=True)
class WeeklyAlignmentConfig:
    ema_span: int = DEFAULT_WEEKLY_EMA_SPAN
    slope_lookback: int = 2  # bars over which the slope is measured


def _weekly_ema_slope(
    prices: pd.Series,
    config: WeeklyAlignmentConfig,
) -> pd.Series:
    """Daily-indexed slope of the weekly-resampled EMA (forward-filled)."""
    if prices.empty:
        return pd.Series(dtype=float)
    weekly = prices.resample("W-FRI").last().dropna()
    if len(weekly) < config.slope_lookback + 1:
        return pd.Series(0.0, index=prices.index)
    ema = weekly.ewm(span=config.ema_span, adjust=False).mean()
    slope = ema.diff(config.slope_lookback)
    slope_daily = slope.reindex(prices.index, method="ffill")
    return slope_daily.fillna(0.0)


def add_weekly_alignment(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    daily_trend_col: str = "daily_trend",
    out_col: str = "weekly_alignment_ok",
    config: WeeklyAlignmentConfig | None = None,
) -> pd.DataFrame:
    """Add a boolean column indicating whether daily trend agrees with weekly EMA slope.

    The input frame must be indexed by a ``DatetimeIndex`` (daily). A
    ``symbol`` column is optional; if present, the slope is computed per
    symbol independently.

    Returns a **copy** of the input with two extra columns:

    - ``weekly_ema_slope``  — forward-filled slope of weekly EMA on close.
    - ``weekly_alignment_ok`` — True when daily trend sign matches slope sign.
    """
    if price_col not in df.columns:
        raise ValueError(f"column {price_col!r} missing from frame")
    if daily_trend_col not in df.columns:
        raise ValueError(f"column {daily_trend_col!r} missing from frame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("frame must be datetime-indexed")

    cfg = config or WeeklyAlignmentConfig()
    out = df.copy()

    if "symbol" in out.columns:
        slopes = []
        for sym, grp in out.groupby("symbol", sort=False):
            s = _weekly_ema_slope(grp[price_col], cfg).rename("weekly_ema_slope")
            s.index = grp.index  # keep source order
            slopes.append(s)
        out["weekly_ema_slope"] = pd.concat(slopes).reindex(out.index)
    else:
        out["weekly_ema_slope"] = _weekly_ema_slope(out[price_col], cfg)

    trend = out[daily_trend_col]
    slope = out["weekly_ema_slope"]
    out[out_col] = np.where(
        (trend > 0) & (slope > 0),
        True,
        np.where((trend < 0) & (slope < 0), True, False),
    )
    return out
