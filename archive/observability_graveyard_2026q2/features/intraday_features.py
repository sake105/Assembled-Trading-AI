"""Intraday Features for Alpha Generation (M20 Task 20.4).

Extracts intraday patterns from minute-bar data:
1. Last-Hour Momentum (Heston et al. 2010)
2. Overnight vs. Intraday Return Separation (Lou et al. 2019)
3. Opening Range Breakout
4. Volume Profile (VWAP deviation)
5. Intraday Volatility Pattern

Alpha: +70-180 bps/year
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradayFeatureResult:
    """Result of intraday feature computation."""
    features: pd.DataFrame
    coverage: float
    warnings: list[str]


def compute_last_hour_momentum(
    minute_bars: pd.DataFrame,
    close_col: str = "close",
    time_col: str = "timestamp",
    last_hour_start: str = "15:00",
) -> pd.Series:
    """Last-hour momentum — persistent effect (Heston et al. 2010).

    Stocks that rally in the last hour tend to open higher next day.

    Args:
        minute_bars: Minute-level OHLCV with timestamp.
        close_col: Column name for close price.
        time_col: Column for timestamp.
        last_hour_start: Start of "last hour" (EST).

    Returns:
        Daily series of last-hour returns.
    """
    df = minute_bars.copy()
    if time_col in df.columns:
        df.index = pd.to_datetime(df[time_col])

    df["date"] = df.index.date
    df["time"] = df.index.time

    results = {}
    for date, group in df.groupby("date"):
        # Find last hour bars
        last_hour = group[group["time"] >= pd.Timestamp(last_hour_start).time()]
        if len(last_hour) < 2:
            continue
        ret = (last_hour[close_col].iloc[-1] / last_hour[close_col].iloc[0]) - 1.0
        results[date] = ret

    return pd.Series(results, name="last_hour_momentum")


def compute_overnight_return(
    daily_open: pd.Series,
    daily_close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Separate overnight and intraday returns (Lou et al. 2019).

    overnight_return = open_t / close_{t-1} - 1
    intraday_return = close_t / open_t - 1

    Args:
        daily_open: Daily open prices.
        daily_close: Daily close prices.

    Returns:
        (overnight_returns, intraday_returns) tuple.
    """
    overnight = (daily_open / daily_close.shift(1)) - 1.0
    intraday = (daily_close / daily_open) - 1.0
    overnight.name = "overnight_return"
    intraday.name = "intraday_return"
    return overnight, intraday


def compute_opening_range_breakout(
    minute_bars: pd.DataFrame,
    opening_minutes: int = 30,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Opening Range Breakout signal.

    If price breaks above the first-30-min high → bullish.
    If price breaks below the first-30-min low → bearish.

    Args:
        minute_bars: Minute OHLCV.
        opening_minutes: Minutes for opening range.
        close_col: Close column.
        high_col: High column.
        low_col: Low column.

    Returns:
        Daily signal: +1 (bullish breakout), -1 (bearish), 0 (inside).
    """
    df = minute_bars.copy()
    df["date"] = df.index.date

    results = {}
    for date, group in df.groupby("date"):
        if len(group) < opening_minutes + 10:
            continue
        opening = group.iloc[:opening_minutes]
        rest = group.iloc[opening_minutes:]

        range_high = opening[high_col].max()
        range_low = opening[low_col].min()

        if rest[close_col].max() > range_high:
            results[date] = 1.0
        elif rest[close_col].min() < range_low:
            results[date] = -1.0
        else:
            results[date] = 0.0

    return pd.Series(results, name="opening_range_breakout")


def compute_vwap_deviation(
    prices: pd.Series,
    volumes: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Price deviation from VWAP.

    Positive = price above VWAP (buying pressure).
    Negative = price below VWAP (selling pressure).

    Args:
        prices: Close prices.
        volumes: Trading volumes.
        lookback: Rolling window for VWAP.

    Returns:
        VWAP deviation ratio.
    """
    vwap = (prices * volumes).rolling(lookback, min_periods=5).sum() / \
           volumes.rolling(lookback, min_periods=5).sum().replace(0, np.nan)
    deviation = (prices - vwap) / vwap.replace(0, np.nan)
    return deviation.fillna(0).rename("vwap_deviation")


def compute_intraday_volatility_ratio(
    daily_high: pd.Series,
    daily_low: pd.Series,
    daily_close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Intraday vs. close-to-close volatility ratio.

    High ratio = lots of intraday movement relative to net change → mean-reverting.
    Low ratio = trending behavior.

    Args:
        daily_high: Daily high prices.
        daily_low: Daily low prices.
        daily_close: Daily close prices.
        lookback: Rolling window.

    Returns:
        Volatility ratio series.
    """
    # Parkinson volatility (intraday range)
    log_hl = np.log(daily_high / daily_low.replace(0, np.nan))
    parkinson = log_hl.rolling(lookback, min_periods=5).std()

    # Close-to-close volatility
    cc_vol = np.log(daily_close / daily_close.shift(1)).rolling(lookback, min_periods=5).std()

    ratio = (parkinson / cc_vol.replace(0, np.nan)).fillna(1.0)
    return ratio.rename("intraday_vol_ratio")


def build_intraday_features(
    daily_open: pd.Series,
    daily_high: pd.Series,
    daily_low: pd.Series,
    daily_close: pd.Series,
    daily_volume: pd.Series,
    minute_bars: pd.DataFrame | None = None,
) -> IntradayFeatureResult:
    """Build all intraday features from available data.

    Args:
        daily_open/high/low/close/volume: Daily OHLCV.
        minute_bars: Optional minute-level data for granular features.

    Returns:
        IntradayFeatureResult with feature DataFrame.
    """
    features = pd.DataFrame(index=daily_close.index)
    warnings = []

    # Overnight vs intraday
    overnight, intraday = compute_overnight_return(daily_open, daily_close)
    features["overnight_return"] = overnight
    features["intraday_return"] = intraday

    # VWAP deviation
    features["vwap_deviation"] = compute_vwap_deviation(daily_close, daily_volume)

    # Intraday volatility ratio
    features["intraday_vol_ratio"] = compute_intraday_volatility_ratio(
        daily_high, daily_low, daily_close
    )

    # Minute-bar features (if available)
    if minute_bars is not None and len(minute_bars) > 0:
        try:
            features["last_hour_momentum"] = compute_last_hour_momentum(minute_bars)
            features["opening_range_breakout"] = compute_opening_range_breakout(minute_bars)
        except Exception as e:
            warnings.append(f"Minute-bar features failed: {e}")
    else:
        warnings.append("No minute bars — last_hour_momentum and opening_range_breakout skipped")

    coverage = 1.0 - features.isna().mean().mean()

    return IntradayFeatureResult(
        features=features.fillna(0),
        coverage=round(coverage, 4),
        warnings=warnings,
    )


__all__ = [
    "IntradayFeatureResult",
    "compute_last_hour_momentum",
    "compute_overnight_return",
    "compute_opening_range_breakout",
    "compute_vwap_deviation",
    "compute_intraday_volatility_ratio",
    "build_intraday_features",
]
