"""Candlestick Pattern Recognition for OHLC Price Data.

Pure pandas/numpy implementation — no ta-lib dependency. Each pattern
function returns a float column: +1.0 = bullish pattern, -1.0 = bearish,
0.0 = no pattern.

Main entry point:
    build_candlestick_features(prices_df) -> pd.DataFrame

Patterns implemented:
    Doji, Hammer, Hanging Man, Bullish/Bearish Engulfing,
    Morning Star, Evening Star, Shooting Star,
    Three White Soldiers, Three Black Crows, Bullish/Bearish Harami
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OPEN = "open"
_HIGH = "high"
_LOW = "low"
_CLOSE = "close"
_SYMBOL = "symbol"
_TIMESTAMP = "timestamp"

# Threshold ratios used across patterns
_DOJI_BODY_RATIO = 0.1       # body/range <= 10% → doji
_SMALL_BODY_RATIO = 0.3      # body/range <= 30% → small body
_LONG_SHADOW_RATIO = 2.0     # shadow >= 2x body length → long shadow


def _body(df: pd.DataFrame) -> pd.Series:
    return (df[_CLOSE] - df[_OPEN]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    r = df[_HIGH] - df[_LOW]
    return r.replace(0, np.nan)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df[_HIGH] - df[[_OPEN, _CLOSE]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[[_OPEN, _CLOSE]].min(axis=1) - df[_LOW]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df[_CLOSE] > df[_OPEN]


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    return df[_CLOSE] < df[_OPEN]


# ---------------------------------------------------------------------------
# Individual pattern functions — all operate per-symbol on a sorted DataFrame
# ---------------------------------------------------------------------------


def _add_doji(df: pd.DataFrame) -> pd.Series:
    """Body/range ratio <= DOJI_BODY_RATIO and range > 0."""
    body = _body(df)
    rng = _range(df)
    return pd.Series(
        np.where((body / rng).fillna(1.0) <= _DOJI_BODY_RATIO, 1.0, 0.0),
        index=df.index, name="cs_doji_v1",
    )


def _add_hammer(df: pd.DataFrame) -> pd.Series:
    """Small body near top, long lower shadow, small upper shadow — bullish."""
    body = _body(df)
    rng = _range(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    cond = (
        ((body / rng.fillna(1e-9)).fillna(1.0) <= _SMALL_BODY_RATIO)
        & ((lower >= _LONG_SHADOW_RATIO * body.replace(0, np.nan)).fillna(False))
        & ((upper <= 0.1 * rng.fillna(1e-9)).fillna(False))
    )
    return pd.Series(np.where(cond, 1.0, 0.0), index=df.index, name="cs_hammer_v1")


def _add_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Same shape as Hammer but occurs after uptrend — bearish."""
    body = _body(df)
    rng = _range(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    cond = (
        ((body / rng.fillna(1e-9)).fillna(1.0) <= _SMALL_BODY_RATIO)
        & ((lower >= _LONG_SHADOW_RATIO * body.replace(0, np.nan)).fillna(False))
        & ((upper <= 0.1 * rng.fillna(1e-9)).fillna(False))
    )
    return pd.Series(np.where(cond, -1.0, 0.0), index=df.index, name="cs_hanging_man_v1")


def _add_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Small body near bottom, long upper shadow — bearish."""
    body = _body(df)
    rng = _range(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    cond = (
        ((body / rng.fillna(1e-9)).fillna(1.0) <= _SMALL_BODY_RATIO)
        & ((upper >= _LONG_SHADOW_RATIO * body.replace(0, np.nan)).fillna(False))
        & ((lower <= 0.1 * rng.fillna(1e-9)).fillna(False))
    )
    return pd.Series(np.where(cond, -1.0, 0.0), index=df.index, name="cs_shooting_star_v1")


def _add_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish (+1) or Bearish (-1) Engulfing pattern."""
    open_ = df[_OPEN].values
    close_ = df[_CLOSE].values
    result = np.zeros(len(df))
    for i in range(1, len(df)):
        prev_bear = close_[i - 1] < open_[i - 1]
        prev_bull = close_[i - 1] > open_[i - 1]
        curr_bull = close_[i] > open_[i]
        curr_bear = close_[i] < open_[i]
        # Bullish engulfing: previous candle bearish, current candle bullish and engulfs
        if prev_bear and curr_bull:
            if open_[i] <= close_[i - 1] and close_[i] >= open_[i - 1]:
                result[i] = 1.0
        # Bearish engulfing: previous candle bullish, current candle bearish and engulfs
        elif prev_bull and curr_bear:
            if open_[i] >= close_[i - 1] and close_[i] <= open_[i - 1]:
                result[i] = -1.0
    return pd.Series(result, index=df.index, name="cs_engulfing_v1")


def _add_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish (+1) or Bearish (-1) Harami — small body inside previous large body."""
    open_ = df[_OPEN].values
    close_ = df[_CLOSE].values
    result = np.zeros(len(df))
    for i in range(1, len(df)):
        prev_top = max(open_[i - 1], close_[i - 1])
        prev_bot = min(open_[i - 1], close_[i - 1])
        curr_top = max(open_[i], close_[i])
        curr_bot = min(open_[i], close_[i])
        inside = curr_top < prev_top and curr_bot > prev_bot
        if not inside:
            continue
        # Bullish harami: previous bearish, current bullish
        if close_[i - 1] < open_[i - 1] and close_[i] > open_[i]:
            result[i] = 1.0
        # Bearish harami: previous bullish, current bearish
        elif close_[i - 1] > open_[i - 1] and close_[i] < open_[i]:
            result[i] = -1.0
    return pd.Series(result, index=df.index, name="cs_harami_v1")


def _add_morning_star(df: pd.DataFrame) -> pd.Series:
    """Three-candle bullish reversal: large bearish, small body, large bullish."""
    open_ = df[_OPEN].values
    close_ = df[_CLOSE].values
    result = np.zeros(len(df))
    for i in range(2, len(df)):
        c1_bear = close_[i - 2] < open_[i - 2]
        c1_large = abs(close_[i - 2] - open_[i - 2]) > 0.5 * (df[_HIGH].iloc[i - 2] - df[_LOW].iloc[i - 2]) if (df[_HIGH].iloc[i - 2] - df[_LOW].iloc[i - 2]) > 0 else False
        c2_small = abs(close_[i - 1] - open_[i - 1]) < 0.3 * abs(close_[i - 2] - open_[i - 2]) if abs(close_[i - 2] - open_[i - 2]) > 0 else False
        c3_bull = close_[i] > open_[i]
        c3_close_above_midpoint = close_[i] > (open_[i - 2] + close_[i - 2]) / 2
        if c1_bear and c1_large and c2_small and c3_bull and c3_close_above_midpoint:
            result[i] = 1.0
    return pd.Series(result, index=df.index, name="cs_morning_star_v1")


def _add_evening_star(df: pd.DataFrame) -> pd.Series:
    """Three-candle bearish reversal: large bullish, small body, large bearish."""
    open_ = df[_OPEN].values
    close_ = df[_CLOSE].values
    result = np.zeros(len(df))
    for i in range(2, len(df)):
        c1_bull = close_[i - 2] > open_[i - 2]
        c1_large = abs(close_[i - 2] - open_[i - 2]) > 0.5 * (df[_HIGH].iloc[i - 2] - df[_LOW].iloc[i - 2]) if (df[_HIGH].iloc[i - 2] - df[_LOW].iloc[i - 2]) > 0 else False
        c2_small = abs(close_[i - 1] - open_[i - 1]) < 0.3 * abs(close_[i - 2] - open_[i - 2]) if abs(close_[i - 2] - open_[i - 2]) > 0 else False
        c3_bear = close_[i] < open_[i]
        c3_close_below_midpoint = close_[i] < (open_[i - 2] + close_[i - 2]) / 2
        if c1_bull and c1_large and c2_small and c3_bear and c3_close_below_midpoint:
            result[i] = -1.0
    return pd.Series(result, index=df.index, name="cs_evening_star_v1")


def _add_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three consecutive strong bullish candles — continuation bullish signal."""
    close_ = df[_CLOSE].values
    open_ = df[_OPEN].values
    result = np.zeros(len(df))
    for i in range(2, len(df)):
        bullish = [close_[j] > open_[j] for j in range(i - 2, i + 1)]
        rising = close_[i - 2] < close_[i - 1] < close_[i]
        opens_within = (
            open_[i - 1] > open_[i - 2] and open_[i - 1] < close_[i - 2]
            and open_[i] > open_[i - 1] and open_[i] < close_[i - 1]
        )
        if all(bullish) and rising and opens_within:
            result[i] = 1.0
    return pd.Series(result, index=df.index, name="cs_three_white_soldiers_v1")


def _add_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three consecutive strong bearish candles — continuation bearish signal."""
    close_ = df[_CLOSE].values
    open_ = df[_OPEN].values
    result = np.zeros(len(df))
    for i in range(2, len(df)):
        bearish = [close_[j] < open_[j] for j in range(i - 2, i + 1)]
        falling = close_[i - 2] > close_[i - 1] > close_[i]
        opens_within = (
            open_[i - 1] < open_[i - 2] and open_[i - 1] > close_[i - 2]
            and open_[i] < open_[i - 1] and open_[i] > close_[i - 1]
        )
        if all(bearish) and falling and opens_within:
            result[i] = -1.0
    return pd.Series(result, index=df.index, name="cs_three_black_crows_v1")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_PATTERN_FUNCTIONS = [
    (_add_doji, "cs_doji_v1"),
    (_add_hammer, "cs_hammer_v1"),
    (_add_hanging_man, "cs_hanging_man_v1"),
    (_add_shooting_star, "cs_shooting_star_v1"),
    (_add_engulfing, "cs_engulfing_v1"),
    (_add_harami, "cs_harami_v1"),
    (_add_morning_star, "cs_morning_star_v1"),
    (_add_evening_star, "cs_evening_star_v1"),
    (_add_three_white_soldiers, "cs_three_white_soldiers_v1"),
    (_add_three_black_crows, "cs_three_black_crows_v1"),
]


def build_candlestick_features(
    prices: pd.DataFrame,
    symbol_col: str = _SYMBOL,
    timestamp_col: str = _TIMESTAMP,
) -> pd.DataFrame:
    """Compute all candlestick pattern features for a price panel.

    Args:
        prices: OHLCV panel with columns: symbol, timestamp, open, high, low, close.
                Must include at least open, high, low, close.
        symbol_col: Name of symbol column (default: "symbol")
        timestamp_col: Name of timestamp column (default: "timestamp")

    Returns:
        Original DataFrame with added candlestick pattern columns (cs_*_v1).
        Pattern values: +1.0 = bullish, -1.0 = bearish, 0.0 = no pattern.
    """
    required = [_OPEN, _HIGH, _LOW, _CLOSE]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        logger.warning("[Candlestick] Missing OHLC columns: %s — skipping", missing)
        return prices.copy()

    result = prices.copy()
    result = result.sort_values([symbol_col, timestamp_col])

    if symbol_col in result.columns:
        for col_fn, col_name in _PATTERN_FUNCTIONS:
            result[col_name] = (
                result.groupby(symbol_col, group_keys=False)
                .apply(lambda g: col_fn(g))
                .reset_index(level=0, drop=True)
            )
    else:
        for col_fn, col_name in _PATTERN_FUNCTIONS:
            result[col_name] = col_fn(result)

    return result


def get_candlestick_feature_names() -> list[str]:
    """Return list of all candlestick feature column names."""
    return [name for _, name in _PATTERN_FUNCTIONS]
