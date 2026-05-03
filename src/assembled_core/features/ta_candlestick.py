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
    o = df[_OPEN]
    c = df[_CLOSE]
    prev_bear = c.shift(1) < o.shift(1)
    prev_bull = c.shift(1) > o.shift(1)
    curr_bull = c > o
    curr_bear = c < o
    bullish = prev_bear & curr_bull & (o <= c.shift(1)) & (c >= o.shift(1))
    bearish = prev_bull & curr_bear & (o >= c.shift(1)) & (c <= o.shift(1))
    return pd.Series(
        np.where(bullish, 1.0, np.where(bearish, -1.0, 0.0)),
        index=df.index, name="cs_engulfing_v1",
    )


def _add_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish (+1) or Bearish (-1) Harami — small body inside previous large body."""
    o = df[_OPEN].values
    c = df[_CLOSE].values
    prev_top = np.maximum(o[:-1], c[:-1])
    prev_bot = np.minimum(o[:-1], c[:-1])
    curr_top = np.maximum(o[1:], c[1:])
    curr_bot = np.minimum(o[1:], c[1:])
    inside = (curr_top < prev_top) & (curr_bot > prev_bot)
    bullish = inside & (c[:-1] < o[:-1]) & (c[1:] > o[1:])
    bearish = inside & (c[:-1] > o[:-1]) & (c[1:] < o[1:])
    result = np.zeros(len(df))
    result[1:] = np.where(bullish, 1.0, np.where(bearish, -1.0, 0.0))
    return pd.Series(result, index=df.index, name="cs_harami_v1")


def _add_morning_star(df: pd.DataFrame) -> pd.Series:
    """Three-candle bullish reversal: large bearish, small body, large bullish."""
    o = df[_OPEN]
    c = df[_CLOSE]
    body = (c - o).abs()
    rng = df[_HIGH] - df[_LOW]
    body2 = body.shift(2)
    rng2 = rng.shift(2)
    c1_bear = c.shift(2) < o.shift(2)
    c1_large = body2 > 0.5 * rng2.where(rng2 > 0)
    c2_small = body.shift(1) < 0.3 * body2.where(body2 > 0)
    c3_bull = c > o
    c3_above = c > (o.shift(2) + c.shift(2)) / 2
    cond = c1_bear & c1_large.fillna(False) & c2_small.fillna(False) & c3_bull & c3_above
    return pd.Series(np.where(cond, 1.0, 0.0), index=df.index, name="cs_morning_star_v1")


def _add_evening_star(df: pd.DataFrame) -> pd.Series:
    """Three-candle bearish reversal: large bullish, small body, large bearish."""
    o = df[_OPEN]
    c = df[_CLOSE]
    body = (c - o).abs()
    rng = df[_HIGH] - df[_LOW]
    body2 = body.shift(2)
    rng2 = rng.shift(2)
    c1_bull = c.shift(2) > o.shift(2)
    c1_large = body2 > 0.5 * rng2.where(rng2 > 0)
    c2_small = body.shift(1) < 0.3 * body2.where(body2 > 0)
    c3_bear = c < o
    c3_below = c < (o.shift(2) + c.shift(2)) / 2
    cond = c1_bull & c1_large.fillna(False) & c2_small.fillna(False) & c3_bear & c3_below
    return pd.Series(np.where(cond, -1.0, 0.0), index=df.index, name="cs_evening_star_v1")


def _add_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three consecutive strong bullish candles — continuation bullish signal."""
    o = df[_OPEN]
    c = df[_CLOSE]
    all_bull = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))
    rising = (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    opens_within = (
        (o.shift(1) > o.shift(2)) & (o.shift(1) < c.shift(2))
        & (o > o.shift(1)) & (o < c.shift(1))
    )
    cond = all_bull & rising & opens_within
    return pd.Series(np.where(cond, 1.0, 0.0), index=df.index, name="cs_three_white_soldiers_v1")


def _add_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three consecutive strong bearish candles — continuation bearish signal."""
    o = df[_OPEN]
    c = df[_CLOSE]
    all_bear = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2))
    falling = (c.shift(2) > c.shift(1)) & (c.shift(1) > c)
    opens_within = (
        (o.shift(1) < o.shift(2)) & (o.shift(1) > c.shift(2))
        & (o < o.shift(1)) & (o > c.shift(1))
    )
    cond = all_bear & falling & opens_within
    return pd.Series(np.where(cond, -1.0, 0.0), index=df.index, name="cs_three_black_crows_v1")


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
                .apply(lambda g: col_fn(g), include_groups=False)
                .reset_index(level=0, drop=True)
            )
    else:
        for col_fn, col_name in _PATTERN_FUNCTIONS:
            result[col_name] = col_fn(result)

    return result


def get_candlestick_feature_names() -> list[str]:
    """Return list of all candlestick feature column names."""
    return [name for _, name in _PATTERN_FUNCTIONS]
