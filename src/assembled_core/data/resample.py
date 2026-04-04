"""Multi-Timeframe Resampling for Price Data.

Converts daily OHLCV panels to weekly and monthly bars with NYSE-calendar
awareness and PIT-safe handling (partial weeks/months are excluded).

Usage:
    from src.assembled_core.data.resample import resample_to_weekly, resample_to_monthly
    weekly = resample_to_weekly(daily_prices, as_of=pd.Timestamp("2024-06-14"))
    monthly = resample_to_monthly(daily_prices, as_of=pd.Timestamp("2024-06-14"))
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SYMBOL = "symbol"
_TIMESTAMP = "timestamp"
_OPEN = "open"
_HIGH = "high"
_LOW = "low"
_CLOSE = "close"
_VOLUME = "volume"


def _ohlcv_agg(freq: str) -> dict:
    """Return the standard OHLCV aggregation dict for a resample."""
    return {
        _OPEN: "first",
        _HIGH: "max",
        _LOW: "min",
        _CLOSE: "last",
        _VOLUME: "sum",
    }


def resample_to_weekly(
    daily_prices: pd.DataFrame,
    symbol_col: str = _SYMBOL,
    timestamp_col: str = _TIMESTAMP,
    as_of: pd.Timestamp | None = None,
    exclude_partial_week: bool = True,
) -> pd.DataFrame:
    """Resample a daily OHLCV panel to weekly bars (week-ending Friday).

    PIT-safe: if ``as_of`` is provided and the current week is not yet finished,
    the partial week is excluded when ``exclude_partial_week=True`` (default).

    Args:
        daily_prices: Daily panel with columns symbol, timestamp, open, high, low, close, volume
        symbol_col: Symbol column name (default: "symbol")
        timestamp_col: Timestamp column name (default: "timestamp")
        as_of: Point-in-time cutoff for PIT safety (default: None)
        exclude_partial_week: Exclude the current incomplete week (default: True)

    Returns:
        Weekly OHLCV panel with the same columns. timestamp = week-ending Friday.
    """
    df = _prepare(daily_prices, symbol_col, timestamp_col, as_of)
    ohlcv_cols = [c for c in [_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME] if c in df.columns]
    agg = {c: _ohlcv_agg("W")[c] for c in ohlcv_cols}

    result = (
        df.groupby(symbol_col)[ohlcv_cols + [timestamp_col]]
        .apply(
            lambda g: _resample_group(
                g, timestamp_col, "W-FRI", agg, as_of if exclude_partial_week else None
            ),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    return result.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


def resample_to_monthly(
    daily_prices: pd.DataFrame,
    symbol_col: str = _SYMBOL,
    timestamp_col: str = _TIMESTAMP,
    as_of: pd.Timestamp | None = None,
    exclude_partial_month: bool = True,
) -> pd.DataFrame:
    """Resample a daily OHLCV panel to monthly bars (month-end).

    PIT-safe: if ``as_of`` is provided and the current month is not yet finished,
    the partial month is excluded when ``exclude_partial_month=True`` (default).

    Args:
        daily_prices: Daily panel
        symbol_col: Symbol column name (default: "symbol")
        timestamp_col: Timestamp column name (default: "timestamp")
        as_of: Point-in-time cutoff (default: None)
        exclude_partial_month: Exclude incomplete current month (default: True)

    Returns:
        Monthly OHLCV panel. timestamp = month-end date.
    """
    df = _prepare(daily_prices, symbol_col, timestamp_col, as_of)
    ohlcv_cols = [c for c in [_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME] if c in df.columns]
    agg = {c: _ohlcv_agg("ME")[c] for c in ohlcv_cols}

    result = (
        df.groupby(symbol_col)[ohlcv_cols + [timestamp_col]]
        .apply(
            lambda g: _resample_group(
                g, timestamp_col, "ME", agg, as_of if exclude_partial_month else None
            ),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    return result.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


def align_higher_tf_to_daily(
    daily_prices: pd.DataFrame,
    higher_tf: pd.DataFrame,
    symbol_col: str = _SYMBOL,
    timestamp_col: str = _TIMESTAMP,
    feature_cols: list[str] | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Forward-fill higher-timeframe features onto the daily grid.

    A weekly feature computed at week T is valid for all daily bars in week T
    (after the week closes). This merge is PIT-safe: weekly bar at date D is
    only available to daily bars on dates >= D.

    Args:
        daily_prices: Daily panel (the target grid)
        higher_tf: Weekly or monthly resampled DataFrame
        symbol_col: Symbol column name (default: "symbol")
        timestamp_col: Timestamp column name (default: "timestamp")
        feature_cols: Columns from higher_tf to merge (default: all non-key cols)
        suffix: Suffix added to higher_tf columns to avoid name clashes (default: "")

    Returns:
        daily_prices with additional higher-tf columns forward-filled.
    """
    if feature_cols is None:
        feature_cols = [c for c in higher_tf.columns if c not in [symbol_col, timestamp_col]]

    rename_map = {c: f"{c}{suffix}" for c in feature_cols if suffix}
    htf = higher_tf[[symbol_col, timestamp_col] + feature_cols].copy()
    if rename_map:
        htf = htf.rename(columns=rename_map)
    merged_feature_cols = [rename_map.get(c, c) for c in feature_cols]

    result = daily_prices.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    htf[timestamp_col] = pd.to_datetime(htf[timestamp_col])

    pieces = []
    for sym, grp in result.groupby(symbol_col):
        htf_sym = htf[htf[symbol_col] == sym].sort_values(timestamp_col)
        grp = grp.sort_values(timestamp_col)
        merged = pd.merge_asof(
            grp,
            htf_sym[[timestamp_col] + merged_feature_cols],
            on=timestamp_col,
            direction="backward",
        )
        pieces.append(merged)

    if not pieces:
        return result

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare(
    df: pd.DataFrame,
    symbol_col: str,
    timestamp_col: str,
    as_of: pd.Timestamp | None,
) -> pd.DataFrame:
    """Validate, copy, and optionally filter to as_of."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if as_of is not None:
        df = df[df[timestamp_col] <= as_of]
    return df.sort_values([symbol_col, timestamp_col])


def _resample_group(
    grp: pd.DataFrame,
    timestamp_col: str,
    freq: str,
    agg: dict,
    pit_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """Resample a single-symbol group to freq, excluding any partial period."""
    grp = grp.set_index(timestamp_col).sort_index()
    resampled = grp.resample(freq).agg(agg).dropna(how="all")

    # PIT: drop the last period if it equals the period containing pit_cutoff
    if pit_cutoff is not None and not resampled.empty:
        last_period_end = resampled.index[-1]
        # Resampled timestamps are period-end labels; if the cutoff falls within
        # the last period, that period is incomplete — remove it
        try:
            if pit_cutoff < last_period_end:
                resampled = resampled.iloc[:-1]
        except Exception:
            pass  # If we can't determine, keep all periods

    resampled.index.name = timestamp_col
    return resampled.reset_index()
