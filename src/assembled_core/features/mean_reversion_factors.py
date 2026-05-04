"""Mean-reversion factor sidecar for multifactor_v2.

Computes three mean-reversion factor columns as a pure sidecar. No wiring
into multifactor_v1 or multifactor_v2 is performed here — this module only
produces a DataFrame with the factor columns, so the caller can blend them
into the multifactor frame in a separate, explicit step.

Factors
-------
1. ``mr_zscore_reversal_3d``
   3-day return z-score over a 60-day rolling window, inverted so that an
   extreme down-move produces a positive signal. Clipped to ``[-3, 3]``.

2. ``mr_rsi_extreme_uptrend``
   Long-only oversold-in-uptrend signal. Active only when RSI(14) < 30
   **and** EMA(50) > EMA(200). Output lies in ``[0, 1]``.

3. ``mr_bollinger_squeeze_break``
   Mean-reversion signal that weights ``(1 - %B)`` by the inverse of the
   current squeeze ratio, emphasising tags of the lower band inside a
   compressed range.

Design notes
------------
* All operations are grouped by symbol so that symbols do not contaminate
  each other (no global rolling across the stacked frame).
* Rows with insufficient history return ``NaN`` in the affected column
  rather than being dropped — the caller decides on the masking policy.
* The module never raises on all-NaN input; missing closes simply
  propagate as ``NaN`` factor values.
* ``rsi_14`` is implemented with **Wilder's smoothing** (the standard RSI
  formulation), computed via an exponentially weighted mean with
  ``alpha = 1 / 14`` on the positive and negative price-change series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["compute_mean_reversion_factors"]


_FACTOR_COLUMNS = [
    "mr_zscore_reversal_3d",
    "mr_rsi_extreme_uptrend",
    "mr_bollinger_squeeze_break",
]


def _rsi_14(close: pd.Series) -> pd.Series:
    """Wilder-smoothed 14-period RSI.

    Uses ``ewm(alpha=1/14, adjust=False)`` which is the canonical
    discrete form of Wilder's smoothing.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 - 0.0) / (1.0 + rs)
    # When avg_loss is 0 and avg_gain > 0 → RSI = 100
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    # When both are 0 → undefined; leave NaN.
    return rsi


def _zscore_reversal_3d(close: pd.Series) -> pd.Series:
    ret_3d = close.pct_change(3, fill_method=None)
    rolling_mean = ret_3d.rolling(60, min_periods=60).mean()
    rolling_std = ret_3d.rolling(60, min_periods=60).std()
    z = (ret_3d - rolling_mean) / rolling_std.replace(0.0, np.nan)
    factor = -z
    return factor.clip(lower=-3.0, upper=3.0)


def _rsi_extreme_uptrend(close: pd.Series) -> pd.Series:
    rsi = _rsi_14(close)
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    uptrend_flag = (ema_50 > ema_200).astype(float)
    # Propagate NaNs from RSI so that insufficient-history rows stay NaN.
    scaled = (30.0 - rsi).clip(lower=0.0) / 30.0
    return scaled * uptrend_flag


def _bollinger_squeeze_break(close: pd.Series) -> pd.Series:
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = ma20 + 2.0 * std20
    lower = ma20 - 2.0 * std20
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    squeeze_width = (upper - lower) / ma20.replace(0.0, np.nan)
    squeeze_mean = squeeze_width.rolling(60, min_periods=60).mean()
    squeeze_ratio = squeeze_width / squeeze_mean.replace(0.0, np.nan)

    inv_squeeze = (1.0 / squeeze_ratio.clip(lower=0.5)).clip(upper=2.0)
    factor = (1.0 - pct_b).clip(lower=0.0, upper=1.0) * inv_squeeze
    return factor


def _compute_for_group(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mr_zscore_reversal_3d": _zscore_reversal_3d(close),
            "mr_rsi_extreme_uptrend": _rsi_extreme_uptrend(close),
            "mr_bollinger_squeeze_break": _bollinger_squeeze_break(close),
        },
        index=close.index,
    )


def compute_mean_reversion_factors(
    prices: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute three mean-reversion factors per symbol.

    Parameters
    ----------
    prices:
        Long-format price frame. Must contain ``symbol_col``,
        ``timestamp_col`` and ``close_col``.
    symbol_col, timestamp_col, close_col:
        Column name overrides.

    Returns
    -------
    pandas.DataFrame
        ``[timestamp, symbol, mr_zscore_reversal_3d,
        mr_rsi_extreme_uptrend, mr_bollinger_squeeze_break]``.
        Rows where a factor cannot yet be computed carry ``NaN`` in that
        column rather than being dropped.
    """
    required = {symbol_col, timestamp_col, close_col}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing required columns: {sorted(missing)}")

    if prices.empty:
        return pd.DataFrame(columns=[timestamp_col, symbol_col, *_FACTOR_COLUMNS])

    # Stable ordering: symbol, then timestamp. We preserve original row
    # order in the output by remembering the source index.
    work = prices[[symbol_col, timestamp_col, close_col]].copy()
    work = work.sort_values([symbol_col, timestamp_col], kind="mergesort")

    frames: list[pd.DataFrame] = []
    for sym, grp in work.groupby(symbol_col, sort=False):
        close = grp[close_col].astype(float).reset_index(drop=True)
        factor_frame = _compute_for_group(close)
        out = factor_frame.copy()
        out[timestamp_col] = grp[timestamp_col].to_numpy()
        out[symbol_col] = sym
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    return result[[timestamp_col, symbol_col, *_FACTOR_COLUMNS]]
