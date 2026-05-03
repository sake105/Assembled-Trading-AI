"""High-Low-Open-Close volatility estimators (Parkinson 1980, Garman-Klass 1980).

Significantly more efficient than close-to-close estimators because they
incorporate intra-day range information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_volatility(
    high: pd.Series, low: pd.Series, period: int = 20
) -> pd.Series:
    """Parkinson (1980) estimator using only High/Low.

    More efficient than close-to-close at ~5× the statistical efficiency.
    """
    log_hl = np.log((high / low).clip(lower=1e-10)) ** 2
    return np.sqrt(log_hl.rolling(period).sum() / (4 * period * np.log(2)))


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Garman-Klass (1980) estimator using OHLC.

    ~8× more efficient than close-to-close; the dominant bar-level vol estimator.
    """
    log_hl = np.log((high / low).clip(lower=1e-10)) ** 2
    log_co = np.log((close / open_).clip(lower=1e-10)) ** 2
    daily_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(daily_var.rolling(period).sum() / period)


def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rogers-Satchell (1991) estimator — handles drift, no open-gap bias.

    Preferred when underlying has non-zero drift.
    """
    log_ho = np.log((high / open_).clip(lower=1e-10))
    log_hc = np.log((high / close).clip(lower=1e-10))
    log_lo = np.log((low / open_).clip(lower=1e-10))
    log_lc = np.log((low / close).clip(lower=1e-10))
    daily_var = log_ho * log_hc + log_lo * log_lc
    return np.sqrt(daily_var.rolling(period).mean().clip(lower=0))


def tick_rule_signs(prices: pd.Series) -> pd.Series:
    """Lee-Ready (1991) tick rule: +1 if price uptick, -1 if downtick.

    Allows buyer/seller classification of trades without bid/ask data.
    """
    diffs = prices.diff()
    signs = np.sign(diffs).replace(0, np.nan).ffill().fillna(1)
    return signs.astype(int)


def close_to_close_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Standard close-to-close volatility (baseline for comparison)."""
    return np.log(close.clip(lower=1e-10)).diff().rolling(period).std() * np.sqrt(252)


def volatility_panel(
    ohlc: pd.DataFrame,
    period: int = 20,
    annualise: bool = True,
) -> pd.DataFrame:
    """Compute all available estimators from an OHLC DataFrame.

    Parameters
    ----------
    ohlc:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    period:
        Rolling window in trading days.
    annualise:
        If True, multiply by sqrt(252).

    Returns
    -------
    DataFrame with columns: parkinson, garman_klass, rogers_satchell, close_to_close.
    """
    scale = np.sqrt(252) if annualise else 1.0
    o, h, low, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    return pd.DataFrame(
        {
            "parkinson": parkinson_volatility(h, low, period) * scale,
            "garman_klass": garman_klass_volatility(o, h, low, c, period) * scale,
            "rogers_satchell": rogers_satchell_volatility(o, h, low, c, period) * scale,
            "close_to_close": close_to_close_volatility(c, period),
        },
        index=ohlc.index,
    )
