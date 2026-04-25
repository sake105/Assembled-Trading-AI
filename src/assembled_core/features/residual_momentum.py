"""Residual Momentum (FF5 factor-adjusted).

From 13_FREE_MODULE.md §13.7.
Blitz et al. 2011: ~2× Sharpe vs total-return momentum.

Method:
1. Download Ken French FF5 + Momentum factors (free via pandas_datareader).
2. Rolling OLS regression of ticker returns on factors (252-bar window).
3. Compute 12-1 momentum on the residuals (11 months, skip last month).
4. Cross-sectional rank as the signal.

Install: pip install pandas-datareader statsmodels
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_ff_factors(start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame | None:
    try:
        from pandas_datareader import data as pdr
        ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start, end=end)[0]
        mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start, end=end)[0]
        factors = pd.concat([ff5, mom], axis=1) / 100.0  # convert from percent
        return factors
    except Exception as exc:
        logger.warning("FF5 factor download failed: %s — install pandas-datareader", exc)
        return None


def compute_residual_momentum(
    ticker_returns: pd.Series,
    factors: pd.DataFrame | None = None,
    window: int = 252,
    skip_last_month: int = 21,
    momentum_window: int = 252,
) -> pd.Series:
    """Compute factor-residual momentum for a single ticker.

    Args:
        ticker_returns: Daily log-return series for the ticker.
        factors: DataFrame with FF5+Mom factor returns (same index).
                 If None, downloads from Ken French website.
        window: Rolling OLS window (bars, default 252 = 1 year).
        skip_last_month: Skip most recent N bars to avoid reversal (default 21).
        momentum_window: Total momentum lookback (default 252 = 12 months).

    Returns:
        Series of residual momentum values (one per bar where calculable).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not installed — pip install statsmodels")
        return pd.Series(dtype=float)

    if factors is None:
        factors = _try_import_ff_factors()
        if factors is None:
            return pd.Series(dtype=float)

    # Align to common index
    common = ticker_returns.index.intersection(factors.index)
    if len(common) < window + skip_last_month + 20:
        logger.debug("Insufficient data for residual momentum: %d bars", len(common))
        return pd.Series(dtype=float)

    ret = ticker_returns.loc[common]
    fac = factors.loc[common]

    results = []
    indices = []

    for i in range(window, len(common)):
        y = ret.iloc[i - window:i]
        X = fac.iloc[i - window:i]
        X_with_const = sm.add_constant(X, has_constant="add")
        try:
            ols_res = sm.OLS(y, X_with_const).fit()
            residuals = ols_res.resid
        except Exception:
            continue

        # 11-1 momentum on residuals: window minus last skip_last_month bars
        start_idx = 0
        end_idx = len(residuals) - skip_last_month
        if end_idx <= start_idx + 20:
            continue

        mom_val = float(residuals.iloc[start_idx:end_idx].mean())
        results.append(mom_val)
        indices.append(common[i])

    if not results:
        return pd.Series(dtype=float)

    return pd.Series(results, index=indices, name="residual_momentum_ff5")


def cross_sectional_residual_momentum(
    returns_panel: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    window: int = 252,
    skip_last_month: int = 21,
) -> pd.DataFrame:
    """Compute residual momentum for all tickers in a panel.

    Args:
        returns_panel: Wide DataFrame, columns=tickers, index=dates, values=returns.
        factors: FF5+Mom factors (downloaded if None).
        window: OLS rolling window.
        skip_last_month: Bars to skip for reversal avoidance.

    Returns:
        Wide DataFrame of residual momentum scores, same shape as input.
    """
    if factors is None:
        factors = _try_import_ff_factors()
        if factors is None:
            return pd.DataFrame()

    scores = {}
    for ticker in returns_panel.columns:
        s = compute_residual_momentum(
            returns_panel[ticker].dropna(),
            factors=factors,
            window=window,
            skip_last_month=skip_last_month,
        )
        if not s.empty:
            scores[ticker] = s

    if not scores:
        return pd.DataFrame()

    result = pd.DataFrame(scores)
    # Cross-sectional z-score per date (ranking signal)
    result_z = result.sub(result.mean(axis=1), axis=0).div(
        result.std(axis=1).replace(0, np.nan), axis=0
    )
    result_z.columns.name = "ticker"
    return result_z


__all__ = [
    "compute_residual_momentum",
    "cross_sectional_residual_momentum",
]
