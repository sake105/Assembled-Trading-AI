"""Sektor-Rotation-Strategien.

Klassiker
---------
- **Faber Trend** (Faber 2007): Long Sektor wenn close > 10M-MA, sonst Cash.
- **Relative Strength**: Top-N-Sektoren nach 12-1-Momentum.
- **Risk-Parity zwischen Sektoren**.

Reference
---------
- Faber, M. (2007). A Quantitative Approach to Tactical Asset Allocation.
  *J. Wealth Management*.
- Asness, C. (2011). Momentum in Japan: The Exception That Proves the Rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def faber_trend_filter(prices: pd.Series, lookback: int = 200) -> pd.Series:
    """Faber-Filter: 1 wenn price > MA(lookback), sonst 0.

    Long-only Sector-Rotation Default.
    """
    s = pd.Series(prices)
    ma = s.rolling(lookback, min_periods=lookback // 2).mean()
    return (s > ma).astype(float).shift(1).fillna(0)  # PIT-shift


def relative_strength_ranking(
    sector_prices: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """Rank-Sektoren nach (lookback-skip)-Momentum.

    Args:
        sector_prices: DataFrame (T, n_sectors) — Preise je Sektor.
        lookback, skip: Klassisches 12-1.

    Returns:
        DataFrame (T, n_sectors) mit ranks (0 = best, n-1 = worst).
    """
    s = sector_prices.copy()
    mom = s.shift(skip) / s.shift(lookback) - 1
    return mom.rank(axis=1, ascending=False)


def top_n_sector_strategy(
    sector_prices: pd.DataFrame,
    n_top: int = 3,
    lookback: int = 252,
    skip: int = 21,
    with_trend_filter: bool = True,
    trend_lookback: int = 200,
) -> pd.DataFrame:
    """Long Top-N momentum Sektoren, optional mit Trend-Filter.

    Returns:
        DataFrame (T, n_sectors) mit Gewichten.
    """
    ranks = relative_strength_ranking(sector_prices, lookback=lookback, skip=skip)
    # Long if rank <= n_top
    long_mask = (ranks <= n_top).astype(float)

    if with_trend_filter:
        trend = sector_prices.apply(
            lambda col: faber_trend_filter(col, lookback=trend_lookback)
        )
        long_mask = long_mask * trend

    # Equal weight across active sectors
    n_active = long_mask.sum(axis=1)
    weights = long_mask.div(n_active.replace(0, np.nan), axis=0).fillna(0)
    # PIT shift
    return weights.shift(1).fillna(0)


def sector_rotation_returns(
    weights: pd.DataFrame, sector_returns: pd.DataFrame
) -> pd.Series:
    """Combine weights with returns."""
    common = weights.columns.intersection(sector_returns.columns)
    aligned = weights[common].reindex(sector_returns.index).fillna(0)
    return (aligned * sector_returns[common]).sum(axis=1)


__all__ = [
    "faber_trend_filter",
    "relative_strength_ranking",
    "top_n_sector_strategy",
    "sector_rotation_returns",
]
