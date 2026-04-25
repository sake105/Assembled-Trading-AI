"""Liquidity-Condition-Index (LCI) — Regime Gate for all composite signals.

~30 LOC core. Z-score aggregation of HY-spread, DXY, VIX, yield-curve.
From 13_FREE_MODULE.md §13.1 — highest ROI/effort signal in the plan.

Interpretation:
  LCI < -1  → Risk-On,  activate Momentum signals
  -1..+1    → Normal regime
  LCI > +1  → Risk-Off, defensive mode
  LCI > +2  → Crisis,   long-vol and cash only
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    roll = series.rolling(window=window, min_periods=max(window // 4, 20))
    return (series - roll.mean()) / roll.std().replace(0, np.nan)


def compute_lci(
    hy_spread: pd.Series,
    ig_spread: pd.Series,
    dxy: pd.Series,
    vix: pd.Series,
    yield_curve_slope: pd.Series,
    lookback_days: int = 252,
) -> pd.Series:
    """Compute Liquidity-Condition-Index as weighted Z-score composite.

    Args:
        hy_spread: High-Yield OAS spread (FRED: BAMLH0A0HYM2)
        ig_spread: IG OAS spread (FRED: BAMLC0A0CM)
        dxy: Broad dollar index (FRED: DTWEXBGS)
        vix: VIX close (FRED: VIXCLS)
        yield_curve_slope: 2s10s slope (FRED: T10Y2Y) — inversion raises risk
        lookback_days: Rolling window for Z-score (default 252 = 1 year)

    Returns:
        LCI Series aligned to common index.
    """
    # HY/IG ratio proxy for credit stress
    hy_ig_ratio = hy_spread / ig_spread.replace(0, np.nan)

    # Align all to common index
    common = hy_ig_ratio.index.intersection(dxy.index).intersection(vix.index).intersection(yield_curve_slope.index)
    if len(common) == 0:
        logger.warning("LCI: no common index across inputs — returning empty Series")
        return pd.Series(dtype=float)

    hy_ig_z = _rolling_zscore(hy_ig_ratio.loc[common], lookback_days)
    dxy_z = _rolling_zscore(dxy.loc[common], lookback_days)
    vix_z = _rolling_zscore(vix.loc[common], lookback_days)
    curve_z = _rolling_zscore(yield_curve_slope.loc[common], lookback_days) * -1  # inversion → risk

    lci = (
        0.30 * hy_ig_z
        + 0.20 * dxy_z
        + 0.30 * vix_z
        + 0.20 * curve_z
    )
    lci.name = "lci"
    return lci


def lci_regime(lci: float | pd.Series) -> str | pd.Series:
    """Map LCI value(s) to regime label.

    Returns one of: 'risk_on', 'normal', 'risk_off', 'crisis'
    """
    def _map(v: float) -> str:
        if v < -1.0:
            return "risk_on"
        if v > 2.0:
            return "crisis"
        if v > 1.0:
            return "risk_off"
        return "normal"

    if isinstance(lci, pd.Series):
        return lci.map(_map)
    return _map(float(lci))


def lci_exposure_multiplier(lci: float) -> float:
    """Return a [0, 1] exposure multiplier based on LCI.

    Crisis → 0.0 (cash only), Risk-Off → 0.5, Normal → 1.0, Risk-On → 1.0
    """
    if lci > 2.0:
        return 0.0
    if lci > 1.0:
        return 0.5
    return 1.0


__all__ = [
    "compute_lci",
    "lci_regime",
    "lci_exposure_multiplier",
]
