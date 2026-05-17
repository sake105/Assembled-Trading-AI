"""Stationarity and half-life diagnostics for pairs trading spreads.

Provides the statistical-validity layer that audit C4-084 (KNOWN_ISSUES §8.13)
flagged as missing from the pairs-trading path:

- `engle_granger_cointegration(y, x)` — bivariate cointegration test via the
  two-step Engle-Granger procedure (statsmodels.tsa.stattools.coint wrapper)
- `ou_half_life(spread)` — Ornstein-Uhlenbeck half-life estimator via AR(1)
  regression on Δspread = -λ·spread_{t-1} + const + ε_t

Complements `signals/pairs_trading.py` (Kalman hedge-ratio, z-score, signal
generation). Use these to PRE-FILTER pair candidates before generating signals:

    from src.assembled_core.signals.pairs_diagnostics import (
        engle_granger_cointegration, ou_half_life,
    )
    coint = engle_granger_cointegration(y, x)
    if not coint.is_cointegrated_at_5pct:
        skip_pair()
    hl = ou_half_life(spread)
    if not (1 < hl < 60):  # too fast or too slow → unstable signal
        skip_pair()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CointegrationResult:
    """Result of an Engle-Granger cointegration test."""

    statistic: float  # Augmented Dickey-Fuller test statistic on the residual
    pvalue: float  # MacKinnon p-value (0..1)
    crit_values: dict[str, float]  # critical t-stats for 1%/5%/10% rejection levels
    is_cointegrated_at_5pct: bool  # pvalue < 0.05


def engle_granger_cointegration(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    trend: str = "c",
    maxlag: int | None = 1,
) -> CointegrationResult:
    """Two-step Engle-Granger cointegration test for a bivariate spread.

    Args:
        y: First series (dependent variable in the cointegrating regression).
        x: Second series (regressor).
        trend: Trend specification for the underlying ADF test on residuals
            ("c" = constant, "ct" = constant + linear trend, "n" = no constant).
        maxlag: Maximum lag for the ADF augmentation. ``None`` uses
            statsmodels' default selection.

    Returns:
        CointegrationResult with ADF statistic, MacKinnon p-value, critical
        values, and a convenience boolean at the 5% rejection level.

    Raises:
        ValueError: If inputs are empty, length-mismatched, or contain NaN/inf.
        ImportError: If statsmodels is not installed.
    """
    from statsmodels.tsa.stattools import coint

    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)

    if y_arr.size == 0 or x_arr.size == 0:
        raise ValueError("engle_granger_cointegration: empty input series")
    if y_arr.size != x_arr.size:
        raise ValueError(
            f"engle_granger_cointegration: length mismatch y={y_arr.size}, x={x_arr.size}"
        )
    if not np.all(np.isfinite(y_arr)) or not np.all(np.isfinite(x_arr)):
        raise ValueError(
            "engle_granger_cointegration: inputs contain NaN/inf — clean before calling"
        )

    stat, pvalue, crit = coint(y_arr, x_arr, trend=trend, maxlag=maxlag, autolag=None)
    return CointegrationResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        crit_values={
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        },
        is_cointegrated_at_5pct=bool(float(pvalue) < 0.05),
    )


def ou_half_life(spread: pd.Series | np.ndarray) -> float:
    """Estimate half-life of mean-reversion via Ornstein-Uhlenbeck AR(1) regression.

    Discrete OU model: Δs_t = -λ · s_{t-1} + μ + ε_t (with λ > 0 for mean reversion).

    Procedure:
        1. Compute Δspread_t = spread_t - spread_{t-1}
        2. OLS regression: Δspread ~ spread_{t-1} + const
        3. λ = -slope; half-life = ln(2) / λ

    Args:
        spread: Series or array of cointegrating-residual / spread values.
            Should be stationary for the half-life to be meaningful — verify
            via `engle_granger_cointegration` before relying on this.

    Returns:
        Half-life in periods (positive float).
        - ``inf`` if the spread is explosive (slope ≥ 0, no mean-reversion).
        - ``NaN`` if input has < 30 observations or all-constant.

    Notes:
        Reference: Chan, "Algorithmic Trading: Winning Strategies and Their
        Rationale" (2013), §3.2. Also Lopez de Prado AFML §13.2.
        For typical equity-pair spreads, useful half-lives are 1–60 periods
        (daily bars); shorter is too fast for slippage, longer suggests the
        cointegration is weak.
    """
    s = pd.Series(spread, dtype=float).dropna()
    if len(s) < 30:
        return float("nan")
    if s.std(ddof=1) == 0:
        return float("nan")  # all-constant → undefined half-life

    delta = s.diff().dropna()
    lagged = s.shift(1).dropna()
    # Align indices (intersection of delta and lagged after dropna)
    common_idx = delta.index.intersection(lagged.index)
    if len(common_idx) < 30:
        return float("nan")
    delta = delta.loc[common_idx]
    lagged = lagged.loc[common_idx]

    # OLS via np.linalg.lstsq: delta = slope * lagged + intercept + epsilon
    design = np.column_stack([lagged.to_numpy(), np.ones(len(lagged))])
    coef, *_ = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)
    slope = float(coef[0])

    if slope >= 0:
        # No mean reversion (explosive or random-walk): half-life is undefined / infinite
        return float("inf")

    lambda_coef = -slope  # positive: mean-reversion rate
    if lambda_coef <= 0:
        return float("inf")
    return float(math.log(2.0) / lambda_coef)


__all__ = ["CointegrationResult", "engle_granger_cointegration", "ou_half_life"]
