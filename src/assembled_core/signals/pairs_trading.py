"""Pairs Trading via Kalman-Filter Dynamic Hedge Ratio.

From 11_FREE_MODELLE.md §11.11.
pykalman KalmanFilter for time-varying beta (spread stationarity).
Entry: |z| > 2, Exit: |z| < 0.5, Stop: |z| > 4.

Install: pip install pykalman filterpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairsSignal:
    spread: pd.Series
    z_score: pd.Series
    beta: pd.Series
    alpha: pd.Series
    entry_long: pd.Series
    entry_short: pd.Series
    exit_signal: pd.Series


def _try_pykalman():
    try:
        from pykalman import KalmanFilter
        return KalmanFilter
    except ImportError:
        try:
            from filterpy.kalman import KalmanFilter as FilterPyKF
            return FilterPyKF
        except ImportError:
            logger.warning(
                "pykalman/filterpy not installed — pip install pykalman filterpy"
            )
            return None


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_var: float = 1e-2,
) -> tuple[pd.Series, pd.Series]:
    """Estimate time-varying hedge ratio [beta, alpha] via Kalman Filter.

    State: [beta_t, alpha_t]
    Observation: y_t = beta_t * x_t + alpha_t + eps_t

    Args:
        y: Dependent asset price series (leg A)
        x: Independent asset price series (leg B)
        delta: State-transition noise (smaller = smoother beta)
        obs_var: Observation noise variance

    Returns:
        Tuple of (beta_series, alpha_series) — same index as y.
        Falls back to OLS constants if pykalman unavailable.
    """
    KF = _try_pykalman()
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]

    if KF is None or len(y) < 30:
        # OLS fallback
        beta_ols = float(np.cov(y.values, x.values)[0, 1] / np.var(x.values))
        alpha_ols = float(y.mean() - beta_ols * x.mean())
        beta_s = pd.Series(beta_ols, index=y.index, name="beta")
        alpha_s = pd.Series(alpha_ols, index=y.index, name="alpha")
        return beta_s, alpha_s

    try:
        n = len(y)
        # Build 2-state KF manually using pykalman
        # State vector: [beta, alpha]
        trans_cov = np.eye(2) * delta
        obs_cov = np.array([[obs_var]])

        # Initial state from OLS
        beta_init = float(np.cov(y.values, x.values)[0, 1] / (np.var(x.values) + 1e-9))
        alpha_init = float(y.mean() - beta_init * x.mean())

        kf = KF(
            transition_matrices=np.eye(2),
            observation_covariance=obs_cov,
            transition_covariance=trans_cov,
            initial_state_mean=[beta_init, alpha_init],
            initial_state_covariance=np.eye(2),
        )

        # Build time-varying observation matrix: [[x_t, 1]]
        obs_matrices = np.column_stack([x.values, np.ones(n)])
        obs_matrices = obs_matrices.reshape(n, 1, 2)

        state_means, _ = kf.filter(
            y.values.reshape(-1, 1),
            observation_matrices=obs_matrices,
        )

        beta_s = pd.Series(state_means[:, 0], index=y.index, name="beta")
        alpha_s = pd.Series(state_means[:, 1], index=y.index, name="alpha")
        return beta_s, alpha_s

    except Exception as exc:
        logger.debug("Kalman filter failed: %s", exc)
        # OLS fallback
        beta_ols = float(np.cov(y.values, x.values)[0, 1] / (np.var(x.values) + 1e-9))
        alpha_ols = float(y.mean() - beta_ols * x.mean())
        return (
            pd.Series(beta_ols, index=y.index, name="beta"),
            pd.Series(alpha_ols, index=y.index, name="alpha"),
        )


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
    alpha: pd.Series,
) -> pd.Series:
    """Compute the residual spread: y - beta*x - alpha."""
    common = y.index.intersection(x.index).intersection(beta.index)
    spread = y.loc[common] - beta.loc[common] * x.loc[common] - alpha.loc[common]
    return spread.rename("spread")


def spread_z_score(
    spread: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling Z-score of the spread."""
    mean = spread.rolling(window, min_periods=max(5, window // 4)).mean()
    std = spread.rolling(window, min_periods=max(5, window // 4)).std()
    z = (spread - mean) / (std + 1e-9)
    return z.rename("z_score")


def generate_pairs_signals(
    y: pd.Series,
    x: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    window: int = 60,
    delta: float = 1e-4,
) -> PairsSignal:
    """Generate long/short pairs trading signals.

    Long spread when z < -entry_z (y cheap vs x).
    Short spread when z > +entry_z (y expensive vs x).
    Exit when |z| < exit_z.
    Stop when |z| > stop_z.

    Returns:
        PairsSignal with spread, z-score, beta, entry/exit signals.
    """
    beta, alpha = kalman_hedge_ratio(y, x, delta=delta)
    spread = compute_spread(y, x, beta, alpha)
    z = spread_z_score(spread, window=window)

    entry_long = z < -entry_z
    entry_short = z > entry_z
    exit_signal = z.abs() < exit_z

    return PairsSignal(
        spread=spread,
        z_score=z,
        beta=beta,
        alpha=alpha,
        entry_long=entry_long,
        entry_short=entry_short,
        exit_signal=exit_signal,
    )


def cointegration_score(y: pd.Series, x: pd.Series) -> float:
    """Quick Engle-Granger cointegration p-value (via statsmodels).

    Returns:
        p-value in [0, 1]. Lower = more cointegrated. Returns 0.5 on failure.
    """
    try:
        from statsmodels.tsa.stattools import coint
        common = y.index.intersection(x.index)
        _, pval, _ = coint(y.loc[common].dropna(), x.loc[common].dropna())
        return float(pval)
    except Exception:
        return 0.5


__all__ = [
    "PairsSignal",
    "kalman_hedge_ratio",
    "compute_spread",
    "spread_z_score",
    "generate_pairs_signals",
    "cointegration_score",
]
