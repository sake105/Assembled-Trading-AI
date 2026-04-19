"""E4 — Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Context
-------

Plain Sharpe is biased upward when a strategy is selected out of ``N`` trials
(multiple-testing) or when returns are non-normal (skew / excess kurtosis).
The Deflated Sharpe Ratio (DSR) corrects both effects and returns the
probability that the observed Sharpe is *not* a lucky draw.

Reference
~~~~~~~~~

Bailey, D. H. and López de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
and Non-Normality". Journal of Portfolio Management 40(5).

Formulas
--------

Given a return series ``r`` of length ``T``:

- ``SR_hat = mean(r) / std(r, ddof=1)``  (periodic Sharpe)
- Skew ``γ3`` and excess kurtosis ``γ4`` of ``r``.
- Standard error of the Sharpe under non-normal returns::

      sigma(SR) = sqrt( (1 - γ3 * SR + (γ4 - 1)/4 * SR^2) / (T - 1) )

- The "Sharpe ratio threshold" for ``N`` independent trials::

      SR_0 = sqrt(Var(SR)_across_trials) *
             [ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e)) ]

  where ``γ`` ≈ 0.5772 (Euler-Mascheroni) and ``Φ^{-1}`` is the inverse
  standard normal CDF.

- Deflated Sharpe probability::

      DSR = Φ( (SR_hat - SR_0) / sigma(SR) )

A DSR > 0.95 is the conventional "significant at 5%" level. Plan's E-Exit
gate is DSR > 0.5 as a minimum bar — anything below is overfit noise.

Design notes
------------

* All inputs are periodic (whatever frequency the returns are in). Callers
  who want an annualised reporting layer should annualise *after* the DSR
  is computed; annualising first and feeding back corrupts ``sigma(SR)``.
* ``n_trials`` must include every strategy evaluated — not just the winning
  one. Silently using ``n_trials=1`` while truly sweeping a grid is exactly
  the overfit failure mode the DSR exists to catch.
* Returns an ``DSRResult`` dataclass so every input / intermediate is
  visible to auditors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    sharpe_observed: float
    sharpe_std_error: float
    sharpe_threshold: float
    deflated_sharpe_probability: float
    n_observations: int
    n_trials: int
    skew: float
    excess_kurtosis: float
    passes_5pct: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "sharpe_observed": self.sharpe_observed,
            "sharpe_std_error": self.sharpe_std_error,
            "sharpe_threshold": self.sharpe_threshold,
            "deflated_sharpe_probability": self.deflated_sharpe_probability,
            "n_observations": self.n_observations,
            "n_trials": self.n_trials,
            "skew": self.skew,
            "excess_kurtosis": self.excess_kurtosis,
            "passes_5pct": self.passes_5pct,
        }


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std_ddof1, skew, excess_kurtosis)."""
    n = r.size
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    if std <= 0:
        return mean, 0.0, 0.0, 0.0
    centered = r - mean
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    # Excess kurtosis (normal = 0)
    ex_kurt = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0
    return mean, std, skew, ex_kurt


def sharpe_std_error(
    sharpe: float, n_obs: int, skew: float, excess_kurtosis: float
) -> float:
    """Standard error of the Sharpe under non-normal returns (BLP 2014)."""
    if n_obs <= 1:
        return float("nan")
    inside = 1.0 - skew * sharpe + ((excess_kurtosis - 1.0) / 4.0) * (sharpe**2)
    # Numerical guard: inside must be non-negative for a real stderr. Under
    # heavy non-normality it can dip slightly below zero for small samples.
    inside = max(inside, 0.0)
    return math.sqrt(inside / (n_obs - 1))


def sharpe_threshold(
    n_trials: int,
    variance_across_trials: float,
) -> float:
    """Critical Sharpe threshold for ``n_trials`` strategies.

    ``variance_across_trials`` is the empirical variance of the Sharpe
    estimates across every trial run. When only one trial is available,
    BLP recommends approximating this via an assumed IID backtest noise
    estimate; callers who know better should pass it explicitly.
    """
    if n_trials <= 1 or variance_across_trials <= 0.0:
        return 0.0
    gamma = _EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sd = math.sqrt(variance_across_trials)
    return sd * ((1.0 - gamma) * q1 + gamma * q2)


def deflated_sharpe(
    returns: pd.Series | np.ndarray | list[float],
    *,
    n_trials: int = 1,
    variance_across_trials: float | None = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio for a return stream.

    Args:
        returns: Periodic returns (not annualised).
        n_trials: Number of strategies evaluated. ``1`` is only valid for
            a pre-registered single strategy — not for grid-selected winners.
        variance_across_trials: Empirical variance of the Sharpe estimate
            across all trials. If ``None`` and ``n_trials>1``, a conservative
            approximation ``1/(T-1)`` is used (IID Gaussian assumption).

    Returns:
        ``DSRResult`` with observed SR, stderr, threshold, DSR probability.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n_obs = r.size
    if n_obs < 2:
        return DSRResult(
            sharpe_observed=float("nan"),
            sharpe_std_error=float("nan"),
            sharpe_threshold=0.0,
            deflated_sharpe_probability=float("nan"),
            n_observations=n_obs,
            n_trials=n_trials,
            skew=0.0,
            excess_kurtosis=0.0,
            passes_5pct=False,
        )

    mean, std, skew, ex_kurt = _moments(r)
    if std == 0.0:
        # Zero-variance returns mean the strategy produced no measurable
        # variation (stuck/no trades). The previous sr=0.0 collapse let
        # callers log "DSR computed" for a non-strategy; propagate NaN so
        # the downstream se<=0 / !isfinite guard (see below) routes the
        # whole result to dsr_prob=NaN and passes_5pct=False via the
        # explicit isfinite check in the return statement. Schema is
        # preserved (sharpe_observed stays float).
        sr = float("nan")
    else:
        sr = mean / std

    se = sharpe_std_error(sr, n_obs, skew, ex_kurt)

    if n_trials <= 1:
        threshold = 0.0
    else:
        if variance_across_trials is None:
            # Conservative IID fallback.
            variance_across_trials = 1.0 / max(n_obs - 1, 1)
        threshold = sharpe_threshold(n_trials, variance_across_trials)

    if not math.isfinite(se) or se <= 0.0:
        dsr_prob = float("nan")
    else:
        dsr_prob = float(norm.cdf((sr - threshold) / se))

    return DSRResult(
        sharpe_observed=sr,
        sharpe_std_error=se,
        sharpe_threshold=threshold,
        deflated_sharpe_probability=dsr_prob,
        n_observations=n_obs,
        n_trials=n_trials,
        skew=skew,
        excess_kurtosis=ex_kurt,
        passes_5pct=bool(math.isfinite(dsr_prob) and dsr_prob >= 0.95),
    )
