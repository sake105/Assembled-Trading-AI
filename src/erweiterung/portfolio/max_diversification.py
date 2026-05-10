"""Maximum-Diversification Portfolio (Choueifaty/Coignard 2008) + Max-Sharpe.

Max-Diversification
-------------------
Choueifaty & Coignard (2008): max w'σ / √(w'Σw)
subject to Σw = 1, w >= 0.

Idee: maximiere "Diversification-Ratio" = weighted-avg-vol / portfolio-vol.

Max-Sharpe
----------
Analytische Lösung im unconstrained Fall:
    w* = Σ⁻¹ (μ - r_f) / (1' Σ⁻¹ (μ - r_f))

Constrained (long-only, max-weight) via scipy.optimize.

Reference
---------
- Choueifaty, Y. & Coignard, Y. (2008). Toward Maximum Diversification.
  *J. Portfolio Management* 35(1).
- Lopez de Prado: HRP-Vergleich, "Quasi-Diagonalization".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_diversification_weights(
    cov: pd.DataFrame,
    long_only: bool = True,
    max_weight: float = 0.30,
) -> pd.Series:
    """Maximum-Diversification-Portfolio.

    Args:
        cov: Cov-Matrix.
        long_only: Wenn True, w_i ≥ 0.
        max_weight: Cap pro Asset.

    Returns:
        Series of weights summing to 1.
    """
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError:
        # Fallback: vol-weighted (inv-vol)
        sigma_i = np.sqrt(np.diag(cov.values))
        w = 1.0 / sigma_i
        w = w / w.sum()
        return pd.Series(w, index=cov.index)

    Sigma = cov.values
    sigma_i = np.sqrt(np.diag(Sigma))
    n = len(sigma_i)

    def neg_diversification(w: np.ndarray) -> float:
        port_vol = float(np.sqrt(w @ Sigma @ w))
        if port_vol == 0:
            return 0.0
        weighted_avg_vol = float(w @ sigma_i)
        # negate (we minimize)
        return -weighted_avg_vol / port_vol

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1)}]
    bounds = [(0 if long_only else -max_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n
    res = minimize(
        neg_diversification, x0, method="SLSQP", bounds=bounds, constraints=cons
    )
    if not res.success:
        # fallback inv-vol
        w = 1.0 / sigma_i
        w = w / w.sum()
        return pd.Series(w, index=cov.index)
    return pd.Series(res.x, index=cov.index)


def max_sharpe_weights_analytical(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
) -> pd.Series:
    """Analytische Max-Sharpe-Lösung (unconstrained).

    w* = Σ⁻¹ (μ - r_f) / (1' Σ⁻¹ (μ - r_f))

    Returns:
        Series — kann negative weights enthalten (short selling).
    """
    excess = mu.values - risk_free
    Sigma_inv = np.linalg.pinv(cov.values)
    raw = Sigma_inv @ excess
    s = raw.sum()
    if s == 0:
        return pd.Series(np.ones(len(mu)) / len(mu), index=mu.index)
    return pd.Series(raw / s, index=mu.index)


def max_sharpe_weights_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    long_only: bool = True,
    max_weight: float = 0.30,
) -> pd.Series:
    """Constrained Max-Sharpe via scipy.optimize."""
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError:
        return max_sharpe_weights_analytical(mu, cov, risk_free)

    mu_v = mu.values
    Sigma = cov.values
    n = len(mu_v)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu_v) - risk_free
        port_vol = float(np.sqrt(w @ Sigma @ w))
        if port_vol == 0:
            return 0.0
        return -port_ret / port_vol

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1)}]
    bounds = [(0 if long_only else -max_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return max_sharpe_weights_analytical(mu, cov, risk_free)
    return pd.Series(res.x, index=mu.index)


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Choueifaty Diversification Ratio = Σ w_i σ_i / √(w'Σw)."""
    w = weights.values
    Sigma = cov.values
    sigma_i = np.sqrt(np.diag(Sigma))
    port_vol = float(np.sqrt(w @ Sigma @ w))
    if port_vol == 0:
        return float("nan")
    return float((w @ sigma_i) / port_vol)


__all__ = [
    "max_diversification_weights",
    "max_sharpe_weights_analytical",
    "max_sharpe_weights_constrained",
    "diversification_ratio",
]
