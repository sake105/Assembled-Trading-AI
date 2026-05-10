"""DCC-GARCH — Dynamic Conditional Correlation (Engle 2002).

Reference
---------
Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate
GARCH Models. *J. Bus. Econ. Stat.* 20.

Modell
------
Zweistufiger Ansatz:
1. **Stage 1**: univariate GARCH(1,1) pro Asset → standardisierte Residuen ε_t.
2. **Stage 2**: zeitvariable Quasi-Korrelationsmatrix
       Q_t = (1 - α - β) Q̄ + α ε_{t-1} ε'_{t-1} + β Q_{t-1}
       R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

Anwendung
---------
- Multi-Asset Cov-Forecast für Portfolio-Vola
- Crisis-Correlation-Spike-Detection
- Stress-Test-Scenarios mit dynamic correlations

Implementation
--------------
Pure NumPy ohne externe Lib. Stage-1 GARCH via simplified MLE; Stage-2 mit
Quasi-Maximum-Likelihood-Estimation der (α, β)-Parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DCCFit:
    alpha: float  # DCC innovation weight
    beta: float  # DCC persistence
    Q_bar: np.ndarray  # unconditional correlation
    R_path: np.ndarray  # (T, N, N) — conditional correlations
    sigma_path: np.ndarray  # (T, N) — conditional vols
    log_lik: float


def _fit_garch_univariate(
    r: np.ndarray, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Simple GARCH(1,1) MLE.

    Returns (params, sigma_path) with params=(omega, alpha, beta), sigma_path of length T.
    """
    var0 = float(np.var(r))
    omega, alpha, beta = 0.05 * var0, 0.1, 0.85

    def var_path(omega: float, alpha: float, beta: float) -> np.ndarray:
        var = np.zeros(len(r))
        var[0] = var0
        for t in range(1, len(r)):
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        return var

    try:
        from scipy.optimize import minimize  # type: ignore

        def neg_ll(theta: np.ndarray) -> float:
            o, a, b = theta
            if o <= 0 or a < 0 or b < 0 or a + b >= 1:
                return 1e10
            v = var_path(o, a, b)
            return 0.5 * np.sum(np.log(v) + r**2 / v)

        res = minimize(
            neg_ll,
            x0=np.array([omega, alpha, beta]),
            method="L-BFGS-B",
            bounds=[(1e-9, None), (0, 0.999), (0, 0.999)],
        )
        omega, alpha, beta = res.x
    except ImportError:
        pass  # use defaults
    var = var_path(omega, alpha, beta)
    return np.array([omega, alpha, beta]), np.sqrt(var)


def fit_dcc_garch(returns: pd.DataFrame, max_iter: int = 50) -> DCCFit:
    """Fit DCC-GARCH on a multivariate returns DataFrame.

    Args:
        returns: DataFrame (T × N), no NaN.

    Returns:
        DCCFit mit conditional vols + correlations per period.
    """
    R = returns.dropna(how="any").values
    T, N = R.shape
    if T < 50:
        raise ValueError("need >= 50 observations")

    # Stage 1: univariate GARCH per series
    sigmas = np.zeros((T, N))
    eps = np.zeros((T, N))  # standardized residuals
    for j in range(N):
        _, sig = _fit_garch_univariate(R[:, j])
        sigmas[:, j] = sig
        eps[:, j] = R[:, j] / np.maximum(sig, 1e-9)

    # Unconditional correlation matrix
    Q_bar = (eps.T @ eps) / T

    # Stage 2: DCC parameters via QMLE
    def dcc_path(alpha: float, beta: float) -> np.ndarray:
        Q = np.zeros((T, N, N))
        Q[0] = Q_bar
        for t in range(1, T):
            Q[t] = (
                (1 - alpha - beta) * Q_bar
                + alpha * np.outer(eps[t - 1], eps[t - 1])
                + beta * Q[t - 1]
            )
        return Q

    def neg_ll_dcc(theta: np.ndarray) -> float:
        a, b = theta
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        Q = dcc_path(a, b)
        ll = 0.0
        for t in range(T):
            q_diag = np.diag(Q[t])
            d = 1.0 / np.sqrt(np.maximum(q_diag, 1e-9))
            R_t = (d[:, None] * Q[t]) * d[None, :]
            # determinant + quadratic form
            try:
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    return 1e10
                inv_R = np.linalg.pinv(R_t)
            except np.linalg.LinAlgError:
                return 1e10
            ll += 0.5 * (logdet + eps[t] @ inv_R @ eps[t])
        return float(ll)

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(
            neg_ll_dcc,
            x0=np.array([0.05, 0.90]),
            method="L-BFGS-B",
            bounds=[(0.001, 0.999), (0.001, 0.999)],
        )
        alpha_dcc, beta_dcc = res.x
        log_lik = -res.fun
    except ImportError:
        alpha_dcc, beta_dcc = 0.05, 0.90
        log_lik = -neg_ll_dcc(np.array([alpha_dcc, beta_dcc]))

    Q_path = dcc_path(alpha_dcc, beta_dcc)
    R_path = np.zeros_like(Q_path)
    for t in range(T):
        q_diag = np.diag(Q_path[t])
        d = 1.0 / np.sqrt(np.maximum(q_diag, 1e-9))
        R_path[t] = (d[:, None] * Q_path[t]) * d[None, :]

    return DCCFit(
        alpha=float(alpha_dcc),
        beta=float(beta_dcc),
        Q_bar=Q_bar,
        R_path=R_path,
        sigma_path=sigmas,
        log_lik=log_lik,
    )


def dcc_covariance_at(fit: DCCFit, t: int) -> np.ndarray:
    """Conditional covariance at time t: Σ_t = diag(σ_t) R_t diag(σ_t)."""
    if t < 0 or t >= len(fit.sigma_path):
        raise IndexError(f"t={t} out of range")
    sigma = fit.sigma_path[t]
    return np.diag(sigma) @ fit.R_path[t] @ np.diag(sigma)


__all__ = ["DCCFit", "fit_dcc_garch", "dcc_covariance_at"]
