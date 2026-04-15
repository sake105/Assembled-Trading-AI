"""Robust Portfolio Optimization under Parameter Uncertainty.

Implements Goldfarb & Iyengar (2003) worst-case optimization:
    max_w min_mu { w'mu - lambda * w'Sigma*w }
    s.t. mu in Uncertainty_Set(mu_hat, epsilon)

The uncertainty set is an ellipsoid around the estimated returns:
    U = { mu : (mu - mu_hat)' Sigma_mu^{-1} (mu - mu_hat) <= epsilon^2 }

The worst-case reduces to:
    max_w { w'mu_hat - epsilon * ||Sigma_mu^{1/2} w|| - lambda * w'Sigma*w }

This penalizes portfolios sensitive to estimation error, producing
more diversified, stable allocations than standard MVO.

References:
    Goldfarb & Iyengar (2003) "Robust Portfolio Selection Problems"
    Tutuncu & Koenig (2004) "Robust Asset Allocation"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_minimize = None  # type: ignore

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class RobustOptResult:
    """Result of robust portfolio optimization."""
    weights: dict[str, float]
    expected_return: float
    worst_case_return: float
    portfolio_volatility: float
    epsilon: float  # uncertainty radius used
    method: str
    converged: bool


def _estimate_return_uncertainty(
    cov: np.ndarray,
    n_obs: int = 252,
    method: str = "asymptotic",
) -> np.ndarray:
    """Estimate uncertainty covariance for expected returns.

    Sigma_mu = Sigma / n_obs (asymptotic standard error of sample mean).

    Args:
        cov: Asset covariance matrix (N x N).
        n_obs: Number of observations.
        method: "asymptotic" or "bootstrap" (only asymptotic implemented).

    Returns:
        Return uncertainty covariance matrix (N x N).
    """
    return cov / max(n_obs, 1)


def compute_robust_weights(
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    symbols: list[str] | None = None,
    epsilon: float | None = None,
    n_obs: int = 252,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
) -> RobustOptResult:
    """Compute robust portfolio weights via worst-case optimization.

    Args:
        expected_returns: Expected returns vector.
        covariance: Covariance matrix.
        symbols: Asset names.
        epsilon: Uncertainty radius. If None, calibrated from n_obs.
        n_obs: Number of observations (for automatic epsilon).
        risk_aversion: Risk aversion parameter lambda.
        long_only: Enforce non-negative weights.
        max_weight: Maximum per-asset weight.
        min_weight: Minimum per-asset weight.

    Returns:
        RobustOptResult with optimal weights.
    """
    # Parse inputs
    if isinstance(expected_returns, pd.Series):
        symbols = symbols or list(expected_returns.index)
        mu = expected_returns.values.astype(float)
    else:
        mu = np.asarray(expected_returns, dtype=float)

    if isinstance(covariance, pd.DataFrame):
        symbols = symbols or list(covariance.columns)
        cov = covariance.values.astype(float)
    else:
        cov = np.asarray(covariance, dtype=float)

    n = len(mu)
    symbols = symbols or [f"asset_{i}" for i in range(n)]

    # Uncertainty covariance
    sigma_mu = _estimate_return_uncertainty(cov, n_obs)

    # Calibrate epsilon if not provided: chi-squared 95% quantile approx
    if epsilon is None:
        epsilon = np.sqrt(n + 2 * np.sqrt(n))  # ~chi2(n) 95th percentile approx

    # Cholesky of uncertainty matrix for norm computation
    try:
        sigma_mu_half = np.linalg.cholesky(sigma_mu + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal
        sigma_mu_half = np.diag(np.sqrt(np.maximum(np.diag(sigma_mu), 1e-10)))

    if CVXPY_AVAILABLE:
        return _robust_cvxpy(symbols, mu, cov, sigma_mu_half, epsilon,
                             risk_aversion, long_only, max_weight, min_weight)
    elif SCIPY_AVAILABLE:
        return _robust_scipy(symbols, mu, cov, sigma_mu_half, epsilon,
                             risk_aversion, long_only, max_weight, min_weight)
    else:
        return _robust_fallback(symbols, mu, cov, sigma_mu_half, epsilon,
                                risk_aversion)


def _robust_cvxpy(
    symbols: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    sigma_mu_half: np.ndarray,
    epsilon: float,
    risk_aversion: float,
    long_only: bool,
    max_weight: float,
    min_weight: float,
) -> RobustOptResult:
    """CVXPY implementation of robust optimization."""
    n = len(symbols)
    w = cp.Variable(n)

    # Worst-case objective:
    # max { w'mu - epsilon * ||Sigma_mu^{1/2} w||_2 - lambda * w'Sigma*w }
    ret = mu @ w
    uncertainty_penalty = epsilon * cp.norm(sigma_mu_half @ w, 2)
    risk = cp.quad_form(w, cov)

    objective = cp.Maximize(
        ret - uncertainty_penalty - risk_aversion * risk
    )

    constraints = [
        cp.sum(w) == 1.0,
        w <= max_weight,
    ]
    if long_only:
        constraints.append(w >= 0)
    else:
        constraints.append(w >= min_weight)

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000)

        if prob.status in ("optimal", "optimal_inaccurate"):
            w_opt = np.array(w.value).flatten()
            w_opt = np.maximum(w_opt, 0) if long_only else w_opt
            w_opt /= w_opt.sum() if w_opt.sum() > 1e-8 else 1.0

            exp_ret = float(mu @ w_opt)
            wc_ret = exp_ret - epsilon * float(np.linalg.norm(sigma_mu_half @ w_opt))
            vol = float(np.sqrt(w_opt @ cov @ w_opt))

            return RobustOptResult(
                weights={s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
                expected_return=round(exp_ret, 6),
                worst_case_return=round(wc_ret, 6),
                portfolio_volatility=round(vol, 6),
                epsilon=round(epsilon, 4),
                method="cvxpy_robust",
                converged=True,
            )
    except Exception as e:
        logger.warning("[RobustOpt] CVXPY failed: %s — falling back to scipy", e)

    if SCIPY_AVAILABLE:
        return _robust_scipy(symbols, mu, cov, sigma_mu_half, epsilon,
                             risk_aversion, long_only, max_weight, min_weight)
    return _robust_fallback(symbols, mu, cov, sigma_mu_half, epsilon, risk_aversion)


def _robust_scipy(
    symbols: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    sigma_mu_half: np.ndarray,
    epsilon: float,
    risk_aversion: float,
    long_only: bool,
    max_weight: float,
    min_weight: float,
) -> RobustOptResult:
    """Scipy SLSQP implementation of robust optimization."""
    n = len(symbols)

    def objective(w: np.ndarray) -> float:
        ret = float(mu @ w)
        unc = epsilon * float(np.linalg.norm(sigma_mu_half @ w))
        risk = float(w @ cov @ w)
        return -(ret - unc - risk_aversion * risk)

    lb = 0.001 if long_only else min_weight
    bounds = [(lb, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    best_result = None
    best_obj = float("inf")

    for seed in range(5):
        rng = np.random.default_rng(42 + seed)
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, lb, max_weight)
        w0 /= w0.sum()

        try:
            res = scipy_minimize(
                objective, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or not best_result.success:
        logger.warning("[RobustOpt] Scipy failed — using shrinkage fallback")
        return _robust_fallback(symbols, mu, cov, sigma_mu_half, epsilon, risk_aversion)

    w_opt = best_result.x
    w_opt = np.maximum(w_opt, 0) if long_only else w_opt
    w_opt /= w_opt.sum() if w_opt.sum() > 1e-8 else 1.0

    exp_ret = float(mu @ w_opt)
    wc_ret = exp_ret - epsilon * float(np.linalg.norm(sigma_mu_half @ w_opt))
    vol = float(np.sqrt(w_opt @ cov @ w_opt))

    return RobustOptResult(
        weights={s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
        expected_return=round(exp_ret, 6),
        worst_case_return=round(wc_ret, 6),
        portfolio_volatility=round(vol, 6),
        epsilon=round(epsilon, 4),
        method="scipy_robust",
        converged=best_result.success,
    )


def _robust_fallback(
    symbols: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    sigma_mu_half: np.ndarray,
    epsilon: float,
    risk_aversion: float,
) -> RobustOptResult:
    """Analytical fallback: shrink returns then use inverse-vol weights."""
    n = len(symbols)
    # Shrink returns by uncertainty
    mu_shrunk = mu * max(0.0, 1.0 - epsilon / (np.linalg.norm(mu) + 1e-10))

    vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    w = (1.0 / vols)
    w /= w.sum()

    exp_ret = float(mu @ w)
    wc_ret = exp_ret - epsilon * float(np.linalg.norm(sigma_mu_half @ w))
    vol = float(np.sqrt(w @ cov @ w))

    return RobustOptResult(
        weights={s: round(float(w[i]), 6) for i, s in enumerate(symbols)},
        expected_return=round(exp_ret, 6),
        worst_case_return=round(wc_ret, 6),
        portfolio_volatility=round(vol, 6),
        epsilon=round(epsilon, 4),
        method="shrinkage_fallback",
        converged=True,
    )


__all__ = [
    "RobustOptResult",
    "compute_robust_weights",
]
