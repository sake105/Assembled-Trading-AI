"""MVO with Cardinality Constraints (Plan 5.10).

Mean-Variance Optimization with a maximum number of positions.
Uses greedy heuristic (no CVXPY required).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mvo_with_cardinality(
    mu: np.ndarray,
    sigma: np.ndarray,
    max_positions: int = 20,
    risk_aversion: float = 1.0,
    min_weight: float = 0.01,
) -> np.ndarray:
    """Mean-Variance Optimization with cardinality constraint.

    Greedy approach: select top-N assets by Sharpe-proxy, then optimize
    within that subset using analytical MVO.

    Args:
        mu: Expected returns (N,).
        sigma: Covariance matrix (N×N).
        max_positions: Maximum number of positions.
        risk_aversion: Lambda for risk penalty.
        min_weight: Minimum weight per position.

    Returns:
        Array of weights (N,) with at most max_positions non-zero.
    """
    n = len(mu)
    if n <= max_positions:
        # No cardinality constraint needed
        return _analytical_mvo(mu, sigma, risk_aversion)

    # Greedy selection: rank by expected return / volatility
    vols = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
    sharpe_proxy = mu / vols
    selected = np.argsort(sharpe_proxy)[-max_positions:]

    # Sub-problem MVO
    mu_sub = mu[selected]
    sigma_sub = sigma[np.ix_(selected, selected)]
    w_sub = _analytical_mvo(mu_sub, sigma_sub, risk_aversion)

    # Zero out tiny weights
    w_sub[w_sub < min_weight] = 0.0
    total = w_sub.sum()
    if total > 0:
        w_sub /= total

    # Map back to full weight vector
    w = np.zeros(n)
    w[selected] = w_sub
    return w


def _analytical_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """Simple analytical MVO: w* = (1/lambda) * Sigma^{-1} * mu, then normalize."""
    try:
        sigma_inv = np.linalg.inv(sigma + np.eye(len(sigma)) * 1e-6)
    except np.linalg.LinAlgError:
        return np.ones(len(mu)) / len(mu)

    w = sigma_inv @ mu / max(risk_aversion, 1e-6)
    w = np.maximum(w, 0.0)  # long-only
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w = np.ones(len(mu)) / len(mu)
    return w


__all__ = ["mvo_with_cardinality"]
