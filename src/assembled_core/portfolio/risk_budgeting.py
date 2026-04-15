"""Risk Budgeting and Equal Risk Contribution (ERC) Portfolio.

Implements Maillard, Roncalli & Teïletche (2010):
- Each asset contributes equally to portfolio risk
- More robust than MVO, less concentrated than HRP
- Uses scipy optimize or CVXPY if available

Risk contribution of asset i:
    RC_i = w_i * (Sigma @ w)_i / sqrt(w' @ Sigma @ w)

ERC constraint: RC_i = RC_j for all i, j

References:
    Maillard, Roncalli, Teïletche (2010) "The Properties of Equally Weighted Risk Contribution Portfolios"
    Roncalli (2014) "Introduction to Risk Parity and Budgeting"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None  # type: ignore

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class RiskBudgetResult:
    """Result of risk budgeting optimization."""
    weights: dict[str, float]
    risk_contributions: dict[str, float]
    portfolio_volatility: float
    max_rc_deviation: float  # max |RC_i - target_i|
    method: str
    converged: bool


def _portfolio_risk(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility."""
    return float(np.sqrt(w @ cov @ w))


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset.

    RC_i = w_i * (Sigma @ w)_i / sigma_p
    """
    sigma_p = _portfolio_risk(w, cov)
    if sigma_p < 1e-12:
        return np.zeros_like(w)
    marginal = cov @ w
    return w * marginal / sigma_p


def compute_erc_weights(
    covariance: pd.DataFrame | np.ndarray,
    symbols: list[str] | None = None,
    risk_budget: dict[str, float] | np.ndarray | None = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> RiskBudgetResult:
    """Compute Equal Risk Contribution (or risk-budgeted) portfolio weights.

    Args:
        covariance: Covariance matrix (DataFrame or ndarray).
        symbols: Asset names (inferred from DataFrame index if not provided).
        risk_budget: Target risk budget per asset. If None, uses equal (ERC).
            Dict mapping symbol -> target fraction, or array summing to 1.
        long_only: If True, enforce w >= 0.
        max_weight: Maximum per-asset weight.
        min_weight: Minimum per-asset weight (if long_only=False).

    Returns:
        RiskBudgetResult with optimized weights and risk contributions.
    """
    if isinstance(covariance, pd.DataFrame):
        symbols = symbols or list(covariance.columns)
        cov = covariance.values.astype(float)
    else:
        cov = np.asarray(covariance, dtype=float)
        n = cov.shape[0]
        symbols = symbols or [f"asset_{i}" for i in range(n)]

    n = len(symbols)

    # Target risk budget
    if risk_budget is None:
        target_rc = np.ones(n) / n
    elif isinstance(risk_budget, dict):
        target_rc = np.array([risk_budget.get(s, 1.0 / n) for s in symbols])
        target_rc /= target_rc.sum()
    else:
        target_rc = np.asarray(risk_budget, dtype=float)
        target_rc /= target_rc.sum()

    if SCIPY_AVAILABLE:
        return _erc_scipy(cov, symbols, target_rc, long_only, max_weight, min_weight)
    else:
        return _erc_analytical_fallback(cov, symbols, target_rc)


def _erc_scipy(
    cov: np.ndarray,
    symbols: list[str],
    target_rc: np.ndarray,
    long_only: bool,
    max_weight: float,
    min_weight: float,
) -> RiskBudgetResult:
    """Scipy-based ERC via minimizing sum of squared RC deviations."""
    n = len(symbols)

    def objective(w: np.ndarray) -> float:
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-12:
            return 1e6
        marginal = cov @ w
        rc = w * marginal / sigma_p
        rc_pct = rc / rc.sum() if rc.sum() > 1e-12 else np.ones(n) / n
        return float(np.sum((rc_pct - target_rc) ** 2))

    # Bounds
    if long_only:
        bounds = [(max(0.001, min_weight), max_weight)] * n
    else:
        bounds = [(min_weight, max_weight)] * n

    # Constraint: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Multiple restarts for robustness
    best_result = None
    best_obj = float("inf")

    for seed in range(5):
        rng = np.random.default_rng(42 + seed)
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, bounds[0][0], bounds[0][1])
        w0 /= w0.sum()

        try:
            res = minimize(
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
        logger.warning("[RiskBudget] Optimization failed, using inverse-vol fallback")
        return _erc_analytical_fallback(cov, symbols, target_rc)

    w_opt = best_result.x
    w_opt = np.maximum(w_opt, 0) if long_only else w_opt
    w_opt /= w_opt.sum()

    rc = _risk_contributions(w_opt, cov)
    rc_pct = rc / rc.sum() if rc.sum() > 1e-12 else np.ones(n) / n
    max_dev = float(np.max(np.abs(rc_pct - target_rc)))

    return RiskBudgetResult(
        weights={s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
        risk_contributions={s: round(float(rc_pct[i]), 6) for i, s in enumerate(symbols)},
        portfolio_volatility=round(_portfolio_risk(w_opt, cov), 6),
        max_rc_deviation=round(max_dev, 6),
        method="scipy_slsqp",
        converged=best_result.success,
    )


def _erc_analytical_fallback(
    cov: np.ndarray,
    symbols: list[str],
    target_rc: np.ndarray,
) -> RiskBudgetResult:
    """Inverse-volatility heuristic when scipy unavailable."""
    n = len(symbols)
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-10)

    # Weight proportional to target_budget / vol
    w = target_rc / vols
    w /= w.sum()

    rc = _risk_contributions(w, cov)
    rc_pct = rc / rc.sum() if rc.sum() > 1e-12 else np.ones(n) / n
    max_dev = float(np.max(np.abs(rc_pct - target_rc)))

    return RiskBudgetResult(
        weights={s: round(float(w[i]), 6) for i, s in enumerate(symbols)},
        risk_contributions={s: round(float(rc_pct[i]), 6) for i, s in enumerate(symbols)},
        portfolio_volatility=round(_portfolio_risk(w, cov), 6),
        max_rc_deviation=round(max_dev, 6),
        method="inverse_vol_fallback",
        converged=True,
    )


def risk_parity_with_views(
    covariance: pd.DataFrame,
    views_confidence: dict[str, float] | None = None,
    base_budget: dict[str, float] | None = None,
    confidence_scale: float = 0.5,
    **kwargs,
) -> RiskBudgetResult:
    """Risk parity that tilts risk budget based on view confidence.

    Higher confidence in an asset → higher risk budget allocation.

    Args:
        covariance: Covariance matrix.
        views_confidence: Symbol -> confidence (0-1) from model.
        base_budget: Base risk budget (defaults to ERC).
        confidence_scale: How much views tilt the budget (0=none, 1=full).
        **kwargs: Passed to compute_erc_weights.

    Returns:
        RiskBudgetResult with tilted risk budgets.
    """
    symbols = list(covariance.columns)
    n = len(symbols)

    if base_budget is None:
        budget = np.ones(n) / n
    else:
        budget = np.array([base_budget.get(s, 1.0 / n) for s in symbols])
        budget /= budget.sum()

    if views_confidence:
        conf = np.array([views_confidence.get(s, 0.5) for s in symbols])
        # Tilt: higher confidence → higher risk budget
        tilt = 1.0 + confidence_scale * (conf - 0.5)
        budget = budget * tilt
        budget /= budget.sum()

    return compute_erc_weights(
        covariance,
        symbols=symbols,
        risk_budget={s: budget[i] for i, s in enumerate(symbols)},
        **kwargs,
    )


__all__ = [
    "RiskBudgetResult",
    "compute_erc_weights",
    "risk_parity_with_views",
]
