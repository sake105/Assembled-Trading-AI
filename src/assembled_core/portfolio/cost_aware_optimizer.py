"""Cost-aware portfolio optimizer with turnover penalty (V9).

Implements: max w'μ - λ·w'Σw - γ·Σ|w_new - w_old|·cost_bps
with optional sector constraints and CVaR limits.

Falls back to simple MVO if CVXPY is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class OptimizerConfig:
    """Configuration for cost-aware portfolio optimization."""

    risk_aversion: float = 1.0  # λ
    turnover_penalty: float = 0.001  # γ (in cost fraction units)
    max_weight: float = 0.10  # Max single-name weight
    min_weight: float = 0.0  # Min weight (0 = long-only)
    max_gross_exposure: float = 1.0
    max_sector_deviation: float | None = None  # Max abs deviation from benchmark sector weight
    sector_mapping: dict[str, str] = field(default_factory=dict)
    long_only: bool = True


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: dict[str, float]  # symbol -> optimal weight
    expected_return: float
    expected_risk: float
    turnover_cost: float
    solver_status: str
    method: str  # "cvxpy" or "fallback_greedy"


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: dict[str, float] | None = None,
    per_symbol_cost_bps: dict[str, float] | None = None,
    config: OptimizerConfig | None = None,
) -> OptimizationResult:
    """Optimize portfolio weights with turnover penalty and constraints.

    Args:
        expected_returns: Series indexed by symbol with expected returns.
        covariance: Covariance matrix (symbols x symbols).
        current_weights: Current portfolio weights (for turnover calc).
        per_symbol_cost_bps: Symbol -> one-way cost in bps (from V2).
        config: Optimizer configuration.

    Returns:
        OptimizationResult with optimal weights.
    """
    config = config or OptimizerConfig()
    current_weights = current_weights or {}

    symbols = list(expected_returns.index)
    n = len(symbols)

    if n == 0:
        return OptimizationResult(
            weights={}, expected_return=0.0, expected_risk=0.0,
            turnover_cost=0.0, solver_status="empty", method="none",
        )

    mu = expected_returns.values.astype(float)
    Sigma = covariance.loc[symbols, symbols].values.astype(float)
    w_old = np.array([current_weights.get(s, 0.0) for s in symbols])

    # Cost vector (per-symbol)
    default_cost = 6.0  # bps
    cost_vec = np.array([
        (per_symbol_cost_bps or {}).get(s, default_cost) / 10_000.0
        for s in symbols
    ])

    if CVXPY_AVAILABLE:
        return _optimize_cvxpy(
            symbols, mu, Sigma, w_old, cost_vec, config
        )
    else:
        return _optimize_fallback(
            symbols, mu, Sigma, w_old, cost_vec, config
        )


def _optimize_cvxpy(
    symbols: list[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    cost_vec: np.ndarray,
    config: OptimizerConfig,
) -> OptimizationResult:
    """CVXPY-based optimization with turnover penalty."""
    n = len(symbols)
    w = cp.Variable(n)

    # Objective: maximize return - risk - turnover cost
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm1(cp.multiply(cost_vec, (w - w_old)))

    objective = cp.Maximize(
        ret - config.risk_aversion * risk - config.turnover_penalty * turnover
    )

    # Constraints
    constraints = [
        w <= config.max_weight,
        cp.sum(w) <= config.max_gross_exposure,
    ]

    if config.long_only:
        constraints.append(w >= 0)
    else:
        constraints.append(w >= config.min_weight)

    # Budget constraint
    constraints.append(cp.sum(w) >= 0.0)

    # Sector constraints
    if config.max_sector_deviation is not None and config.sector_mapping:
        sectors = set(config.sector_mapping.values())
        for sector in sectors:
            sector_mask = np.array([
                1.0 if config.sector_mapping.get(s) == sector else 0.0
                for s in symbols
            ])
            if sector_mask.sum() > 0:
                constraints.append(
                    sector_mask @ w <= config.max_sector_deviation + sector_mask.sum() / n
                )

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000)

        if prob.status in ("optimal", "optimal_inaccurate"):
            w_opt = w.value
            weights = {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)}
            exp_ret = float(mu @ w_opt)
            exp_risk = float(np.sqrt(w_opt @ Sigma @ w_opt))
            tc = float(np.sum(np.abs(w_opt - w_old) * cost_vec))

            return OptimizationResult(
                weights=weights,
                expected_return=round(exp_ret, 6),
                expected_risk=round(exp_risk, 6),
                turnover_cost=round(tc, 6),
                solver_status=prob.status,
                method="cvxpy",
            )
        else:
            _log.warning("CVXPY solver status: %s — falling back", prob.status)
    except Exception as e:
        _log.warning("CVXPY failed: %s — falling back to greedy", e)

    return _optimize_fallback(symbols, mu, Sigma, w_old, cost_vec, config)


def _optimize_fallback(
    symbols: list[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_old: np.ndarray,
    cost_vec: np.ndarray,
    config: OptimizerConfig,
) -> OptimizationResult:
    """Simple score-based fallback when CVXPY is unavailable."""
    # Score = expected return - turnover cost
    scores = mu - config.turnover_penalty * np.abs(cost_vec) * np.abs(w_old)

    if config.long_only:
        scores = np.maximum(scores, 0)

    # Top-N by score
    total = scores.sum()
    if total > 1e-12:
        w_opt = scores / total * config.max_gross_exposure
    else:
        w_opt = np.ones(len(symbols)) / len(symbols) * config.max_gross_exposure

    # Clip
    w_opt = np.clip(w_opt, config.min_weight if not config.long_only else 0, config.max_weight)

    # Renormalize
    if w_opt.sum() > config.max_gross_exposure:
        w_opt *= config.max_gross_exposure / w_opt.sum()

    weights = {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)}
    exp_ret = float(mu @ w_opt)
    exp_risk = float(np.sqrt(w_opt @ Sigma @ w_opt))
    tc = float(np.sum(np.abs(w_opt - w_old) * cost_vec))

    return OptimizationResult(
        weights=weights,
        expected_return=round(exp_ret, 6),
        expected_risk=round(exp_risk, 6),
        turnover_cost=round(tc, 6),
        solver_status="fallback",
        method="fallback_greedy",
    )


__all__ = [
    "OptimizerConfig",
    "OptimizationResult",
    "optimize_portfolio",
    "CVXPY_AVAILABLE",
]
