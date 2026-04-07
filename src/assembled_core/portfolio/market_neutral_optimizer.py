"""Market-neutral portfolio construction (V17).

Constructs dollar-neutral, beta-neutral, and optionally sector-neutral
portfolios using CVXPY constraints on top of the cost-aware optimizer (V9).

Constraints:
- Dollar-neutral: sum(long_weights) approx sum(short_weights)
- Beta-neutral: portfolio beta approx 0
- Sector-neutral: max net sector exposure <= threshold

Falls back to simple long-short if CVXPY is not installed.

Reference: Citadel/Millennium market-neutral construction patterns.
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
class MarketNeutralConfig:
    """Configuration for market-neutral portfolio construction."""

    risk_aversion: float = 1.0
    turnover_penalty: float = 0.001
    max_weight: float = 0.10
    max_gross_exposure: float = 2.0  # Long + Short gross
    dollar_neutral_tolerance: float = 0.02  # Max net exposure
    beta_neutral: bool = True
    beta_neutral_tolerance: float = 0.05  # Max abs portfolio beta
    sector_neutral: bool = False
    max_sector_net_exposure: float = 0.05
    sector_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class MarketNeutralResult:
    """Result of market-neutral optimization."""

    long_weights: dict[str, float]
    short_weights: dict[str, float]
    net_exposure: float  # sum(long) - sum(|short|)
    gross_exposure: float  # sum(long) + sum(|short|)
    portfolio_beta: float
    expected_return: float
    expected_risk: float
    turnover_cost: float
    solver_status: str
    method: str


def optimize_market_neutral(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    betas: pd.Series | None = None,
    current_weights: dict[str, float] | None = None,
    per_symbol_cost_bps: dict[str, float] | None = None,
    config: MarketNeutralConfig | None = None,
) -> MarketNeutralResult:
    """Optimize a market-neutral portfolio.

    Args:
        expected_returns: Symbol -> expected return.
        covariance: Covariance matrix.
        betas: Symbol -> market beta (for beta neutrality).
        current_weights: Current portfolio weights.
        per_symbol_cost_bps: Per-symbol transaction costs.
        config: Market-neutral configuration.

    Returns:
        MarketNeutralResult with long/short weights.
    """
    config = config or MarketNeutralConfig()
    current_weights = current_weights or {}

    symbols = list(expected_returns.index)
    n = len(symbols)

    if n == 0:
        return MarketNeutralResult(
            long_weights={}, short_weights={}, net_exposure=0.0,
            gross_exposure=0.0, portfolio_beta=0.0, expected_return=0.0,
            expected_risk=0.0, turnover_cost=0.0, solver_status="empty",
            method="none",
        )

    mu = expected_returns.values.astype(float)
    Sigma = covariance.loc[symbols, symbols].values.astype(float)
    w_old = np.array([current_weights.get(s, 0.0) for s in symbols])

    # Betas
    if betas is not None:
        beta_vec = np.array([betas.get(s, 1.0) for s in symbols])
    else:
        beta_vec = np.ones(n)  # Assume beta=1 if not provided

    # Cost vector
    default_cost = 6.0
    cost_vec = np.array([
        (per_symbol_cost_bps or {}).get(s, default_cost) / 10_000.0
        for s in symbols
    ])

    if CVXPY_AVAILABLE:
        return _optimize_cvxpy_neutral(
            symbols, mu, Sigma, beta_vec, w_old, cost_vec, config
        )
    else:
        return _optimize_fallback_neutral(
            symbols, mu, Sigma, beta_vec, w_old, cost_vec, config
        )


def _optimize_cvxpy_neutral(
    symbols: list[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    beta_vec: np.ndarray,
    w_old: np.ndarray,
    cost_vec: np.ndarray,
    config: MarketNeutralConfig,
) -> MarketNeutralResult:
    """CVXPY market-neutral optimization."""
    n = len(symbols)
    w = cp.Variable(n)

    # Objective
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm1(cp.multiply(cost_vec, (w - w_old)))

    objective = cp.Maximize(
        ret - config.risk_aversion * risk - config.turnover_penalty * turnover
    )

    # Constraints
    constraints = [
        w >= -config.max_weight,  # Allow shorts
        w <= config.max_weight,
        cp.norm1(w) <= config.max_gross_exposure,  # Gross exposure limit
    ]

    # Dollar neutrality: |sum(w)| <= tolerance
    constraints.append(cp.abs(cp.sum(w)) <= config.dollar_neutral_tolerance)

    # Beta neutrality
    if config.beta_neutral:
        constraints.append(
            cp.abs(beta_vec @ w) <= config.beta_neutral_tolerance
        )

    # Sector neutrality
    if config.sector_neutral and config.sector_mapping:
        sectors = set(config.sector_mapping.values())
        for sector in sectors:
            mask = np.array([
                1.0 if config.sector_mapping.get(s) == sector else 0.0
                for s in symbols
            ])
            if mask.sum() > 0:
                constraints.append(
                    cp.abs(mask @ w) <= config.max_sector_net_exposure
                )

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000)

        if prob.status in ("optimal", "optimal_inaccurate"):
            w_opt = w.value
            return _build_result(symbols, mu, Sigma, beta_vec, w_opt, w_old, cost_vec, prob.status, "cvxpy")
        else:
            _log.warning("CVXPY neutral solver: %s — falling back", prob.status)
    except Exception as e:
        _log.warning("CVXPY neutral failed: %s — fallback", e)

    return _optimize_fallback_neutral(symbols, mu, Sigma, beta_vec, w_old, cost_vec, config)


def _optimize_fallback_neutral(
    symbols: list[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    beta_vec: np.ndarray,
    w_old: np.ndarray,
    cost_vec: np.ndarray,
    config: MarketNeutralConfig,
) -> MarketNeutralResult:
    """Simple long-short fallback: long top-N, short bottom-N by expected return."""
    n = len(symbols)
    ranks = np.argsort(mu)  # ascending

    # Short bottom quartile, long top quartile
    n_each = max(n // 4, 1)
    w_opt = np.zeros(n)

    # Long
    long_idx = ranks[-n_each:]
    for i in long_idx:
        w_opt[i] = config.max_weight

    # Short
    short_idx = ranks[:n_each]
    for i in short_idx:
        w_opt[i] = -config.max_weight

    # Dollar neutral: adjust to zero net
    net = w_opt.sum()
    if abs(net) > 1e-10:
        # Scale the larger side down
        if net > 0:
            long_total = w_opt[w_opt > 0].sum()
            if long_total > 0:
                w_opt[w_opt > 0] *= (long_total - net) / long_total
        else:
            short_total = abs(w_opt[w_opt < 0].sum())
            if short_total > 0:
                w_opt[w_opt < 0] *= (short_total + net) / short_total

    return _build_result(symbols, mu, Sigma, beta_vec, w_opt, w_old, cost_vec, "fallback", "fallback_longshort")


def _build_result(
    symbols: list[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    beta_vec: np.ndarray,
    w_opt: np.ndarray,
    w_old: np.ndarray,
    cost_vec: np.ndarray,
    status: str,
    method: str,
) -> MarketNeutralResult:
    """Build MarketNeutralResult from optimized weights."""
    long_w = {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols) if w_opt[i] > 1e-8}
    short_w = {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols) if w_opt[i] < -1e-8}

    net = float(w_opt.sum())
    gross = float(np.abs(w_opt).sum())
    port_beta = float(beta_vec @ w_opt)
    exp_ret = float(mu @ w_opt)
    exp_risk = float(np.sqrt(max(w_opt @ Sigma @ w_opt, 0.0)))
    tc = float(np.sum(np.abs(w_opt - w_old) * cost_vec))

    return MarketNeutralResult(
        long_weights=long_w,
        short_weights=short_w,
        net_exposure=round(net, 6),
        gross_exposure=round(gross, 6),
        portfolio_beta=round(port_beta, 6),
        expected_return=round(exp_ret, 6),
        expected_risk=round(exp_risk, 6),
        turnover_cost=round(tc, 6),
        solver_status=status,
        method=method,
    )


__all__ = [
    "MarketNeutralConfig",
    "MarketNeutralResult",
    "optimize_market_neutral",
]
