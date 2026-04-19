"""Multi-Period Portfolio Optimization.

Implements Garleanu & Pedersen (2013) dynamic trading:
    w*_t = aim_t - trade_speed * (w_{t-1} - aim_t)

Where aim_t is the frictionless Markowitz portfolio and trade_speed
controls how fast we converge (function of transaction costs).

This reduces turnover 20-40% vs. single-period rebalancing by
anticipating future trading costs in today's decision.

Also provides a simple rolling multi-period optimizer that plans
K periods ahead using expected returns forecasts.

References:
    Garleanu & Pedersen (2013) "Dynamic Trading with Predictable Returns
    and Transaction Costs", Journal of Finance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_minimize = None  # type: ignore


@dataclass
class MultiPeriodResult:
    """Result of multi-period optimization."""
    target_weights: dict[str, float]  # final target for this period
    aim_portfolio: dict[str, float]  # frictionless aim portfolio
    trade_speed: float  # how fast to converge (0-1)
    expected_turnover: float  # expected turnover vs current
    periods_ahead: int
    method: str


def compute_trade_speed(
    risk_aversion: float,
    transaction_cost: float,
    autocorrelation: float = 0.0,
) -> float:
    """Compute optimal trade speed from Garleanu-Pedersen.

    Trade speed = how much of the gap (aim - current) to close per period.

    Higher transaction cost → slower trading.
    Higher risk aversion → faster trading (opportunity cost of suboptimal portfolio).
    Higher return autocorrelation → slower (returns will persist, no rush).

    Args:
        risk_aversion: Lambda parameter.
        transaction_cost: Average one-way cost as fraction (e.g., 0.001 = 10bps).
        autocorrelation: Estimated alpha autocorrelation (-1 to 1).

    Returns:
        Trade speed in (0, 1].
    """
    if transaction_cost < 1e-10:
        return 1.0  # zero cost → trade instantly

    # Garleanu-Pedersen approximate trade speed:
    # speed ≈ sqrt(risk_aversion / (2 * tc_cost)) * (1 - autocorrelation)
    # Clamped to (0, 1]
    raw_speed = np.sqrt(risk_aversion / (2.0 * transaction_cost + 1e-10))
    raw_speed *= (1.0 - max(-0.99, min(0.99, autocorrelation)))
    return float(min(1.0, max(0.01, raw_speed)))


def garleanu_pedersen_target(
    aim_weights: dict[str, float],
    current_weights: dict[str, float],
    trade_speed: float | None = None,
    risk_aversion: float = 1.0,
    transaction_cost: float = 0.0006,
    autocorrelation: float = 0.0,
) -> MultiPeriodResult:
    """Compute multi-period target weights via Garleanu-Pedersen.

    w*_t = w_{t-1} + speed * (aim_t - w_{t-1})

    This is a partial adjustment model: instead of jumping to the
    frictionless optimum, we move part-way there to balance
    portfolio quality against transaction costs.

    Args:
        aim_weights: Frictionless optimal weights (from any single-period optimizer).
        current_weights: Current portfolio weights.
        trade_speed: Override trade speed (0-1). If None, computed from params.
        risk_aversion: Risk aversion for trade speed computation.
        transaction_cost: Average one-way cost fraction.
        autocorrelation: Estimated alpha autocorrelation.

    Returns:
        MultiPeriodResult with adjusted target weights.
    """
    symbols = sorted(set(aim_weights) | set(current_weights))
    n = len(symbols)  # noqa: F841

    if trade_speed is None:
        trade_speed = compute_trade_speed(risk_aversion, transaction_cost, autocorrelation)

    w_aim = np.array([aim_weights.get(s, 0.0) for s in symbols])
    w_curr = np.array([current_weights.get(s, 0.0) for s in symbols])

    # Partial adjustment
    w_target = w_curr + trade_speed * (w_aim - w_curr)

    # Ensure non-negative and normalize
    w_target = np.maximum(w_target, 0.0)
    total = w_target.sum()
    if total > 1e-8:
        w_target /= total

    turnover = float(np.sum(np.abs(w_target - w_curr)))

    return MultiPeriodResult(
        target_weights={s: round(float(w_target[i]), 6) for i, s in enumerate(symbols)},
        aim_portfolio={s: round(float(w_aim[i]), 6) for i, s in enumerate(symbols)},
        trade_speed=round(trade_speed, 4),
        expected_turnover=round(turnover, 6),
        periods_ahead=1,
        method="garleanu_pedersen",
    )


def multi_period_optimize(
    expected_returns_path: list[pd.Series],
    covariance: pd.DataFrame,
    current_weights: dict[str, float],
    risk_aversion: float = 1.0,
    transaction_cost: float = 0.0006,
    max_weight: float = 0.10,
    long_only: bool = True,
) -> MultiPeriodResult:
    """Multi-period optimization with K-period lookahead.

    Uses dynamic programming to find optimal trades considering
    expected returns over multiple future periods.

    Args:
        expected_returns_path: List of expected return Series for periods 1..K.
        covariance: Covariance matrix (assumed stationary).
        current_weights: Current portfolio weights.
        risk_aversion: Risk aversion parameter.
        transaction_cost: One-way transaction cost fraction.
        max_weight: Maximum per-asset weight.
        long_only: Enforce non-negative weights.

    Returns:
        MultiPeriodResult with optimal target for period 1.
    """
    K = len(expected_returns_path)
    if K == 0:
        return MultiPeriodResult(
            target_weights=current_weights,
            aim_portfolio=current_weights,
            trade_speed=0.0,
            expected_turnover=0.0,
            periods_ahead=0,
            method="multi_period_dp",
        )

    symbols = list(covariance.columns)
    n = len(symbols)
    cov = covariance.values.astype(float)

    w_curr = np.array([current_weights.get(s, 0.0) for s in symbols])

    if not SCIPY_AVAILABLE:
        # Fallback without scipy: discounted-sum aim + Garleanu-Pedersen partial
        # adjustment. We inline the math so that periods_ahead=K is preserved
        # (the contract of this function is independent of solver availability).
        gamma = 0.95
        mu_combined = np.zeros(n)
        for t, mu_t in enumerate(expected_returns_path):
            mu_vals = mu_t.reindex(symbols).fillna(0).values.astype(float)
            mu_combined += (gamma ** t) * mu_vals
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        aim = mu_combined / (risk_aversion * vols + 1e-10)
        aim = np.maximum(aim, 0) if long_only else aim
        total = aim.sum()
        if total > 1e-8:
            aim /= total
        speed = compute_trade_speed(risk_aversion, transaction_cost)
        w_final = w_curr + speed * (aim - w_curr)
        w_final = np.maximum(w_final, 0) if long_only else w_final
        tot = w_final.sum()
        if tot > 1e-8:
            w_final /= tot
        turnover = float(np.sum(np.abs(w_final - w_curr)))
        return MultiPeriodResult(
            target_weights={s: round(float(w_final[i]), 6) for i, s in enumerate(symbols)},
            aim_portfolio={s: round(float(aim[i]), 6) for i, s in enumerate(symbols)},
            trade_speed=round(speed, 4),
            expected_turnover=round(turnover, 6),
            periods_ahead=K,
            method="multi_period_fallback_no_scipy",
        )

    # Multi-period DP: backward induction
    # V_K(w) = 0 (terminal value)
    # V_t(w) = max_{w'} { w'*mu_t - lambda*w'*Sigma*w' - tc*||w'-w||_1 + V_{t+1}(w') }
    # Approximate: solve period 1 with discounted future value

    # Discount factor for future returns
    gamma = 0.95

    # Weighted average of future returns
    mu_combined = np.zeros(n)
    for t, mu_t in enumerate(expected_returns_path):
        mu_vals = mu_t.reindex(symbols).fillna(0).values.astype(float)
        mu_combined += (gamma ** t) * mu_vals

    # Solve single optimization with combined returns
    def objective(w: np.ndarray) -> float:
        ret = float(mu_combined @ w)
        risk = float(w @ cov @ w)
        tc = transaction_cost * float(np.sum(np.abs(w - w_curr)))
        return -(ret - risk_aversion * risk - tc)

    lb = 0.001 if long_only else -max_weight
    bounds = [(lb, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    best_result = None
    best_obj = float("inf")

    for seed in range(3):
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
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback to Garleanu-Pedersen with first period
        mu_1 = expected_returns_path[0].reindex(symbols).fillna(0).values
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        aim = mu_1 / (risk_aversion * vols + 1e-10)
        aim = np.maximum(aim, 0) if long_only else aim
        total = aim.sum()
        if total > 1e-8:
            aim /= total
        aim_dict = {s: float(aim[i]) for i, s in enumerate(symbols)}
        return garleanu_pedersen_target(
            aim_dict, current_weights,
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
        )

    w_opt = best_result.x
    w_opt = np.maximum(w_opt, 0) if long_only else w_opt
    w_opt /= w_opt.sum() if w_opt.sum() > 1e-8 else 1.0

    # Apply Garleanu-Pedersen partial adjustment
    speed = compute_trade_speed(risk_aversion, transaction_cost)
    w_final = w_curr + speed * (w_opt - w_curr)
    w_final = np.maximum(w_final, 0) if long_only else w_final
    total = w_final.sum()
    if total > 1e-8:
        w_final /= total

    turnover = float(np.sum(np.abs(w_final - w_curr)))

    return MultiPeriodResult(
        target_weights={s: round(float(w_final[i]), 6) for i, s in enumerate(symbols)},
        aim_portfolio={s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
        trade_speed=round(speed, 4),
        expected_turnover=round(turnover, 6),
        periods_ahead=K,
        method="multi_period_dp",
    )


__all__ = [
    "MultiPeriodResult",
    "compute_trade_speed",
    "garleanu_pedersen_target",
    "multi_period_optimize",
]
