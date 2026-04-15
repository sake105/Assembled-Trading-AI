"""Almgren-Chriss Optimal Execution Model (M20.2).

Implements the Almgren-Chriss (2001) framework for optimal trade execution
that minimizes the sum of:
  - Permanent market impact (information leakage)
  - Temporary market impact (liquidity demand)
  - Execution risk (price volatility during execution)

The model produces an optimal execution trajectory that balances urgency
(risk aversion) against market impact cost.

Reference:
    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio
    transactions." Journal of Risk, 3(2), 5-39.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlmgrenChrissParams:
    """Parameters for the Almgren-Chriss model.

    Attributes:
        sigma: Daily volatility of the asset (decimal, e.g. 0.02 = 2%).
        gamma: Permanent impact coefficient (price moves gamma * (shares/ADV) permanently).
        eta: Temporary impact coefficient (price moves eta * (trading_rate/ADV) temporarily).
        risk_aversion: Lambda - trader's risk aversion parameter.
            Higher = faster execution (less risk tolerance).
            Lower = slower execution (more patient, less impact).
            Typical range: 1e-6 to 1e-4 for institutional accounts.
        adv: Average daily volume (shares).
        daily_volume_fraction: Fraction of day available for trading (default: 1.0).
    """

    sigma: float = 0.02
    gamma: float = 0.1
    eta: float = 0.05
    risk_aversion: float = 1e-5
    adv: float = 1_000_000.0
    daily_volume_fraction: float = 1.0


@dataclass
class ExecutionTrajectory:
    """Result of Almgren-Chriss optimal execution computation.

    Attributes:
        time_steps: Array of time points (0 to T).
        holdings: Optimal shares remaining at each time step.
        trade_list: Shares to trade in each interval (negative = sell).
        trading_rate: Trading rate (shares per interval) at each step.
        expected_cost_bps: Expected total execution cost in basis points.
        permanent_impact_bps: Permanent impact component in bps.
        temporary_impact_bps: Temporary impact component in bps.
        risk_penalty_bps: Execution risk penalty in bps.
        optimal_horizon_days: Suggested optimal execution horizon.
        participation_rates: Fraction of ADV at each interval.
    """

    time_steps: np.ndarray
    holdings: np.ndarray
    trade_list: np.ndarray
    trading_rate: np.ndarray
    expected_cost_bps: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    risk_penalty_bps: float
    optimal_horizon_days: float
    participation_rates: np.ndarray


def compute_optimal_trajectory(
    total_shares: float,
    price: float,
    n_intervals: int,
    horizon_days: float,
    params: AlmgrenChrissParams | None = None,
) -> ExecutionTrajectory:
    """Compute Almgren-Chriss optimal execution trajectory.

    The optimal strategy follows a sinh-based trajectory that front-loads
    or back-loads execution depending on risk aversion.

    Args:
        total_shares: Total shares to execute (positive = buy, negative = sell).
        price: Current market price per share.
        n_intervals: Number of trading intervals to divide the horizon into.
        horizon_days: Execution horizon in trading days.
        params: Model parameters (defaults used if None).

    Returns:
        ExecutionTrajectory with optimal schedule and cost breakdown.
    """
    if params is None:
        params = AlmgrenChrissParams()

    n = max(n_intervals, 1)
    X = abs(total_shares)
    T = max(horizon_days, 0.01)
    tau = T / n  # length of each interval in days

    sigma = params.sigma
    gamma = params.gamma
    eta = params.eta
    lam = params.risk_aversion
    adv = max(params.adv, 1.0)

    # Normalize impact coefficients by ADV for dimensional consistency
    # gamma_hat: permanent impact per share as fraction of price
    # eta_hat: temporary impact per share-per-day as fraction of price
    gamma_hat = gamma / adv
    eta_hat = eta / adv

    # Almgren-Chriss kappa: urgency parameter
    # kappa = sqrt(lambda * sigma^2 / eta_hat)
    # Higher kappa -> more front-loaded execution
    if eta_hat > 0:
        kappa_sq = lam * sigma**2 / (eta_hat / tau)
        kappa = np.sqrt(max(kappa_sq, 0.0))
    else:
        kappa = 0.0

    # Optimal trajectory: holdings at time j
    # x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
    time_steps = np.linspace(0, T, n + 1)
    kappa_T = kappa * T

    if kappa_T < 1e-10:
        # Low urgency: linear trajectory (TWAP)
        holdings = X * (1.0 - time_steps / T)
    else:
        # sinh-based trajectory
        sinh_kT = np.sinh(kappa_T)
        if sinh_kT < 1e-15:
            holdings = X * (1.0 - time_steps / T)
        else:
            holdings = X * np.sinh(kappa * (T - time_steps)) / sinh_kT

    # Ensure boundary conditions
    holdings[0] = X
    holdings[-1] = 0.0

    # Trade list: shares to execute in each interval
    trade_list = -np.diff(holdings)  # positive = shares executed

    # Trading rate: shares per day
    trading_rate = trade_list / tau

    # Participation rates
    interval_adv = adv * tau * params.daily_volume_fraction
    participation_rates = np.abs(trade_list) / max(interval_adv, 1.0)

    # Cost computation
    notional = X * price

    # Permanent impact cost: 0.5 * gamma_hat * X^2 * price
    perm_cost = 0.5 * gamma_hat * X * price
    perm_bps = (perm_cost / notional * 10000) if notional > 0 else 0.0

    # Temporary impact cost: eta_hat * sum(n_j^2 / tau) * price
    temp_cost = eta_hat * np.sum(trade_list**2 / tau) * price / X if X > 0 else 0.0
    temp_bps = (temp_cost / price * 10000) if price > 0 else 0.0

    # Risk penalty: lambda * sigma^2 * sum(x_j^2 * tau)
    risk_cost = lam * sigma**2 * np.sum(holdings[:-1] ** 2 * tau)
    risk_bps = (risk_cost / notional * 10000) if notional > 0 else 0.0

    total_bps = perm_bps + temp_bps + risk_bps

    # Compute suggested optimal horizon
    opt_horizon = _compute_optimal_horizon(X, price, params)

    # Apply sign convention for sells
    sign = 1.0 if total_shares >= 0 else -1.0
    holdings_signed = holdings * sign
    trade_list_signed = trade_list * sign

    result = ExecutionTrajectory(
        time_steps=time_steps,
        holdings=holdings_signed,
        trade_list=trade_list_signed,
        trading_rate=trading_rate * sign,
        expected_cost_bps=round(float(total_bps), 2),
        permanent_impact_bps=round(float(perm_bps), 2),
        temporary_impact_bps=round(float(temp_bps), 2),
        risk_penalty_bps=round(float(risk_bps), 2),
        optimal_horizon_days=round(float(opt_horizon), 2),
        participation_rates=participation_rates,
    )

    logger.info(
        "[Almgren-Chriss] %s %.0f shares @ $%.2f over %.1f days in %d intervals: "
        "cost=%.1f bps (perm=%.1f + temp=%.1f + risk=%.1f), "
        "max participation=%.1f%%",
        "BUY" if total_shares >= 0 else "SELL",
        abs(total_shares), price, horizon_days, n_intervals,
        total_bps, perm_bps, temp_bps, risk_bps,
        float(participation_rates.max()) * 100 if len(participation_rates) > 0 else 0,
    )

    return result


def _compute_optimal_horizon(
    total_shares: float,
    price: float,
    params: AlmgrenChrissParams,
) -> float:
    """Estimate optimal execution horizon in trading days.

    The optimal horizon balances temporary impact (decreases with T)
    against risk penalty (increases with T). Analytically:
        T* = (3/2 * eta * X / (lambda * sigma^2 * ADV))^(1/3)

    Args:
        total_shares: Total shares to execute.
        price: Current price.
        params: Model parameters.

    Returns:
        Optimal horizon in trading days.
    """
    X = abs(total_shares)
    adv = max(params.adv, 1.0)
    sigma = params.sigma
    eta = params.eta
    lam = params.risk_aversion

    denom = lam * sigma**2 * adv
    if denom < 1e-15 or X < 1:
        return 1.0

    T_star = (1.5 * eta * X / denom) ** (1.0 / 3.0)
    # Clamp to reasonable range
    return float(np.clip(T_star, 0.1, 20.0))


def estimate_impact_cost(
    total_shares: float,
    price: float,
    adv: float,
    sigma: float,
    horizon_days: float = 1.0,
    *,
    gamma: float = 0.1,
    eta: float = 0.05,
) -> dict[str, float]:
    """Quick impact cost estimate without full trajectory computation.

    Useful for pre-trade cost estimation in the execution pipeline.

    Args:
        total_shares: Shares to trade.
        price: Current price.
        adv: Average daily volume.
        sigma: Daily volatility.
        horizon_days: Planned execution horizon.
        gamma: Permanent impact coefficient.
        eta: Temporary impact coefficient.

    Returns:
        Dict with permanent_bps, temporary_bps, total_bps, total_cost_usd.
    """
    X = abs(total_shares)
    notional = X * price
    if notional <= 0 or adv <= 0:
        return {
            "permanent_bps": 0.0,
            "temporary_bps": 0.0,
            "total_bps": 0.0,
            "total_cost_usd": 0.0,
        }

    participation = X / (adv * max(horizon_days, 0.01))

    perm_bps = 0.5 * gamma * (X / adv) * 10000
    temp_bps = eta * participation * 10000

    total_bps = perm_bps + temp_bps
    total_usd = notional * total_bps / 10000

    return {
        "permanent_bps": round(float(perm_bps), 2),
        "temporary_bps": round(float(temp_bps), 2),
        "total_bps": round(float(total_bps), 2),
        "total_cost_usd": round(float(total_usd), 2),
    }


def compute_frontier(
    total_shares: float,
    price: float,
    params: AlmgrenChrissParams | None = None,
    n_points: int = 20,
    max_horizon: float = 10.0,
) -> list[dict[str, float]]:
    """Compute the efficient execution frontier: cost vs. risk for different horizons.

    Args:
        total_shares: Total shares to execute.
        price: Current price.
        params: Model parameters.
        n_points: Number of frontier points.
        max_horizon: Maximum horizon in days.

    Returns:
        List of dicts with horizon_days, expected_cost_bps, risk_bps.
    """
    if params is None:
        params = AlmgrenChrissParams()

    horizons = np.linspace(0.1, max_horizon, n_points)
    frontier = []

    for T in horizons:
        traj = compute_optimal_trajectory(
            total_shares, price, n_intervals=max(int(T * 10), 5),
            horizon_days=float(T), params=params,
        )
        frontier.append({
            "horizon_days": round(float(T), 2),
            "expected_cost_bps": traj.expected_cost_bps,
            "permanent_impact_bps": traj.permanent_impact_bps,
            "temporary_impact_bps": traj.temporary_impact_bps,
            "risk_penalty_bps": traj.risk_penalty_bps,
            "max_participation_pct": round(float(traj.participation_rates.max()) * 100, 1)
            if len(traj.participation_rates) > 0 else 0.0,
        })

    return frontier


__all__ = [
    "AlmgrenChrissParams",
    "ExecutionTrajectory",
    "compute_optimal_trajectory",
    "estimate_impact_cost",
    "compute_frontier",
]
