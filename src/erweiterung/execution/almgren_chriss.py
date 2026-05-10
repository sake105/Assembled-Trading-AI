"""Almgren-Chriss Optimal Execution (2000).

Theorie
-------
Bei großen Orders entsteht Market-Impact:
- **Permanent Impact** (γ): Preis bleibt um γ·v dauerhaft verschoben.
- **Temporary Impact** (η): Preis bleibt während Trade um η·(v/T) schlechter.

Almgren-Chriss formulieren das als Optimization über Trajektorien:
    min E[Cost] + λ Var[Cost]

Lösung
------
Optimal trade rate ist exponentiell:
    x_t = X · sinh(κ·(T-t)) / sinh(κ·T)
mit
    κ = √(λ·σ² / η_tilde),  η_tilde = η - 1/2 γ τ

Wobei
- X = total order size
- T = total execution time
- λ = risk aversion
- σ = volatility
- τ = trading interval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketImpactParams:
    permanent_impact_gamma: float = 1e-7  # $/share permanent
    temporary_impact_eta: float = 1e-6  # $/share temporary
    volatility: float = 0.02  # daily vol of underlying
    risk_aversion: float = 1e-6  # λ


def optimal_trade_schedule(
    total_shares: float,
    total_time_steps: int,
    params: MarketImpactParams,
    tau: float = 1.0,
) -> np.ndarray:
    """Almgren-Chriss optimal trade rate per step.

    Args:
        total_shares: X (full order size).
        total_time_steps: T (in units of tau).
        params: MarketImpactParams.
        tau: time per step.

    Returns:
        Array (T,) — shares to trade in each step. Sums to total_shares.
    """
    if total_time_steps < 1 or total_shares == 0:
        return np.array([total_shares])
    eta_tilde = params.temporary_impact_eta - 0.5 * params.permanent_impact_gamma * tau
    if eta_tilde <= 0:
        # Impact dominated by permanent component => uniform schedule (TWAP)
        return np.full(total_time_steps, total_shares / total_time_steps)
    kappa = np.sqrt(params.risk_aversion * params.volatility**2 / eta_tilde)
    if kappa * total_time_steps > 700:
        # Numerical stability — fallback TWAP
        return np.full(total_time_steps, total_shares / total_time_steps)
    schedule = np.zeros(total_time_steps)
    sinh_kT = np.sinh(kappa * total_time_steps * tau)
    if sinh_kT == 0:
        return np.full(total_time_steps, total_shares / total_time_steps)
    for t in range(total_time_steps):
        # Holdings at time (t+1)*tau:
        x_next = (
            total_shares * np.sinh(kappa * (total_time_steps - t - 1) * tau) / sinh_kT
        )
        x_now = total_shares * np.sinh(kappa * (total_time_steps - t) * tau) / sinh_kT
        schedule[t] = x_now - x_next
    # Clip negative due to floating point
    schedule = np.clip(schedule, 0, None)
    # Normalize to sum exactly total_shares
    s = schedule.sum()
    if s > 0:
        schedule = schedule * (total_shares / s)
    return schedule


def expected_cost(
    schedule: np.ndarray,
    params: MarketImpactParams,
    tau: float = 1.0,
) -> float:
    """E[Cost] = γ X²/2 + Σ η * (n_k/τ) * n_k."""
    X = float(schedule.sum())
    perm_cost = 0.5 * params.permanent_impact_gamma * X * X
    temp_cost = sum((params.temporary_impact_eta / tau) * n * n for n in schedule)
    return float(perm_cost + temp_cost)


def variance_cost(
    schedule: np.ndarray,
    params: MarketImpactParams,
    tau: float = 1.0,
) -> float:
    """Var[Cost] = σ² τ Σ x_k² wobei x_k = remaining inventory."""
    sigma2 = params.volatility**2
    rem = schedule[::-1].cumsum()[::-1]  # x_k = remaining at step k
    return float(sigma2 * tau * (rem**2).sum())


__all__ = [
    "MarketImpactParams",
    "optimal_trade_schedule",
    "expected_cost",
    "variance_cost",
]
