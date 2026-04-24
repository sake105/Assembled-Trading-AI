"""Reverse Stress Testing — Find scenarios that cause a target loss.

Instead of asking "what happens if X?", reverse stress asks:
"What must happen for us to lose Y%?"

Uses scipy.optimize to find the minimum-norm shock vector that causes
the target portfolio loss. This identifies the most plausible
catastrophe scenarios.

References:
    Bank of England (2019) — Reverse stress testing guidance
    Basel Committee on Banking Supervision — Stress testing principles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReverseStressResult:
    """Result of a reverse stress test optimization."""
    target_loss: float
    achieved_loss: float
    shock_vector: np.ndarray
    shock_labels: list[str]
    shock_norm: float
    plausibility_score: float  # Lower = more plausible (smaller shock needed)
    converged: bool
    top_shocks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Historical crisis shock profiles (for plausibility bounds)
# ---------------------------------------------------------------------------

HISTORICAL_CRISES = {
    "black_monday_1987": {
        "description": "Black Monday — single-day crash",
        "equity_shock": -0.226,
        "vol_multiplier": 5.0,
        "duration_days": 1,
    },
    "ltcm_1998": {
        "description": "LTCM / Russian / Asian crisis",
        "equity_shock": -0.20,
        "vol_multiplier": 3.0,
        "duration_days": 60,
    },
    "dotcom_2000": {
        "description": "Dot-com crash",
        "equity_shock": -0.49,
        "vol_multiplier": 2.0,
        "duration_days": 550,
    },
    "gfc_2008": {
        "description": "Global Financial Crisis",
        "equity_shock": -0.57,
        "vol_multiplier": 4.0,
        "duration_days": 350,
    },
    "flash_crash_2010": {
        "description": "Flash Crash — intraday",
        "equity_shock": -0.092,
        "vol_multiplier": 6.0,
        "duration_days": 1,
    },
    "china_deval_2015": {
        "description": "China devaluation / growth scare",
        "equity_shock": -0.12,
        "vol_multiplier": 2.5,
        "duration_days": 30,
    },
    "volmageddon_2018": {
        "description": "Volatility selling blowup Feb 2018",
        "equity_shock": -0.10,
        "vol_multiplier": 4.0,
        "duration_days": 5,
    },
    "covid_2020": {
        "description": "COVID-19 crash",
        "equity_shock": -0.34,
        "vol_multiplier": 5.0,
        "duration_days": 23,
    },
    "rate_shock_2022": {
        "description": "Fed rate hiking cycle",
        "equity_shock": -0.25,
        "vol_multiplier": 1.8,
        "duration_days": 280,
    },
    "svb_2023": {
        "description": "SVB bank run / regional banking",
        "equity_shock": -0.08,
        "vol_multiplier": 2.5,
        "duration_days": 10,
    },
    "yen_carry_2024": {
        "description": "Yen carry trade unwind Aug 2024",
        "equity_shock": -0.06,
        "vol_multiplier": 3.0,
        "duration_days": 3,
    },
    "gamestop_2021": {
        "description": "Meme stock squeeze",
        "equity_shock": -0.05,
        "vol_multiplier": 4.0,
        "duration_days": 10,
    },
    "opec_2014": {
        "description": "OPEC price war — oil collapse",
        "equity_shock": -0.08,
        "vol_multiplier": 2.0,
        "duration_days": 120,
    },
    "eu_debt_2011": {
        "description": "European sovereign debt crisis",
        "equity_shock": -0.19,
        "vol_multiplier": 2.5,
        "duration_days": 90,
    },
    "taper_tantrum_2013": {
        "description": "Fed taper announcement shock",
        "equity_shock": -0.06,
        "vol_multiplier": 1.8,
        "duration_days": 30,
    },
}

# Hypothetical extreme scenarios
HYPOTHETICAL_SCENARIOS = {
    "correlation_crisis": {
        "description": "All correlations → 0.9",
        "equity_shock": -0.20,
        "vol_multiplier": 3.0,
        "correlation_override": 0.9,
    },
    "liquidity_crisis": {
        "description": "Spreads ×5, ADV ×0.3",
        "equity_shock": -0.15,
        "vol_multiplier": 3.0,
        "spread_multiplier": 5.0,
        "volume_multiplier": 0.3,
    },
    "triple_down": {
        "description": "3 consecutive -5% days",
        "equity_shock": -0.1426,  # (0.95^3 - 1)
        "vol_multiplier": 5.0,
    },
    "vix_80": {
        "description": "VIX springs from 15 to 80",
        "equity_shock": -0.25,
        "vol_multiplier": 5.33,
    },
    "tech_sector_collapse": {
        "description": "Tech -30%, rest ±5%",
        "equity_shock": -0.10,  # portfolio-dependent
        "vol_multiplier": 3.0,
    },
    "usd_squeeze": {
        "description": "USD +15% rapid appreciation",
        "equity_shock": -0.08,
        "vol_multiplier": 2.0,
    },
    "oil_supply_shock": {
        "description": "Oil +100% supply disruption",
        "equity_shock": -0.12,
        "vol_multiplier": 2.5,
    },
    "emergency_rate_hike": {
        "description": "Fed emergency +200bps overnight",
        "equity_shock": -0.10,
        "vol_multiplier": 4.0,
    },
}


def get_all_scenario_names() -> list[str]:
    """Return names of all historical + hypothetical scenarios."""
    return list(HISTORICAL_CRISES.keys()) + list(HYPOTHETICAL_SCENARIOS.keys())


def get_scenario(name: str) -> dict:
    """Look up a scenario by name."""
    if name in HISTORICAL_CRISES:
        return HISTORICAL_CRISES[name]
    if name in HYPOTHETICAL_SCENARIOS:
        return HYPOTHETICAL_SCENARIOS[name]
    raise KeyError(f"Unknown scenario: {name}")


# ---------------------------------------------------------------------------
# Reverse stress optimization
# ---------------------------------------------------------------------------


def reverse_stress_test(
    weights: np.ndarray,
    covariance: np.ndarray,
    expected_returns: np.ndarray | None = None,
    target_loss: float = -0.20,
    plausibility_bound: float = 3.0,
    n_restarts: int = 5,
    seed: int = 42,
) -> ReverseStressResult:
    """Find the minimum-norm shock vector causing target_loss.

    Solves: min ||s|| s.t. w' @ (mu + s) <= target_loss, ||s|| <= bound

    Args:
        weights: Portfolio weights (n_assets,).
        covariance: Asset covariance matrix (n_assets, n_assets).
        expected_returns: Expected returns per asset. If None, uses zero.
        target_loss: Target portfolio loss (negative, e.g. -0.20 for 20% loss).
        plausibility_bound: Max L2-norm of shock vector (in std-dev units).
        n_restarts: Number of random restarts for optimization.
        seed: Random seed.

    Returns:
        ReverseStressResult with the optimal shock vector and metadata.
    """
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        logger.warning("[ReverseStress] scipy not available — returning empty result")
        n = len(weights)
        return ReverseStressResult(
            target_loss=target_loss, achieved_loss=0.0,
            shock_vector=np.zeros(n), shock_labels=[f"asset_{i}" for i in range(n)],
            shock_norm=0.0, plausibility_score=999.0, converged=False,
        )

    n = len(weights)
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    mu = np.asarray(expected_returns, dtype=float) if expected_returns is not None else np.zeros(n)

    # Asset volatilities for scaling
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols > 0, vols, 1e-6)

    rng = np.random.default_rng(seed)
    best_result = None
    best_norm = float("inf")

    for _ in range(n_restarts):
        # Random starting point (scaled by asset vols)
        s0 = rng.normal(0, 1, n) * vols * 0.5

        def objective(s: np.ndarray) -> float:
            """Minimize shock norm."""
            return float(np.sum((s / vols) ** 2))

        def loss_constraint(s: np.ndarray) -> float:
            """Portfolio return with shock must be <= target_loss."""
            port_return = float(w @ (mu + s))
            return target_loss - port_return  # >= 0 when loss achieved

        def bound_constraint(s: np.ndarray) -> float:
            """Shock norm must be <= plausibility_bound."""
            return plausibility_bound ** 2 - float(np.sum((s / vols) ** 2))

        constraints = [
            {"type": "ineq", "fun": loss_constraint},
            {"type": "ineq", "fun": bound_constraint},
        ]

        try:
            res = scipy_minimize(
                objective, s0, method="SLSQP",
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )

            if res.success or res.fun < best_norm:
                port_loss = float(w @ (mu + res.x))
                norm = float(np.sqrt(np.sum((res.x / vols) ** 2)))

                if port_loss <= target_loss * 0.95:  # Close enough
                    if norm < best_norm:
                        best_norm = norm
                        best_result = res
        except Exception:
            continue

    if best_result is None:
        return ReverseStressResult(
            target_loss=target_loss, achieved_loss=0.0,
            shock_vector=np.zeros(n),
            shock_labels=[f"asset_{i}" for i in range(n)],
            shock_norm=0.0, plausibility_score=999.0, converged=False,
        )

    shock = best_result.x
    achieved = float(w @ (mu + shock))
    norm = float(np.sqrt(np.sum((shock / vols) ** 2)))

    # Top shocks (sorted by absolute magnitude)
    sorted_idx = np.argsort(-np.abs(shock))
    top_shocks = [
        {"asset_idx": int(i), "shock": float(shock[i]),
         "shock_std": float(shock[i] / vols[i])}
        for i in sorted_idx[:5]
    ]

    return ReverseStressResult(
        target_loss=target_loss,
        achieved_loss=round(achieved, 6),
        shock_vector=shock,
        shock_labels=[f"asset_{i}" for i in range(n)],
        shock_norm=round(norm, 4),
        plausibility_score=round(norm, 4),
        converged=best_result.success,
        top_shocks=top_shocks,
    )


def run_multiple_reverse_stress(
    weights: np.ndarray,
    covariance: np.ndarray,
    expected_returns: np.ndarray | None = None,
    target_losses: list[float] | None = None,
    plausibility_bound: float = 3.0,
) -> list[ReverseStressResult]:
    """Run reverse stress tests for multiple target loss levels.

    Args:
        weights: Portfolio weights.
        covariance: Asset covariance matrix.
        expected_returns: Expected returns (optional).
        target_losses: List of target losses (default: [-0.05, -0.10, -0.15, -0.20, -0.30]).
        plausibility_bound: Max shock norm.

    Returns:
        List of ReverseStressResult for each target.
    """
    if target_losses is None:
        target_losses = [-0.05, -0.10, -0.15, -0.20, -0.30]

    results = []
    for tl in target_losses:
        res = reverse_stress_test(
            weights, covariance, expected_returns,
            target_loss=tl, plausibility_bound=plausibility_bound,
        )
        results.append(res)
        logger.info("[ReverseStress] target=%.1f%% → achieved=%.2f%%, norm=%.3f, converged=%s",
                    tl * 100, res.achieved_loss * 100, res.shock_norm, res.converged)

    return results


def stress_test_portfolio_against_scenarios(
    weights: np.ndarray,
    asset_returns: pd.DataFrame | None = None,
    scenarios: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Apply historical + hypothetical scenarios to a portfolio.

    Args:
        weights: Portfolio weights.
        asset_returns: Optional return series for scenario replay.
        scenarios: Custom scenarios dict. If None, uses all built-in scenarios.

    Returns:
        DataFrame with scenario_name, description, equity_shock,
        portfolio_impact, severity columns.
    """
    if scenarios is None:
        scenarios = {**HISTORICAL_CRISES, **HYPOTHETICAL_SCENARIOS}

    w = np.asarray(weights, dtype=float)
    rows = []

    for name, spec in scenarios.items():
        eq_shock = spec.get("equity_shock", 0.0)
        # Simplified: apply uniform shock to all assets
        portfolio_impact = float(w.sum() * eq_shock)

        severity = "LOW"
        if abs(portfolio_impact) > 0.05:
            severity = "MEDIUM"
        if abs(portfolio_impact) > 0.15:
            severity = "HIGH"
        if abs(portfolio_impact) > 0.25:
            severity = "CRITICAL"

        rows.append({
            "scenario_name": name,
            "description": spec.get("description", ""),
            "equity_shock": eq_shock,
            "vol_multiplier": spec.get("vol_multiplier", 1.0),
            "portfolio_impact": round(portfolio_impact, 4),
            "severity": severity,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("portfolio_impact").reset_index(drop=True)
    logger.info("[StressTest] Evaluated %d scenarios, %d CRITICAL",
                len(result), (result["severity"] == "CRITICAL").sum())
    return result
