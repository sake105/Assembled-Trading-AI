"""Monte-Carlo Scenario Stress-Testing für Portfolios.

Simuliert synthetische Schock-Szenarien und berechnet Portfolio-Verhalten:
- Volatility Spike: σ ×3 für N Tage
- Correlation Collapse: alle ρ → 1.0 (oder 0.0)
- Liquidity Drought: Slippage ×5, Volume ÷3
- Drawdown Shock: 1-tägiger -10% Move des Markts
- Regime-Switch: plötzlicher Wechsel von Low- zu High-Vol

Pro Szenario: VaR/CVaR/MaxDD auf synthetischen Returns.

PIT-Invariante: Baseline-Statistiken aus historischen Returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    mean_return: float
    std_return: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    n_simulations: int = 0
    shock_magnitude: float = 0.0


@dataclass
class StressTestReport:
    baseline_metrics: dict = field(default_factory=dict)
    scenarios: list[ScenarioResult] = field(default_factory=list)
    worst_scenario: str = ""
    worst_cvar: float = 0.0


def _compute_var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if len(returns) == 0:
        return 0.0, 0.0
    var = float(np.quantile(returns, alpha))
    cvar = float(returns[returns <= var].mean()) if (returns <= var).any() else var
    return var, cvar


def simulate_vol_spike_scenario(
    baseline_returns: pd.Series,
    vol_multiplier: float = 3.0,
    duration: int = 20,
    n_simulations: int = 1000,
    seed: int = 42,
) -> ScenarioResult:
    """Vol-Spike: σ × vol_multiplier für duration Tage."""
    rng = np.random.default_rng(seed)
    mu = float(baseline_returns.mean())
    sigma = float(baseline_returns.std()) * vol_multiplier

    # n_simulations × duration Matrix
    sim = rng.normal(mu, sigma, (n_simulations, duration))
    final_rets = sim.sum(axis=1)

    var, cvar = _compute_var_cvar(final_rets)
    equity_paths = np.cumsum(sim, axis=1)
    max_dds = (equity_paths - np.maximum.accumulate(equity_paths, axis=1)).min(axis=1)

    return ScenarioResult(
        scenario_name="VolSpike",
        mean_return=round(float(final_rets.mean()), 4),
        std_return=round(float(final_rets.std()), 4),
        var_95=round(var, 4),
        cvar_95=round(cvar, 4),
        max_drawdown=round(float(max_dds.mean()), 4),
        n_simulations=n_simulations,
        shock_magnitude=vol_multiplier,
    )


def simulate_crash_scenario(
    baseline_returns: pd.Series,
    crash_magnitude: float = -0.10,
    recovery_days: int = 30,
    n_simulations: int = 1000,
    seed: int = 43,
) -> ScenarioResult:
    """1-Tages-Crash gefolgt von recovery_days normaler Returns."""
    rng = np.random.default_rng(seed)
    mu = float(baseline_returns.mean())
    sigma = float(baseline_returns.std())

    sim = rng.normal(mu, sigma, (n_simulations, recovery_days))
    sim[:, 0] = crash_magnitude
    final_rets = sim.sum(axis=1)

    var, cvar = _compute_var_cvar(final_rets)
    equity_paths = np.cumsum(sim, axis=1)
    max_dds = (equity_paths - np.maximum.accumulate(equity_paths, axis=1)).min(axis=1)

    return ScenarioResult(
        scenario_name="Crash",
        mean_return=round(float(final_rets.mean()), 4),
        std_return=round(float(final_rets.std()), 4),
        var_95=round(var, 4),
        cvar_95=round(cvar, 4),
        max_drawdown=round(float(max_dds.mean()), 4),
        n_simulations=n_simulations,
        shock_magnitude=abs(crash_magnitude),
    )


def simulate_correlation_breakdown_scenario(
    portfolio_returns: pd.DataFrame,
    duration: int = 20,
    target_correlation: float = 1.0,
    n_simulations: int = 500,
    seed: int = 44,
) -> ScenarioResult:
    """Correlation-Breakdown: Alle Asset-Korrelationen → target_correlation.

    Typisch für Krisen: Diversifikation bricht zusammen.
    """
    rng = np.random.default_rng(seed)
    n_assets = portfolio_returns.shape[1]
    mu = portfolio_returns.mean().values
    # Original cov → target cov
    orig_std = portfolio_returns.std().values
    # build target cov matrix with target_correlation
    target_cov = np.outer(orig_std, orig_std) * target_correlation
    np.fill_diagonal(target_cov, orig_std ** 2)

    # Equal weighted portfolio (simpel)
    weights = np.ones(n_assets) / n_assets

    final_rets = np.empty(n_simulations)
    max_dds = np.empty(n_simulations)
    for s in range(n_simulations):
        sim = rng.multivariate_normal(mu, target_cov, size=duration)
        port_rets = sim @ weights
        final_rets[s] = port_rets.sum()
        eq = np.cumsum(port_rets)
        max_dds[s] = float(np.min(eq - np.maximum.accumulate(eq)))

    var, cvar = _compute_var_cvar(final_rets)

    return ScenarioResult(
        scenario_name="CorrelationBreakdown",
        mean_return=round(float(final_rets.mean()), 4),
        std_return=round(float(final_rets.std()), 4),
        var_95=round(var, 4),
        cvar_95=round(cvar, 4),
        max_drawdown=round(float(max_dds.mean()), 4),
        n_simulations=n_simulations,
        shock_magnitude=target_correlation,
    )


def run_stress_test(
    baseline_returns: pd.Series,
    portfolio_returns: pd.DataFrame | None = None,
    include_vol_spike: bool = True,
    include_crash: bool = True,
    include_correlation: bool = True,
) -> StressTestReport:
    """Führt alle ausgewählten Szenarien durch.

    Args:
        baseline_returns: Portfolio-Returns historisch (für μ/σ Baseline)
        portfolio_returns: DataFrame mit Per-Asset-Returns (für Korrelations-Szenarien)
    """
    # Baseline Metriken
    base_var, base_cvar = _compute_var_cvar(baseline_returns.values)
    baseline = {
        "mean_return": round(float(baseline_returns.mean()), 6),
        "std_return": round(float(baseline_returns.std()), 4),
        "var_95": round(base_var, 4),
        "cvar_95": round(base_cvar, 4),
    }

    scenarios: list[ScenarioResult] = []
    if include_vol_spike:
        scenarios.append(simulate_vol_spike_scenario(baseline_returns))
    if include_crash:
        scenarios.append(simulate_crash_scenario(baseline_returns))
    if include_correlation and portfolio_returns is not None and portfolio_returns.shape[1] > 1:
        scenarios.append(simulate_correlation_breakdown_scenario(portfolio_returns))

    worst = min(scenarios, key=lambda s: s.cvar_95) if scenarios else None
    return StressTestReport(
        baseline_metrics=baseline,
        scenarios=scenarios,
        worst_scenario=worst.scenario_name if worst else "",
        worst_cvar=worst.cvar_95 if worst else 0.0,
    )


__all__ = [
    "ScenarioResult",
    "StressTestReport",
    "simulate_vol_spike_scenario",
    "simulate_crash_scenario",
    "simulate_correlation_breakdown_scenario",
    "run_stress_test",
]
