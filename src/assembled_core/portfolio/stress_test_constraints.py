"""Portfolio-level stress testing constraints for optimizer (V19).

Adds stress-scenario constraints to CVXPY optimization:
  scenario_matrix @ w >= loss_floor_vector

Pre-built scenarios: COVID, rate shock, oil spike, China escalation.
Integrates with the existing scenario_engine.py for return vectors.

Reference: Meucci (2009) "Risk and Asset Allocation", Ch. 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)

# Pre-defined stress return vectors (per-symbol shocks as fraction)
# These represent worst-case sector-level returns under each scenario.
STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "covid_crash": {
        "default": -0.25,
        "Technology": -0.20,
        "Healthcare": -0.10,
        "Energy": -0.45,
        "Financials": -0.35,
        "Consumer Discretionary": -0.30,
        "Consumer Staples": -0.10,
        "Utilities": -0.15,
        "Real Estate": -0.25,
        "Industrials": -0.30,
        "Materials": -0.25,
        "Communication Services": -0.15,
    },
    "rate_shock": {
        "default": -0.10,
        "Technology": -0.18,
        "Healthcare": -0.08,
        "Energy": -0.05,
        "Financials": 0.05,
        "Consumer Discretionary": -0.12,
        "Consumer Staples": -0.05,
        "Utilities": -0.15,
        "Real Estate": -0.20,
        "Industrials": -0.08,
        "Materials": -0.06,
        "Communication Services": -0.12,
    },
    "oil_spike": {
        "default": -0.08,
        "Technology": -0.05,
        "Healthcare": -0.03,
        "Energy": 0.15,
        "Financials": -0.05,
        "Consumer Discretionary": -0.12,
        "Consumer Staples": -0.06,
        "Utilities": -0.10,
        "Real Estate": -0.05,
        "Industrials": -0.10,
        "Materials": 0.05,
        "Communication Services": -0.04,
    },
    "china_escalation": {
        "default": -0.15,
        "Technology": -0.25,
        "Healthcare": -0.08,
        "Energy": -0.10,
        "Financials": -0.12,
        "Consumer Discretionary": -0.20,
        "Consumer Staples": -0.05,
        "Utilities": -0.03,
        "Real Estate": -0.10,
        "Industrials": -0.18,
        "Materials": -0.15,
        "Communication Services": -0.12,
    },
}

# Default loss floors (maximum acceptable portfolio loss per scenario)
DEFAULT_LOSS_FLOORS: dict[str, float] = {
    "covid_crash": -0.10,
    "rate_shock": -0.05,
    "oil_spike": -0.08,
    "china_escalation": -0.08,
}


@dataclass
class StressTestConfig:
    """Configuration for stress-test constraints."""

    enabled: bool = True
    scenarios: list[str] = field(default_factory=lambda: list(STRESS_SCENARIOS.keys()))
    loss_floors: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_LOSS_FLOORS)
    )
    sector_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of portfolio stress testing."""

    scenario_losses: dict[str, float]  # Scenario -> portfolio loss
    worst_scenario: str
    worst_loss: float
    all_within_floors: bool
    violated_scenarios: list[str]


def build_scenario_return_matrix(
    symbols: list[str],
    sector_mapping: dict[str, str],
    scenarios: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build scenario return matrix (n_scenarios x n_symbols).

    Each row is a stress scenario, each column is a symbol.
    Values are the expected return under that scenario.

    Returns:
        (matrix, scenario_names)
    """
    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())

    n_scenarios = len(scenarios)
    n_symbols = len(symbols)
    matrix = np.zeros((n_scenarios, n_symbols))

    for i, sc_name in enumerate(scenarios):
        sc_returns = STRESS_SCENARIOS.get(sc_name, {})
        default_ret = sc_returns.get("default", -0.10)
        for j, sym in enumerate(symbols):
            sector = sector_mapping.get(sym, "")
            matrix[i, j] = sc_returns.get(sector, default_ret)

    return matrix, scenarios


def evaluate_stress_scenarios(
    weights: np.ndarray | dict[str, float],
    symbols: list[str],
    sector_mapping: dict[str, str],
    config: StressTestConfig | None = None,
) -> StressTestResult:
    """Evaluate portfolio weights against stress scenarios.

    Args:
        weights: Portfolio weights (array or dict).
        symbols: Symbol list.
        sector_mapping: Symbol -> sector.
        config: Stress test configuration.

    Returns:
        StressTestResult with per-scenario losses.
    """
    config = config or StressTestConfig()

    if isinstance(weights, dict):
        w = np.array([weights.get(s, 0.0) for s in symbols])
    else:
        w = np.asarray(weights)

    matrix, sc_names = build_scenario_return_matrix(
        symbols, sector_mapping, config.scenarios
    )

    # Portfolio loss per scenario: scenario_returns @ weights
    losses = {}
    violated = []
    for i, sc_name in enumerate(sc_names):
        loss = float(matrix[i] @ w)
        losses[sc_name] = round(loss, 6)
        floor = config.loss_floors.get(sc_name, -0.15)
        if loss < floor:
            violated.append(sc_name)

    worst_sc = min(losses, key=lambda k: losses[k]) if losses else "none"
    worst_loss = losses.get(worst_sc, 0.0)

    if violated:
        _log.warning(
            "STRESS TEST VIOLATIONS: %s (worst: %s = %.2f%%)",
            violated,
            worst_sc,
            worst_loss * 100,
        )

    return StressTestResult(
        scenario_losses=losses,
        worst_scenario=worst_sc,
        worst_loss=worst_loss,
        all_within_floors=len(violated) == 0,
        violated_scenarios=violated,
    )


def get_cvxpy_stress_constraints(
    w_var,  # cp.Variable
    symbols: list[str],
    sector_mapping: dict[str, str],
    config: StressTestConfig | None = None,
):
    """Generate CVXPY constraints for stress scenarios.

    Returns list of CVXPY constraints: scenario_matrix @ w >= loss_floor

    Args:
        w_var: CVXPY Variable for weights.
        symbols: Symbol list.
        sector_mapping: Symbol -> sector.
        config: Stress test configuration.

    Returns:
        List of CVXPY constraints.
    """
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError:
        _log.warning("CVXPY not available — no stress constraints")
        return []

    config = config or StressTestConfig()
    if not config.enabled:
        return []

    matrix, sc_names = build_scenario_return_matrix(
        symbols, sector_mapping, config.scenarios
    )

    constraints = []
    for i, sc_name in enumerate(sc_names):
        floor = config.loss_floors.get(sc_name, -0.15)
        # scenario_return_vector @ w >= floor
        constraints.append(matrix[i] @ w_var >= floor)

    _log.info("Added %d stress-test constraints to optimizer", len(constraints))
    return constraints


__all__ = [
    "STRESS_SCENARIOS",
    "DEFAULT_LOSS_FLOORS",
    "StressTestConfig",
    "StressTestResult",
    "build_scenario_return_matrix",
    "evaluate_stress_scenarios",
    "get_cvxpy_stress_constraints",
]
