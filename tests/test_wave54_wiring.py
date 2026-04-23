"""Tests for wave-54 module wiring into trading_cycle.py.

Covers:
  Step 8.55 — ml.rl_portfolio (RLPortfolioConfig / GYM_AVAILABLE / SB3_AVAILABLE)
  Step 5.96 — ml.rl_execution (QLearningExecutionAgent / N_ACTIONS)
  Step 8.56 — ml.symbolic_regression (discover_formulas / SymbolicSearchResult)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.rl_portfolio import (
    RLPortfolioConfig,
    GYM_AVAILABLE,
    SB3_AVAILABLE,
)
from src.assembled_core.ml.rl_execution import (
    QLearningExecutionAgent,
    ExecutionState,
    N_ACTIONS,
)
from src.assembled_core.ml.symbolic_regression import (
    SymbolicSearchResult,
    DiscoveredFormula,
    discover_formulas,
    GPLEARN_AVAILABLE,
)


# ---------------------------------------------------------------------------
# rl_portfolio (Step 8.55)
# ---------------------------------------------------------------------------

def test_rl_portfolio_config_creates():
    cfg = RLPortfolioConfig()
    assert isinstance(cfg, RLPortfolioConfig)


def test_rl_portfolio_config_max_position():
    cfg = RLPortfolioConfig()
    assert 0 < cfg.max_position <= 1.0


def test_rl_portfolio_config_risk_aversion():
    cfg = RLPortfolioConfig()
    assert cfg.risk_aversion > 0


def test_gym_available_flag():
    assert isinstance(GYM_AVAILABLE, bool)


def test_sb3_available_flag():
    assert isinstance(SB3_AVAILABLE, bool)


def test_rl_portfolio_config_custom():
    cfg = RLPortfolioConfig(max_position=0.2, risk_aversion=2.0)
    assert cfg.max_position == 0.2
    assert cfg.risk_aversion == 2.0


# ---------------------------------------------------------------------------
# rl_execution (Step 5.96)
# ---------------------------------------------------------------------------

def test_q_agent_creates():
    agent = QLearningExecutionAgent()
    assert isinstance(agent, QLearningExecutionAgent)


def test_q_agent_defaults():
    agent = QLearningExecutionAgent()
    assert 0 < agent.epsilon <= 1.0
    assert 0 < agent.alpha <= 1.0


def test_n_actions_positive_int():
    assert isinstance(N_ACTIONS, int)
    assert N_ACTIONS > 0


def test_q_agent_select_action():
    agent = QLearningExecutionAgent()
    state = ExecutionState(
        remaining_qty=0.5,
        time_remaining=0.5,
        spread_bps=5.0,
        volume_ratio=1.0,
        volatility_ratio=1.0,
        momentum=0.0,
        inventory_risk=0.1,
    )
    action = agent.select_action(state, training=False)
    assert action is not None


def test_execution_state_creates():
    state = ExecutionState(
        remaining_qty=1.0,
        time_remaining=1.0,
        spread_bps=3.0,
        volume_ratio=0.8,
        volatility_ratio=1.2,
        momentum=0.01,
        inventory_risk=0.05,
    )
    assert state.remaining_qty == 1.0


# ---------------------------------------------------------------------------
# symbolic_regression (Step 8.56)
# ---------------------------------------------------------------------------

def test_gplearn_available_flag():
    assert isinstance(GPLEARN_AVAILABLE, bool)


def _make_factor_returns(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    features = pd.DataFrame(
        rng.normal(0, 1, (n, 3)),
        index=idx,
        columns=["momentum", "value", "quality"],
    )
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
    return features, returns


def test_discover_formulas_returns_result():
    features, returns = _make_factor_returns()
    result = discover_formulas(features, returns)
    assert isinstance(result, SymbolicSearchResult)


def test_discover_formulas_has_formulas():
    features, returns = _make_factor_returns()
    result = discover_formulas(features, returns)
    assert isinstance(result.formulas, list)
    assert result.n_evaluated >= 0


def test_discovered_formula_fields():
    features, returns = _make_factor_returns()
    result = discover_formulas(features, returns)
    if result.best_formula is not None:
        f = result.best_formula
        assert isinstance(f, DiscoveredFormula)
        assert isinstance(f.expression, str)
        assert isinstance(f.oos_ic, float)
