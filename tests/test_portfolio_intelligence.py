"""Tests for M27 Portfolio Intelligence modules.

Covers:
- Task 27.1: Risk Budgeting / ERC (risk_budgeting.py)
- Task 27.2: Robust Optimization (robust_optimizer.py)
- Task 27.3: Liquidity CVXPY term (cost_aware_optimizer.py extension)
- Task 27.4: Multi-Period Optimization (multi_period.py)
- Task 27.5: Turnover penalty in BL and HRP
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cov(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create a valid positive-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.02, (100, n))
    symbols = [f"SYM{i}" for i in range(n)]
    df = pd.DataFrame(A, columns=symbols)
    return df.cov()


def _make_returns(n_days: int = 200, n_assets: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_assets)]
    data = rng.normal(0.0005, 0.02, (n_days, n_assets))
    return pd.DataFrame(data, columns=symbols)


def _make_expected_returns(n: int = 5, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n)]
    return pd.Series(rng.normal(0.05, 0.03, n), index=symbols)


# ===========================================================================
# Task 27.1: Risk Budgeting / ERC
# ===========================================================================


# ===========================================================================
# Task 27.2: Robust Optimization
# ===========================================================================


# ===========================================================================
# Task 27.3: Liquidity CVXPY term
# ===========================================================================


@pytest.mark.fast
class TestLiquidityPenalty:
    def test_no_liquidity_penalty(self):
        from src.assembled_core.portfolio.cost_aware_optimizer import (
            optimize_portfolio,
            OptimizerConfig,
        )

        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        config = OptimizerConfig(liquidity_penalty=0.0)
        result = optimize_portfolio(mu, cov, config=config)
        assert len(result.weights) == 5

    def test_with_liquidity_penalty(self):
        from src.assembled_core.portfolio.cost_aware_optimizer import (
            optimize_portfolio,
            OptimizerConfig,
        )

        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        symbols = list(mu.index)
        adv = {s: 1e6 for s in symbols}
        adv[symbols[0]] = 1e3  # very illiquid
        config = OptimizerConfig(liquidity_penalty=1.0, max_weight=0.5)
        result = optimize_portfolio(mu, cov, config=config, adv_shares=adv)
        # Illiquid asset should have lower weight
        assert result.weights[symbols[0]] < 0.3

    def test_liquidity_alpha_param(self):
        from src.assembled_core.portfolio.cost_aware_optimizer import OptimizerConfig

        config = OptimizerConfig(liquidity_penalty=0.5, liquidity_alpha=0.7)
        assert config.liquidity_alpha == 0.7

    def test_default_adv(self):
        """With no ADV provided and penalty > 0, should still work (default high ADV)."""
        from src.assembled_core.portfolio.cost_aware_optimizer import (
            optimize_portfolio,
            OptimizerConfig,
        )

        mu = _make_expected_returns(3)
        cov = _make_cov(3)
        config = OptimizerConfig(liquidity_penalty=0.1, max_weight=0.5)
        result = optimize_portfolio(mu, cov, config=config)
        assert len(result.weights) == 3


# ===========================================================================
# Task 27.4: Multi-Period Optimization
# ===========================================================================


# ===========================================================================
# Task 27.5: Turnover penalty in BL and HRP
# ===========================================================================


@pytest.mark.fast
class TestBLTurnoverPenalty:
    def test_bl_no_turnover_penalty(self):
        """BL optimize without turnover penalty should work as before."""
        from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer

        cov = _make_cov(4)
        symbols = list(cov.columns)
        mkt_w = pd.Series([0.25] * 4, index=symbols)
        views = {symbols[0]: 0.08, symbols[1]: -0.02}
        bl = BlackLittermanOptimizer(max_position=0.5)
        weights = bl.optimize(mkt_w, cov, views)
        assert abs(weights.sum() - 1.0) < 0.01

    def test_bl_with_turnover_penalty(self):
        from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer

        cov = _make_cov(4)
        symbols = list(cov.columns)
        mkt_w = pd.Series([0.25] * 4, index=symbols)
        views = {symbols[0]: 0.08, symbols[1]: -0.02}
        curr = {s: 0.25 for s in symbols}
        bl = BlackLittermanOptimizer(max_position=0.5)
        w_no_tc = bl.optimize(mkt_w, cov, views, turnover_penalty=0.0)
        w_tc = bl.optimize(
            mkt_w, cov, views, current_weights=curr, turnover_penalty=0.5
        )
        # With turnover penalty, should be closer to current weights
        turnover_no_tc = float(np.sum(np.abs(w_no_tc.values - 0.25)))
        turnover_tc = float(np.sum(np.abs(w_tc.values - 0.25)))
        assert turnover_tc <= turnover_no_tc + 0.05

    def test_bl_high_penalty_stays_close(self):
        from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer

        cov = _make_cov(4)
        symbols = list(cov.columns)
        mkt_w = pd.Series([0.25] * 4, index=symbols)
        views = {symbols[0]: 0.08}
        curr = {s: 0.25 for s in symbols}
        bl = BlackLittermanOptimizer(max_position=0.5)
        w = bl.optimize(mkt_w, cov, views, current_weights=curr, turnover_penalty=10.0)
        # Very high penalty → should stay close to current
        turnover = float(np.sum(np.abs(w.values - 0.25)))
        assert turnover < 0.5


@pytest.mark.fast
class TestHRPTurnoverControl:
    def test_hrp_turnover_within_budget(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            hrp_with_turnover_control,
        )

        returns = _make_returns(200, 5)
        symbols = list(returns.columns)
        curr = {s: 0.2 for s in symbols}
        result = hrp_with_turnover_control(returns, curr, max_turnover=10.0)
        assert len(result) == 5
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_hrp_turnover_exceeded_partial(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            hrp_with_turnover_control,
        )

        returns = _make_returns(200, 5)
        symbols = list(returns.columns)
        # Current: all in one asset
        curr = {symbols[0]: 1.0}
        for s in symbols[1:]:
            curr[s] = 0.0
        result = hrp_with_turnover_control(
            returns, curr, max_turnover=0.3, blend_speed=0.5
        )
        # Should not fully rebalance — turnover limited
        w_arr = np.array([result.get(s, 0.0) for s in symbols])
        w_curr = np.array([curr.get(s, 0.0) for s in symbols])
        actual_turnover = float(np.sum(np.abs(w_arr - w_curr)))
        assert actual_turnover <= 0.35  # small tolerance

    def test_hrp_turnover_preserves_sum(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            hrp_with_turnover_control,
        )

        returns = _make_returns(200, 4)
        curr = {s: 0.25 for s in returns.columns}
        result = hrp_with_turnover_control(returns, curr, max_turnover=0.1)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_hrp_turnover_zero_speed(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            hrp_with_turnover_control,
        )

        returns = _make_returns(200, 3)
        curr = {s: 1.0 / 3 for s in returns.columns}
        result = hrp_with_turnover_control(
            returns, curr, blend_speed=0.0, max_turnover=0.01
        )
        # With zero speed and tight turnover, should stay at current
        for s in returns.columns:
            assert abs(result.get(s, 0) - 1.0 / 3) < 0.02
