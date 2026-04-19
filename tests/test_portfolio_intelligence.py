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

@pytest.mark.phase12
class TestRiskBudgeting:
    def test_erc_basic(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights, RiskBudgetResult
        cov = _make_cov(5)
        result = compute_erc_weights(cov)
        assert isinstance(result, RiskBudgetResult)
        assert len(result.weights) == 5
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.portfolio_volatility > 0

    def test_erc_equal_risk_contributions(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
        cov = _make_cov(4)
        result = compute_erc_weights(cov)
        rc_vals = list(result.risk_contributions.values())  # noqa: F841
        # All risk contributions should be approximately equal
        assert result.max_rc_deviation < 0.05

    def test_erc_custom_budget(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
        cov = _make_cov(3)
        symbols = list(cov.columns)
        budget = {symbols[0]: 0.5, symbols[1]: 0.3, symbols[2]: 0.2}
        result = compute_erc_weights(cov, risk_budget=budget)
        assert len(result.weights) == 3
        assert result.converged

    def test_erc_ndarray_input(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
        cov_arr = _make_cov(3).values
        result = compute_erc_weights(cov_arr)
        assert len(result.weights) == 3

    def test_risk_parity_with_views(self):
        from src.assembled_core.portfolio.risk_budgeting import risk_parity_with_views
        cov = _make_cov(4)
        symbols = list(cov.columns)
        views = {symbols[0]: 0.9, symbols[1]: 0.3}
        result = risk_parity_with_views(cov, views_confidence=views, confidence_scale=0.5)
        assert len(result.weights) == 4
        # Higher confidence asset should have higher risk budget
        assert result.risk_contributions[symbols[0]] > result.risk_contributions[symbols[1]]

    def test_erc_max_weight(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
        cov = _make_cov(3)
        result = compute_erc_weights(cov, max_weight=0.5)
        for w in result.weights.values():
            assert w <= 0.51  # small tolerance

    def test_erc_method_reported(self):
        from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
        cov = _make_cov(3)
        result = compute_erc_weights(cov)
        assert result.method in ("scipy_slsqp", "inverse_vol_fallback")


# ===========================================================================
# Task 27.2: Robust Optimization
# ===========================================================================

@pytest.mark.phase12
class TestRobustOptimizer:
    def test_basic(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights, RobustOptResult
        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        result = compute_robust_weights(mu, cov, symbols=list(mu.index))
        assert isinstance(result, RobustOptResult)
        assert len(result.weights) == 5
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_worst_case_lower(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        result = compute_robust_weights(mu, cov)
        # Worst-case return should be <= expected return
        assert result.worst_case_return <= result.expected_return + 1e-6

    def test_higher_epsilon_more_conservative(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        r_low = compute_robust_weights(mu, cov, epsilon=0.5)
        r_high = compute_robust_weights(mu, cov, epsilon=5.0)
        # Higher epsilon → more diversified → lower concentration
        w_low = np.array(list(r_low.weights.values()))
        w_high = np.array(list(r_high.weights.values()))
        hhi_low = float(np.sum(w_low ** 2))
        hhi_high = float(np.sum(w_high ** 2))
        # More conservative should be more diversified (lower HHI) or similar
        assert hhi_high <= hhi_low + 0.05

    def test_ndarray_input(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
        mu = np.array([0.05, 0.03, 0.07])
        cov = _make_cov(3).values
        result = compute_robust_weights(mu, cov)
        assert len(result.weights) == 3

    def test_converged(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
        mu = _make_expected_returns(4)
        cov = _make_cov(4)
        result = compute_robust_weights(mu, cov)
        assert result.converged

    def test_epsilon_in_result(self):
        from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
        mu = _make_expected_returns(3)
        cov = _make_cov(3)
        result = compute_robust_weights(mu, cov, epsilon=2.0)
        assert result.epsilon == 2.0


# ===========================================================================
# Task 27.3: Liquidity CVXPY term
# ===========================================================================

@pytest.mark.phase12
class TestLiquidityPenalty:
    def test_no_liquidity_penalty(self):
        from src.assembled_core.portfolio.cost_aware_optimizer import (
            optimize_portfolio, OptimizerConfig,
        )
        mu = _make_expected_returns(5)
        cov = _make_cov(5)
        config = OptimizerConfig(liquidity_penalty=0.0)
        result = optimize_portfolio(mu, cov, config=config)
        assert len(result.weights) == 5

    def test_with_liquidity_penalty(self):
        from src.assembled_core.portfolio.cost_aware_optimizer import (
            optimize_portfolio, OptimizerConfig,
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
            optimize_portfolio, OptimizerConfig,
        )
        mu = _make_expected_returns(3)
        cov = _make_cov(3)
        config = OptimizerConfig(liquidity_penalty=0.1, max_weight=0.5)
        result = optimize_portfolio(mu, cov, config=config)
        assert len(result.weights) == 3


# ===========================================================================
# Task 27.4: Multi-Period Optimization
# ===========================================================================

@pytest.mark.phase12
class TestMultiPeriod:
    def test_trade_speed_basic(self):
        from src.assembled_core.portfolio.multi_period import compute_trade_speed
        speed = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.001)
        assert 0.0 < speed <= 1.0

    def test_trade_speed_zero_cost(self):
        from src.assembled_core.portfolio.multi_period import compute_trade_speed
        speed = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.0)
        assert speed == 1.0

    def test_trade_speed_high_cost_slower(self):
        from src.assembled_core.portfolio.multi_period import compute_trade_speed
        s_low = compute_trade_speed(risk_aversion=0.01, transaction_cost=0.001)
        s_high = compute_trade_speed(risk_aversion=0.01, transaction_cost=1.0)
        assert s_high < s_low

    def test_garleanu_pedersen_basic(self):
        from src.assembled_core.portfolio.multi_period import garleanu_pedersen_target, MultiPeriodResult
        aim = {"A": 0.4, "B": 0.3, "C": 0.3}
        curr = {"A": 0.33, "B": 0.33, "C": 0.34}
        result = garleanu_pedersen_target(aim, curr)
        assert isinstance(result, MultiPeriodResult)
        assert abs(sum(result.target_weights.values()) - 1.0) < 0.01
        assert result.method == "garleanu_pedersen"

    def test_garleanu_pedersen_partial_adjustment(self):
        from src.assembled_core.portfolio.multi_period import garleanu_pedersen_target
        aim = {"A": 1.0, "B": 0.0}
        curr = {"A": 0.5, "B": 0.5}
        result = garleanu_pedersen_target(aim, curr, trade_speed=0.5)
        # Should move halfway
        assert abs(result.target_weights["A"] - 0.75) < 0.02
        assert abs(result.target_weights["B"] - 0.25) < 0.02

    def test_garleanu_pedersen_full_speed(self):
        from src.assembled_core.portfolio.multi_period import garleanu_pedersen_target
        aim = {"A": 0.6, "B": 0.4}
        curr = {"A": 0.3, "B": 0.7}
        result = garleanu_pedersen_target(aim, curr, trade_speed=1.0)
        assert abs(result.target_weights["A"] - 0.6) < 0.01
        assert abs(result.target_weights["B"] - 0.4) < 0.01

    def test_multi_period_optimize_basic(self):
        from src.assembled_core.portfolio.multi_period import multi_period_optimize
        cov = _make_cov(4)
        symbols = list(cov.columns)
        mu_path = [
            pd.Series(np.random.default_rng(42 + i).normal(0.05, 0.02, 4), index=symbols)
            for i in range(3)
        ]
        curr = {s: 0.25 for s in symbols}
        result = multi_period_optimize(mu_path, cov, curr)
        assert len(result.target_weights) == 4
        assert result.periods_ahead == 3

    def test_multi_period_empty_path(self):
        from src.assembled_core.portfolio.multi_period import multi_period_optimize
        cov = _make_cov(3)
        curr = {s: 1.0 / 3 for s in cov.columns}
        result = multi_period_optimize([], cov, curr)
        assert result.periods_ahead == 0

    def test_multi_period_reduces_turnover(self):
        from src.assembled_core.portfolio.multi_period import garleanu_pedersen_target
        aim = {"A": 0.8, "B": 0.2}
        curr = {"A": 0.2, "B": 0.8}
        # With cost, should trade slower
        r_fast = garleanu_pedersen_target(aim, curr, trade_speed=1.0)
        r_slow = garleanu_pedersen_target(aim, curr, transaction_cost=0.01)
        assert r_slow.expected_turnover <= r_fast.expected_turnover + 0.01


# ===========================================================================
# Task 27.5: Turnover penalty in BL and HRP
# ===========================================================================

@pytest.mark.phase12
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
        w_tc = bl.optimize(mkt_w, cov, views, current_weights=curr, turnover_penalty=0.5)
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


@pytest.mark.phase12
class TestHRPTurnoverControl:
    def test_hrp_turnover_within_budget(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import hrp_with_turnover_control
        returns = _make_returns(200, 5)
        symbols = list(returns.columns)
        curr = {s: 0.2 for s in symbols}
        result = hrp_with_turnover_control(returns, curr, max_turnover=10.0)
        assert len(result) == 5
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_hrp_turnover_exceeded_partial(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import hrp_with_turnover_control
        returns = _make_returns(200, 5)
        symbols = list(returns.columns)
        # Current: all in one asset
        curr = {symbols[0]: 1.0}
        for s in symbols[1:]:
            curr[s] = 0.0
        result = hrp_with_turnover_control(returns, curr, max_turnover=0.3, blend_speed=0.5)
        # Should not fully rebalance — turnover limited
        w_arr = np.array([result.get(s, 0.0) for s in symbols])
        w_curr = np.array([curr.get(s, 0.0) for s in symbols])
        actual_turnover = float(np.sum(np.abs(w_arr - w_curr)))
        assert actual_turnover <= 0.35  # small tolerance

    def test_hrp_turnover_preserves_sum(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import hrp_with_turnover_control
        returns = _make_returns(200, 4)
        curr = {s: 0.25 for s in returns.columns}
        result = hrp_with_turnover_control(returns, curr, max_turnover=0.1)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_hrp_turnover_zero_speed(self):
        from src.assembled_core.portfolio.hierarchical_risk_parity import hrp_with_turnover_control
        returns = _make_returns(200, 3)
        curr = {s: 1.0 / 3 for s in returns.columns}
        result = hrp_with_turnover_control(returns, curr, blend_speed=0.0, max_turnover=0.01)
        # With zero speed and tight turnover, should stay at current
        for s in returns.columns:
            assert abs(result.get(s, 0) - 1.0 / 3) < 0.02
