"""Tests for M17 Wave 1: execution, risk, QA, and accounting items."""

import numpy as np
import pandas as pd
import pytest


# ── Circuit Breaker (6.1) ─────────────────────────────────────────────

class TestCircuitBreaker:

    def test_import(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        assert check_circuit_breaker is not None

    def test_no_halt_normal_day(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        halted, reason = check_circuit_breaker(market_return_today=-0.02)
        assert not halted
        assert reason == ""

    def test_level1_halt(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        halted, reason = check_circuit_breaker(market_return_today=-0.08)
        assert halted
        assert "L1" in reason

    def test_level2_halt(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        halted, reason = check_circuit_breaker(market_return_today=-0.14)
        assert halted
        assert "L2" in reason

    def test_level3_halt(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        halted, reason = check_circuit_breaker(market_return_today=-0.21)
        assert halted
        assert "L3" in reason

    def test_luld_halt(self):
        from src.assembled_core.execution.fill_model import check_circuit_breaker
        halted, reason = check_circuit_breaker(
            market_return_today=-0.01, symbol_5min_return=0.07,
        )
        assert halted
        assert "LULD" in reason


# ── Adversarial Fill (6.3) ────────────────────────────────────────────

class TestAdversarialFill:

    def test_import(self):
        from src.assembled_core.execution.fill_model import (
            compute_adversarial_fill_cost,
            apply_adversarial_fill_adjustment,
        )
        assert compute_adversarial_fill_cost is not None

    def test_zero_signal_no_cost(self):
        from src.assembled_core.execution.fill_model import compute_adversarial_fill_cost
        cost = compute_adversarial_fill_cost(
            order_size=10000, signal_strength=0.0, adv=1_000_000,
        )
        assert cost == 0.0

    def test_high_signal_higher_cost(self):
        from src.assembled_core.execution.fill_model import compute_adversarial_fill_cost
        cost_low = compute_adversarial_fill_cost(
            order_size=10000, signal_strength=0.2, adv=1_000_000,
        )
        cost_high = compute_adversarial_fill_cost(
            order_size=10000, signal_strength=0.8, adv=1_000_000,
        )
        assert cost_high > cost_low

    def test_fill_adjustment_buy(self):
        from src.assembled_core.execution.fill_model import apply_adversarial_fill_adjustment
        adjusted = apply_adversarial_fill_adjustment(100.0, "BUY", 10.0)
        assert adjusted > 100.0  # buy fills higher (worse)

    def test_fill_adjustment_sell(self):
        from src.assembled_core.execution.fill_model import apply_adversarial_fill_adjustment
        adjusted = apply_adversarial_fill_adjustment(100.0, "SELL", 10.0)
        assert adjusted < 100.0  # sell fills lower (worse)


# ── Monte Carlo VaR (7.1) – already tested in test_m17_wave1_features ─

# ── Brinson-Fachler (7.5) – already tested in test_m17_wave1_features ─


# ── Benchmark Metrics (9.2) ───────────────────────────────────────────

class TestBenchmarkMetrics:

    def test_import(self):
        from src.assembled_core.qa.metrics import compute_benchmark_relative_metrics
        assert compute_benchmark_relative_metrics is not None

    def test_basic_metrics(self):
        from src.assembled_core.qa.metrics import compute_benchmark_relative_metrics

        np.random.seed(42)
        n = 500
        bench = pd.Series(np.random.normal(0.0004, 0.01, n))
        # Portfolio outperforms
        port = bench + pd.Series(np.random.normal(0.0001, 0.002, n))

        result = compute_benchmark_relative_metrics(port, bench)
        assert result["information_ratio"] is not None
        assert result["active_return"] > 0  # portfolio outperforms
        assert result["tracking_error"] > 0
        assert result["beta"] is not None

    def test_up_down_capture(self):
        from src.assembled_core.qa.metrics import compute_benchmark_relative_metrics

        np.random.seed(42)
        n = 500
        bench = pd.Series(np.random.normal(0, 0.01, n))
        # Defensive portfolio: captures less upside but also less downside
        port = bench * 0.5

        result = compute_benchmark_relative_metrics(port, bench)
        assert result["up_capture"] is not None
        # Both captures should be ~0.5 for half-beta portfolio
        if result["up_capture"] is not None:
            assert 0.3 < result["up_capture"] < 0.7

    def test_insufficient_data(self):
        from src.assembled_core.qa.metrics import compute_benchmark_relative_metrics

        result = compute_benchmark_relative_metrics(
            pd.Series([0.01, -0.01]),
            pd.Series([0.005, -0.005]),
        )
        assert result["information_ratio"] is None


# ── Permutation Test (9.4) ────────────────────────────────────────────

class TestPermutationTest:

    def test_import(self):
        from src.assembled_core.qa.metrics import permutation_test_sharpe
        assert permutation_test_sharpe is not None

    def test_basic_output(self):
        from src.assembled_core.qa.metrics import permutation_test_sharpe

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 500))

        result = permutation_test_sharpe(returns)
        assert "observed_sharpe" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert "mean_permuted_sharpe" in result
        assert "sharpe_percentile" in result

    def test_noise_strategy(self):
        from src.assembled_core.qa.metrics import permutation_test_sharpe

        np.random.seed(42)
        # Pure noise → should NOT be significant
        returns = pd.Series(np.random.normal(0, 0.01, 500))

        result = permutation_test_sharpe(returns)
        assert result["p_value"] > 0.05  # not significant

    def test_deterministic(self):
        from src.assembled_core.qa.metrics import permutation_test_sharpe

        returns = pd.Series(np.random.normal(0.001, 0.01, 200))
        r1 = permutation_test_sharpe(returns, seed=42)
        r2 = permutation_test_sharpe(returns, seed=42)
        assert r1["p_value"] == r2["p_value"]


# ── Daily P&L Reconciliation (8.1) ────────────────────────────────────

class TestDailyPnLReconciliation:

    def test_import(self):
        from src.assembled_core.accounting.reconciliation import reconcile_daily_pnl
        assert reconcile_daily_pnl is not None

    def test_matching_pnl(self):
        from src.assembled_core.accounting.reconciliation import reconcile_daily_pnl

        positions = {"AAPL": 0.5, "MSFT": 0.5}
        prices_start = {"AAPL": 100.0, "MSFT": 200.0}
        prices_end = {"AAPL": 102.0, "MSFT": 198.0}

        # Expected: 0.5 * 2% + 0.5 * (-1%) = 0.5%
        portfolio_return = 0.005

        result = reconcile_daily_pnl(
            positions, prices_start, prices_end, portfolio_return,
        )
        assert result["ok"]
        assert abs(result["unexplained_return"]) < 0.001

    def test_pnl_break(self):
        from src.assembled_core.accounting.reconciliation import reconcile_daily_pnl

        positions = {"AAPL": 0.5, "MSFT": 0.5}
        prices_start = {"AAPL": 100.0, "MSFT": 200.0}
        prices_end = {"AAPL": 102.0, "MSFT": 198.0}

        # Reported return doesn't match
        portfolio_return = 0.02  # 2% but should be 0.5%

        result = reconcile_daily_pnl(
            positions, prices_start, prices_end, portfolio_return,
        )
        assert not result["ok"]
        assert result["break_reason"] != ""

    def test_position_contributions(self):
        from src.assembled_core.accounting.reconciliation import reconcile_daily_pnl

        positions = {"A": 0.6, "B": 0.4}
        prices_start = {"A": 50.0, "B": 100.0}
        prices_end = {"A": 55.0, "B": 95.0}

        result = reconcile_daily_pnl(
            positions, prices_start, prices_end, 0.04,
        )
        assert "A" in result["position_contributions"]
        assert result["position_contributions"]["A"] > 0  # A went up
        assert result["position_contributions"]["B"] < 0  # B went down
