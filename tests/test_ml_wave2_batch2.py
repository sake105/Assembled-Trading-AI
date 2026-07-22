"""Tests for M17 Wave 2 — remaining STANDARD items."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestDiffusionIndex:
    def test_basic(self):
        from src.assembled_core.features.macro_features import compute_diffusion_index

        values = {
            "GDP": pd.Series([100, 101, 102, 103, 104]),
            "CPI": pd.Series([200, 201, 200, 199, 198]),
        }
        result = compute_diffusion_index(values)
        assert len(result) == 5


class TestACLED:
    def test_parse(self):
        from src.assembled_core.events.news.fetch_acled import parse_acled_events

        df = pd.DataFrame(
            {
                "event_date": ["2024-01-01", "2024-01-02"],
                "event_type": ["Battles", "Protests"],
                "country": ["Ukraine", "France"],
                "fatalities": [10, 0],
                "notes": ["clash", "protest"],
            }
        )
        events = parse_acled_events(df)
        assert len(events) == 2
        assert events[0].trigger_type == "MILITARY_BUILDUP"
        assert events[1].trigger_type == "REGIME_CHANGE_RISK"

    def test_aggregate(self):
        from src.assembled_core.events.news.fetch_acled import (
            parse_acled_events,
            aggregate_acled_by_country,
        )

        df = pd.DataFrame(
            {
                "event_date": ["2024-01-01", "2024-01-02"],
                "event_type": ["Battles", "Battles"],
                "country": ["Ukraine", "Ukraine"],
                "fatalities": [5, 3],
                "notes": ["a", "b"],
            }
        )
        events = parse_acled_events(df)
        agg = aggregate_acled_by_country(events)
        assert "Ukraine" in agg
        assert agg["Ukraine"]["total_fatalities"] == 8


class TestFeedbackLoopTracker:
    def test_partial_activation(self):
        from src.assembled_core.intel.feedback_loops import (
            FeedbackLoop,
            track_loop_activation,
        )

        loop = FeedbackLoop(
            loop_id="test",
            name="Test",
            chain=["a", "b", "c", "d", "e"],
        )
        history = [["a", "b"], ["a", "c"]]
        result = track_loop_activation(loop, history)
        assert result["activated_elements"] == 3
        assert result["activation_score"] == 0.6
        assert result["alert"] is False  # exactly 0.6, not > 0.6


class TestMaxDiversification:
    def test_weights_sum_to_one(self):
        from src.assembled_core.portfolio.position_sizing import (
            compute_max_diversification_weights,
        )

        np.random.seed(42)
        n = 5
        cov = np.eye(n) * 0.04
        w = compute_max_diversification_weights(cov)
        assert abs(sum(w) - 1.0) < 1e-6
        assert all(wi >= 0 for wi in w)


class TestRobustBL:
    def test_shrinkage(self):
        from src.assembled_core.portfolio.black_litterman import robust_bl_shrinkage

        mu = np.array([0.10, 0.05, -0.02])
        sigma = np.eye(3) * 0.04
        mu_robust = robust_bl_shrinkage(mu, sigma, n_obs=252)
        # Shrunk toward zero
        assert np.linalg.norm(mu_robust) <= np.linalg.norm(mu)


class TestTailRiskParity:
    def test_basic_v2(self):
        from src.assembled_core.portfolio.position_sizing import (
            compute_tail_risk_parity_weights,
        )

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 200),
                "B": np.random.normal(0, 0.02, 200),
            }
        )
        w = compute_tail_risk_parity_weights(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # Higher vol asset should get less weight
        assert w["A"] > w["B"]


class TestTCAFeedback:
    def test_flagging(self):
        from src.assembled_core.execution.transaction_costs import compute_tca_feedback

        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 30 + ["MSFT"] * 30,
                "slippage_bps": [20.0] * 30 + [3.0] * 30,
                "date": list(range(30)) * 2,
            }
        )
        result = compute_tca_feedback(df, model_slippage_bps=5.0)
        assert result["AAPL"]["high_slippage_flag"] is True
        assert result["MSFT"]["high_slippage_flag"] is False


class TestCashDrag:
    def test_basic_v3(self):
        from src.assembled_core.accounting.ledger import compute_cash_drag

        result = compute_cash_drag(100_000, 1_000_000)
        assert result["cash_pct"] == 0.1
        assert result["daily_interest"] > 0


class TestInterestAccrual:
    def test_basic_v4(self):
        from src.assembled_core.accounting.ledger import compute_daily_interest_accrual

        result = compute_daily_interest_accrual(
            {"AAPL": -100_000},
            margin_balance=50_000,
        )
        assert result["borrow_fees"] > 0
        assert result["margin_interest"] > 0


class TestCorporateActions:
    def test_split(self):
        from src.assembled_core.accounting.position_engine import adjust_for_stock_split

        positions = {"AAPL": {"qty": 100, "cost_basis": 15000, "realized_pnl": 0}}
        result = adjust_for_stock_split(positions, "AAPL", 4.0)
        assert result["AAPL"]["qty"] == 400

    def test_spinoff(self):
        from src.assembled_core.accounting.position_engine import adjust_for_spinoff

        positions = {"PARENT": {"qty": 100, "cost_basis": 10000, "realized_pnl": 0}}
        result = adjust_for_spinoff(positions, "PARENT", "CHILD", 0.85, 0.1)
        assert "CHILD" in result
        assert abs(result["PARENT"]["cost_basis"] - 8500) < 0.01
        assert abs(result["CHILD"]["cost_basis"] - 1500) < 0.01


class TestFXConverter:
    def test_conversion(self):
        from src.assembled_core.accounting.currency import FXConverter

        fx = FXConverter()
        usd = fx.to_usd(1000, "EUR")
        assert usd > 1000  # EUR > USD


class TestBacktestRealism:
    def test_score(self):
        from src.assembled_core.qa.validation import compute_backtest_realism_score

        result = compute_backtest_realism_score(
            has_transaction_costs=True,
            has_slippage=True,
            has_pit_features=True,
        )
        assert result["score"] == 45
        assert result["grade"] == "C"


class TestCorrelatedStress:
    def test_basic_v5(self):
        from src.assembled_core.qa.scenario_engine import run_correlated_stress_test

        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 200),
                "B": np.random.normal(0, 0.01, 200),
            }
        )
        result = run_correlated_stress_test({"A": 0.5, "B": 0.5}, returns)
        assert result["var_95"] < 0
        assert result["cvar_95"] <= result["var_95"]


class TestSyntheticGenerator:
    def test_crisis(self):
        from src.assembled_core.data.synthetic_generator import generate_crisis_returns

        result = generate_crisis_returns("2008_gfc", n_assets=3)
        assert result.shape == (252, 3)
        assert result.mean().mean() < 0  # should be negative

    def test_normal(self):
        from src.assembled_core.data.synthetic_generator import generate_normal_returns

        result = generate_normal_returns(n_days=100, n_assets=5)
        assert result.shape == (100, 5)


class TestDelistedDetection:
    def test_stale_symbol(self):
        from src.assembled_core.data.universe import detect_delisted_symbols

        prices = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, tz="UTC"),
                "symbol": ["A"] * 100,
                "close": range(100, 200),
            }
        )
        result = detect_delisted_symbols(prices, "2024-01-01")
        assert "A" in result["delisted"]


class TestJSONLogging:
    def test_formatter(self):
        from src.assembled_core.logging_config import JSONFormatter
        import json
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "test message", (), None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["event"] == "test message"
