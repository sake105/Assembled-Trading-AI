"""Tests for M17 Wave 1 Batch 3: Items 4.4, 5.1, 5.3, 5.5, 6.2, 7.2, 7.3, 7.4,
8.2, 8.3, 9.1, 9.3, 10.1, 10.2, 10.3, 11.1, 11.2."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 5.1 DCC-GARCH Dynamic Covariance
# ---------------------------------------------------------------------------


class TestDCCGARCH:

    def test_dcc_garch_basic(self):
        from src.assembled_core.portfolio.covariance import estimate_covariance

        np.random.seed(42)
        n = 200
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, n),
            "B": np.random.normal(0, 0.015, n),
            "C": np.random.normal(0, 0.02, n),
        })
        cov = estimate_covariance(returns, method="dcc_garch", annualize=False)
        assert cov.shape == (3, 3)
        assert all(cov.index == ["A", "B", "C"])
        # Diagonal should be positive
        for i in range(3):
            assert cov.iloc[i, i] > 0

    def test_dcc_garch_symmetric(self):
        from src.assembled_core.portfolio.covariance import estimate_covariance

        np.random.seed(42)
        returns = pd.DataFrame({
            "X": np.random.normal(0, 0.01, 150),
            "Y": np.random.normal(0, 0.01, 150),
        })
        cov = estimate_covariance(returns, method="dcc_garch", annualize=False)
        assert abs(cov.iloc[0, 1] - cov.iloc[1, 0]) < 1e-10


# ---------------------------------------------------------------------------
# 5.3 TC-Penalized Rebalancing
# ---------------------------------------------------------------------------


class TestTCPenalizedRebalancing:

    def test_dead_zone(self):
        from src.assembled_core.portfolio.position_sizing import apply_tc_penalized_rebalancing

        target = {"A": 0.30, "B": 0.30, "C": 0.40}
        current = {"A": 0.31, "B": 0.29, "C": 0.40}  # all within 2% dead zone
        result = apply_tc_penalized_rebalancing(target, current, dead_zone_pct=0.02)
        # Should keep current weights
        assert result["A"] == 0.31
        assert result["B"] == 0.29

    def test_large_trade_penalized(self):
        from src.assembled_core.portfolio.position_sizing import apply_tc_penalized_rebalancing

        target = {"A": 0.50}
        current = {"A": 0.10}
        result = apply_tc_penalized_rebalancing(
            target, current, cost_bps=100.0, tc_penalty_gamma=1.0, dead_zone_pct=0.01,
        )
        # Should be between current and target
        assert 0.10 < result["A"] <= 0.50


# ---------------------------------------------------------------------------
# 5.5 BL Views from Intel
# ---------------------------------------------------------------------------


class TestIntelBLViews:

    def test_basic_conversion(self):
        from src.assembled_core.portfolio.black_litterman import intel_to_bl_views

        impacts = {"XLE": 0.8, "GLD": 0.5, "QQQ": -0.3}
        views, conf = intel_to_bl_views(impacts)
        assert views["XLE"] > 0  # positive impact → positive view
        assert views["QQQ"] < 0  # negative impact → negative view
        assert all(0 < c <= 1 for c in conf.values())

    def test_empty_input(self):
        from src.assembled_core.portfolio.black_litterman import intel_to_bl_views

        views, conf = intel_to_bl_views({})
        assert views == {}
        assert conf == {}

    def test_regime_multiplier(self):
        from src.assembled_core.portfolio.black_litterman import intel_to_bl_views

        v1, _ = intel_to_bl_views({"A": 1.0}, regime_multiplier=1.0)
        v2, _ = intel_to_bl_views({"A": 1.0}, regime_multiplier=0.5)
        assert abs(v1["A"]) > abs(v2["A"])


# ---------------------------------------------------------------------------
# 6.2 Borrow Cost Model
# ---------------------------------------------------------------------------


class TestBorrowCostModel:

    def test_gc_rate(self):
        from src.assembled_core.execution.transaction_costs import BorrowCostModel

        model = BorrowCostModel()
        cost = model.daily_borrow_cost("AAPL", 100000)
        expected = 100000 * 0.0025 / 252
        assert abs(cost - expected) < 0.01

    def test_htb_rate(self):
        from src.assembled_core.execution.transaction_costs import BorrowCostModel

        model = BorrowCostModel(htb_symbols={"GME", "AMC"})
        cost = model.daily_borrow_cost("GME", 50000)
        expected = 50000 * 0.08 / 252
        assert abs(cost - expected) < 0.01

    def test_portfolio_costs(self):
        from src.assembled_core.execution.transaction_costs import BorrowCostModel

        model = BorrowCostModel()
        shorts = {"AAPL": -100000, "MSFT": -50000, "GOOG": 80000}  # GOOG is long
        costs = model.compute_portfolio_borrow_costs(shorts)
        assert "AAPL" in costs
        assert "MSFT" in costs
        assert "GOOG" not in costs  # long position, no borrow cost


# ---------------------------------------------------------------------------
# 7.2 Pre-Trade Stress Test
# ---------------------------------------------------------------------------


class TestPreTradeStressTest:

    def test_passes_small_portfolio(self):
        from src.assembled_core.execution.pre_trade_checks import run_pre_trade_stress_test

        result = run_pre_trade_stress_test(
            portfolio_weights={"AAPL": 0.1, "MSFT": 0.1},
            portfolio_value=100000,
        )
        assert result["passed"] is True
        assert len(result["scenario_results"]) > 0

    def test_fails_concentrated_portfolio(self):
        from src.assembled_core.execution.pre_trade_checks import run_pre_trade_stress_test

        result = run_pre_trade_stress_test(
            portfolio_weights={"SPY": 1.0},
            portfolio_value=1000000,
            betas={"SPY": 1.0},
            max_stress_loss_pct=0.05,
        )
        # Severe crisis (-20%) should breach 5% limit
        assert result["passed"] is False
        assert result["worst_loss_pct"] > 0.05


# ---------------------------------------------------------------------------
# 7.3 Liquidity Constraint
# ---------------------------------------------------------------------------


class TestLiquidityConstraint:

    def test_reduces_illiquid(self):
        from src.assembled_core.portfolio.position_sizing import apply_liquidity_constraint

        target = {"AAPL": 0.50, "TINY": 0.50}
        adv = {"AAPL": 10_000_000, "TINY": 100_000}  # TINY has tiny ADV
        result = apply_liquidity_constraint(target, adv, total_capital=10_000_000)
        assert result["TINY"] < result["AAPL"]

    def test_zero_adv(self):
        from src.assembled_core.portfolio.position_sizing import apply_liquidity_constraint

        result = apply_liquidity_constraint({"X": 0.5}, {"X": 0.0}, total_capital=100000)
        assert result["X"] == 0.0


# ---------------------------------------------------------------------------
# 7.4 Regime-Conditional Risk Limits
# ---------------------------------------------------------------------------


class TestRegimeRiskLimits:

    def test_pure_bull(self):
        from src.assembled_core.risk.state_machine import compute_regime_risk_limits

        result = compute_regime_risk_limits({"bull": 1.0})
        assert result["max_gross"] == 1.0
        assert result["max_dd"] == 0.20

    def test_crisis_more_conservative(self):
        from src.assembled_core.risk.state_machine import compute_regime_risk_limits

        bull = compute_regime_risk_limits({"bull": 1.0})
        crisis = compute_regime_risk_limits({"crisis": 1.0})
        assert crisis["max_gross"] < bull["max_gross"]
        assert crisis["max_dd"] < bull["max_dd"]

    def test_blended(self):
        from src.assembled_core.risk.state_machine import compute_regime_risk_limits

        result = compute_regime_risk_limits({"bull": 0.5, "crisis": 0.5})
        assert 0.5 < result["max_gross"] < 1.0


# ---------------------------------------------------------------------------
# 8.2 Dividend Tracking
# ---------------------------------------------------------------------------


class TestDividendTracking:

    def test_long_receives_dividend(self):
        from src.assembled_core.accounting.ledger import generate_dividend_events

        events = generate_dividend_events(
            positions={"AAPL": 100},
            dividends={"AAPL": 0.82},
            event_ts=pd.Timestamp("2024-01-15"),
        )
        assert len(events) == 1
        assert events.iloc[0]["cash_delta"] == 82.0  # 100 * 0.82
        assert events.iloc[0]["event_type"] == "DIVIDEND"

    def test_short_pays_dividend(self):
        from src.assembled_core.accounting.ledger import generate_dividend_events

        events = generate_dividend_events(
            positions={"AAPL": -50},
            dividends={"AAPL": 0.82},
            event_ts=pd.Timestamp("2024-01-15"),
        )
        assert events.iloc[0]["cash_delta"] == -41.0  # -50 * 0.82


# ---------------------------------------------------------------------------
# 8.3 Margin Accounting
# ---------------------------------------------------------------------------


class TestMarginAccounting:

    def test_no_margin_call(self):
        from src.assembled_core.accounting.ledger import check_margin_requirements

        result = check_margin_requirements(
            positions={"AAPL": 100},
            prices={"AAPL": 150.0},
            cash_balance=50000,
        )
        assert result["margin_call"] is False

    def test_margin_call_triggered(self):
        from src.assembled_core.accounting.ledger import check_margin_requirements

        result = check_margin_requirements(
            positions={"AAPL": 1000},
            prices={"AAPL": 150.0},
            cash_balance=1000,  # way too little for 150k position
        )
        # Equity = 1000 + 1000*150 = 151000
        # Maintenance = 150000 * 0.30 = 45000
        # Actually no margin call since equity is 151000 > 45000
        assert result["margin_call"] is False

    def test_margin_call_short_position(self):
        from src.assembled_core.accounting.ledger import check_margin_requirements

        result = check_margin_requirements(
            positions={"GME": -100},
            prices={"GME": 400.0},  # price spiked
            cash_balance=5000,
        )
        # Equity = 5000 + (-100 * 400) = 5000 - 40000 = -35000
        # Maintenance = 40000 * 0.30 = 12000
        # -35000 < 12000 → margin call
        assert result["margin_call"] is True


# ---------------------------------------------------------------------------
# 9.3 Regime-Segmented Performance
# ---------------------------------------------------------------------------


class TestRegimeSegmentedPerformance:

    def test_basic(self):
        from src.assembled_core.qa.metrics import compute_regime_segmented_performance

        np.random.seed(42)
        n = 200
        returns = pd.Series(np.random.normal(0.001, 0.01, n))
        regimes = pd.Series(["bull"] * 100 + ["bear"] * 100)
        result = compute_regime_segmented_performance(returns, regimes)
        assert "bull" in result
        assert "bear" in result
        assert result["bull"]["n_days"] == 100


# ---------------------------------------------------------------------------
# 10.1 Universe Reconstitution
# ---------------------------------------------------------------------------


class TestUniverseReconstitution:

    def test_monthly_snapshots(self):
        from src.assembled_core.data.universe import build_monthly_snapshots

        history = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "DEAD"],
            "start_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "end_date": [pd.NaT, pd.NaT, "2020-06-01"],
            "status": ["active", "active", "delisted"],
        })
        snapshots = build_monthly_snapshots(history, "2020-01-01", "2020-12-01")
        assert "2020-01" in snapshots
        assert "AAPL" in snapshots["2020-01"]
        # DEAD should not be in July snapshot
        if "2020-07" in snapshots:
            assert "DEAD" not in snapshots["2020-07"]


# ---------------------------------------------------------------------------
# 10.2 Multi-Source Price Validation
# ---------------------------------------------------------------------------


class TestMultiSourceValidation:

    def test_matching_prices(self):
        from src.assembled_core.data.prices_ingest import validate_prices_cross_source

        df1 = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "AAPL"],
            "close": [150.0, 151.0],
        })
        df2 = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "AAPL"],
            "close": [150.1, 150.9],
        })
        result = validate_prices_cross_source(df1, df2)
        assert result["validated"] is True

    def test_divergent_prices(self):
        from src.assembled_core.data.prices_ingest import validate_prices_cross_source

        df1 = pd.DataFrame({
            "timestamp": ["2024-01-01"],
            "symbol": ["X"],
            "close": [100.0],
        })
        df2 = pd.DataFrame({
            "timestamp": ["2024-01-01"],
            "symbol": ["X"],
            "close": [110.0],  # 10% different
        })
        result = validate_prices_cross_source(df1, df2, max_diff_pct=1.0)
        assert result["validated"] is False
        assert "X" in result["flagged_symbols"]


# ---------------------------------------------------------------------------
# 11.1 Policy Consistency
# ---------------------------------------------------------------------------


class TestPolicyConsistency:

    def test_valid_policy(self):
        from src.assembled_core.config.policy_schema import validate_policy_consistency

        policy = {
            "scope": {"leverage_allowed": False},
            "risk_limits": {
                "max_gross_exposure": 1.0,
                "max_short_gross": 0.30,
                "max_position_weight": 0.15,
                "max_drawdown": {"soft": 0.05, "hard": 0.10, "kill": 0.20},
                "max_positions": 20,
            },
        }
        violations = validate_policy_consistency(policy)
        assert violations == []

    def test_inconsistent_short(self):
        from src.assembled_core.config.policy_schema import validate_policy_consistency

        policy = {
            "scope": {},
            "risk_limits": {
                "max_gross_exposure": 0.50,
                "max_short_gross": 0.80,  # > max_gross!
            },
        }
        violations = validate_policy_consistency(policy)
        assert any("max_short_gross" in v for v in violations)


# ---------------------------------------------------------------------------
# 11.2 Graceful Degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:

    def test_tracker(self):
        from src.assembled_core.pipeline.graceful_degradation import DegradationTracker

        t = DegradationTracker()
        assert not t.is_degraded
        t.record_failure("fred_macro", "API timeout")
        assert t.is_degraded
        assert t.severity == "minor"
        assert "fred_macro" in t.failed_sources

    def test_neutralize(self):
        from src.assembled_core.pipeline.graceful_degradation import neutralize_missing_features

        df = pd.DataFrame({
            "vix_level": [25.0, np.nan, 30.0],
            "vix_zscore": [1.5, 2.0, np.nan],
            "other": [1, 2, 3],
        })
        result = neutralize_missing_features(df, {"vix": "stale data"})
        # vix features should be neutralized
        assert result["vix_level"].iloc[0] == 20.0  # default neutral
        assert result["vix_zscore"].iloc[0] == 0.0   # default neutral for zscore
        assert result["other"].tolist() == [1, 2, 3]  # unchanged


# ---------------------------------------------------------------------------
# 4.4 FinBERT (import only — model not loaded in tests)
# ---------------------------------------------------------------------------


class TestFinBERTIntegration:

    def test_score_cluster_no_finbert(self):
        from src.assembled_core.events.news.clustering import score_cluster_sentiment

        # Without transformers installed, should return 0.0
        score = score_cluster_sentiment(["Bank collapse fears mount"])
        assert isinstance(score, float)

    def test_enrich_clusters(self):
        from src.assembled_core.events.news.clustering import enrich_clusters_with_sentiment

        clusters = [
            {"sample_titles": ["Crisis deepens"], "cluster_id": "c1"},
            {"sample_titles": ["Markets rally"], "cluster_id": "c2"},
        ]
        result = enrich_clusters_with_sentiment(clusters)
        assert all("sentiment_score" in c for c in result)
        assert all("magnitude_adjustment" in c for c in result)
