"""Tests for M17 Wave 2 — STANDARD items (Welle 2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


class TestMeanReversion:
    def test_bull_regime_generates_signals(self):
        from src.assembled_core.signals.mean_reversion import compute_mean_reversion_signals

        np.random.seed(42)
        n = 100
        prices = pd.DataFrame({
            "symbol": ["A"] * n,
            "close": np.cumsum(np.random.normal(0, 1, n)) + 100,
        })
        result = compute_mean_reversion_signals(prices, regime="bull")
        assert isinstance(result, pd.DataFrame)

    def test_bear_regime_inactive(self):
        from src.assembled_core.signals.mean_reversion import compute_mean_reversion_signals

        prices = pd.DataFrame({"symbol": ["A"] * 50, "close": range(50)})
        result = compute_mean_reversion_signals(prices, regime="bear")
        assert result.empty


class TestSignalDiagnostics:
    def test_compute_health(self):
        from src.assembled_core.signals.signal_diagnostics import compute_signal_health

        np.random.seed(42)
        n_ts = 20
        n_sym = 10  # need >=5 symbols per timestamp for IC
        total = n_ts * n_sym
        df = pd.DataFrame({
            "timestamp": list(range(n_ts)) * n_sym,
            "symbol": sum([[f"S{i}"] * n_ts for i in range(n_sym)], []),
            "factor1": np.random.normal(0, 1, total),
            "fwd_ret": np.random.normal(0, 0.01, total),
        })
        result = compute_signal_health(df, "fwd_ret", ["factor1"])
        assert not result.empty
        assert "ic" in result.columns


class TestCrashPredictionThresholds:
    def test_rolling_percentiles(self):
        from src.assembled_core.signals.crash_prediction import compute_rolling_percentile_thresholds

        series = pd.Series(np.random.normal(20, 5, 300))
        result = compute_rolling_percentile_thresholds(series, window=100)
        assert "p75" in result.columns
        assert "p90" in result.columns


class TestFeatureDrift:
    def test_no_drift(self):
        from src.assembled_core.ml.model_monitoring import detect_feature_drift

        np.random.seed(42)
        train = pd.DataFrame({"f1": np.random.normal(0, 1, 200)})
        recent = pd.DataFrame({"f1": np.random.normal(0, 1, 50)})
        result = detect_feature_drift(train, recent, ["f1"])
        assert result["alert_level"] == "OK"


class TestSatelliteFeatures:
    def test_copper_gold(self):
        from src.assembled_core.features.satellite_proxy_features import compute_copper_gold_ratio

        copper = pd.Series([4.0, 4.1, 4.2])
        gold = pd.Series([1800, 1810, 1790])
        result = compute_copper_gold_ratio(copper, gold)
        assert len(result) == 3
        assert result.iloc[0] > 0


class TestDisclosureFeatures:
    def test_fog_index(self):
        from src.assembled_core.features.disclosure_features import compute_fog_index

        text = "The company faces significant risks. Market conditions are challenging. " * 10
        fog = compute_fog_index(text)
        assert fog > 0

    def test_empty_text(self):
        from src.assembled_core.features.disclosure_features import compute_fog_index
        assert compute_fog_index("") == 0.0


class TestWildCardDetector:
    def test_no_anomaly(self):
        from src.assembled_core.intel.wild_card_detector import detect_volume_anomaly

        counts = pd.Series([100] * 40)
        result = detect_volume_anomaly(counts)
        assert result["is_anomaly"] is False

    def test_anomaly_detected(self):
        from src.assembled_core.intel.wild_card_detector import detect_volume_anomaly

        np.random.seed(42)
        # baseline with natural variance, then a huge spike
        baseline = np.random.normal(100, 10, 39).astype(int).tolist()
        counts = pd.Series(baseline + [500])
        result = detect_volume_anomaly(counts)
        assert result["is_anomaly"] is True


class TestWargaming:
    def test_prisoners_dilemma(self):
        from src.assembled_core.intel.wargaming import find_nash_2x2

        # Classic prisoner's dilemma
        payoff_a = np.array([[3, 0], [5, 1]])
        payoff_b = np.array([[3, 5], [0, 1]])
        result = find_nash_2x2(payoff_a, payoff_b)
        assert result.equilibrium_type in ("pure", "mixed", "dominant")
        assert result.confidence > 0


class TestStructuralCycles:
    def test_normal_environment(self):
        from src.assembled_core.intel.structural_cycles import compute_structural_cycle_score

        result = compute_structural_cycle_score(
            debt_gdp_pct=80, gini_index=0.35, trust_index=0.60, rivalry_index=0.30,
        )
        assert result.risk_multiplier >= 1.0
        assert result.composite >= 0


class TestRegimePortfolio:
    def test_blend_templates(self):
        from src.assembled_core.portfolio.regime_portfolio import blend_regime_templates

        result = blend_regime_templates({"bull": 0.7, "bear": 0.3})
        assert sum(result.values()) > 0
        assert all(v >= 0 for v in result.values())


class TestSmartOrderRouter:
    def test_route_order(self):
        from src.assembled_core.execution.smart_order_router import route_order

        result = route_order(100000, signal_urgency=0.9, seed=42)
        assert "venue" in result
        assert result["venue"] in ("primary", "dark_pool", "ats")


class TestSystemicRisk:
    def test_centrality(self):
        from src.assembled_core.risk.systemic_risk import compute_return_network_centrality

        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 100),
            "B": np.random.normal(0, 0.01, 100),
            "C": np.random.normal(0, 0.01, 100),
        })
        result = compute_return_network_centrality(returns)
        assert all(0 <= v <= 1 for v in result.values())


class TestAntifragility:
    def test_score(self):
        from src.assembled_core.risk.antifragility import compute_antifragility_score

        np.random.seed(42)
        port = pd.Series(np.random.normal(0, 0.01, 100))
        market = pd.Series(np.random.normal(0, 0.01, 100))
        result = compute_antifragility_score(port, market)
        assert len(result) == 100


class TestTaxLots:
    def test_fifo_pnl(self):
        from src.assembled_core.accounting.tax_lots import TaxLotTracker

        tracker = TaxLotTracker()
        tracker.buy("AAPL", 100, 150.0, date(2024, 1, 1))
        tracker.buy("AAPL", 50, 160.0, date(2024, 2, 1))
        pnl = tracker.sell("AAPL", 100, 170.0, date(2024, 3, 1))
        assert pnl == 2000.0  # (170-150)*100 FIFO


class TestRoundTrips:
    def test_basic(self):
        from src.assembled_core.accounting.round_trips import compute_round_trips, round_trip_summary

        trades = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": ["2024-01-01", "2024-01-10"],
            "side": ["BUY", "SELL"],
            "price": [150.0, 160.0],
            "quantity": [100, 100],
            "commission": [1.0, 1.0],
        })
        trips = compute_round_trips(trades)
        assert len(trips) == 1
        summary = round_trip_summary(trips)
        assert summary["n_trips"] == 1
        assert summary["total_pnl"] > 0


class TestDecisionAudit:
    def test_record_and_summary(self):
        from src.assembled_core.accounting.decision_audit import DecisionAuditTrail, DecisionRecord

        audit = DecisionAuditTrail()
        audit.record(DecisionRecord(
            timestamp="2024-01-01", symbol="AAPL", direction="LONG",
            signal_score=0.8, regime="bull",
        ))
        assert audit.summary()["n_records"] == 1


class TestFreshnessMonitor:
    def test_stale_detection(self):
        from src.assembled_core.data.freshness_monitor import FreshnessMonitor

        monitor = FreshnessMonitor()
        monitor.register("test_source", max_age_hours=0.001)  # very short TTL
        alerts = monitor.check_all()
        assert len(alerts) == 1  # never updated → stale


class TestDataVersioning:
    def test_hash(self):
        from src.assembled_core.data.data_versioning import compute_data_hash

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = compute_data_hash(df)
        h2 = compute_data_hash(df)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestAlertManager:
    def test_alert_and_rate_limit(self):
        from src.assembled_core.ops.alert_manager import AlertManager

        mgr = AlertManager(rate_limit_seconds=60)
        assert mgr.alert("WARNING", "test", "msg1") is True
        assert mgr.alert("WARNING", "test", "msg1") is False  # rate limited
        assert mgr.pending_count == 1


class TestDrawdownDuration:
    def test_duration(self):
        from src.assembled_core.risk.risk_metrics import compute_drawdown_duration

        equity = pd.Series([100, 101, 99, 98, 97, 100, 101, 99, 100])
        result = compute_drawdown_duration(equity)
        assert result["max_dd_duration_days"] > 0
        assert "n_drawdown_periods" in result


class TestCDaR:
    def test_cdar(self):
        from src.assembled_core.risk.risk_metrics import compute_cdar

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        cdar = compute_cdar(returns)
        assert cdar < 0  # CDaR is negative


class TestImplementationShortfall:
    def test_buy_is(self):
        from src.assembled_core.execution.algo_execution import compute_implementation_shortfall

        is_bps = compute_implementation_shortfall(100.0, 100.5, "BUY")
        assert is_bps == 50.0  # 50 bps

    def test_sell_is(self):
        from src.assembled_core.execution.algo_execution import compute_implementation_shortfall

        is_bps = compute_implementation_shortfall(100.0, 99.5, "SELL")
        assert is_bps == 50.0


class TestOrderNetting:
    def test_net_opposing(self):
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "qty": [100, -60, 50],
        })
        result = net_orders(orders)
        assert len(result) == 2  # AAPL net=40, MSFT=50
        aapl_net = result[result["symbol"] == "AAPL"]["qty"].iloc[0]
        assert aapl_net == 40

    def test_net_opposing_with_side_column(self):
        """Unsigned qty + side column (the generate_orders_from_targets format)
        must net to 0 for fully offsetting BUY+SELL, not sum to 2× qty."""
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "side": ["BUY", "SELL", "BUY"],
            "qty": [100, 100, 50],
        })
        result = net_orders(orders)
        assert len(result) == 1  # AAPL nets to zero, MSFT survives
        assert result["symbol"].iloc[0] == "MSFT"
        assert result["side"].iloc[0] == "BUY"
        assert result["qty"].iloc[0] == 50

    def test_net_partial_with_side_column(self):
        """BUY 100 + SELL 40 → net BUY 60 (not BUY 140)."""
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "side": ["BUY", "SELL"],
            "qty": [100, 40],
        })
        result = net_orders(orders)
        assert len(result) == 1
        assert result["side"].iloc[0] == "BUY"
        assert abs(result["qty"].iloc[0] - 60) < 1e-10
