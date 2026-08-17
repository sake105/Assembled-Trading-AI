"""Tests for M17 Wave 2 — STANDARD items (Welle 2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


class TestSignalDiagnostics:
    def test_compute_health(self):
        import pytest

        pytest.importorskip("src.assembled_core.signals.signal_diagnostics")
        from src.assembled_core.signals.signal_diagnostics import compute_signal_health

        np.random.seed(42)
        n_ts = 20
        n_sym = 10  # need >=5 symbols per timestamp for IC
        total = n_ts * n_sym
        df = pd.DataFrame(
            {
                "timestamp": list(range(n_ts)) * n_sym,
                "symbol": sum([[f"S{i}"] * n_ts for i in range(n_sym)], []),
                "factor1": np.random.normal(0, 1, total),
                "fwd_ret": np.random.normal(0, 0.01, total),
            }
        )
        result = compute_signal_health(df, "fwd_ret", ["factor1"])
        assert not result.empty
        assert "ic" in result.columns


class TestCrashPredictionThresholds:
    def test_rolling_percentiles(self):
        from src.assembled_core.signals.crash_prediction import (
            compute_rolling_percentile_thresholds,
        )

        series = pd.Series(np.random.normal(20, 5, 300))
        result = compute_rolling_percentile_thresholds(series, window=100)
        assert "p75" in result.columns
        assert "p90" in result.columns


# ENTFERNT 2026-08-17: testete intel/wild_card_detector + intel/structural_cycles, archiviert in Tranche 2, s. archive/orphaned_code_2026-08-17/README.md


class TestSmartOrderRouter:
    def test_route_order(self):
        from src.assembled_core.execution.smart_order_router import (
            RoutingResult,
            route_order,
        )

        result = route_order(100000, signal_urgency=0.9, seed=42)
        assert isinstance(result, RoutingResult)
        assert result.allocations, "expected at least one venue allocation"
        assert result.total_expected_cost_bps >= 0
        assert result.total_expected_fill_pct > 0


class TestTaxLots:
    def test_fifo_pnl(self):
        import pytest

        pytest.importorskip("src.assembled_core.accounting.tax_lots")
        from src.assembled_core.accounting.tax_lots import TaxLotTracker

        tracker = TaxLotTracker()
        tracker.buy("AAPL", 100, 150.0, date(2024, 1, 1))
        tracker.buy("AAPL", 50, 160.0, date(2024, 2, 1))
        pnl = tracker.sell("AAPL", 100, 170.0, date(2024, 3, 1))
        assert pnl == 2000.0  # (170-150)*100 FIFO


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
        from src.assembled_core.execution.algo_execution import (
            compute_implementation_shortfall,
        )

        is_bps = compute_implementation_shortfall(100.0, 100.5, "BUY")
        assert is_bps == 50.0  # 50 bps

    def test_sell_is(self):
        from src.assembled_core.execution.algo_execution import (
            compute_implementation_shortfall,
        )

        is_bps = compute_implementation_shortfall(100.0, 99.5, "SELL")
        assert is_bps == 50.0


class TestOrderNetting:
    def test_net_opposing(self):
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "qty": [100, -60, 50],
            }
        )
        result = net_orders(orders)
        assert len(result) == 2  # AAPL net=40, MSFT=50
        aapl_net = result[result["symbol"] == "AAPL"]["qty"].iloc[0]
        assert aapl_net == 40

    def test_net_opposing_with_side_column(self):
        """Unsigned qty + side column (the generate_orders_from_targets format)
        must net to 0 for fully offsetting BUY+SELL, not sum to 2× qty."""
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "side": ["BUY", "SELL", "BUY"],
                "qty": [100, 100, 50],
            }
        )
        result = net_orders(orders)
        assert len(result) == 1  # AAPL nets to zero, MSFT survives
        assert result["symbol"].iloc[0] == "MSFT"
        assert result["side"].iloc[0] == "BUY"
        assert result["qty"].iloc[0] == 50

    def test_net_partial_with_side_column(self):
        """BUY 100 + SELL 40 → net BUY 60 (not BUY 140)."""
        from src.assembled_core.execution.order_generation import net_orders

        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "side": ["BUY", "SELL"],
                "qty": [100, 40],
            }
        )
        result = net_orders(orders)
        assert len(result) == 1
        assert result["side"].iloc[0] == "BUY"
        assert abs(result["qty"].iloc[0] - 60) < 1e-10
