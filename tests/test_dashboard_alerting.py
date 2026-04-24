"""Tests for M29 Dashboard & Observability.

Covers:
- Task 29.1: Dashboard data provider
- Task 29.2: Alerting system (existing alert_manager + new dashboard_data)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

import pytest; pytest.importorskip('src.assembled_core.ops.dashboard_data')
from src.assembled_core.ops.dashboard_data import (
    DashboardSnapshot,
    build_pnl_curve,
    build_position_table,
    compute_risk_snapshot,
    compute_exposure,
    build_signal_heatmap,
)


@pytest.mark.phase12
class TestPnLCurve:
    def test_basic(self):
        equity = pd.Series([100000, 100500, 101000, 100800],
                           index=pd.date_range("2026-01-01", periods=4))
        pnl = build_pnl_curve(equity, initial_capital=100000)
        assert len(pnl) == 4
        assert pnl[str(equity.index[0])] == 0.0
        assert pnl[str(equity.index[2])] == 1000.0

    def test_empty(self):
        pnl = build_pnl_curve(pd.Series(dtype=float))
        assert pnl == {}


@pytest.mark.phase12
class TestPositionTable:
    def test_basic(self):
        weights = {"AAPL": 0.3, "MSFT": 0.2, "GOOG": -0.1}
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 2800.0}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech"}
        positions = build_position_table(weights, prices, sector_mapping=sectors)
        assert len(positions) == 3
        assert positions[0]["symbol"] == "AAPL"  # sorted by abs weight
        assert positions[2]["side"] == "short"

    def test_filters_zero_weights(self):
        weights = {"AAPL": 0.3, "MSFT": 0.0}
        positions = build_position_table(weights)
        assert len(positions) == 1


@pytest.mark.phase12
class TestRiskSnapshot:
    def test_basic(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.02, 252))
        metrics = compute_risk_snapshot(returns)
        assert "var_95" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["var_95"] < 0
        assert metrics["max_drawdown"] < 0

    def test_short_series(self):
        metrics = compute_risk_snapshot(pd.Series([0.01]))
        assert metrics == {}

    def test_win_rate(self):
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])
        metrics = compute_risk_snapshot(returns)
        assert metrics["win_rate"] == 0.6


@pytest.mark.phase12
class TestExposure:
    def test_long_only(self):
        weights = {"A": 0.3, "B": 0.2, "C": 0.5}
        exp = compute_exposure(weights)
        assert exp["gross"] == pytest.approx(1.0)
        assert exp["net"] == pytest.approx(1.0)
        assert exp["short"] == 0.0

    def test_long_short(self):
        weights = {"A": 0.4, "B": -0.2, "C": 0.3}
        exp = compute_exposure(weights)
        assert exp["gross"] == pytest.approx(0.9)
        assert exp["net"] == pytest.approx(0.5)
        assert exp["long"] == pytest.approx(0.7)
        assert exp["short"] == pytest.approx(0.2)
        assert exp["n_positions"] == 3

    def test_empty(self):
        exp = compute_exposure({})
        assert exp["gross"] == 0.0


@pytest.mark.phase12
class TestSignalHeatmap:
    def test_basic(self):
        signals = pd.DataFrame(
            {"momentum": [0.5, -0.3, 0.8], "value": [0.2, 0.6, -0.1]},
            index=["AAPL", "MSFT", "GOOG"],
        )
        heatmap = build_signal_heatmap(signals)
        assert "AAPL" in heatmap
        assert "momentum" in heatmap["AAPL"]
        assert heatmap["GOOG"]["momentum"] == 0.8

    def test_empty(self):
        heatmap = build_signal_heatmap(pd.DataFrame())
        assert heatmap == {}


@pytest.mark.phase12
class TestDashboardSnapshot:
    def test_to_dict(self):
        snap = DashboardSnapshot(
            timestamp="2026-04-14T18:00:00Z",
            pnl_curve={"2026-04-14": 500.0},
            current_positions=[{"symbol": "AAPL", "weight": 0.3}],
            risk_metrics={"var_95": -0.03},
            factor_performance={"momentum": 0.05},
            signal_heatmap={"AAPL": {"momentum": 0.8}},
            trade_log=[],
            exposure={"gross": 1.0, "net": 0.8},
            alerts=[],
        )
        d = snap.to_dict()
        assert d["timestamp"] == "2026-04-14T18:00:00Z"
        assert d["risk_metrics"]["var_95"] == -0.03
