"""Tests for M26: Tail Hedging — Portfolio Insurance and Tail Risk Management."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("src.assembled_core.risk.tail_hedging")
from src.assembled_core.risk.tail_hedging import (
    TailHedgeConfig,
    HedgeRecommendation,
    compute_hedge_ratio,
    compute_put_cost_estimate,
    recommend_hedge,
    compute_tail_risk_metrics,
)


@pytest.mark.phase12
class TestComputeHedgeRatio:
    def test_below_trigger_returns_zero(self):
        ratio = compute_hedge_ratio(current_vix=20.0, portfolio_vol=0.12)
        assert ratio == 0.0

    def test_at_trigger_returns_min(self):
        cfg = TailHedgeConfig(vix_hedge_trigger=25.0, min_hedge_ratio=0.05)
        ratio = compute_hedge_ratio(current_vix=25.0, portfolio_vol=0.12, config=cfg)
        assert ratio == pytest.approx(0.05, abs=0.01)

    def test_at_full_hedge_returns_max(self):
        cfg = TailHedgeConfig(
            vix_hedge_trigger=25.0,
            vix_full_hedge_level=35.0,
            max_hedge_ratio=0.30,
        )
        ratio = compute_hedge_ratio(current_vix=35.0, portfolio_vol=0.12, config=cfg)
        assert ratio == pytest.approx(0.30, abs=0.01)

    def test_above_full_hedge_capped(self):
        ratio = compute_hedge_ratio(current_vix=50.0, portfolio_vol=0.12)
        cfg = TailHedgeConfig()
        assert ratio <= cfg.max_hedge_ratio

    def test_midpoint_interpolation(self):
        cfg = TailHedgeConfig(
            vix_hedge_trigger=25.0,
            vix_full_hedge_level=35.0,
            min_hedge_ratio=0.05,
            max_hedge_ratio=0.30,
        )
        ratio = compute_hedge_ratio(current_vix=30.0, portfolio_vol=0.10, config=cfg)
        assert 0.05 < ratio < 0.30

    def test_dynamic_sizing_high_vol(self):
        cfg = TailHedgeConfig(use_dynamic_sizing=True)
        ratio_low = compute_hedge_ratio(
            current_vix=30.0, portfolio_vol=0.10, config=cfg
        )
        ratio_high = compute_hedge_ratio(
            current_vix=30.0, portfolio_vol=0.25, config=cfg
        )
        assert ratio_high >= ratio_low

    def test_dynamic_sizing_disabled(self):
        cfg = TailHedgeConfig(use_dynamic_sizing=False)
        ratio = compute_hedge_ratio(current_vix=30.0, portfolio_vol=0.30, config=cfg)
        assert ratio > 0

    def test_equal_trigger_and_full_returns_max(self):
        cfg = TailHedgeConfig(
            vix_hedge_trigger=30.0,
            vix_full_hedge_level=30.0,
            max_hedge_ratio=0.25,
        )
        ratio = compute_hedge_ratio(current_vix=30.0, portfolio_vol=0.15, config=cfg)
        assert ratio == pytest.approx(0.25, abs=0.01)


@pytest.mark.phase12
class TestPutCostEstimate:
    def test_zero_hedge_ratio(self):
        cost = compute_put_cost_estimate(1_000_000, hedge_ratio=0.0, current_vol=0.20)
        assert cost == 0.0

    def test_positive_cost(self):
        cost = compute_put_cost_estimate(
            1_000_000,
            hedge_ratio=0.10,
            current_vol=0.20,
        )
        assert cost > 0

    def test_higher_vol_higher_cost(self):
        cost_low = compute_put_cost_estimate(1_000_000, 0.10, current_vol=0.15)
        cost_high = compute_put_cost_estimate(1_000_000, 0.10, current_vol=0.30)
        assert cost_high > cost_low

    def test_zero_vol(self):
        cost = compute_put_cost_estimate(1_000_000, 0.10, current_vol=0.0)
        assert cost == 0.0


@pytest.mark.phase12
class TestRecommendHedge:
    def test_no_trigger_active(self):
        rec = recommend_hedge(
            portfolio_value=1_000_000,
            current_vix=18.0,
            portfolio_vol=0.10,
        )
        assert isinstance(rec, HedgeRecommendation)
        assert rec.hedge_ratio == 0.0
        assert "No hedge trigger active" in rec.trigger_reason

    def test_vix_trigger(self):
        rec = recommend_hedge(
            portfolio_value=1_000_000,
            current_vix=30.0,
            portfolio_vol=0.12,
        )
        assert rec.hedge_ratio > 0
        assert "VIX=" in rec.trigger_reason

    def test_full_hedge_urgency(self):
        rec = recommend_hedge(
            portfolio_value=1_000_000,
            current_vix=40.0,
            portfolio_vol=0.20,
        )
        assert rec.urgency == 1.0

    def test_drawdown_trigger(self):
        rec = recommend_hedge(
            portfolio_value=1_000_000,
            current_vix=20.0,
            portfolio_vol=0.12,
            recent_max_drawdown=-0.15,
        )
        assert rec.hedge_ratio > 0
        assert "drawdown" in rec.trigger_reason.lower()

    def test_elevated_vol_trigger(self):
        rec = recommend_hedge(
            portfolio_value=1_000_000,
            current_vix=20.0,
            portfolio_vol=0.30,
        )
        assert "elevated" in rec.trigger_reason.lower()

    def test_notional_calculation(self):
        rec = recommend_hedge(
            portfolio_value=2_000_000,
            current_vix=30.0,
            portfolio_vol=0.15,
        )
        assert rec.notional_to_hedge == pytest.approx(
            2_000_000 * rec.hedge_ratio,
            abs=1.0,
        )

    def test_put_strike_pct(self):
        cfg = TailHedgeConfig(put_otm_pct=0.05)
        rec = recommend_hedge(1_000_000, 30.0, 0.15, config=cfg)
        assert rec.put_strike_pct == pytest.approx(0.95, abs=0.001)


@pytest.mark.phase12
class TestTailRiskMetrics:
    def test_basic_metrics(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.015, 500)
        metrics = compute_tail_risk_metrics(returns)
        assert "var_pct" in metrics
        assert "expected_shortfall_pct" in metrics
        assert "tail_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics

    def test_var_is_negative(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.02, 500)
        metrics = compute_tail_risk_metrics(returns)
        assert metrics["var_pct"] < 0

    def test_es_worse_than_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 500)
        metrics = compute_tail_risk_metrics(returns)
        assert metrics["expected_shortfall_pct"] <= metrics["var_pct"]

    def test_short_series_defaults(self):
        metrics = compute_tail_risk_metrics(np.array([0.01, 0.02, -0.01]))
        assert metrics["var_pct"] == 0.0
        assert metrics["tail_ratio"] == 1.0

    def test_max_drawdown_negative(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 500)
        metrics = compute_tail_risk_metrics(returns)
        assert metrics["max_drawdown"] <= 0
