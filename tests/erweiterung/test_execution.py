"""Tests for erweiterung.execution."""

from __future__ import annotations

import numpy as np

from erweiterung.execution import adaptive_slippage, almgren_chriss


def test_almgren_chriss_schedule_sums_to_total():
    params = almgren_chriss.MarketImpactParams(
        permanent_impact_gamma=1e-7,
        temporary_impact_eta=1e-6,
        volatility=0.02,
        risk_aversion=1e-6,
    )
    schedule = almgren_chriss.optimal_trade_schedule(
        total_shares=10000, total_time_steps=10, params=params, tau=1.0
    )
    assert abs(schedule.sum() - 10000) < 1e-3
    assert (schedule >= 0).all()


def test_almgren_chriss_high_risk_aversion_front_loaded():
    params_low = almgren_chriss.MarketImpactParams(risk_aversion=1e-9)
    params_high = almgren_chriss.MarketImpactParams(risk_aversion=1e-3)
    s_low = almgren_chriss.optimal_trade_schedule(10000, 20, params_low)
    s_high = almgren_chriss.optimal_trade_schedule(10000, 20, params_high)
    # High risk aversion -> trade earlier (more front-loaded)
    assert s_high[:5].sum() >= s_low[:5].sum()


def test_expected_cost_positive():
    params = almgren_chriss.MarketImpactParams()
    schedule = np.array([1000.0, 1000.0, 1000.0])
    c = almgren_chriss.expected_cost(schedule, params)
    assert c > 0


def test_slippage_increases_with_size():
    s_small = adaptive_slippage.slippage_bps(100, avg_daily_volume=1_000_000)
    s_large = adaptive_slippage.slippage_bps(100_000, avg_daily_volume=1_000_000)
    assert s_large > s_small


def test_slippage_zero_adv_returns_default():
    out = adaptive_slippage.slippage_bps(100, avg_daily_volume=0)
    assert out > 0
    assert np.isfinite(out)


def test_execution_price_buy_higher():
    p_buy = adaptive_slippage.execution_price(100, side=+1, slippage_bps_value=10)
    p_sell = adaptive_slippage.execution_price(100, side=-1, slippage_bps_value=10)
    assert p_buy > 100
    assert p_sell < 100
