"""Test that cumulative cash gate rejects orders that would drive cash negative."""

from __future__ import annotations

import pytest

import pandas as pd

from src.assembled_core.execution.fill_model import REJECT_INSUFFICIENT_CASH, apply_cash_gate


@pytest.mark.unit
def test_cash_gate_rejects_when_three_buys_exceed_cash():
    """Three BUY orders at same timestamp that exceed available_cash: at least one rejected with INSUFFICIENT_CASH."""
    # Cash = 10000. Three BUYs: 4000 + 4000 + 4000 = 12000 > 10000 -> last one (or more) must be rejected
    ts = pd.Timestamp("2025-01-15 16:00", tz="UTC")
    orders = pd.DataFrame([
        {"timestamp": ts, "symbol": "AAPL", "side": "BUY", "qty": 40.0, "price": 100.0},
        {"timestamp": ts, "symbol": "MSFT", "side": "BUY", "qty": 30.0, "price": 133.34},
        {"timestamp": ts, "symbol": "GOOG", "side": "BUY", "qty": 20.0, "price": 200.0},
    ])
    available_cash = 10000.0

    out = apply_cash_gate(orders, available_cash)

    rejected = out[out["reject_reason"] == REJECT_INSUFFICIENT_CASH]
    assert len(rejected) >= 1, (
        f"At least one order should be rejected with INSUFFICIENT_CASH; reject_reason values: {out['reject_reason'].tolist()}"
    )
    assert (rejected["fill_qty"] == 0.0).all(), "Rejected orders should have fill_qty=0"


@pytest.mark.unit
def test_cash_gate_rejects_single_order_exceeding_cash():
    """Single BUY with notional > available_cash is rejected."""
    ts = pd.Timestamp("2025-01-15 16:00", tz="UTC")
    orders = pd.DataFrame([
        {"timestamp": ts, "symbol": "AAPL", "side": "BUY", "qty": 150.0, "price": 100.0},
    ])
    available_cash = 10000.0  # notional = 15000

    out = apply_cash_gate(orders, available_cash)

    assert out["reject_reason"].iloc[0] == REJECT_INSUFFICIENT_CASH
    assert out["fill_qty"].iloc[0] == 0.0


@pytest.mark.unit
def test_cash_gate_with_estimated_costs():
    """When total_cost_cash is present, notional + cost is used; can reject due to cost pushing over."""
    ts = pd.Timestamp("2025-01-15 16:00", tz="UTC")
    # Notional 9990, but total_cost_cash 20 -> need 10010 > 10000
    orders = pd.DataFrame([
        {"timestamp": ts, "symbol": "AAPL", "side": "BUY", "qty": 99.9, "price": 100.0,
         "total_cost_cash": 20.0},
    ])
    available_cash = 10000.0

    out = apply_cash_gate(orders, available_cash)

    assert out["reject_reason"].iloc[0] == REJECT_INSUFFICIENT_CASH
    assert out["fill_qty"].iloc[0] == 0.0


@pytest.mark.unit
def test_cash_curve_min_non_negative_after_backtest():
    """After full backtest with costs, cash_curve min should be >= -1e-6 (no overspend from gate)."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
    from tests.test_equity_curve_is_mtm import (
        _dummy_position_sizing_fn,
        _dummy_signal_fn,
        _synthetic_prices_three_symbols_upward,
    )
    prices = _synthetic_prices_three_symbols_upward()
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=_dummy_signal_fn,
        position_sizing_fn=_dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        commission_bps=5.0,
        spread_w=0.25,
        impact_w=0.25,
        strict_session_gate=False,
        include_trades=True,
    )
    assert "cash" in result.equity.columns
    cash_min = result.equity["cash"].min()
    assert cash_min >= -1e-6, (
        f"Cash gate should prevent overspend: cash_curve min={cash_min} must be >= -1e-6"
    )
