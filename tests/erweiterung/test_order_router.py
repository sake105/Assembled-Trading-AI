"""Tests für Order-Router."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erweiterung.live.order_router import (
    Order,
    OrderRouterConfig,
    apply_pre_trade_checks,
    compute_orders,
    compute_target_notionals,
    decision_to_orders,
    orders_to_dataframe,
)


def test_target_notionals_basic():
    weights = pd.Series({"NVDA": 0.5, "GOOGL": 0.5})
    notionals = compute_target_notionals(weights, equity=100_000.0, exposure_cap=1.0)
    assert notionals["NVDA"] == 50_000
    assert notionals["GOOGL"] == 50_000


def test_target_notionals_exposure_cap_renormalize():
    """Sum > Cap → re-normalize."""
    weights = pd.Series({"NVDA": 0.8, "GOOGL": 0.8})  # sum = 1.6
    notionals = compute_target_notionals(weights, equity=100_000.0, exposure_cap=1.0)
    # Should sum to 100_000 (cap × equity)
    assert abs(notionals.sum() - 100_000) < 1


def test_compute_orders_new_position():
    targets = pd.Series({"NVDA": 5000})  # buy $5000 NVDA
    current = pd.Series(dtype=float)  # no current
    prices = pd.Series({"NVDA": 100.0})
    orders = compute_orders(targets, current, prices, OrderRouterConfig(equity=10_000))
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].qty == 50  # 5000 / 100
    assert orders[0].reason == "NEW_POSITION"


def test_compute_orders_exit_position():
    targets = pd.Series({"NVDA": 0.0})  # exit
    current = pd.Series({"NVDA": 50.0})
    prices = pd.Series({"NVDA": 100.0})
    orders = compute_orders(targets, current, prices, OrderRouterConfig(
        equity=10_000, rebalance_threshold=0.001,
    ))
    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert "EXIT_POSITION" in orders[0].reason


def test_compute_orders_anti_churn_threshold():
    """Position-Diff unter min_order_notional → kein Trade."""
    targets = pd.Series({"NVDA": 100.0})  # only $100 diff
    current = pd.Series({"NVDA": 0})
    prices = pd.Series({"NVDA": 100.0})
    cfg = OrderRouterConfig(equity=100_000, min_order_notional=200.0)
    orders = compute_orders(targets, current, prices, cfg)
    assert len(orders) == 0  # skipped due to anti-churn


def test_compute_orders_rebalance_threshold():
    """Weight-Diff < threshold → kein Trade."""
    targets = pd.Series({"NVDA": 50_100})  # tiny change
    current = pd.Series({"NVDA": 500})  # current 500 × $100 = $50_000
    prices = pd.Series({"NVDA": 100.0})
    cfg = OrderRouterConfig(equity=100_000, rebalance_threshold=0.05)
    orders = compute_orders(targets, current, prices, cfg)
    # weight-diff 50100/100000 - 50000/100000 = 0.001 < 0.05 threshold
    assert len(orders) == 0


def test_compute_orders_lot_size_rounding():
    targets = pd.Series({"NVDA": 5050})  # 50.5 shares
    current = pd.Series(dtype=float)
    prices = pd.Series({"NVDA": 100.0})
    cfg = OrderRouterConfig(equity=10_000, lot_size=10.0)
    orders = compute_orders(targets, current, prices, cfg)
    # Round 50.5 to nearest 10 → 50
    assert orders[0].qty == 50
    assert orders[0].target_position == 50


def test_pre_trade_blacklist_flag():
    orders = [
        Order(symbol="BLACK", side="BUY", qty=10, target_notional=1000,
              current_position=0, target_position=10, price=100, reason="NEW"),
        Order(symbol="GOOD", side="BUY", qty=10, target_notional=1000,
              current_position=0, target_position=10, price=100, reason="NEW"),
    ]
    flagged = apply_pre_trade_checks(orders, {"blacklist": ["BLACK"]})
    assert "BLACKLISTED" in flagged[0].pre_trade_flags
    assert "BLACKLISTED" not in flagged[1].pre_trade_flags


def test_pre_trade_max_position_flag():
    orders = [
        Order(symbol="BIG", side="BUY", qty=100, target_notional=50_000,
              current_position=0, target_position=100, price=500, reason="NEW"),
        Order(symbol="SMALL", side="BUY", qty=10, target_notional=5_000,
              current_position=0, target_position=10, price=500, reason="NEW"),
    ]
    flagged = apply_pre_trade_checks(orders, {"max_position_pct": 0.40})
    # BIG = 50k / 55k = 91% > 40% threshold
    assert any("OVER_MAX_POS" in f for f in flagged[0].pre_trade_flags)
    assert not any("OVER_MAX_POS" in f for f in flagged[1].pre_trade_flags)


def test_orders_to_dataframe():
    orders = [
        Order(symbol="A", side="BUY", qty=10, target_notional=1000,
              current_position=0, target_position=10, price=100, reason="NEW"),
    ]
    df = orders_to_dataframe(orders)
    assert "symbol" in df.columns
    assert "side" in df.columns
    assert df.iloc[0]["qty"] == 10


def test_orders_to_dataframe_empty():
    df = orders_to_dataframe([])
    assert df.empty


def test_decision_to_orders_e2e():
    """LiveDecisionEngine-Output → Order-Liste."""
    decision = {
        "sa_weight": 0.70,
        "sa_leverage": 1.0,
        "eq_top_weights": pd.Series({"NVDA": 0.5, "GOOGL": 0.5}),
        "xa_hybrid_weights": pd.Series({"SPY": 0.6, "TLT": 0.4}),
    }
    current = pd.Series(dtype=float)
    prices = pd.Series({"NVDA": 100, "GOOGL": 200, "SPY": 500, "TLT": 90})
    cfg = OrderRouterConfig(equity=100_000)
    orders = decision_to_orders(decision, current, prices, cfg)
    assert len(orders) > 0
    # NVDA: sa_w=0.70 × sa_lev=1.0 × 0.5 = 0.35 of equity = $35k → 350 shares
    nvda_orders = [o for o in orders if o.symbol == "NVDA"]
    assert len(nvda_orders) == 1
    assert nvda_orders[0].qty == 350


def test_compute_orders_invalid_price_skipped():
    targets = pd.Series({"NVDA": 5000})
    current = pd.Series(dtype=float)
    prices = pd.Series({"NVDA": np.nan})
    orders = compute_orders(targets, current, prices)
    assert len(orders) == 0


def test_empty_decision():
    decision = {"sa_weight": 0.7, "sa_leverage": 1.0,
                "eq_top_weights": pd.Series(dtype=float),
                "xa_hybrid_weights": pd.Series(dtype=float)}
    orders = decision_to_orders(decision, pd.Series(dtype=float), pd.Series(dtype=float))
    assert orders == []
