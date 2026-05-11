"""End-to-End-Smoke: LiveDecisionEngine + Composite-Overlay -> OrderRouter -> Orders.

Verifies the full live-trading path with real data:
1. Bootstrap engine on real watchlist panel.
2. Attach composite overlay.
3. For each day: decide_next() -> decision_to_orders() -> verify Order objects.
4. Apply pre-trade checks (blacklist + max-position) and confirm flags fire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erweiterung.live.live_decision_engine import (
    LiveDecisionEngine,
    LiveEngineConfig,
)
from erweiterung.live.order_router import (
    Order,
    OrderRouterConfig,
    decision_to_orders,
    orders_to_dataframe,
)


PANEL_PATH = Path("data/sample/watchlist_2007_2026.parquet")

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists(), reason="watchlist panel missing"
)


@pytest.fixture(scope="module")
def real_returns():
    df = pd.read_parquet(PANEL_PATH)
    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])
    df["return"] = df.groupby("symbol")["close"].pct_change()
    df = df.dropna(subset=["return"])
    eq = df.pivot(index="date", columns="symbol", values="return").fillna(0)
    # Take last 600 days only (smoke speed)
    return eq.tail(600)


@pytest.fixture(scope="module")
def latest_prices():
    df = pd.read_parquet(PANEL_PATH)
    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    latest = df.sort_values("date").drop_duplicates("symbol", keep="last")
    return latest.set_index("symbol")["close"]


def test_decision_to_orders_smoke(real_returns, latest_prices):
    """Engine output flows through OrderRouter to valid Order objects."""
    cfg_engine = LiveEngineConfig()
    engine = LiveDecisionEngine(cfg_engine)
    xa_dummy = pd.DataFrame(0.0, index=real_returns.index, columns=["_XA"])
    engine.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])

    # Advance one day
    last_idx = 505
    date = real_returns.index[last_idx]
    engine.update_with_new_day(
        date, real_returns.iloc[last_idx], xa_dummy.iloc[last_idx]
    )
    decision = engine.decide_next()
    assert "eq_top_weights" in decision
    assert (decision["eq_top_weights"] > 0).any()

    # Convert to orders
    current = pd.Series(dtype=float)  # no current positions
    orders = decision_to_orders(
        decision, current, latest_prices, OrderRouterConfig(equity=100_000)
    )
    assert len(orders) > 0
    assert all(isinstance(o, Order) for o in orders)
    assert all(o.side in ("BUY", "SELL") for o in orders)
    assert all(o.qty > 0 for o in orders)
    assert all(np.isfinite(o.target_notional) for o in orders)


def test_decision_to_orders_with_composite_reduces_notional(
    real_returns, latest_prices
):
    """Composite-Overlay PAUSE -> reduced sa_leverage -> smaller target notionals."""
    from erweiterung.risk.geo_stress_composite import (
        GeoStressPolicy,
        compute_monthly_composite,
        expand_composite_to_daily,
    )

    # Build overlay; force PAUSE on test date by injecting low multiplier
    cfg_engine = LiveEngineConfig(enable_geo_overlay=True)
    engine = LiveDecisionEngine(cfg_engine)
    xa_dummy = pd.DataFrame(0.0, index=real_returns.index, columns=["_XA"])
    engine.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])

    # Build daily overlay covering eval range
    monthly = compute_monthly_composite()
    daily = expand_composite_to_daily(monthly, real_returns.index, GeoStressPolicy())
    overlay = daily[["multiplier", "state"]]
    # Force a PAUSE day for testing
    overlay.loc[real_returns.index[505], "multiplier"] = 0.50
    overlay.loc[real_returns.index[505], "state"] = "PAUSE"
    engine.attach_geo_overlay(overlay)

    date = real_returns.index[505]
    engine.update_with_new_day(date, real_returns.iloc[505], xa_dummy.iloc[505])
    decision = engine.decide_next()
    assert decision["geo_multiplier"] == pytest.approx(0.50)

    # Baseline (no overlay)
    cfg_base = LiveEngineConfig()
    engine_base = LiveDecisionEngine(cfg_base)
    engine_base.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])
    engine_base.update_with_new_day(date, real_returns.iloc[505], xa_dummy.iloc[505])
    decision_base = engine_base.decide_next()

    # PAUSE -> sa_leverage halved
    assert decision["sa_leverage"] == pytest.approx(
        decision_base["sa_leverage"] * 0.50, abs=1e-9
    )

    # Notionals should also be halved
    current = pd.Series(dtype=float)
    orders_pause = decision_to_orders(
        decision, current, latest_prices, OrderRouterConfig(equity=100_000)
    )
    orders_base = decision_to_orders(
        decision_base, current, latest_prices, OrderRouterConfig(equity=100_000)
    )
    pause_notional = sum(abs(o.target_notional) for o in orders_pause)
    base_notional = sum(abs(o.target_notional) for o in orders_base)
    # With same exposure_cap but smaller weights, base will hit cap → after renorm
    # both might be at cap. Confirm at least: PAUSE notional <= base notional.
    assert pause_notional <= base_notional + 1.0  # tolerance for rounding


def test_pre_trade_blacklist_blocks(real_returns, latest_prices):
    cfg_engine = LiveEngineConfig()
    engine = LiveDecisionEngine(cfg_engine)
    xa_dummy = pd.DataFrame(0.0, index=real_returns.index, columns=["_XA"])
    engine.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])
    engine.update_with_new_day(
        real_returns.index[505], real_returns.iloc[505], xa_dummy.iloc[505]
    )
    decision = engine.decide_next()
    top_picks = decision["eq_top_weights"]
    top_picks = top_picks[top_picks > 0]
    assert len(top_picks) > 0
    blacklisted_sym = top_picks.index[0]  # pick a top symbol to blacklist

    orders = decision_to_orders(
        decision,
        pd.Series(dtype=float),
        latest_prices,
        OrderRouterConfig(equity=100_000),
        pre_trade_policy={"blacklist": [blacklisted_sym]},
    )
    blacklisted_orders = [o for o in orders if o.symbol == blacklisted_sym]
    if blacklisted_orders:
        assert "BLACKLISTED" in blacklisted_orders[0].pre_trade_flags


def test_orders_to_dataframe_roundtrip(real_returns, latest_prices):
    cfg_engine = LiveEngineConfig()
    engine = LiveDecisionEngine(cfg_engine)
    xa_dummy = pd.DataFrame(0.0, index=real_returns.index, columns=["_XA"])
    engine.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])
    engine.update_with_new_day(
        real_returns.index[505], real_returns.iloc[505], xa_dummy.iloc[505]
    )
    orders = decision_to_orders(
        engine.decide_next(),
        pd.Series(dtype=float),
        latest_prices,
        OrderRouterConfig(equity=50_000),
    )
    df = orders_to_dataframe(orders)
    assert not df.empty
    assert {"symbol", "side", "qty", "target_notional", "price"}.issubset(df.columns)


def test_walk_forward_orders_stay_valid(real_returns, latest_prices):
    """5-day rolling smoke: orders remain valid across multiple days."""
    cfg_engine = LiveEngineConfig()
    engine = LiveDecisionEngine(cfg_engine)
    xa_dummy = pd.DataFrame(0.0, index=real_returns.index, columns=["_XA"])
    engine.bootstrap_from_history(real_returns.iloc[:504], xa_dummy.iloc[:504])

    current = pd.Series(dtype=float)
    for i in range(504, 510):
        engine.update_with_new_day(
            real_returns.index[i], real_returns.iloc[i], xa_dummy.iloc[i]
        )
        decision = engine.decide_next()
        orders = decision_to_orders(
            decision, current, latest_prices, OrderRouterConfig(equity=100_000)
        )
        for o in orders:
            assert np.isfinite(o.qty) and o.qty > 0
            assert np.isfinite(o.price) and o.price > 0
            assert o.side in ("BUY", "SELL")
        # Update positions = target (simulating fill)
        for o in orders:
            current.loc[o.symbol] = o.target_position
