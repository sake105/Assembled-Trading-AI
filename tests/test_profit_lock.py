"""Tests for Soft Profit Lock overlay (INT-6.2): trigger, cooldown, combined multiplier."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier

pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_profit_lock_triggers_on_return() -> None:
    """When lookback return >= trigger_return, multiplier_on_trigger is returned (clamped by floor)."""
    # +8% over 20 bars: equity 100 -> 108
    equity = pd.Series([100.0] * 20 + [108.0])
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    mult, state = compute_profit_lock_multiplier(equity, policy, now_idx=20, state=None)
    assert mult == pytest.approx(0.80)
    assert state.get("trigger_idx") == 20


def test_profit_lock_no_trigger_below_return() -> None:
    """When lookback return < trigger_return, 1.0 is returned."""
    equity = pd.Series([100.0] * 20 + [107.0])  # +7%
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    mult, state = compute_profit_lock_multiplier(equity, policy, now_idx=20, state=None)
    assert mult == pytest.approx(1.0)
    assert "trigger_idx" not in state or state.get("trigger_idx") is None


def test_profit_lock_cooldown_applies() -> None:
    """After trigger, multiplier stays for cooldown_days then re-evaluates."""
    # Bars 0..19 = 100, bar 20 = 108 (+8%), then flat 108
    equity = pd.Series([100.0] * 20 + [108.0] * 15)
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    state: dict = {}
    mult0, state = compute_profit_lock_multiplier(
        equity, policy, now_idx=20, state=state
    )
    assert mult0 == pytest.approx(0.80)
    assert state.get("trigger_idx") == 20
    # Within cooldown: bar 25
    mult1, state = compute_profit_lock_multiplier(
        equity, policy, now_idx=25, state=state
    )
    assert mult1 == pytest.approx(0.80)
    # Bar 31: 31 - 20 = 11 > cooldown_days(10); re-check. Lookback 31-20=11 -> start_idx=11, now_val=108, start_val=100, ret=0.08 -> re-trigger
    mult2, state = compute_profit_lock_multiplier(
        equity, policy, now_idx=31, state=state
    )
    assert mult2 == pytest.approx(0.80)
    assert state.get("trigger_idx") == 31


def test_profit_lock_cooldown_expires_then_no_retrigger() -> None:
    """After cooldown expires, if lookback return < trigger, return 1.0."""
    # Need 32 points: indices 0..31. start_idx=11, now_idx=31 -> ret = equity[31]/equity[11] - 1.
    # Set equity[11]=108, equity[31]=107 -> ret = 107/108 - 1 < 0.08
    equity = pd.Series([100.0] * 11 + [108.0] * 20 + [107.0])
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    state = {"trigger_idx": 20}
    mult, state = compute_profit_lock_multiplier(
        equity, policy, now_idx=31, state=state
    )
    assert mult == pytest.approx(1.0)
    assert "trigger_idx" not in state or state.get("trigger_idx") is None


def test_combined_multiplier_clamped() -> None:
    """Profit lock multiplier is clamped to [floor, 1.0]; combined = geo * profit_lock."""
    equity = pd.Series([100.0] * 20 + [108.0])
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    mult, _ = compute_profit_lock_multiplier(equity, policy, now_idx=20, state=None)
    assert 0.50 <= mult <= 1.0
    assert mult == pytest.approx(0.80)
    # Simulate combined: final = geo * profit_lock
    geo_mult = 0.70
    final = geo_mult * mult
    assert final == pytest.approx(0.56)


def test_profit_lock_disabled_returns_one() -> None:
    """When policy enabled is False, always 1.0."""
    equity = pd.Series([100.0] * 20 + [108.0])
    policy = {"enabled": False, "lookback_days": 20, "trigger_return": 0.08}
    mult, _ = compute_profit_lock_multiplier(equity, policy, now_idx=20, state=None)
    assert mult == pytest.approx(1.0)


def test_profit_lock_curve_too_short_returns_one() -> None:
    """When equity curve is too short, return 1.0."""
    equity = pd.Series([100.0, 108.0])
    policy = {"enabled": True, "lookback_days": 20, "trigger_return": 0.08}
    mult, _ = compute_profit_lock_multiplier(equity, policy, now_idx=1, state=None)
    assert mult == pytest.approx(1.0)


def test_equity_curve_in_ctx_triggers_profit_lock() -> None:
    """With equity_curve and equity_curve_index set in ctx (e.g. from backtest), profit_lock can trigger."""
    from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
    from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle

    def _signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        ts = prices_df["timestamp"].iloc[0]
        syms = prices_df["symbol"].unique().tolist()
        return pd.DataFrame(
            {
                "timestamp": [ts] * len(syms),
                "symbol": syms,
                "direction": ["LONG"] * len(syms),
                "score": [1.0] * len(syms),
            }
        )

    def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions(
            signals, total_capital=capital, equal_weight=True
        )

    # Minimal prices so cycle runs
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-15", tz="UTC")] * 2,
            "symbol": ["A", "B"],
            "close": [10.0, 20.0],
        }
    )
    # Up 8% over 20 bars: triggers profit_lock (policy from configs/policy.yaml has profit_lock.enabled)
    equity_curve = pd.Series([100.0] * 20 + [108.0])
    ctx = TradingContext(
        prices=prices,
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        capital=10000.0,
        equity_curve=equity_curve,
        equity_curve_index=20,
    )
    result = run_trading_cycle(ctx)
    assert result.status == "success"
    pl = result.meta.get("profit_lock") or {}
    assert "multiplier" in pl
    assert pl["multiplier"] != 1.0
    assert pl["multiplier"] == pytest.approx(0.80)


def test_profit_lock_state_roundtrip_keeps_cooldown() -> None:
    """State passed from run1 (trigger) into run2 (within cooldown) keeps multiplier_on_trigger."""
    equity = pd.Series([100.0] * 20 + [108.0] * 15)
    policy = {
        "enabled": True,
        "lookback_days": 20,
        "trigger_return": 0.08,
        "multiplier_on_trigger": 0.80,
        "floor": 0.50,
        "cooldown_days": 10,
    }
    # Run1: trigger at bar 20 -> returns state with trigger_idx=20
    mult1, state_after_run1 = compute_profit_lock_multiplier(
        equity, policy, now_idx=20, state=None
    )
    assert mult1 == pytest.approx(0.80)
    assert state_after_run1.get("trigger_idx") == 20
    # Run2: same state, bar 25 (within cooldown) -> still multiplier_on_trigger
    mult2, _ = compute_profit_lock_multiplier(
        equity, policy, now_idx=25, state=state_after_run1
    )
    assert mult2 == pytest.approx(0.80)
