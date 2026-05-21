"""§9.6 (b) Phase-2 pre-condition (i) regression guards: trend_baseline
strategy module exit-discipline.

Tests the new src/assembled_core/strategies/trend_baseline.py:
  - compute_signals / compute_target_positions interfaces
  - check_exit_signals: stop_loss, trailing_stop, take_profit gates with
    proper priority (stop_loss > trailing > take_profit) and HWM tracking

Without exit-discipline, trend_baseline would have only LONG→FLAT MA-flip
as a position-closing mechanism — unacceptable risk for the $91k pilot
already at -7.5%. Phase-2 promotion is gated on these tests passing.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.strategies.trend_baseline import (
    check_exit_signals,
    compute_signals,
    compute_target_positions,
)


def _make_uptrend_prices() -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D", tz="UTC")
    rows = []
    for sym, base in [("AAPL", 100.0), ("MSFT", 200.0)]:
        for i, d in enumerate(dates):
            close = base * (1 + 0.002 * i)
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_compute_signals_produces_canonical_schema():
    sig = compute_signals(_make_uptrend_prices(), ma_fast=20, ma_slow=60)
    for col in ("timestamp", "symbol", "direction", "score"):
        assert col in sig.columns
    assert (sig["direction"] == "LONG").any()


def test_compute_target_positions_equal_weight_on_longs():
    sig = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "direction": ["LONG", "LONG", "FLAT"],
            "score": [0.9, 0.8, 0.0],
        }
    )
    targets = compute_target_positions(sig, capital=100_000.0)
    assert set(targets.columns) == {"symbol", "target_weight", "target_qty"}
    assert len(targets) == 2  # only LONGs
    assert abs(targets["target_weight"].sum() - 1.0) < 1e-9


def test_compute_target_positions_respects_max_positions_by_score():
    sig = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "direction": ["LONG"] * 4,
            "score": [0.1, 0.9, 0.5, 0.7],
        }
    )
    targets = compute_target_positions(sig, capital=100_000.0, max_positions=2)
    assert len(targets) == 2
    assert set(targets["symbol"]) == {"BBB", "DDD"}  # top-2 by score


def test_check_exit_signals_stop_loss_triggers_at_threshold():
    positions = {"AAA": {"qty": 10.0, "avg_price": 100.0}}
    prices_latest = pd.DataFrame({"symbol": ["AAA"], "close": [91.0]})  # -9%
    exits = check_exit_signals(
        positions,
        prices_latest,
        strategy_cfg={
            "stop_loss_pct": 0.08,
            "trailing_stop_pct": 0.10,
            "take_profit_pct": 0.15,
        },
    )
    assert len(exits) == 1
    assert exits.iloc[0]["symbol"] == "AAA"
    assert exits.iloc[0]["direction"] == "FLAT"
    assert "stop_loss" in exits.iloc[0]["exit_reason"]
    assert exits.iloc[0]["exit_qty_pct"] == 1.0


def test_check_exit_signals_trailing_stop_uses_hwm():
    """Position up to hwm=110 then dropped to 98 → 98 < 110*(1-0.10)=99 → exit."""
    positions = {
        "AAA": {"qty": 10.0, "avg_price": 100.0, "hwm": 110.0},
    }
    prices_latest = pd.DataFrame({"symbol": ["AAA"], "close": [98.0]})
    exits = check_exit_signals(
        positions,
        prices_latest,
        strategy_cfg={"stop_loss_pct": 0.0, "trailing_stop_pct": 0.10},
    )
    assert len(exits) == 1
    assert "trailing_stop" in exits.iloc[0]["exit_reason"]
    assert "hwm=110" in exits.iloc[0]["exit_reason"]


def test_check_exit_signals_take_profit_partial_50pct():
    positions = {"AAA": {"qty": 10.0, "avg_price": 100.0}}
    prices_latest = pd.DataFrame({"symbol": ["AAA"], "close": [120.0]})
    exits = check_exit_signals(
        positions,
        prices_latest,
        strategy_cfg={
            "take_profit_pct": 0.15,
            "stop_loss_pct": 0.0,
            "trailing_stop_pct": 0.0,
        },
    )
    assert len(exits) == 1
    assert "take_profit" in exits.iloc[0]["exit_reason"]
    assert exits.iloc[0]["exit_qty_pct"] == 0.5


def test_check_exit_signals_stop_loss_has_priority_over_trailing():
    """When both fire, stop_loss wins (first in sequence)."""
    positions = {
        "AAA": {"qty": 10.0, "avg_price": 100.0, "hwm": 110.0},
    }
    # Price 90: stop_loss triggers (90 <= 100*0.92), trailing also (90 <= 110*0.90)
    prices_latest = pd.DataFrame({"symbol": ["AAA"], "close": [90.0]})
    exits = check_exit_signals(
        positions,
        prices_latest,
        strategy_cfg={"stop_loss_pct": 0.08, "trailing_stop_pct": 0.10},
    )
    assert len(exits) == 1
    assert "stop_loss" in exits.iloc[0]["exit_reason"]


def test_check_exit_signals_empty_positions_returns_empty():
    exits = check_exit_signals(
        {}, pd.DataFrame({"symbol": [], "close": []}), strategy_cfg={}
    )
    assert exits.empty


def test_check_exit_signals_zero_pcts_disable_gates():
    positions = {"AAA": {"qty": 10.0, "avg_price": 100.0}}
    prices_latest = pd.DataFrame({"symbol": ["AAA"], "close": [50.0]})  # -50%
    exits = check_exit_signals(
        positions,
        prices_latest,
        strategy_cfg={
            "stop_loss_pct": 0.0,
            "trailing_stop_pct": 0.0,
            "take_profit_pct": 0.0,
        },
    )
    # All gates disabled → no exit even on catastrophic drop
    assert exits.empty
