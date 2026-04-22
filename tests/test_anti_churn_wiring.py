"""Tests for anti-churn wiring in trading_cycle.py.

Covers Step 3.6 (ranking_hysteresis) and Step 6.6 (deadzone + rebalance_filter)
that were wired in wave-4 of Part B.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.paper.ranking_hysteresis import apply_ranking_hysteresis
from src.assembled_core.paper.deadzone_rebalance import filter_deadzone_orders
from src.assembled_core.paper.rebalance_filter import filter_small_rebalances


# ---------------------------------------------------------------------------
# ranking_hysteresis
# ---------------------------------------------------------------------------

def _make_signals(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "direction": ["LONG"] * n,
        "score": [1.0 - i * 0.1 for i in range(n)],
    })


def test_ranking_hysteresis_blocks_new_entries_beyond_entry_n():
    sig = _make_signals(8)
    out, meta = apply_ranking_hysteresis(sig, held_symbols=set(), entry_n=5, hold_n=7)
    flat = out[out["direction"] == "FLAT"]["symbol"].tolist()
    assert len(flat) == 3  # S5, S6, S7 blocked
    assert meta["blocked_entry"] == 3


def test_ranking_hysteresis_keeps_held_symbols_within_hold_n():
    sig = _make_signals(8)
    held = {"S5", "S6"}  # rank 6 and 7 — within hold_n=7
    out, meta = apply_ranking_hysteresis(sig, held_symbols=held, entry_n=5, hold_n=7)
    # S5 held, rank=6 ≤ hold_n=7 → kept LONG
    assert out.loc[out["symbol"] == "S5", "direction"].values[0] == "LONG"
    assert meta["kept_by_hysteresis"] == 2


def test_ranking_hysteresis_drops_held_beyond_hold_n():
    sig = _make_signals(9)
    held = {"S8"}  # rank 9 > hold_n=7 → should be dropped to FLAT
    out, meta = apply_ranking_hysteresis(sig, held_symbols=held, entry_n=5, hold_n=7)
    assert out.loc[out["symbol"] == "S8", "direction"].values[0] == "FLAT"


def test_ranking_hysteresis_empty_signals_passthrough():
    empty = pd.DataFrame(columns=["symbol", "direction", "score"])
    out, meta = apply_ranking_hysteresis(empty, held_symbols=set())
    assert out.empty


# ---------------------------------------------------------------------------
# deadzone_rebalance
# ---------------------------------------------------------------------------

def _make_orders(symbols, qtys) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": symbols,
        "side": ["buy"] * len(symbols),
        "qty": qtys,
        "price": [100.0] * len(symbols),
        "timestamp": [0] * len(symbols),
    })


def test_deadzone_drops_small_relative_change():
    orders = _make_orders(["A"], [2.0])        # order_qty=2 vs current=100 → ratio=0.02
    positions = pd.DataFrame({"symbol": ["A"], "qty": [100.0]})
    out, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
    assert stats["orders_dropped"] == 1
    assert out.empty


def test_deadzone_keeps_large_relative_change():
    orders = _make_orders(["A"], [20.0])       # ratio=0.2 > 0.05
    positions = pd.DataFrame({"symbol": ["A"], "qty": [100.0]})
    out, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
    assert stats["orders_dropped"] == 0


def test_deadzone_new_position_always_passes():
    orders = _make_orders(["NEW"], [1.0])
    positions = pd.DataFrame({"symbol": ["OTHER"], "qty": [100.0]})
    out, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.99)
    assert stats["orders_dropped"] == 0


def test_deadzone_no_positions_passthrough():
    orders = _make_orders(["A", "B"], [1.0, 2.0])
    out, stats = filter_deadzone_orders(orders, current_positions=None)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# rebalance_filter (min_notional)
# ---------------------------------------------------------------------------

def test_rebalance_filter_drops_below_notional():
    orders = _make_orders(["SMALL"], [4.0])     # 4 × 100 = $400 < $500
    out, stats = filter_small_rebalances(orders, min_notional=500.0)
    assert stats["orders_dropped"] == 1


def test_rebalance_filter_keeps_above_notional():
    orders = _make_orders(["BIG"], [6.0])       # 6 × 100 = $600 ≥ $500
    out, stats = filter_small_rebalances(orders, min_notional=500.0)
    assert stats["orders_dropped"] == 0


def test_rebalance_filter_uses_prices_lookup():
    orders = pd.DataFrame({
        "symbol": ["A"],
        "side": ["buy"],
        "qty": [10.0],
        "price": [0.0],    # price 0 → must use fallback prices
        "timestamp": [0],
    })
    prices = pd.DataFrame({"symbol": ["A"], "close": [60.0]})
    out, stats = filter_small_rebalances(orders, min_notional=500.0, prices=prices)
    # 10 × 60 = $600 ≥ $500 → kept
    assert stats["orders_dropped"] == 0


def test_rebalance_filter_zero_min_notional_passthrough():
    orders = _make_orders(["A"], [1.0])
    out, stats = filter_small_rebalances(orders, min_notional=0.0)
    assert len(out) == 1
