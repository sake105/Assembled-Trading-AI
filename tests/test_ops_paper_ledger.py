"""Tests for OPS-4 paper ledger: load/save, simulate_fills, apply_fills, mark_to_market, equity_curve + profit_lock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.assembled_core.ops.paper_ledger import (
    apply_fills_to_ledger,
    load_ledger_state,
    mark_to_market_equity,
    save_ledger_state,
    simulate_fills,
    write_ledger_snapshot,
)


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_load_save_ledger_state_roundtrip(tmp_path: Path) -> None:
    """Load/save roundtrip preserves cash, positions, equity_curve."""
    path = tmp_path / "ledger_state.json"
    state: dict[str, Any] = {
        "schema_version": "paper.ledger_state.v1",
        "updated_utc": "2025-01-15T12:00:00+00:00",
        "cash": 5000.0,
        "positions": {
            "AAA": {"qty": 10.0, "avg_price": 100.0},
            "BBB": {"qty": 20.0, "avg_price": 50.0},
        },
        "equity_curve": [
            {"utc": "2025-01-14T12:00:00+00:00", "equity": 10000.0},
            {"utc": "2025-01-15T12:00:00+00:00", "equity": 10050.0},
        ],
    }
    save_ledger_state(state, path)
    assert path.exists()
    loaded = load_ledger_state(path, start_capital=10000.0)
    assert loaded["cash"] == 5000.0
    assert loaded["positions"]["AAA"]["qty"] == 10.0
    assert loaded["positions"]["AAA"]["avg_price"] == 100.0
    assert len(loaded["equity_curve"]) == 2
    assert loaded["equity_curve"][1]["equity"] == 10050.0


def test_load_ledger_state_missing_returns_fresh(tmp_path: Path) -> None:
    """Missing file returns fresh state with start_capital and empty positions."""
    loaded = load_ledger_state(tmp_path / "nonexistent.json", start_capital=20000.0)
    assert loaded["cash"] == 20000.0
    assert loaded["positions"] == {}
    assert loaded["equity_curve"] == []


def test_apply_fills_updates_cash_and_positions() -> None:
    """apply_fills_to_ledger updates cash and positions (buy adds position, sell reduces)."""
    state: dict[str, Any] = {
        "schema_version": "paper.ledger_state.v1",
        "cash": 10000.0,
        "positions": {},
        "equity_curve": [],
    }
    fills = [
        {"symbol": "A", "side": "BUY", "qty": 10.0, "price": 100.0},
        {"symbol": "B", "side": "BUY", "qty": 20.0, "price": 50.0},
    ]
    after = apply_fills_to_ledger(state, fills)
    assert after["cash"] == 10000.0 - 10 * 100 - 20 * 50  # 10000 - 1000 - 1000 = 8000
    assert after["positions"]["A"]["qty"] == 10.0
    assert after["positions"]["A"]["avg_price"] == 100.0
    assert after["positions"]["B"]["qty"] == 20.0
    assert after["positions"]["B"]["avg_price"] == 50.0

    # Sell half of A
    after2 = apply_fills_to_ledger(after, [{"symbol": "A", "side": "SELL", "qty": 5.0, "price": 102.0}])
    assert after2["positions"]["A"]["qty"] == 5.0
    assert after2["positions"]["A"]["avg_price"] == 100.0
    assert after2["cash"] == 8000 + 5 * 102  # 8510


def test_mark_to_market_equity() -> None:
    """mark_to_market_equity = cash + sum(qty * latest_price)."""
    state: dict[str, Any] = {
        "cash": 1000.0,
        "positions": {"X": {"qty": 10.0, "avg_price": 20.0}, "Y": {"qty": 5.0, "avg_price": 100.0}},
    }
    prices = pd.DataFrame({"symbol": ["X", "Y"], "close": [22.0, 110.0]})
    eq = mark_to_market_equity(state, prices)
    assert eq == 1000.0 + 10 * 22 + 5 * 110  # 1000 + 220 + 550 = 1770


def test_simulate_fills_uses_close_price() -> None:
    """simulate_fills produces fills at close price from prices_latest."""
    orders = pd.DataFrame([
        {"timestamp": pd.Timestamp("2025-01-15"), "symbol": "A", "side": "BUY", "qty": 5.0, "price": 0.0},
        {"timestamp": pd.Timestamp("2025-01-15"), "symbol": "B", "side": "SELL", "qty": 10.0, "price": 0.0},
    ])
    prices = pd.DataFrame({"symbol": ["A", "B"], "close": [100.0, 50.0]})
    fills = simulate_fills(orders, prices, None)
    assert len(fills) == 2
    by_sym = {f["symbol"]: f for f in fills}
    assert by_sym["A"]["side"] == "BUY" and by_sym["A"]["qty"] == 5.0 and by_sym["A"]["price"] == 100.0
    assert by_sym["B"]["side"] == "SELL" and by_sym["B"]["qty"] == 10.0 and by_sym["B"]["price"] == 50.0


def test_paper_run_updates_equity_curve_and_profit_lock_can_trigger(tmp_path: Path) -> None:
    """After applying fills and appending equity_curve, profit_lock can see curve and trigger."""
    from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier

    # Start ledger, apply one buy fill, mtm, append to equity_curve, save
    state = load_ledger_state(tmp_path / "ledger.json", start_capital=10000.0)
    orders = pd.DataFrame([{"symbol": "A", "side": "BUY", "qty": 10.0, "price": 100.0}])
    prices = pd.DataFrame({"symbol": ["A"], "close": [100.0]})
    fills = simulate_fills(orders, prices, None)
    state = apply_fills_to_ledger(state, fills)
    state["equity_curve"] = [
        {"utc": "2025-01-01T12:00:00+00:00", "equity": 10000.0},
        {"utc": "2025-01-02T12:00:00+00:00", "equity": 10100.0},
        {"utc": "2025-01-03T12:00:00+00:00", "equity": 10250.0},
    ]
    save_ledger_state(state, tmp_path / "ledger.json")

    # Load and build Series for profit_lock (index = 0..n-1)
    loaded = load_ledger_state(tmp_path / "ledger.json", start_capital=10000.0)
    curve = loaded["equity_curve"]
    equity_series = pd.Series([float(c["equity"]) for c in curve], dtype=float)
    # Trigger: lookback 2 bars, return (10250/10100)-1 ~ 1.49% >= 1%
    policy: dict[str, Any] = {
        "enabled": True,
        "lookback_days": 2,
        "trigger_return": 0.01,
        "multiplier_on_trigger": 0.8,
        "floor": 0.5,
        "cooldown_days": 10,
    }
    mult, _ = compute_profit_lock_multiplier(equity_series, policy, now_idx=2, state=None)
    assert mult < 1.0


def test_write_ledger_snapshot(tmp_path: Path) -> None:
    """write_ledger_snapshot produces ledger_snapshot.json with schema paper.ledger_snapshot.v1."""
    state: dict[str, Any] = {"cash": 8000.0, "positions": {"A": {"qty": 10.0, "avg_price": 100.0}}, "updated_utc": "2025-01-15T12:00:00+00:00"}
    path = write_ledger_snapshot(tmp_path, state, equity=9000.0)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "paper.ledger_snapshot.v1"
    assert data["cash"] == 8000.0
    assert data["equity"] == 9000.0
