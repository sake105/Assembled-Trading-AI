"""Tests for execution/symbol_kill_switch.py (Sprint 4 / Plan C27)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.symbol_kill_switch import (  # noqa: E402
    block_symbol,
    filter_orders_by_symbol_blocks,
    filter_orders_from_policy,
    is_symbol_blocked,
    list_blocked_symbols,
    unblock_symbol,
)


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "qty": 10.0, "price": 100.0},
            {"symbol": "BBB", "qty": 20.0, "price": 50.0},
            {"symbol": "CCC", "qty": 5.0, "price": 200.0},
        ]
    )


def test_block_and_list(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    block_symbol("BBB", "earnings blackout", state_path=state)

    blocked = list_blocked_symbols(state_path=state)
    assert set(blocked.keys()) == {"AAA", "BBB"}
    assert blocked["AAA"]["reason"] == "halted"
    assert "blocked_at" in blocked["AAA"]


def test_is_symbol_blocked(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    assert is_symbol_blocked("AAA", state_path=state) is False
    block_symbol("AAA", "halted", state_path=state)
    assert is_symbol_blocked("AAA", state_path=state) is True


def test_unblock(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    assert unblock_symbol("AAA", state_path=state) is True
    assert unblock_symbol("AAA", state_path=state) is False
    assert list_blocked_symbols(state_path=state) == {}


def test_filter_orders_drops_blocked_symbols(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("BBB", "delisting", state_path=state)

    filtered, reasons = filter_orders_by_symbol_blocks(_orders(), state_path=state)
    assert list(filtered["symbol"]) == ["AAA", "CCC"]
    assert len(reasons) == 1
    assert "BBB" in reasons[0]
    assert "delisting" in reasons[0]


def test_filter_orders_no_blocks_is_passthrough(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"  # file does not exist yet
    orders = _orders()
    filtered, reasons = filter_orders_by_symbol_blocks(orders, state_path=state)
    assert len(filtered) == len(orders)
    assert reasons == []


def test_filter_orders_empty_frame(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    empty = pd.DataFrame(columns=["symbol", "qty", "price"])
    filtered, reasons = filter_orders_by_symbol_blocks(empty, state_path=state)
    assert filtered.empty
    assert reasons == []


def test_state_file_persists_across_calls(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    # Second process-style read: should still see the block
    assert is_symbol_blocked("AAA", state_path=state) is True


def test_corrupt_state_file_recovers(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    state.write_text("not valid json", encoding="utf-8")
    # Should not crash; treat as empty
    assert list_blocked_symbols(state_path=state) == {}


def test_from_policy_disabled_is_passthrough(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    orders = _orders()
    filtered, reasons = filter_orders_from_policy(
        orders, {"symbol_kill_switch": {"enabled": False}}, state_path=state
    )
    assert len(filtered) == len(orders)
    assert reasons == []


def test_from_policy_enabled_applies_blocks(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    block_symbol("AAA", "halted", state_path=state)
    orders = _orders()
    filtered, reasons = filter_orders_from_policy(
        orders, {"symbol_kill_switch": {"enabled": True}}, state_path=state
    )
    assert list(filtered["symbol"]) == ["BBB", "CCC"]
    assert len(reasons) == 1
