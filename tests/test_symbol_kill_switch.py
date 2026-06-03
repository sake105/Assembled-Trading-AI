"""Tests for execution/symbol_kill_switch.py (Sprint 4 / Plan C27)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

import pytest  # noqa: E402

import src.assembled_core.execution.kill_switch as kill_switch_mod  # noqa: E402
from src.assembled_core.execution.symbol_kill_switch import (  # noqa: E402
    block_symbol,
    filter_orders_by_symbol_blocks,
    filter_orders_from_policy,
    filter_orders_with_kill_switches,
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


# ----------------------------------------------------------------------------
# B-exec-3: filter_orders_with_kill_switches must FAIL-CLOSED.
# ----------------------------------------------------------------------------


def test_unified_filter_disengaged_passes_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal path (switch disengaged, no per-symbol blocks): orders unchanged."""
    state = tmp_path / "sks.json"
    monkeypatch.setattr(kill_switch_mod, "is_kill_switch_engaged", lambda: False)
    orders = _orders()
    out = filter_orders_with_kill_switches(orders, state_path=state)
    assert list(out["symbol"]) == ["AAA", "BBB", "CCC"]


def test_unified_filter_engaged_blocks_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Genuine engaged kill switch: returns an empty frame (all blocked)."""
    state = tmp_path / "sks.json"
    monkeypatch.setattr(kill_switch_mod, "is_kill_switch_engaged", lambda: True)
    out = filter_orders_with_kill_switches(_orders(), state_path=state)
    assert out.empty
    # same column shape as input (iloc[0:0] preserves schema)
    assert list(out.columns) == ["symbol", "qty", "price"]


def test_unified_filter_state_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """B-exec-3: if the kill-switch state cannot be read (raises), block ALL
    orders (fail-closed) and log at ERROR — never fail-open."""
    state = tmp_path / "sks.json"

    def _boom() -> bool:
        raise OSError("state file unreadable")

    monkeypatch.setattr(kill_switch_mod, "is_kill_switch_engaged", _boom)

    with caplog.at_level(logging.ERROR):
        out = filter_orders_with_kill_switches(_orders(), state_path=state)

    # All orders blocked (empty frame), same shape as a genuine engaged switch.
    assert out.empty
    assert list(out.columns) == ["symbol", "qty", "price"]
    # ERROR (not debug) names the could-not-determine / fail-closed condition.
    assert any(
        r.levelno == logging.ERROR and "fail-closed" in r.getMessage()
        for r in caplog.records
    )
