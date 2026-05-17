"""Tests for Sprint 5: symbol_kill_switch threading lock fix.

Verifies that concurrent block/unblock calls on different symbols
no longer corrupt each other's writes after adding per-file locks.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.symbol_kill_switch import (  # noqa: E402
    block_symbol,
    is_symbol_blocked,
    list_blocked_symbols,
    unblock_symbol,
)

pytestmark = pytest.mark.fast


def test_concurrent_blocks_preserve_all_symbols(tmp_path: Path) -> None:
    """Block 20 symbols concurrently — all must appear in final state."""
    state = tmp_path / "sks.json"
    symbols = [f"SYM{i:02d}" for i in range(20)]
    errors: list[Exception] = []

    def _block(sym: str) -> None:
        try:
            block_symbol(sym, f"test_{sym}", state_path=state)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_block, args=(s,)) for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"workers raised: {errors}"
    blocked = list_blocked_symbols(state_path=state)
    # With locking, ALL symbols must be present
    for sym in symbols:
        assert sym in blocked, f"{sym} missing after concurrent blocks"


def test_block_then_query_immediate(tmp_path: Path) -> None:
    """A blocked symbol is immediately observable, even under load."""
    state = tmp_path / "sks.json"

    # Start noise threads
    def _noise() -> None:
        for i in range(10):
            block_symbol(f"NOISE{i}", "noise", state_path=state)

    noise_threads = [threading.Thread(target=_noise) for _ in range(4)]
    for t in noise_threads:
        t.start()

    block_symbol("TARGET", "race_under_load", state_path=state)
    assert is_symbol_blocked("TARGET", state_path=state) is True

    for t in noise_threads:
        t.join(timeout=5.0)

    assert is_symbol_blocked("TARGET", state_path=state) is True


def test_unblock_removes_symbol(tmp_path: Path) -> None:
    """Unblock actually removes the symbol from state."""
    state = tmp_path / "sks.json"
    block_symbol("A", "test", state_path=state)
    block_symbol("B", "test", state_path=state)
    assert is_symbol_blocked("A", state_path=state)
    assert is_symbol_blocked("B", state_path=state)

    removed = unblock_symbol("A", state_path=state)
    assert removed is True
    assert not is_symbol_blocked("A", state_path=state)
    assert is_symbol_blocked("B", state_path=state)


def test_unblock_nonexistent_returns_false(tmp_path: Path) -> None:
    state = tmp_path / "sks.json"
    assert unblock_symbol("NOPE", state_path=state) is False
