"""A3: symbol_kill_switch _write_state must be atomic (tmp+rename pattern)."""
from __future__ import annotations

import json
import pytest


@pytest.mark.fast
def test_write_state_uses_atomic_helper():
    """_write_state must call atomic_write_json_with_retry, not path.write_text."""
    import inspect
    from src.assembled_core.execution import symbol_kill_switch as ks
    src = inspect.getsource(ks._write_state)
    assert "atomic_write_json_with_retry" in src, "_write_state must use atomic helper"
    assert "write_text" not in src, "_write_state must NOT use plain write_text"


@pytest.mark.fast
def test_write_state_crash_leaves_no_corrupt_file(tmp_path, monkeypatch):
    """If the rename step fails, the original state file stays intact."""
    import os
    from src.assembled_core.execution.symbol_kill_switch import _write_state

    # Write known-good initial state
    state_file = tmp_path / "state.json"
    good_state = {"blocked": {}}
    state_file.write_text(json.dumps(good_state), encoding="utf-8")

    # Patch os.replace to simulate crash during rename
    original_replace = os.replace

    def boom(src, dst):
        raise OSError("Simulated crash during rename")

    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError):
        _write_state(state_file, {"blocked": {"AAPL": {"reason": "test"}}})

    # Original file must still be valid JSON (or not exist — but NOT corrupted)
    if state_file.exists():
        content = state_file.read_text()
        parsed = json.loads(content)  # must not raise
        assert parsed == good_state, "Original state must be preserved after crash"


@pytest.mark.fast
def test_block_and_unblock_roundtrip(tmp_path):
    """block_symbol + unblock_symbol roundtrip produces correct state."""
    from src.assembled_core.execution.symbol_kill_switch import (
        block_symbol,
        unblock_symbol,
        list_blocked_symbols,
        is_symbol_blocked,
    )
    state_path = tmp_path / "ks.json"
    block_symbol("TSLA", "test reason", state_path=state_path)
    assert is_symbol_blocked("TSLA", state_path=state_path)
    assert "TSLA" in list_blocked_symbols(state_path=state_path)
    unblock_symbol("TSLA", state_path=state_path)
    assert not is_symbol_blocked("TSLA", state_path=state_path)
