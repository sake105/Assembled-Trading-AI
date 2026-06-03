"""Regression: ``pre_trade_gate`` must FAIL CLOSED on corrupt kill-switch state.

Sibling finding (Bucket-B): ``is_kill_switch_engaged()`` now fails closed on a
present-but-unreadable persistent state file, but a SECOND kill-switch decision
path — ``pre_trade_gate``'s ``enforce_kill_switch`` branch — read
``get_kill_switch_state()["engaged"]`` directly. On a corrupt file
``get_kill_switch_state`` flattens (via ``_read_state() -> {}``) to
``{engaged: False, throttle_pct: 1.0}`` and does NOT raise, so the block
condition was False and the ``except`` never fired -> corrupt-present state
FAILED OPEN here. This path is LATENT (no ``src/`` caller of ``pre_trade_gate``
today) but is pinned fail-closed so it cannot be wired live while fail-open.

Contract pinned here (mirrors tests/test_kill_switch_corrupt_fail_closed.py
setup — write a corrupt state file at the env-resolved path):
  - present-but-corrupt persistent state -> PreTradeGateBlocked + ERROR log,
    indistinguishable from a fully-engaged kill-switch (check="kill_switch",
    reasons=["kill_switch_engaged"]),
  - MISSING state file -> NOT blocked (orders pass, unchanged),
  - healthy ENGAGED (throttle_pct=0) -> blocked (unchanged),
  - healthy DISENGAGED -> NOT blocked (unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.execution.pre_trade_checks import (
    PreTradeGateBlocked,
    pre_trade_gate,
)


@pytest.fixture
def _isolated_state(monkeypatch, tmp_path: Path) -> Path:
    """Point the kill switch at an isolated tmp state file with NO other source.

    Clears env/sentinel sources so the persistent-state path is the only signal
    under test (same setup as test_kill_switch_corrupt_fail_closed.py).
    """
    state = tmp_path / "kill_switch_state.json"
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(state))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".no_sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    return state


def _orders() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": "AAPL", "side": "BUY", "qty": 1.0, "price": 100.0}])


def test_corrupt_state_blocks_with_killswitch_semantics(
    _isolated_state: Path, caplog
) -> None:
    """Present-but-corrupt persistent state -> PreTradeGateBlocked + ERROR log.

    The block must be indistinguishable from a fully-engaged kill-switch so a
    downstream caller handles it identically (fail-closed).
    """
    _isolated_state.write_text("}{ this is not json @@@", encoding="utf-8")
    with caplog.at_level("ERROR"):
        with pytest.raises(PreTradeGateBlocked) as ei:
            pre_trade_gate(_orders())
    assert ei.value.check == "kill_switch"
    assert "kill_switch_engaged" in ei.value.reasons
    assert any(
        "corrupt" in r.message.lower() or "unreadable" in r.message.lower()
        for r in caplog.records
    ), "expected an ERROR log naming the corrupt/unreadable kill-switch state"


def test_corrupt_non_object_json_blocks(_isolated_state: Path) -> None:
    """Valid JSON but not an object (list) -> not a state doc -> blocked."""
    _isolated_state.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PreTradeGateBlocked) as ei:
        pre_trade_gate(_orders())
    assert ei.value.check == "kill_switch"


def test_missing_state_not_blocked(_isolated_state: Path) -> None:
    """A legitimately MISSING state file -> disengaged -> orders pass (unchanged)."""
    assert not _isolated_state.exists()
    filtered = pre_trade_gate(_orders())
    assert len(filtered) == 1


def test_healthy_engaged_state_blocked(_isolated_state: Path) -> None:
    """Healthy engaged state with throttle_pct=0 -> blocked (unchanged)."""
    _isolated_state.write_text(
        json.dumps({"engaged": True, "throttle_pct": 0.0, "reason": "test"}),
        encoding="utf-8",
    )
    with pytest.raises(PreTradeGateBlocked) as ei:
        pre_trade_gate(_orders())
    assert ei.value.check == "kill_switch"


def test_healthy_disengaged_state_not_blocked(_isolated_state: Path) -> None:
    """Healthy disengaged state -> orders pass (unchanged)."""
    _isolated_state.write_text(json.dumps({"engaged": False}), encoding="utf-8")
    filtered = pre_trade_gate(_orders())
    assert len(filtered) == 1


def test_enforce_kill_switch_false_skips_corrupt_check(_isolated_state: Path) -> None:
    """With enforce_kill_switch=False the corrupt pre-check is skipped entirely."""
    _isolated_state.write_text("}{ not json", encoding="utf-8")
    filtered = pre_trade_gate(_orders(), enforce_kill_switch=False)
    assert len(filtered) == 1
