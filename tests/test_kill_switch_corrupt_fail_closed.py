"""Regression: corrupt/unreadable persistent kill-switch state must FAIL CLOSED.

Review finding (Bucket-B): the symbol_kill_switch wrapper already fails closed
when ``is_kill_switch_engaged()`` *raises*, but a layer below
``kill_switch._read_state()`` caught a corrupt/unreadable state file and returned
``{}`` -> ``{}.get("engaged") -> False`` -> NO raise -> a CORRUPT kill-switch
state silently FAILED OPEN (all orders passed).

These tests pin the safe contract:
  - present-but-corrupt persistent state file -> is_kill_switch_engaged() True
    (fail-closed / blocked) + ERROR log,
  - MISSING state file -> False (disengaged, unchanged),
  - healthy engaged state -> True,
  - healthy disengaged state -> False,
  - end-to-end: filter_orders_with_kill_switches() blocks ALL orders when the
    persistent state file is corrupt (the deep fix flows through the wrapper).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.execution.kill_switch import is_kill_switch_engaged
from src.assembled_core.execution.symbol_kill_switch import (
    filter_orders_with_kill_switches,
)


@pytest.fixture
def _isolated_state(monkeypatch, tmp_path: Path) -> Path:
    """Point the kill switch at an isolated tmp state file with NO other source.

    Clears env/sentinel sources so the persistent-state path is the only signal
    under test.
    """
    state = tmp_path / "kill_switch_state.json"
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(state))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".no_sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    return state


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 20.0, "price": 50.0},
        ]
    )


# ---------------------------------------------------------------------------
# Deep layer: is_kill_switch_engaged() on the persistent state file
# ---------------------------------------------------------------------------


def test_missing_state_file_disengaged(_isolated_state: Path) -> None:
    """A legitimately MISSING state file -> never engaged -> disengaged."""
    assert not _isolated_state.exists()
    assert is_kill_switch_engaged() is False


def test_healthy_disengaged_state_is_false(_isolated_state: Path) -> None:
    _isolated_state.write_text(json.dumps({"engaged": False}), encoding="utf-8")
    assert is_kill_switch_engaged() is False


def test_healthy_engaged_state_is_true(_isolated_state: Path) -> None:
    _isolated_state.write_text(
        json.dumps({"engaged": True, "throttle_pct": 0.0, "reason": "test"}),
        encoding="utf-8",
    )
    assert is_kill_switch_engaged() is True


def test_corrupt_garbage_state_fails_closed(_isolated_state: Path, caplog) -> None:
    """Present-but-unparseable (garbage bytes) -> fail-closed (engaged) + ERROR."""
    _isolated_state.write_text("}{ this is not json @@@", encoding="utf-8")
    with caplog.at_level("ERROR"):
        engaged = is_kill_switch_engaged()
    assert engaged is True
    assert any(
        "unreadable" in r.message.lower() or "corrupt" in r.message.lower()
        for r in caplog.records
    ), "expected an ERROR log naming the unreadable/corrupt state"


def test_corrupt_truncated_json_fails_closed(_isolated_state: Path) -> None:
    """Present-but-truncated JSON (incomplete object) -> fail-closed (engaged)."""
    _isolated_state.write_text('{"engaged": tr', encoding="utf-8")
    assert is_kill_switch_engaged() is True


def test_non_object_json_fails_closed(_isolated_state: Path) -> None:
    """Valid JSON but not an object (e.g. a list) -> not a state doc -> fail-closed."""
    _isolated_state.write_text("[1, 2, 3]", encoding="utf-8")
    assert is_kill_switch_engaged() is True


def test_empty_file_fails_closed(_isolated_state: Path) -> None:
    """A present but empty file is not valid JSON -> corrupt -> fail-closed."""
    _isolated_state.write_text("", encoding="utf-8")
    assert is_kill_switch_engaged() is True


def test_unreadable_present_file_fails_closed(
    monkeypatch, _isolated_state: Path
) -> None:
    """A present path that raises OSError on read -> corrupt -> fail-closed.

    Simulates a transient I/O failure on a file that ``exists()`` reports as
    present (distinct from a genuinely missing file).
    """
    _isolated_state.write_text(json.dumps({"engaged": False}), encoding="utf-8")

    import src.assembled_core.execution.kill_switch as ks_mod

    _real_read_text = Path.read_text

    def _boom(self, *args, **kwargs):  # noqa: ANN001
        if self == ks_mod._state_path():
            raise OSError("simulated transient unreadable file")
        return _real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _boom)
    assert is_kill_switch_engaged() is True


# ---------------------------------------------------------------------------
# End-to-end: the wrapper inherits the deep fail-closed behaviour
# ---------------------------------------------------------------------------


def test_wrapper_blocks_all_orders_on_corrupt_state(_isolated_state: Path) -> None:
    """filter_orders_with_kill_switches() blocks ALL orders on corrupt state."""
    _isolated_state.write_text("}{ not json", encoding="utf-8")
    orders = _orders()
    filtered = filter_orders_with_kill_switches(orders)
    assert len(filtered) == 0
    assert list(filtered.columns) == list(orders.columns)


def test_wrapper_passes_orders_on_missing_state(_isolated_state: Path) -> None:
    """Sanity: with NO state file (disengaged) the wrapper passes orders through."""
    assert not _isolated_state.exists()
    orders = _orders()
    filtered = filter_orders_with_kill_switches(orders)
    assert len(filtered) == len(orders)
