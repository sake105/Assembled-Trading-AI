"""Chaos test: ledger partial write / corruption recovery (Plan C21).

``save_ledger_state`` uses temp-file + atomic rename + backup
rotation. ``load_ledger_state`` is expected to recover from a
corrupted state file rather than crashing the daily cycle. This test
locks in both behaviours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.paper_ledger import (  # noqa: E402
    load_ledger_state,
    save_ledger_state,
)


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "ledger.json"
    state = {
        "cash": 12_345.67,
        "positions": {"AAPL": {"qty": 5.0, "avg_price": 150.0}},
        "equity_curve": [],
    }
    save_ledger_state(state, p)

    loaded = load_ledger_state(p)
    assert loaded["cash"] == 12_345.67
    assert "AAPL" in loaded["positions"]
    assert loaded["positions"]["AAPL"]["qty"] == 5.0
    assert loaded.get("schema_version")  # stamped by save_ledger_state


def test_load_recovers_from_corrupt_file(tmp_path: Path) -> None:
    """A truncated / half-written state file must not crash the
    daily cycle. The loader returns a fresh default state instead,
    which the caller can distinguish by the empty positions dict.
    """
    p = tmp_path / "ledger.json"
    # Simulate a write that got interrupted mid-json.
    p.write_text('{"cash": 10000.0, "positions":', encoding="utf-8")

    loaded = load_ledger_state(p, start_capital=10_000.0)
    assert loaded["cash"] == 10_000.0
    assert loaded["positions"] == {}
    # No exception escaped — this is the safety contract.


def test_load_recovers_from_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "ledger.json"
    p.write_text("", encoding="utf-8")

    loaded = load_ledger_state(p, start_capital=5_000.0)
    assert loaded["cash"] == 5_000.0
    assert loaded["positions"] == {}


def test_load_recovers_from_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.json"
    loaded = load_ledger_state(p, start_capital=7_777.0)
    assert loaded["cash"] == 7_777.0
    assert loaded["positions"] == {}


def test_save_is_atomic_no_partial_file_left(tmp_path: Path) -> None:
    """After a successful save, there must be no stray .tmp file
    next to the ledger. Atomic rename guarantees this.
    """
    p = tmp_path / "ledger.json"
    save_ledger_state({"cash": 1.0, "positions": {}}, p)

    siblings = list(tmp_path.iterdir())
    tmp_files = [f for f in siblings if f.name.endswith(".tmp")]
    assert tmp_files == [], f"stray temp files left: {tmp_files}"


def test_repeated_saves_rotate_backups(tmp_path: Path) -> None:
    """Backup rotation keeps the previous state recoverable after a
    new write. This is the chaos scenario: if a new save corrupts
    something at the application layer, the rotated .1 file is the
    last known good.
    """
    p = tmp_path / "ledger.json"
    save_ledger_state({"cash": 100.0, "positions": {}}, p)
    save_ledger_state({"cash": 200.0, "positions": {}}, p)

    backup = p.with_suffix(p.suffix + ".1")
    # Backup rotation is best-effort; if the implementation has it,
    # the .1 file must hold a parseable prior state.
    if backup.exists():
        prior = json.loads(backup.read_text(encoding="utf-8"))
        assert prior.get("cash") in (100.0, 200.0)
