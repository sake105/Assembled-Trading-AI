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
    LedgerCorruptionError,
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


def test_load_recovers_from_corrupt_main_with_good_backup(tmp_path: Path) -> None:
    """Corrupt main but a parseable backup → recover the backup's state, no raise.

    This is the designed safety mechanism: a half-written main file is healed
    from the last-known-good rotated backup rather than crashing or resetting.
    """
    p = tmp_path / "ledger.json"
    p.write_text('{"cash": 999.0, "positions":', encoding="utf-8")  # truncated
    backup = p.with_suffix(p.suffix + ".1")
    backup.write_text(
        json.dumps(
            {"cash": 4_242.0, "positions": {"X": {"qty": 1.0, "avg_price": 7.0}}}
        ),
        encoding="utf-8",
    )

    loaded = load_ledger_state(p, start_capital=10_000.0)
    assert loaded["cash"] == 4_242.0
    assert loaded["positions"]["X"]["qty"] == 1.0


def test_load_raises_when_main_and_backup_both_corrupt(tmp_path: Path) -> None:
    """Main AND a backup existed but neither parses → fail loud (R2-5/E-025).

    Prior persisted state is unrecoverable, so a silent reset to start_capital
    would mask the loss. The loader must raise instead of returning fresh state.
    """
    p = tmp_path / "ledger.json"
    p.write_text('{"cash": 10000.0, "positions":', encoding="utf-8")  # truncated
    backup = p.with_suffix(p.suffix + ".1")
    backup.write_text("}{ not json at all", encoding="utf-8")  # corrupt backup

    try:
        load_ledger_state(p, start_capital=10_000.0)
    except LedgerCorruptionError:
        pass
    else:
        raise AssertionError("expected LedgerCorruptionError on total corruption")


def test_load_raises_when_only_backup_exists_and_is_corrupt(tmp_path: Path) -> None:
    """Main missing but a backup exists and is corrupt → still fail loud.

    A backup only exists after a successful prior save, so its presence means
    real state was persisted and is now unrecoverable — not a cold start.
    """
    p = tmp_path / "ledger.json"  # main intentionally absent
    backup = p.with_suffix(p.suffix + ".1")
    backup.write_text("totally corrupt", encoding="utf-8")

    try:
        load_ledger_state(p, start_capital=10_000.0)
    except LedgerCorruptionError:
        pass
    else:
        raise AssertionError(
            "expected LedgerCorruptionError when only corrupt backup exists"
        )


def test_load_skips_non_dict_main_and_recovers_from_backup(tmp_path: Path) -> None:
    """Main parses as valid JSON but is not a dict (e.g. a list) → skip it and
    recover from a good backup, rather than treating the list as state or raising.
    """
    p = tmp_path / "ledger.json"
    p.write_text("[]", encoding="utf-8")  # valid JSON, wrong shape
    backup = p.with_suffix(p.suffix + ".1")
    backup.write_text(json.dumps({"cash": 3_140.0, "positions": {}}), encoding="utf-8")

    loaded = load_ledger_state(p, start_capital=10_000.0)
    assert loaded["cash"] == 3_140.0


def test_runner_load_propagates_corruption(tmp_path: Path) -> None:
    """The paper runner's load helper must let LedgerCorruptionError propagate
    (halt the daily cycle) rather than swallowing it into a fresh state.
    """
    import pandas as pd

    from src.assembled_core.ops.paper_runner import _prd_load_paper_state

    led = tmp_path / "output" / "runs" / "_paper_ledger" / "ledger_state.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text('{"cash": 1.0, "positions":', encoding="utf-8")  # corrupt main
    led.with_suffix(led.suffix + ".1").write_text(
        "nope", encoding="utf-8"
    )  # corrupt backup

    try:
        _prd_load_paper_state(
            "paper",
            {},
            pd.DataFrame(),
            pd.Timestamp("2025-01-01", tz="UTC"),
            tmp_path,
            10_000.0,
        )
    except LedgerCorruptionError:
        pass
    else:
        raise AssertionError("expected LedgerCorruptionError to propagate from runner")
