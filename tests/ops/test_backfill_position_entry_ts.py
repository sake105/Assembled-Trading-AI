"""Tests for scripts/ops/backfill_position_entry_ts.py (operator tool).

Pins the safety model: dry-run default, all-or-nothing validation, no
overwrite without --force, no fresh-ledger creation, round-trip through the
canonical loader (entry_ts survives _norm_position).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.ops.backfill_position_entry_ts as backfill_tool
from scripts.ops.backfill_position_entry_ts import main

pytestmark = pytest.mark.fast

TS = "2026-07-14T00:00:00+00:00"


def _seed_ledger(tmp_path: Path, positions: dict | None = None) -> Path:
    p = tmp_path / "ledger_state.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_utc": "2026-07-24T19:00:00+00:00",
                "cash": 71_440.14,
                "positions": positions
                if positions is not None
                else {
                    "GLD": {"qty": 19.89, "avg_price": 381.80, "hwm": 382.17},
                    "TLT": {"qty": 88.86, "avg_price": 85.09, "hwm": 85.11},
                },
                "equity_curve": [],
            }
        ),
        encoding="utf-8",
    )
    return p


def _positions(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))["positions"]


def test_dry_run_changes_nothing(tmp_path) -> None:
    ledger = _seed_ledger(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    rc = main(["--set", f"GLD={TS}", "--ledger-path", str(ledger)])
    assert rc == 0
    assert ledger.read_text(encoding="utf-8") == before


def test_apply_stamps_both_symbols_and_round_trips(tmp_path) -> None:
    ledger = _seed_ledger(tmp_path)
    rc = main(
        [
            "--set",
            f"GLD={TS}",
            "--set",
            f"TLT={TS}",
            "--ledger-path",
            str(ledger),
            "--apply",
        ]
    )
    assert rc == 0
    pos = _positions(ledger)
    assert pos["GLD"]["entry_ts"] == TS
    assert pos["TLT"]["entry_ts"] == TS
    # untouched fields survive
    assert pos["GLD"]["avg_price"] == pytest.approx(381.80)


def test_unknown_symbol_is_all_or_nothing(tmp_path) -> None:
    ledger = _seed_ledger(tmp_path)
    rc = main(
        [
            "--set",
            f"GLD={TS}",
            "--set",
            f"XXXX={TS}",
            "--ledger-path",
            str(ledger),
            "--apply",
        ]
    )
    assert rc == 2
    assert "entry_ts" not in _positions(ledger)["GLD"], "no partial write allowed"


def test_existing_entry_ts_requires_force(tmp_path) -> None:
    ledger = _seed_ledger(
        tmp_path,
        {"GLD": {"qty": 19.89, "avg_price": 381.80, "hwm": 382.17, "entry_ts": TS}},
    )
    rc = main(
        [
            "--set",
            "GLD=2026-07-20T00:00:00+00:00",
            "--ledger-path",
            str(ledger),
            "--apply",
        ]
    )
    assert rc == 2
    assert _positions(ledger)["GLD"]["entry_ts"] == TS

    rc2 = main(
        [
            "--set",
            "GLD=2026-07-20T00:00:00+00:00",
            "--ledger-path",
            str(ledger),
            "--apply",
            "--force",
        ]
    )
    assert rc2 == 0
    assert _positions(ledger)["GLD"]["entry_ts"] == "2026-07-20T00:00:00+00:00"


def test_naive_timestamp_rejected(tmp_path) -> None:
    ledger = _seed_ledger(tmp_path)
    rc = main(["--set", "GLD=2026-07-14T00:00:00", "--ledger-path", str(ledger)])
    assert rc == 2


def test_future_timestamp_rejected(tmp_path) -> None:
    ledger = _seed_ledger(tmp_path)
    rc = main(["--set", "GLD=2199-01-01T00:00:00+00:00", "--ledger-path", str(ledger)])
    assert rc == 2


def test_missing_ledger_refused_never_created(tmp_path) -> None:
    missing = tmp_path / "nope" / "ledger_state.json"
    rc = main(["--set", f"GLD={TS}", "--ledger-path", str(missing), "--apply"])
    assert rc == 2
    assert not missing.exists()


def test_corrupt_main_file_refused_even_with_readable_backup(tmp_path) -> None:
    """Review F-senior-1 / E-065: a corrupt MAIN file must abort the tool —
    load_ledger_state would silently fall back to an OLDER backup, and a
    subsequent save would promote that stale state as the new truth."""
    ledger = _seed_ledger(tmp_path)
    backup = ledger.with_suffix(ledger.suffix + ".1")
    backup.write_text(ledger.read_text(encoding="utf-8"), encoding="utf-8")
    ledger.write_text("{ this is not json", encoding="utf-8")

    rc = main(["--set", f"GLD={TS}", "--ledger-path", str(ledger), "--apply"])

    assert rc == 2
    assert ledger.read_text(encoding="utf-8") == "{ this is not json", (
        "corrupt main file must stay untouched (no backup promotion)"
    )


def test_concurrent_write_between_load_and_save_aborts(tmp_path, monkeypatch) -> None:
    """mtime-sentinel lost-update guard: a concurrent write between load and
    save must abort with exit 2 and leave the ledger byte-identical.

    Deterministic simulation: wrap the tool's module-level load_ledger_state
    so that AFTER the real load returns, the ledger file's mtime is bumped
    via os.utime (+1ms in ns, immune to filesystem timestamp granularity) —
    exactly the load→save window a concurrent scheduler cycle would hit.
    """
    ledger = _seed_ledger(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    real_load = backfill_tool.load_ledger_state

    def _load_then_touch(path, **kwargs):
        state = real_load(path, **kwargs)
        st = ledger.stat()
        os.utime(ledger, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        return state

    monkeypatch.setattr(backfill_tool, "load_ledger_state", _load_then_touch)

    rc = main(["--set", f"GLD={TS}", "--ledger-path", str(ledger), "--apply"])

    assert rc == 2, "changed mtime between load and save must abort"
    assert ledger.read_text(encoding="utf-8") == before, (
        "aborted run must not write anything (lost-update guard)"
    )
