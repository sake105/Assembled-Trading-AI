"""Functional coverage for the RE-ACTIVATED ledger-events parquet write.

Re-activation (2026-06-04, user-approved Option A) fixed BUG 4: the engine had
imported ``store_ledger_events_parquet`` from the WRONG module
(``accounting.ledger``), so ``_HAS_LEDGER`` was pinned False and
``_write_ledger_events`` never ran. The import now targets
``accounting.ledger_store`` and the write produces the CANONICAL
``<ledger_dir>/ledger_<run_id>/ledger_events.parquet`` artifact (atomic
tmp->rename, dedup-by-event_id) that the accounting chain reads.

This is an ADDITIVE persistence path: it writes a parquet artifact at the
end-of-day persistence step. It does NOT touch order/fill/position/cash logic.
The reconcile ``ledger_exists`` flag now reflects the real artifact (was
False-always); it is observability only and does not gate any reconcile verdict.

These tests lock:
* ``_HAS_LEDGER`` is True (re-activation contract).
* a write produces the canonical artifact with the REQUIRED_COLUMNS schema and
  the expected rows.
* a re-run dedups on ``event_id`` (no row duplication).
* a write failure (unwritable target) is caught, does NOT abort the cycle and
  does NOT corrupt cash/positions.
* the reconcile ``ledger_exists`` flag reads True once the artifact exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.accounting.ledger import REQUIRED_COLUMNS
from src.assembled_core.accounting.ledger_store import ledger_base_path
from src.assembled_core.execution import unified_paper_engine as upe
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    seed: float = 100_000.0,
    ledger_dir: Path | None = None,
    enable_reconciliation: bool = False,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=seed,
        state_dir=tmp_path / "state",
        ledger_dir=ledger_dir if ledger_dir is not None else tmp_path / "ledger",
        enable_reconciliation=enable_reconciliation,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="test_run",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": seed, "positions": {}, "cost_basis": {}}
    return eng


def _canonical_path(eng: UnifiedPaperEngine) -> Path:
    return (
        ledger_base_path(eng.config.ledger_dir, eng.config.run_id)
        / "ledger_events.parquet"
    )


def _fills() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_price": 100.0},
            {"symbol": "BBB", "side": "BUY", "qty": 5.0, "fill_price": 200.0},
        ]
    )


# --- re-activation contract -------------------------------------------------


@pytest.mark.fast
def test_has_ledger_is_true() -> None:
    assert upe._HAS_LEDGER is True


# --- canonical write --------------------------------------------------------


@pytest.mark.fast
def test_write_produces_canonical_artifact(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)

    eng._write_ledger_events(_fills(), "2025-01-15")

    path = _canonical_path(eng)
    assert path.exists(), f"canonical ledger artifact missing: {path}"
    # Layout is <ledger_dir>/ledger_<run_id>/ledger_events.parquet
    assert path.parent.name == "ledger_test_run"
    assert path.name == "ledger_events.parquet"

    df = pd.read_parquet(path)
    assert len(df) == 2
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing required column: {col}"
    assert set(df["event_type"]) == {"FILL"}
    assert set(df["symbol"]) == {"AAA", "BBB"}


@pytest.mark.fast
def test_write_caches_returned_path(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    assert eng._last_ledger_path is None

    eng._write_ledger_events(_fills(), "2025-01-15")

    assert eng._last_ledger_path == _canonical_path(eng)


# --- dedup on re-run --------------------------------------------------------


@pytest.mark.fast
def test_rerun_same_day_dedups_on_event_id(tmp_path: Path) -> None:
    """append-mode + dedup-by-event_id makes a same-day re-run idempotent."""
    eng = _make_engine(tmp_path)

    eng._write_ledger_events(_fills(), "2025-01-15")
    eng._write_ledger_events(_fills(), "2025-01-15")

    df = pd.read_parquet(_canonical_path(eng))
    # 2 unique event_ids; the second write must not duplicate rows.
    assert len(df) == 2
    assert df["event_id"].nunique() == 2


@pytest.mark.fast
def test_distinct_days_accumulate(tmp_path: Path) -> None:
    """Different days have distinct event_ids and accumulate (append mode)."""
    eng = _make_engine(tmp_path)

    eng._write_ledger_events(_fills(), "2025-01-15")
    eng._write_ledger_events(_fills(), "2025-01-16")

    df = pd.read_parquet(_canonical_path(eng))
    assert len(df) == 4
    assert df["event_id"].nunique() == 4


# --- fail-safe --------------------------------------------------------------


@pytest.mark.fast
def test_write_failure_is_caught_and_does_not_corrupt_state(
    tmp_path: Path,
) -> None:
    """A write failure must be swallowed (no raise) and leave state intact.

    The ledger_dir is pointed at a FILE (not a directory) so the store's
    ``mkdir`` raises. The engine must catch it, not abort, and not mutate
    cash/positions (this method runs after positions are updated).
    """
    not_a_dir = tmp_path / "blocker"
    not_a_dir.write_text("i am a file, not a directory", encoding="utf-8")

    eng = _make_engine(tmp_path, ledger_dir=not_a_dir)
    eng._state["positions"]["AAA"] = 7.0
    eng._state["cash"] = 12_345.0

    # Must NOT raise.
    eng._write_ledger_events(_fills(), "2025-01-15")

    # State untouched by the failed persistence write.
    assert eng._state["positions"]["AAA"] == 7.0
    assert eng._state["cash"] == pytest.approx(12_345.0)
    # No artifact produced, cache stays None.
    assert eng._last_ledger_path is None


# --- reconcile flag reflects reality ----------------------------------------


@pytest.mark.fast
def test_reconcile_ledger_exists_true_after_write(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_reconciliation=True)
    eng._state = {"cash": 100_000.0, "positions": {"AAA": 10.0}, "cost_basis": {}}

    # Before any ledger write, the canonical artifact does not exist.
    assert not _canonical_path(eng).exists()

    eng._write_ledger_events(_fills(), "2025-01-15")
    verdict = eng._run_reconciliation("2025-01-15")

    assert verdict is not None
    assert verdict["reconcile"]["ledger_exists"] is True


@pytest.mark.fast
def test_reconcile_ledger_exists_false_without_write(tmp_path: Path) -> None:
    """No ledger write -> flag is False (honest), reconcile still runs."""
    eng = _make_engine(tmp_path, enable_reconciliation=True)
    eng._state = {"cash": 100_000.0, "positions": {"AAA": 10.0}, "cost_basis": {}}

    verdict = eng._run_reconciliation("2025-01-15")

    assert verdict is not None
    assert verdict["reconcile"]["ledger_exists"] is False
