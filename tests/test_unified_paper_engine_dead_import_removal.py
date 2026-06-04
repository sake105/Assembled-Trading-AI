"""Regression locks for the dead-import removal in unified_paper_engine.

Two masked/dead imports were removed (byte-identical to prior live behaviour):

* BUG 3 — ``from ops.experience_log import log_experience_entry``: the function
  ``log_experience_entry`` never existed (the module only ever exposed
  ``append_experience`` / ``load_experience`` / ``compute_experience_summary``),
  so ``_HAS_EXPERIENCE_LOG`` was always False and the call site was a no-op.
  The import, the ``_HAS_EXPERIENCE_LOG`` flag/guard, and the dead
  ``_write_experience_entry`` method were removed.

* BUG 4 — ``from accounting.ledger import store_ledger_events_parquet``: wrong
  module (the symbol lives in ``accounting.ledger_store``), so ``_HAS_LEDGER``
  was always False and the ledger-events write path (``_write_ledger_events``)
  never executed. RE-ACTIVATED 2026-06-04 (user-approved Option A): the import
  now targets the correct module (``accounting.ledger_store``) and
  ``_HAS_LEDGER`` is True when that import succeeds, so the write path now runs.
  The prior ``_HAS_LEDGER is False`` contract lock is therefore intentionally
  INVERTED below — the write-path behaviour change is deliberate, not a
  regression. Functional coverage of the re-activated write lives in
  ``tests/test_unified_paper_engine_ledger_write.py``.

These tests still lock the EXPERIENCE-LOG contract (BUG 3) so the non-existent
symbol cannot creep back in, and lock the re-activated ledger contract (the
correct module is imported and the flag is True).
"""

from __future__ import annotations

import pytest

import src.assembled_core.execution.unified_paper_engine as upe


@pytest.mark.fast
def test_has_ledger_reactivated_true() -> None:
    # Re-activated (user-approved Option A): the ledger-events parquet write is
    # ON. ``_HAS_LEDGER`` is True because the import now targets the correct
    # module (accounting.ledger_store). This inverts the prior False lock.
    assert upe._HAS_LEDGER is True


@pytest.mark.fast
def test_ledger_store_imported_from_correct_module() -> None:
    # The re-activation must import the REAL symbol (correct module), not the
    # old broken ``accounting.ledger`` target. Both the store function and the
    # base-path helper are now module-level attributes and callable.
    assert hasattr(upe, "store_ledger_events_parquet")
    assert upe.store_ledger_events_parquet is not None
    assert callable(upe.store_ledger_events_parquet)
    assert upe.store_ledger_events_parquet.__module__.endswith(
        "accounting.ledger_store"
    )
    assert hasattr(upe, "ledger_base_path")
    assert upe.ledger_base_path is not None


@pytest.mark.fast
def test_experience_log_symbols_removed() -> None:
    # The non-existent experience-log feature must not creep back in.
    # (Ledger symbols are intentionally PRESENT now — see the ledger tests.)
    assert not hasattr(upe, "_HAS_EXPERIENCE_LOG")
    assert not hasattr(upe, "log_experience_entry")


@pytest.mark.fast
def test_write_experience_entry_method_removed() -> None:
    from src.assembled_core.execution.unified_paper_engine import UnifiedPaperEngine

    assert not hasattr(UnifiedPaperEngine, "_write_experience_entry")
