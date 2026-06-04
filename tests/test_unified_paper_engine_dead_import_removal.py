"""Regression locks for the dead-import removal in unified_paper_engine.

Two masked/dead imports were removed (byte-identical to prior live behaviour):

* BUG 3 — ``from ops.experience_log import log_experience_entry``: the function
  ``log_experience_entry`` never existed (the module only ever exposed
  ``append_experience`` / ``load_experience`` / ``compute_experience_summary``),
  so ``_HAS_EXPERIENCE_LOG`` was always False and the call site was a no-op.
  The import, the ``_HAS_EXPERIENCE_LOG`` flag/guard, and the dead
  ``_write_experience_entry`` method were removed.

* BUG 4 — ``from accounting.ledger import store_ledger_events_parquet``: wrong
  module (the symbol lives in ``accounting.ledger_store`` and needs a different
  signature + output layout), so ``_HAS_LEDGER`` was always False and the
  ledger-events write path (``_write_ledger_events``) never executed. The broken
  import was removed and ``_HAS_LEDGER`` pinned to ``False`` to preserve the
  byte-identical (write-path-OFF) behaviour. Re-activation is DEFERRED.

These tests lock the contract so a future change cannot silently re-enable the
ledger write path or re-introduce the non-existent experience-log symbol.
"""

from __future__ import annotations

import pytest

import src.assembled_core.execution.unified_paper_engine as upe


@pytest.mark.fast
def test_has_ledger_pinned_false() -> None:
    # Ledger-events parquet write stays OFF (byte-identical to prior behaviour).
    # Flipping this to True is a deliberate output-layout behaviour change and
    # also requires fixing the import (accounting.ledger_store) + call signature.
    assert upe._HAS_LEDGER is False


@pytest.mark.fast
def test_experience_log_symbols_removed() -> None:
    # The non-existent experience-log feature must not creep back in.
    assert not hasattr(upe, "_HAS_EXPERIENCE_LOG")
    assert not hasattr(upe, "log_experience_entry")
    assert not hasattr(upe, "store_ledger_events_parquet")


@pytest.mark.fast
def test_write_experience_entry_method_removed() -> None:
    from src.assembled_core.execution.unified_paper_engine import UnifiedPaperEngine

    assert not hasattr(UnifiedPaperEngine, "_write_experience_entry")
