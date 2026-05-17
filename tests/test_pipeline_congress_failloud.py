"""Sub-Project A / Task A1b — congress import surfaces missing module loudly.

The production handler is in `trading_cycle_shared._build_features_default`
inside a try/except around `from src.assembled_core.data.congress_trades_ingest`.
A full integration test would require constructing a real `TradingContext` with
`feature_cfg.include_congress=True`, which is research-work-scoped (out of plan).

Senior-review F-senior-3 caught a self-verifying-warning antipattern in the
earlier version of this file — the test emitted its own warning and asserted
on caplog. That has been removed. The remaining tests verify (a) the
ghost-module precondition and (b) static guarantees about the handler text in
the source file. They do NOT depend on production code emitting a log signal.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADING_CYCLE_SHARED = (
    REPO_ROOT / "src" / "assembled_core" / "pipeline" / "trading_cycle_shared.py"
)


def test_congress_module_is_missing_from_repo():
    """Sanity: confirm the ghost-module condition that motivates this test."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.assembled_core.data.congress_trades_ingest")


def test_trading_cycle_shared_imports_cleanly_despite_missing_congress():
    """The shared module must import even though congress_trades_ingest is absent."""
    mod = importlib.import_module("src.assembled_core.pipeline.trading_cycle_shared")
    assert hasattr(mod, "_build_features_default")


def test_production_handler_contains_warn_message():
    """Static check: the production handler text in trading_cycle_shared.py must
    contain the 'SILENTLY DISABLED' WARNING message. This is a string-level
    contract — if a future refactor accidentally removes the warning text, this
    test catches it without requiring a full pipeline run.
    """
    source = TRADING_CYCLE_SHARED.read_text(encoding="utf-8")
    assert "Congress features SILENTLY DISABLED" in source, (
        "Expected the narrowed-except WARNING text in "
        "trading_cycle_shared.py — F-senior-2/A1b handler regression"
    )


def test_production_handler_has_outer_safety_net():
    """Static check: F-senior-2 split the try/except. The outer except must catch
    runtime errors from load_congress_sample / add_congress_features and log
    them as ERROR (not let them crash the pipeline).
    """
    source = TRADING_CYCLE_SHARED.read_text(encoding="utf-8")
    assert "Congress feature load/merge failed" in source, (
        "Expected the outer-safety-net ERROR text in trading_cycle_shared.py — "
        "F-senior-2 regression: data-op errors must not propagate"
    )
