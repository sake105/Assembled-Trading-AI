"""Sub-Project A / Task A1b — congress import handler in trading_cycle_shared.

The production handler is in `trading_cycle_shared._build_features_default`
inside a try/except around `from src.assembled_core.data.congress_trades_ingest`.
A full integration test would require constructing a real `TradingContext` with
`feature_cfg.include_congress=True`, which is research-work-scoped (out of plan).

UPDATE 2026-06-09: the `congress_trades_ingest` module has now been BUILT (free
House+Senate STOCK-Act ingester). The earlier "ghost-module" precondition test
is replaced by one asserting the module exists and exposes `load_congress_sample`.
The defensive try/except in the handler is intentionally KEPT — it still guards
against runtime errors from load/merge and against the module being removed again.

Senior-review F-senior-3 caught a self-verifying-warning antipattern in the
earlier version of this file — the test emitted its own warning and asserted
on caplog. That has been removed. The remaining tests verify static guarantees
about the handler text in the source file. They do NOT depend on production code
emitting a log signal.
"""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADING_CYCLE_SHARED = (
    REPO_ROOT / "src" / "assembled_core" / "pipeline" / "trading_cycle_shared.py"
)


def test_congress_module_exists_and_exposes_loader():
    """The congress ingester is now built and exposes the pipeline entry point."""
    mod = importlib.import_module("src.assembled_core.data.congress_trades_ingest")
    assert hasattr(mod, "load_congress_sample")


def test_trading_cycle_shared_imports_cleanly_despite_missing_congress():
    """The shared module must import cleanly (the defensive handler is kept)."""
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
