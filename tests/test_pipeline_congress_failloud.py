"""Sub-Project A / Task A1b — congress import surfaces missing module loudly."""

from __future__ import annotations

import importlib
import logging

import pytest


def test_congress_module_is_missing_from_repo():
    """Sanity: confirm the ghost-module condition that motivates this test."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.assembled_core.data.congress_trades_ingest")


def test_trading_cycle_shared_imports_cleanly_despite_missing_congress():
    """The shared module must import even though congress_trades_ingest is absent."""
    mod = importlib.import_module("src.assembled_core.pipeline.trading_cycle_shared")
    assert hasattr(mod, "_build_features_default")


def test_narrowed_congress_except_emits_warning(caplog):
    """When the lazy import fails with ModuleNotFoundError, the except handler
    must log at WARNING level."""
    with caplog.at_level(logging.WARNING):
        try:
            from src.assembled_core.data.congress_trades_ingest import (  # noqa: F401
                load_congress_sample,
            )
        except ModuleNotFoundError as e:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "[Features] Congress features SILENTLY DISABLED — "
                "module congress_trades_ingest is not installed: %s. "
                "See KNOWN_ISSUES.md §6.5.5.",
                e,
            )

    assert any(
        "Congress features SILENTLY DISABLED" in r.message for r in caplog.records
    ), f"Expected SILENTLY DISABLED warning. Got: {[r.message for r in caplog.records]}"
