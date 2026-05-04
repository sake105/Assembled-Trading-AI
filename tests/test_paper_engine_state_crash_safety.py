"""E0.3 — Atomic state-save crash safety.

These tests verify that :meth:`UnifiedPaperEngine._save_state` cannot leave
the primary state file in a corrupt state even if the process dies mid-write.
The implementation writes to ``<state>.tmp`` first, fsyncs, then
``os.replace``s onto the target — so reading the target must always yield
either the pre-save or the post-save content, never a truncated JSON.

Regression guard against CLAUDE.md §30 risk/execution safeguards: a corrupt
``paper_state.json`` triggers ``_default_state`` on next load, which silently
resets cash to ``seed_capital`` — a wipe-out scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)

pytestmark = pytest.mark.phase_zero


@pytest.fixture()
def engine(tmp_path: Path) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(state_dir=tmp_path, seed_capital=12_345.0)
    eng = UnifiedPaperEngine(cfg)
    eng._load_state()
    eng._state["cash"] = 7_777.77
    eng._state["positions"] = {"AAPL": 10.0, "MSFT": 5.0}
    eng._equity_curve = [{"date": "2026-04-17", "equity": 12_345.67}]
    return eng


def test_save_state_writes_target_and_cleans_tmp(engine: UnifiedPaperEngine) -> None:
    engine._save_state()
    state_path = engine.config.state_dir / engine._STATE_FILE
    equity_path = engine.config.state_dir / engine._EQUITY_FILE
    assert state_path.exists()
    assert equity_path.exists()
    # tmp siblings must be cleaned up
    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists()
    assert not equity_path.with_suffix(equity_path.suffix + ".tmp").exists()
    # content is well-formed JSON
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["cash"] == pytest.approx(7_777.77)
    assert data["positions"]["AAPL"] == pytest.approx(10.0)


def test_save_state_preserves_prior_state_when_replace_fails(
    engine: UnifiedPaperEngine,
) -> None:
    """Simulated crash: ``os.replace`` raises before the swap happens."""
    engine._save_state()  # baseline good state on disk
    state_path = engine.config.state_dir / engine._STATE_FILE
    baseline = state_path.read_text(encoding="utf-8")

    engine._state["cash"] = 9_999.99  # would-be new value

    with patch(
        "src.assembled_core.execution.unified_paper_engine.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        engine._save_state()  # must not raise

    # Primary file unchanged — this is the key atomic-save guarantee.
    assert state_path.read_text(encoding="utf-8") == baseline
    # Tmp file cleaned up even though replace failed.
    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists()


def test_save_state_tolerates_fsync_refusal(engine: UnifiedPaperEngine) -> None:
    """Filesystems that reject fsync (network, tmpfs) must not break saves."""
    with patch(
        "src.assembled_core.execution.unified_paper_engine.os.fsync",
        side_effect=OSError("fsync not supported"),
    ):
        engine._save_state()  # must not raise

    state_path = engine.config.state_dir / engine._STATE_FILE
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["cash"] == pytest.approx(7_777.77)


def test_save_state_round_trip_reload(engine: UnifiedPaperEngine) -> None:
    """Reload produced content must match what was written."""
    engine._save_state()
    fresh = UnifiedPaperEngine(engine.config)
    reloaded = fresh._load_state()
    assert reloaded["cash"] == pytest.approx(7_777.77)
    assert reloaded["positions"]["AAPL"] == pytest.approx(10.0)


def test_save_state_does_not_leak_tmp_on_write_failure(
    engine: UnifiedPaperEngine, tmp_path: Path
) -> None:
    """If the initial write raises, tmp siblings must be cleaned up."""
    original_open = open

    def failing_open(path, *a, **kw):
        if str(path).endswith(".tmp"):
            raise OSError("simulated disk-full")
        return original_open(path, *a, **kw)

    with patch("builtins.open", side_effect=failing_open):
        engine._save_state()  # must not raise

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"tmp files leaked: {leftovers}"
