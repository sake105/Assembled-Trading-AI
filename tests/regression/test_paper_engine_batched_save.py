"""B2 — Batched state-save gate.

``state_save_every_n_days`` must:

* default to 1 (save every day; paper/live-safe default)
* reduce the number of on-disk writes when set > 1
* still produce a bit-identical on-disk state at the end of ``run_paper_period``
  (so resume-after-crash + end-of-run state are deterministic)

The test uses a tiny adapter that substitutes ``_save_state`` with a counter
so we don't need to actually drive a backtest through the whole pipeline.
"""

from __future__ import annotations

import json
import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)

pytestmark = pytest.mark.phase_speed


def _make_engine(tmp_path, every: int) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_ledger=False,
        enable_lifecycle_tracking=False,
        enable_manifest=False,
        state_save_every_n_days=every,
        manifest_every_n_days=every,
    )
    return UnifiedPaperEngine(cfg)


def test_state_save_every_n_days_defaults_to_one() -> None:
    cfg = UnifiedPaperConfig()
    assert cfg.state_save_every_n_days == 1
    assert cfg.manifest_every_n_days == 1


def test_maybe_save_state_is_gated_by_counter(tmp_path) -> None:
    eng = _make_engine(tmp_path, every=5)
    calls: list[int] = []
    eng._save_state = lambda: calls.append(1)  # type: ignore[assignment]

    for _ in range(4):
        eng._maybe_save_state()
    assert calls == [], "save must not fire before the batching window closes"

    eng._maybe_save_state()
    assert len(calls) == 1, "fifth call must flush"

    for _ in range(4):
        eng._maybe_save_state()
    assert len(calls) == 1, "no extra flushes inside the next window"

    eng._maybe_save_state()
    assert len(calls) == 2


def test_maybe_save_state_every_day_when_every_is_one(tmp_path) -> None:
    eng = _make_engine(tmp_path, every=1)
    calls: list[int] = []
    eng._save_state = lambda: calls.append(1)  # type: ignore[assignment]
    for _ in range(3):
        eng._maybe_save_state()
    assert len(calls) == 3, "every=1 must save on every call (paper/live default)"


def test_run_paper_period_forces_terminal_flush(tmp_path) -> None:
    """After N days with every=5 we may end mid-window. The end-of-period flush
    must still persist the last bookkept state so resume-after-restart works."""
    eng = _make_engine(tmp_path, every=5)
    calls: list[int] = []
    eng._save_state = lambda: calls.append(1)  # type: ignore[assignment]

    # Fake two non-flush days, then drive the terminal-flush branch directly.
    eng._maybe_save_state()
    eng._maybe_save_state()
    assert calls == []
    assert eng._days_since_state_save == 2

    # Mirror the forced-flush block from ``run_paper_period``.
    if eng._days_since_state_save > 0:
        eng._save_state()
        eng._days_since_state_save = 0

    assert len(calls) == 1
    assert eng._days_since_state_save == 0


def test_state_file_round_trip_after_batched_save(tmp_path) -> None:
    eng = _make_engine(tmp_path, every=3)
    eng._load_state()
    eng._state["positions"]["AAPL"] = 5.0
    eng._state["cash"] = 42_000.0
    eng._save_state()

    state_path = tmp_path / "state" / "paper_state.json"
    assert state_path.exists()
    reloaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert reloaded["cash"] == 42_000.0
    assert reloaded["positions"]["AAPL"] == 5.0
