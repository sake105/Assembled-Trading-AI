"""B1 — Pin the hot-path state-save format to the minimal encoding.

The plan's B1 fix removed ``indent=2`` from the per-bar state/equity save so
that a multi-hundred-bar backtest doesn't waste milliseconds on
pretty-printing. Any regression that re-introduces ``indent=`` in
``_atomic_write_json`` (or the paths that call it) will slow every backtest
run proportionally to ``n_bars`` — not a code-correctness bug, but a silent
wall-clock regression the B-phase gate is there to catch.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from src.assembled_core.execution import unified_paper_engine as upe

pytestmark = pytest.mark.phase_speed


def test_atomic_write_json_uses_no_indent() -> None:
    source = inspect.getsource(upe.UnifiedPaperEngine._atomic_write_json)
    # We want `json.dump(payload, fh, default=str)` — no `indent=` keyword.
    assert re.search(
        r"json\.dump\(\s*payload\s*,\s*fh\s*,\s*default=str\s*\)", source
    ), (
        "Hot-path atomic write must call json.dump(payload, fh, default=str) "
        "without indent= — see plan B1. If you need a pretty-printed artifact, "
        "add a separate end-of-run helper; don't re-indent every per-bar save."
    )
    assert "indent=" not in source, (
        "indent= keyword re-introduced into _atomic_write_json. "
        "That's a B1 regression: pretty-printing on every save doubles "
        "per-bar overhead."
    )


def test_save_state_round_trips_via_atomic_writer(tmp_path: Path) -> None:
    payload = {
        "cash": 10_000.0,
        "positions": {"AAPL": 3.0},
        "cost_basis": {"AAPL": 150.25},
        "created_at": "2026-04-17T00:00:00+00:00",
        "last_updated": "2026-04-17T20:00:00+00:00",
    }
    target = tmp_path / "state.json"
    upe.UnifiedPaperEngine._atomic_write_json(target, payload, log_label="state-test")
    assert target.exists()
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded == payload
    # Minimal encoding: no newlines / no multi-line indent after the opening brace.
    raw = target.read_text(encoding="utf-8")
    assert "\n" not in raw.rstrip("\n"), (
        f"State file is multi-line — indent keyword may have slipped back in:\n{raw!r}"
    )
