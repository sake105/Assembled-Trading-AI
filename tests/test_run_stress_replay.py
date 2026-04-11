"""Tests for scripts/run_stress_replay.py (Sprint 3 / Plan C6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# yaml is optional in some environments; skip the whole module when missing.
pytest.importorskip("yaml")

import run_stress_replay  # type: ignore  # noqa: E402


def test_build_synthetic_equity_shape() -> None:
    eq = run_stress_replay.build_synthetic_equity(
        start="2020-01-01", end="2020-02-01", seed=1
    )
    assert len(eq) > 10
    assert eq.iloc[0] > 0
    assert str(eq.index.tz) == "UTC"


def test_run_replay_writes_reports(tmp_path: Path) -> None:
    config = str(ROOT / "configs" / "stress_scenarios.yaml")
    out_dir = tmp_path / "stress_reports"
    # shrink equity to keep the test fast
    eq = run_stress_replay.build_synthetic_equity(
        start="2007-01-01", end="2025-01-01", seed=3
    )
    summaries = run_stress_replay.run_replay(config, str(out_dir), equity=eq)

    assert len(summaries) == 6
    files = sorted(p.name for p in out_dir.glob("*.json"))
    # one file per scenario + summary.json
    assert "summary.json" in files
    assert any("2020_covid" in f for f in files)

    # summary.json shape
    combined = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert combined["scenario_count"] == 6
    assert len(combined["scenarios"]) == 6
    # every scenario must have a status field
    for s in combined["scenarios"]:
        assert s["status"] in ("ok", "error")


def test_run_replay_with_missing_config(tmp_path: Path) -> None:
    out_dir = tmp_path / "stress_reports"
    summaries = run_stress_replay.run_replay(
        str(tmp_path / "does_not_exist.yaml"), str(out_dir)
    )
    assert summaries == []
    # summary.json should still be written, with scenario_count=0
    combined = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert combined["scenario_count"] == 0
