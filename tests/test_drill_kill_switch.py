"""Tests fuer scripts/drills/drill_kill_switch.py (Stage-1-N1, 2026-08-17).

Pinnt das CI-Selbstversorgungs-Gate (E-178): im GitHub-Runner provisioniert
der Drill ein ephemeres OPERATOR_KILL_TOKEN (der Workflow hat kein
Repo-Secret); lokal bleibt das Verhalten unveraendert (tokenlos scheitert
der Deaktivierungs-Schritt kontrolliert). Die Engine wird ueber ihre
ASSEMBLED_KILL_SWITCH_*-Env-Overrides + chdir vollstaendig auf tmp_path
isoliert — der ECHTE Kill-Switch-State wird nie beruehrt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "drills" / "drill_kill_switch.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("drill_kill_switch_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _isolate_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # F-senior-8: env-basierter Engage-Override wuerde initial_state kippen.
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "kill_switch_state.json")
    )
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / "KILL_SWITCH_ENGAGED")
    )
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "kill_switch_audit.jsonl")
    )


def test_ci_gate_provisions_ephemeral_token_and_drill_passes(tmp_path, monkeypatch):
    """GITHUB_ACTIONS + kein Token -> ephemeres Token, voller Drill gruen."""
    _isolate_engine(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    mod = _load()

    rc = mod.main()

    assert rc == 0
    import os

    assert os.environ.get("OPERATOR_KILL_TOKEN"), (
        "CI-Gate muss ein ephemeres Token gesetzt haben"
    )
    reports = list((tmp_path / "output" / "drills").glob("kill_switch_*.json"))
    assert len(reports) == 1  # Drill-Report geschrieben


def test_local_without_token_aborts_before_activation(tmp_path, monkeypatch):
    """F-senior-2: lokal ohne Token bricht der Drill VOR der Aktivierung ab
    (rc=1, FAIL-Report) und laesst den Kill-Switch disengaged — der alte
    Ablauf aktivierte erst und konnte dann nie deaktivieren (der
    09.08.-Engaged-Zustand entstand genau so)."""
    _isolate_engine(tmp_path, monkeypatch)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    mod = _load()

    rc = mod.main()

    assert rc == 1
    import json

    reports = list((tmp_path / "output" / "drills").glob("kill_switch_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    statuses = {s["step"]: s["status"] for s in report["steps"]}
    assert statuses.get("token_present") == "FAIL"
    assert "activation_works" not in statuses  # NIE aktiviert
    assert not (tmp_path / "kill_switch_state.json").exists()  # State unberuehrt
