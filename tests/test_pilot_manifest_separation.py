# -*- coding: utf-8 -*-
"""Regressionstests fuer die CI/Betrieb-Trennung des Pilot-Manifests (E-190).

Befund 2026-08-18: CI-Runner und lokaler Betrieb schrieben DIESELBE Datei
``output/pilot/pilot_manifest.json``, und der Workflow committet sie mit
``git add -f``. Der Runner sieht nach dem Checkout nur die zuletzt
COMMITTETE (kurze) Historie, haengt seinen Tag an und pusht — ein lokales
``git pull`` ersetzte damit die echte Betriebshistorie: gemessen 27 Tage
-> 1 Tag. Betroffen sind die 30-Tage-Bewertung UND der Watchdog-Input
``zero_orders_unexpected``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_pilot.py"


def _load(monkeypatch, *, in_ci: bool):
    if in_ci:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    else:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    spec = importlib.util.spec_from_file_location("run_paper_pilot_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ci_writes_separate_manifest(monkeypatch):
    """Im Runner MUSS ein eigener Pfad benutzt werden — sonst hat der
    `git add -f`-Step wieder etwas zu committen und ueberschreibt die
    Betriebshistorie."""
    mod = _load(monkeypatch, in_ci=True)
    assert mod.PILOT_MANIFEST.name == "pilot_manifest_ci.json"


def test_local_run_keeps_operational_manifest(monkeypatch):
    """Lokal bleibt der Betriebspfad unveraendert (sonst waere die Historie
    fuer Watchdog und Pilot-Bewertung nicht mehr auffindbar)."""
    mod = _load(monkeypatch, in_ci=False)
    assert mod.PILOT_MANIFEST.name == "pilot_manifest.json"


def test_paths_differ_between_environments(monkeypatch):
    ci = _load(monkeypatch, in_ci=True).PILOT_MANIFEST
    local = _load(monkeypatch, in_ci=False).PILOT_MANIFEST
    assert ci != local
    assert ci.parent == local.parent  # beide unter output/pilot/


def test_reconstruction_marks_itself_as_lower_bound():
    """Das Rekonstruktions-Tool darf nie so tun, als sei die Historie
    vollstaendig — Laeufe ohne Artefakte (Kill-Switch-Aborts) fehlen
    zwangslaeufig."""
    spec = importlib.util.spec_from_file_location(
        "rebuild_pilot_manifest",
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ops"
        / "rebuild_pilot_manifest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = Path(mod.__file__).read_text(encoding="utf-8") if mod.__file__ else ""
    assert "is_lower_bound" in src
    assert "UNTERGRENZE" in src
