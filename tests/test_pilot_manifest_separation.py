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


# --- Doppellauf-Schutz (E-191) ----------------------------------------------


def test_daily_marker_is_shared_with_scheduler_daemon(monkeypatch):
    """Beide Pfade MUESSEN denselben Marker nutzen.

    Gemessen 2026-08-18: der Task-Pilot fuhr 21:30 einen Broker-Zyklus
    (8 Fills), der Daemon 21:40 einen zweiten (2 Fills) — der Tages-Cap von
    20 % Turnover gilt PRO Zyklus, war also faktisch verdoppelt, und zwei
    Systeme schrieben dasselbe Ledger.
    """
    import importlib.util as ilu

    pilot = _load(monkeypatch, in_ci=False)
    spec = ilu.spec_from_file_location(
        "paper_trading_scheduler",
        Path(__file__).resolve().parents[1] / "scripts" / "paper_trading_scheduler.py",
    )
    daemon = ilu.module_from_spec(spec)
    spec.loader.exec_module(daemon)
    assert pilot.LAST_RUN_PATH == daemon.LAST_RUN_PATH


def test_pilot_skips_when_marker_is_today(monkeypatch, tmp_path):
    """Marker == heute -> kein zweiter Zyklus (rc 0, kein Broker-Kontakt)."""
    from datetime import datetime, timezone

    mod = _load(monkeypatch, in_ci=False)
    marker = tmp_path / "last_run_date.txt"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    marker.write_text(today, encoding="utf-8")
    monkeypatch.setattr(mod, "LAST_RUN_PATH", marker)

    called = {"startup": 0}
    monkeypatch.setattr(
        mod, "run_startup_checks", lambda: called.__setitem__("startup", 1)
    )
    rc = mod.cmd_run_day()
    assert rc == 0
    assert called["startup"] == 0, "der Zyklus haette gar nicht starten duerfen"


def test_pilot_runs_when_marker_is_stale(monkeypatch, tmp_path):
    """Gegenprobe: Marker von gestern -> der Zyklus laeuft (kein Blockieren
    des Betriebs durch einen alten Marker)."""
    mod = _load(monkeypatch, in_ci=False)
    marker = tmp_path / "last_run_date.txt"
    marker.write_text("2020-01-01", encoding="utf-8")
    monkeypatch.setattr(mod, "LAST_RUN_PATH", marker)
    assert mod._already_ran_today("2026-08-19") is False


def test_marker_not_set_on_failed_cycle(monkeypatch, tmp_path):
    """Ein fehlgeschlagener Zyklus darf den Marker NICHT setzen — sonst
    faellt auch das Daemon-Backup aus."""
    mod = _load(monkeypatch, in_ci=False)
    marker = tmp_path / "last_run_date.txt"
    monkeypatch.setattr(mod, "LAST_RUN_PATH", marker)
    monkeypatch.setattr(mod, "PILOT_MANIFEST", tmp_path / "m.json")
    monkeypatch.setattr(mod, "PILOT_DIR", tmp_path)
    monkeypatch.setattr(mod, "run_startup_checks", lambda: None)

    class _Res:
        returncode = 1
        stdout = "boom"
        stderr = ""

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Res())
    mod.cmd_run_day()
    assert not marker.exists(), "Marker nach fehlgeschlagenem Zyklus gesetzt"
