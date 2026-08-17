# -*- coding: utf-8 -*-
"""Verdrahtungstests fuer Audit-Plan 5.3 (2026-08-16): composite_score ->
attribution + der signal_scores-Producer/-Consumer-Bund.

Vorher war attribution/ Kategorie (b) — fertig, getestet, ohne Producer;
/monitoring/signals suchte Dateien, die niemand schrieb (E-159-Klasse:
Producer-Consumer-Fixes brauchen Bindungstests mit dem ECHTEN Writer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assembled_core.signals.composite_score import (  # noqa: E402
    generate_composite_score_signals,
)


def _panel(n_days: int = 60, symbols=("AAA", "BBB")) -> pd.DataFrame:
    days = pd.bdate_range("2026-01-02", periods=n_days, tz="UTC")
    rows = []
    rng = np.random.default_rng(7)
    for sym in symbols:
        close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n_days))
        for t, c in zip(days, close):
            rows.append({"symbol": sym, "timestamp": t, "close": c, "volume": 1e6})
    return pd.DataFrame(rows)


def test_collect_attributions_matches_scores():
    """Jede Attribution muss zu ihrem Signal passen: Summe der
    dimension_contributions == composite_score (vor Clipping)."""
    attrs: list = []
    signals = generate_composite_score_signals(
        _panel(), regime="normal", collect_attributions=attrs
    )
    assert len(attrs) == len(signals) > 0
    by_ticker = {a.ticker: a for a in attrs}
    for _, row in signals.iterrows():
        a = by_ticker[row["symbol"]]
        contrib_sum = sum(a.dimension_contributions.values())
        assert a.composite_score == pytest.approx(contrib_sum, abs=1e-9)
        # Score im Signal == geclippter Attribution-Score
        assert row["score"] == pytest.approx(
            float(np.clip(a.composite_score, -1.0, 1.0)), abs=1e-9
        )


def test_no_attributions_without_optin():
    """Ohne collect_attributions darf sich nichts am Verhalten aendern."""
    s1 = generate_composite_score_signals(_panel(), regime="normal")
    attrs: list = []
    s2 = generate_composite_score_signals(
        _panel(), regime="normal", collect_attributions=attrs
    )
    pd.testing.assert_frame_equal(s1, s2)


def test_signal_scores_producer_feeds_monitoring_endpoint(tmp_path, monkeypatch):
    """Bindungstest mit dem ECHTEN Producer-Codepfad: ein vom Report-Script-
    Format geschriebenes Artefakt muss vom Endpoint gelesen werden."""
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    # Producer-Format exakt wie scripts/generate_attribution_report.py
    scores_dir = tmp_path / "signals"
    scores_dir.mkdir()
    (scores_dir / "signal_scores_20260816T120000Z.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-08-16T12:00:00+00:00",
                "producer": "scripts/generate_attribution_report.py",
                "scores": {"AAA": 0.42, "BBB": -0.17},
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app())
    r = client.get("/api/v1/monitoring/signals", params={"output_dir": str(scores_dir)})
    # Der Pfad liegt ausserhalb der Monitoring-Sandbox? Dann sagt der Guard
    # unavailable — das waere ein Test-Setup-Problem, kein Produktbefund.
    body = r.json()
    assert r.status_code == 200
    assert body.get("status") == "ok", body
    assert body["n_symbols"] == 2
    assert ["AAA", 0.42] in [list(x) for x in body["top_long"]]


def test_data_quality_reads_real_panel(tmp_path):
    """5.2-Coverage: /monitoring/data-quality gegen ein echtes tmp-Parquet
    (Long-Format!) — n_symbols muss nunique sein, nicht len(columns)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    days = pd.bdate_range("2026-08-01", periods=5, tz="UTC")
    rows = [
        {"timestamp": t, "symbol": s, "close": 100.0, "volume": 1.0}
        for t in days
        for s in ("AAA", "BBB", "CCC")
    ]
    p = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(p)
    client = TestClient(create_app())
    r = client.get("/api/v1/monitoring/data-quality", params={"price_path": str(p)})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_symbols"] == 3  # nunique, NICHT 4 Spalten
    assert body["last_bar"].startswith("2026-08-07")


def test_data_quality_503_when_panel_missing(tmp_path):
    """5.2-Coverage: fehlendes Panel ist ein 503, kein 200-unavailable."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    r = client.get(
        "/api/v1/monitoring/data-quality",
        params={"price_path": str(tmp_path / "nope.parquet")},
    )
    assert r.status_code == 503
    assert "does not exist" in r.json()["detail"]


def test_walk_forward_503_when_artifact_missing(monkeypatch, tmp_path):
    """5.2-Coverage: beide walk_forward-Endpunkte liefern 503 statt
    200-mit-Nullwerten, wenn nie ein Artefakt geschrieben wurde."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.assembled_core.api.routers.qa as qa_router

    monkeypatch.setattr(qa_router, "OUTPUT_DIR", tmp_path)
    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    for url in (
        "/api/v1/qa/walk_forward/1d/windows",
        "/api/v1/qa/walk_forward/1d/sharpe-distribution",
    ):
        r = client.get(url)
        assert r.status_code == 503, (url, r.status_code, r.text[:200])
        assert "has not run" in r.json()["detail"]


def test_signals_endpoint_default_dir_is_producer_dir():
    """F-senior-10: pinnt den 5.2-Umbau — der Query-Default von
    /monitoring/signals muss auf das Producer-Verzeichnis zeigen, nicht auf
    das alte Phantom-Verzeichnis src/output."""
    pytest.importorskip("fastapi")
    import inspect

    from src.assembled_core.api.routers.monitoring import get_signal_scores

    sig = inspect.signature(get_signal_scores)
    default = sig.parameters["output_dir"].default
    assert getattr(default, "default", default) == "output/signals"


def test_data_quality_freshness_from_last_bar_not_mtime(tmp_path):
    """F-auditor-1 (E-163-Pin): freshness MUSS aus dem letzten Bar kommen —
    ein frisch geschriebenes File (mtime=jetzt) mit alten Bars ist stale.
    Ein Rueckbau auf st_mtime macht diesen Test rot."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    old_days = pd.bdate_range(
        end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=20), periods=5
    )
    rows = [
        {"timestamp": t, "symbol": "AAA", "close": 100.0, "volume": 1.0}
        for t in old_days
    ]
    p = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(p)  # mtime = jetzt, Bars = 20 Tage alt
    client = TestClient(create_app())
    r = client.get("/api/v1/monitoring/data-quality", params={"price_path": str(p)})
    assert r.status_code == 200
    body = r.json()
    assert body["freshness"] == "stale", body
    assert body["bar_age_days"] > 4
    assert body["file_age_hours"] < 1  # beweist: mtime war frisch


def test_signals_no_data_yet_names_producer(tmp_path):
    """F-auditor-2 (E-162-Pin): leeres Verzeichnis heisst 'no_data_yet' mit
    producer_exists:true + Referenz — nie wieder 'strukturell tot'."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/api/v1/monitoring/signals", params={"output_dir": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_data_yet", body
    assert body["producer_exists"] is True
    assert body["producer"].endswith("generate_attribution_report.py")


def test_regime_producer_feeds_monitoring_endpoint(tmp_path):
    """Bindungstest (E-159-Regel) fuer den neuen regime-Producer: ein Artefakt
    im Producer-Format muss vom Endpoint gelesen werden; leeres Verzeichnis
    heisst no_data_yet mit Producer-Referenz (E-162)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    # leer -> no_data_yet + producer benannt
    r0 = client.get("/api/v1/monitoring/regime", params={"output_dir": str(tmp_path)})
    assert r0.status_code == 200
    b0 = r0.json()
    assert b0["status"] == "no_data_yet" and b0["producer_exists"] is True
    assert b0["producer"].endswith("write_regime_state.py")
    # Producer-Format -> gelesen
    (tmp_path / "regime_state_20260817T020000Z.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-08-17T02:00:00+00:00",
                "data_as_of": "2026-08-14 00:00:00+00:00",
                "regime": "sideways",
                "regime_score": 0.61,
                "producer": "scripts/ops/write_regime_state.py",
            }
        ),
        encoding="utf-8",
    )
    r1 = client.get("/api/v1/monitoring/regime", params={"output_dir": str(tmp_path)})
    assert r1.status_code == 200
    assert r1.json()["regime"] == "sideways"


def test_prune_keeps_newest_per_family(tmp_path, monkeypatch):
    """Retention-Pin: alte Dateien fallen, die NEUESTE bleibt immer —
    auch wenn sie selbst aelter als die Frist ist."""
    import importlib.util as ilu
    import os
    import time as _t

    # E-169: an __file__ ankern, nicht an die CWD (F-senior-4).
    spec = ilu.spec_from_file_location(
        "prune_ops_artifacts",
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "ops"
            / "prune_ops_artifacts.py"
        ),
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_REPO", tmp_path)
    d = tmp_path / "output" / "ops"
    d.mkdir(parents=True)
    old_ts = _t.time() - 90 * 86400
    for i, name in enumerate(["pull_log_a.json", "pull_log_b.json", "pull_log_c.json"]):
        f = d / name
        f.write_text("{}")
        os.utime(f, (old_ts + i, old_ts + i))  # alle 90 Tage alt
    mod.prune(dry_run=False)
    remaining = sorted(x.name for x in d.glob("pull_log_*.json"))
    assert remaining == ["pull_log_c.json"]  # nur die juengste ueberlebt


def _load_regime_producer(tmp_path, monkeypatch, fake_hmm_result):
    """write_regime_state.py mit gemocktem Detector + Mini-Panel laden
    (F-auditor-4: die Fehler-Zweige sind ohne Mock nicht erreichbar)."""
    import importlib.util as ilu
    from pathlib import Path as _P

    spec = ilu.spec_from_file_location(
        "write_regime_state",
        str(
            _P(__file__).resolve().parents[1]
            / "scripts"
            / "ops"
            / "write_regime_state.py"
        ),
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    panel = tmp_path / "daily.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range(end="2026-08-14", periods=5, freq="D", tz="UTC"),
            "symbol": ["SPY"] * 5,
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
        }
    ).to_parquet(panel, index=False)
    monkeypatch.setattr(mod, "PANEL_PATH", panel)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "regime")

    # main() importiert build_regime_state_hmm zur Laufzeit aus dem echten
    # risk-Modul — dort patchen, damit der from-Import den Mock zieht.
    import src.assembled_core.risk.regime_models as _rm

    monkeypatch.setattr(_rm, "build_regime_state_hmm", lambda **kwargs: fake_hmm_result)
    return mod


def test_regime_producer_nan_label_writes_unknown(tmp_path, monkeypatch):
    """E-175-Pin: NaN ist truthy — ein NaN-regime_label muss als 'unknown'
    landen, nicht als String 'nan'; die Confidence bleibt nutzbar."""
    fake = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14"]),
            "regime_label": [np.nan],
            "regime_confidence": [0.7],
        }
    )
    mod = _load_regime_producer(tmp_path, monkeypatch, fake)
    assert mod.main() == 0
    payload = json.loads(
        next((tmp_path / "regime").glob("regime_state_*.json")).read_text("utf-8")
    )
    assert payload["regime"] == "unknown"
    assert payload["regime_score"] == pytest.approx(0.7)


def test_regime_producer_missing_confidence_writes_none_score(tmp_path, monkeypatch):
    """F-senior-3/E-174-Pin: fehlt die regime_confidence-Spalte, ist
    regime_score explizit null — kein KeyError, kein erfundener Wert."""
    fake = pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-14"]), "regime_label": ["bull"]}
    )
    mod = _load_regime_producer(tmp_path, monkeypatch, fake)
    assert mod.main() == 0
    payload = json.loads(
        next((tmp_path / "regime").glob("regime_state_*.json")).read_text("utf-8")
    )
    assert payload["regime"] == "bull"
    assert payload["regime_score"] is None
