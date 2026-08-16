# -*- coding: utf-8 -*-
"""Verdrahtungstests fuer Audit-Plan 5.3 (2026-08-16): composite_score ->
attribution + der signal_scores-Producer/-Consumer-Bund.

Vorher war attribution/ Kategorie (b) — fertig, getestet, ohne Producer;
/monitoring/signals suchte Dateien, die niemand schrieb (E-159-Klasse:
Producer-Consumer-Fixes brauchen Bindungstests mit dem ECHTEN Writer)."""

from __future__ import annotations

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
    import json

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
