"""Tests fuer scripts/ops/oddlot_forward_scanner.py (F-senior-5 aus d5fc9688).

Die Watchlist ist gitignored und der EINZIGE Speicherort der manuellen
geprueft-Flags — der Erhalt dieser Flags ueber Laeufe hinweg ist die
Invariante, deren Bruch manuelle Arbeit vernichtet. Netzwerk wird gemockt;
kein Test spricht mit EDGAR.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "oddlot_forward_scanner", ROOT / "scripts" / "ops" / "oddlot_forward_scanner.py"
)
scanner = importlib.util.module_from_spec(_SPEC)
sys.modules["oddlot_forward_scanner"] = scanner
_SPEC.loader.exec_module(scanner)


def _edgar_antwort(hits: list[dict], total: int | None = None) -> dict:
    return {
        "took": 1,
        "hits": {
            "total": {"value": len(hits) if total is None else total},
            "hits": hits,
        },
    }


def _hit(accession: str, firma: str, datum: str) -> dict:
    return {
        "_id": f"{accession}:x.htm",
        "_source": {
            "display_names": [firma],
            "file_date": datum,
            "ciks": ["0000000001"],
        },
    }


def _mock_urlopen(monkeypatch, payload: dict) -> None:
    class _H:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(scanner.urllib.request, "urlopen", lambda *a, **k: _H())


def test_suche_ohne_hits_feld_bricht_laut_ab(monkeypatch):
    """HTTP 200 mit Drossel-JSON darf nicht wie ein leerer Markt aussehen (E-103)."""
    _mock_urlopen(monkeypatch, {"error": "rate limited"})
    with pytest.raises(SystemExit, match="ohne 'hits'-Feld"):
        scanner.suche("2026-01-01", "2026-03-01")


def test_suche_deckelung_bricht_laut_ab(monkeypatch):
    """total > geliefert (EDGAR-100er-Deckel) darf nicht still verschluckt werden (E-132)."""
    _mock_urlopen(
        monkeypatch, _edgar_antwort([_hit("a-1", "F1", "2026-01-02")], total=273)
    )
    with pytest.raises(SystemExit, match="273 Treffer"):
        scanner.suche("2026-01-01", "2026-03-01")


def test_suche_leeres_hits_array_ist_gueltig(monkeypatch):
    _mock_urlopen(monkeypatch, _edgar_antwort([]))
    assert scanner.suche("2026-01-01", "2026-03-01") == []


def test_main_erhaelt_geprueft_flags_ueber_laeufe(monkeypatch, tmp_path):
    """Zweimal laufen; zwischen den Laeufen wird ein Fall manuell geprueft=True
    gesetzt. Der zweite Lauf darf das Flag nicht zuruecksetzen."""
    ziel = tmp_path / "oddlot_watchlist.json"
    monkeypatch.setattr(scanner, "ZIEL", ziel)
    _mock_urlopen(
        monkeypatch,
        _edgar_antwort(
            [
                _hit("a-1", "Fonds Eins", "2026-06-01"),
                _hit("a-2", "Fonds Zwei", "2026-06-15"),
            ]
        ),
    )
    assert scanner.main() == 0
    daten = json.loads(ziel.read_text(encoding="utf-8"))
    assert set(daten["faelle"]) == {"a-1", "a-2"}
    assert all(f["geprueft"] is False for f in daten["faelle"].values())

    # manuelle Pruefung simulieren
    daten["faelle"]["a-1"]["geprueft"] = True
    ziel.write_text(json.dumps(daten), encoding="utf-8")

    assert scanner.main() == 0
    danach = json.loads(ziel.read_text(encoding="utf-8"))
    assert danach["faelle"]["a-1"]["geprueft"] is True, "manuelles Flag verloren"
    assert danach["faelle"]["a-2"]["geprueft"] is False
    assert "stand" in danach


def test_main_stand_luecke_oeffnet_fenster(monkeypatch, tmp_path, capsys):
    """Aussetzer laenger als RUECKBLICK_TAGE: Fenster ab letztem Stand (E-132)."""
    ziel = tmp_path / "oddlot_watchlist.json"
    monkeypatch.setattr(scanner, "ZIEL", ziel)
    alter_stand = (scanner.date.today() - scanner.timedelta(days=200)).isoformat()
    ziel.write_text(json.dumps({"stand": alter_stand, "faelle": {}}), encoding="utf-8")

    aufrufe: list[tuple[str, str]] = []

    def _fake_suche(von: str, bis: str) -> list[dict]:
        aufrufe.append((von, bis))
        return []

    monkeypatch.setattr(scanner, "suche", _fake_suche)
    assert scanner.main() == 0
    assert aufrufe[0][0] == alter_stand, "Fenster nicht rueckwirkend geoeffnet"
    # 200 Tage Luecke in Scheiben <= RUECKBLICK_TAGE, lueckenlos verkettet
    assert len(aufrufe) >= 3
    for (_, b1), (v2, _) in zip(aufrufe, aufrufe[1:]):
        assert b1 == v2, "Scheiben nicht lueckenlos"
    heute = scanner.date.today().isoformat()
    assert aufrufe[-1][1] == heute
    assert "[WARN]" in capsys.readouterr().out


def test_main_abbruch_beim_write_laesst_watchlist_intakt(monkeypatch, tmp_path):
    """Atomarer Write als Invariante: scheitert der finale replace, bleibt die
    bestehende Watchlist (inkl. geprueft-Flags) byte-identisch erhalten."""
    ziel = tmp_path / "oddlot_watchlist.json"
    monkeypatch.setattr(scanner, "ZIEL", ziel)
    vorher = json.dumps(
        {
            "stand": scanner.date.today().isoformat(),
            "faelle": {"a-1": {"geprueft": True}},
        }
    )
    ziel.write_text(vorher, encoding="utf-8")
    monkeypatch.setattr(scanner, "suche", lambda von, bis: [])

    def _kaputt(src, dst):
        raise OSError("simulierter Abbruch")

    monkeypatch.setattr(scanner.os, "replace", _kaputt)
    with pytest.raises(OSError):
        scanner.main()
    assert ziel.read_text(encoding="utf-8") == vorher, "Watchlist verstuemmelt"
