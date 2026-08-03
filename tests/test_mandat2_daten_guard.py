"""Der Skip-Guard selbst — er war der Fehler, also braucht er einen Test.

Zwei Fehlermodi sind hier zu verhindern, und beide sind schon eingetreten:

1. **Der Guard greift zu kurz** (Stage-1-Finding F-test-3). Die erste Fassung
   prüfte nur ``prices_verdict.parquet``. Fehlte ``dividends.parquet`` bei
   vorhandenen Preisen, lief der Test in genau den ``FileNotFoundError``, den
   der Guard verhindern soll. Der Regressionstest unten koppelt die Dateiliste
   an die tatsächlichen Lesezugriffe in ``campaign_data`` — kommt dort eine
   vierte Quelle dazu, fällt dieser Test, nicht erst die CI.

2. **Der Guard greift zu weit** (E-092). Beim ersten Anlauf waren sechs Tests
   markiert, von denen fünf datenfrei laufen — darunter die einzige
   CI-Abdeckung der Holdout-Suchsperre. Deshalb wird hier auch geprüft, dass
   die Marke sparsam vergeben ist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from research.mandat2.campaign_data import DATA
from tests.mandat2_daten_guard import NOETIG, braucht_kampagnendaten

pytestmark = pytest.mark.fast

WURZEL = Path(__file__).resolve().parents[1]
CAMPAIGN_DATA_QUELLE = WURZEL / "research" / "mandat2" / "campaign_data.py"


def _gelesene_dateien() -> set[str]:
    """Doppelt gequotete ``DATA / "..."``-Literale in campaign_data.py.

    Bewusst enger formuliert als „was der Loader liest“: die Regex findet
    kein ``DATA / name``, kein f-String und keinen Lesezugriff in einem
    anderen Modul des ``load_campaign()``-Pfades. Sie deckt den heutigen Code
    vollstaendig ab und ist ein Stolperdraht gegen die haeufigste Aenderung
    (eine weitere Datei), kein Vollstaendigkeitsbeweis (F-auditor-3).
    """
    quelle = CAMPAIGN_DATA_QUELLE.read_text(encoding="utf-8")
    return set(re.findall(r'DATA\s*/\s*"([^"]+)"', quelle))


class TestGuardDeckungGegenDrift:
    def test_guard_kennt_jede_von_campaign_data_gelesene_datei(self):
        """Verhindert die Rueckkehr von F-test-3: neue Quelle, blinder Guard."""
        gelesen = _gelesene_dateien()
        assert gelesen, "Regex fand keine DATA/-Zugriffe — Test ist wirkungslos"
        bewacht = {p.name for p in NOETIG}
        fehlend = gelesen - bewacht
        assert not fehlend, (
            f"campaign_data liest {sorted(fehlend)}, der Guard prueft sie nicht. "
            "NOETIG in tests/mandat2_daten_guard.py ergaenzen."
        )

    def test_guard_bewacht_nichts_ueberfluessiges(self):
        """Die Gegenrichtung — sonst skippt der Guard wegen irrelevanter Dateien."""
        ueberzaehlig = {p.name for p in NOETIG} - _gelesene_dateien()
        assert not ueberzaehlig, (
            f"Guard prueft {sorted(ueberzaehlig)}, campaign_data liest sie nicht."
        )

    def test_pfade_haengen_am_produktionsmodul(self):
        """Kein nachgebauter Pfad — sonst driftet der Guard beim naechsten Umzug."""
        for p in NOETIG:
            assert p.parent == DATA


class TestGuardSparsamkeit:
    """E-092: die Marke darf nur dort stehen, wo die Daten wirklich noetig sind."""

    def test_marke_wird_sparsam_vergeben(self):
        """Stolperdraht, kein Beweis.

        Die Grenze prueft Quantitaet, nicht Berechtigung — sie kann eine
        unnoetige Marke nicht von einer noetigen unterscheiden. Sie zwingt aber
        dazu, beim Ueberschreiten zu begruenden, und genau das hat beim ersten
        Anlauf gefehlt.

        Gezaehlt wird ZEILENVERANKERT und ueber ``tests/**/*.py``: eine reine
        ``count()``-Suche zaehlte diese Datei mit (das Literal steht im
        Fehlertext) und war namensgekoppelt an ``test_mandat2_*``
        (Stage-3-Finding F-auditor-2).
        """
        muster = re.compile(r"(?m)^\s*@braucht_kampagnendaten\s*$")
        treffer = []
        for f in sorted(WURZEL.glob("tests/**/*.py")):
            n = len(muster.findall(f.read_text(encoding="utf-8")))
            if n:
                treffer.append((f.name, n))
        gesamt = sum(n for _, n in treffer)
        assert gesamt <= 3, (
            f"{gesamt} Tests tragen die Marke ({treffer}). Fuenf der ersten "
            "sechs Marken waren unnoetig (E-092). Bevor die Grenze steigt: je "
            "Test messen — Datenpfad wegschieben, Test einzeln laufen lassen — "
            "und die Grenze mit diesem Beleg anheben."
        )


class TestSkipGrund:
    def test_anker_steht_vorn(self):
        """`SKIP (nicht geprueft)` muss eine Rechts-Truncation ueberleben (E-066)."""
        grund = braucht_kampagnendaten.kwargs.get("reason", "")
        assert grund, "Skip-Grund ist leer — der CI-Report traegt dann keinen Klartext"
        assert grund.startswith("SKIP (nicht geprueft)")

    def test_grund_ist_ascii(self):
        """Die Windows-Konsole der CI-Matrix zerlegt Nicht-ASCII im Report."""
        grund = braucht_kampagnendaten.kwargs.get("reason", "")
        nicht_ascii = [c for c in grund if ord(c) > 127]
        assert not nicht_ascii, f"Nicht-ASCII im Skip-Grund: {nicht_ascii}"
