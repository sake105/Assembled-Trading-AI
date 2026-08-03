"""Skip-Guard für Tests, die die echten Kampagnendaten brauchen.

WARUM DAS NÖTIG IST
-------------------
``research/mandat/data/`` ist gitignored (~4,1 GB Kursdaten). In CI existiert
das Verzeichnis deshalb nicht — und ein Test, der ``load_campaign()`` aufruft,
scheitert dort mit einem Dateifehler statt zu skippen.

Genau das ist am 2026-08-01 passiert und **zwei Tage lang unbemerkt geblieben**:
CI war ab Commit ``b9656969`` rot (``CI`` und ``Backend CI``), fünf Commits
lang, weil die Mandat-II-Arbeit ausschließlich in ``research/`` stattfand und
der CI-Status nie geprüft wurde. Lokal war alles grün — die Daten liegen hier.

Das ``exists()``+``skip``-Muster war im Repo **längst etabliert** (Dutzende
Testdateien nutzen es gegen gitignorierte Pfade). Es fehlte nicht — es wurde
bei den Mandat-II-Tests nicht angewandt. Diese Datei bündelt es für die
Kampagne, weil ``load_campaign()`` mehrere Dateien braucht und eine einzelne
``exists()``-Prüfung genau deshalb zu kurz greift.

DIE UNTERSCHEIDUNG, AUF DIE ES ANKOMMT
--------------------------------------
Ein *erwarteter Skip* wegen fehlender Umgebungsvoraussetzung ist etwas anderes
als ein Fehler (Rule 40). Aber die Grenze zum Weichspülen ist schmal, und ich
habe sie beim ersten Anlauf überschritten: markiert waren sechs Tests, von
denen **fünf datenfrei laufen** — darunter die einzige CI-Abdeckung der
Holdout-Suchsperre, also des P0-Schutzmechanismus der Kampagne (E-092).

**Regel für diese Marke:** vor dem Setzen nachweisen, dass der Test ohne die
Daten wirklich scheitert — Datenpfad wegschieben, Test laufen lassen. Wer das
nicht gemessen hat, setzt die Marke nicht.

WARUM DER PFAD IMPORTIERT UND NICHT NACHGEBAUT WIRD
---------------------------------------------------
``DATA`` kommt aus ``campaign_data`` selbst. Ein Guard, der die Abhängigkeiten
seines Ziels dupliziert statt sie abzuleiten, veraltet lautlos — genau so war
die erste Fassung entstanden, die nur das Preispanel prüfte und bei fehlender
``dividends.parquet`` in denselben ``FileNotFoundError`` lief, den sie
verhindern sollte. ``test_mandat2_daten_guard.py`` koppelt die Dateiliste
zusätzlich per Regressionstest an die tatsächlichen Lesezugriffe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research.mandat2.campaign_data import DATA

WURZEL = Path(__file__).resolve().parents[1]

# Alle Dateien, die campaign_data.load_campaign() liest, mit Beschaffungsweg.
# Der Hinweis ist vollstaendig im Wert hinterlegt (kein festes "via " im
# f-String), weil nicht jede Datei von einem Pull-Skript erzeugt wird.
NOETIG: dict[Path, str] = {
    DATA / "prices_verdict.parquet": "via research/mandat/pull_eodhd_verdict.py",
    DATA / "dividends.parquet": "via research/mandat/pull_dividends.py",
    DATA / "sp500_historical_constituents.csv": "kein Pull-Skript, separat beschaffen",
}

_fehlend = [p for p in NOETIG if not p.exists()]

# ASCII im reason: der String landet im CI-Report, und die Windows-Konsole der
# CI-Matrix zerlegt Em-Dashes. Der Anker "SKIP (nicht geprueft)" steht vorn,
# damit er eine Rechts-Truncation ueberlebt (vgl. E-066).
braucht_kampagnendaten = pytest.mark.skipif(
    bool(_fehlend),
    reason=(
        "SKIP (nicht geprueft): Kampagnendaten fehlen - "
        + "; ".join(
            f"{p.relative_to(WURZEL).as_posix()} ({NOETIG[p]})" for p in _fehlend
        )
        + ". research/mandat/data/ ist gitignored (~4,1 GB)."
    ),
)
