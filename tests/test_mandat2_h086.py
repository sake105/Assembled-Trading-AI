"""Tests fuer H-086 — Traegt der Trendfilter auch ohne Dauerkrise?

Geprueft wird die Aufspaltung nach Krisen- und krisenfreien Fenstern. Sie
traegt das Verdikt: der Gesamtmedian ist ausdruecklich NICHT die
Entscheidungsgroesse (Welle 47), und die Zahl disjunkter Bloecke ist es, die
aus „338 Fenster" eine belastbare Aussage macht oder eben nicht (E-078).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.mandat2.h086_trendfilter_lang import aufspalten
from research.mandat2.metrics import FensterErgebnis
from research.mandat2.p13c_ereignisabhaengigkeit import KRISEN_DD

ARTEFAKT = (
    Path(__file__).resolve().parents[1]
    / "research/mandat2/results/h086_trendfilter_lang.json"
)


def _f(start: str, bench_dd: float, kand: float, bench: float) -> FensterErgebnis:
    s = pd.Timestamp(start)
    return FensterErgebnis(
        start=s,
        ende=s + pd.DateOffset(years=10),
        kandidat_faktor=kand,
        benchmark_faktor=bench,
        kandidat_maxdd=-0.2,
        benchmark_maxdd=bench_dd,
    )


def test_trennt_an_der_krisenschwelle() -> None:
    """Beide Raender pruefen, nicht die Mitte (E-104)."""
    s = aufspalten([_f("1950-01-01", KRISEN_DD, 2.0, 1.5)], "t")
    assert s["krisenfenster"]["n"] == 1, "die Schwelle selbst zaehlt als Krise"
    assert s["krisenfreie_fenster"]["n"] == 0
    s2 = aufspalten([_f("1950-01-01", KRISEN_DD + 1e-9, 2.0, 1.5)], "t")
    assert s2["krisenfreie_fenster"]["n"] == 1


def test_leere_gruppe_meldet_none_statt_null() -> None:
    """E-103: eine leere Gruppe darf nicht wie ein gemessener Nullvorsprung
    aussehen — im Suchfenster von P13 war genau sie der Befund."""
    s = aufspalten([_f("1950-01-01", -0.5, 2.0, 1.5)], "t")
    leer = s["krisenfreie_fenster"]
    assert leer["n"] == 0
    assert leer["vorsprung_pp"] is None
    assert leer["gewonnen"] is None


def test_vorsprung_ist_in_prozentpunkten() -> None:
    s = aufspalten([_f("1950-01-01", -0.5, 2.0, 1.5)], "t")
    assert s["krisenfenster"]["vorsprung_pp"] == pytest.approx(50.0)


def test_disjunkte_bloecke_fassen_ueberlappende_fenster_zusammen() -> None:
    """Die 338 krisenfreien Fenster ueberlappen monatlich. Wer sie als 338
    unabhaengige Belege liest, wiederholt E-078."""
    dicht = [_f(f"1950-0{m}-01", -0.1, 2.0, 1.5) for m in range(1, 5)]
    assert aufspalten(dicht, "t")["krisenfreie_fenster"]["disjunkte_bloecke"] == 1
    weit = [_f("1950-01-01", -0.1, 2.0, 1.5), _f("1990-01-01", -0.1, 2.0, 1.5)]
    assert aufspalten(weit, "t")["krisenfreie_fenster"]["disjunkte_bloecke"] == 2


def test_gewonnen_zaehlt_nur_echte_siege() -> None:
    s = aufspalten(
        [
            _f("1950-01-01", -0.1, 2.0, 1.5),
            _f("1960-01-01", -0.1, 1.5, 1.5),
            _f("1970-01-01", -0.1, 1.0, 1.5),
        ],
        "t",
    )
    assert s["krisenfreie_fenster"]["gewonnen"] == 1
    assert s["krisenfreie_fenster"]["n"] == 3


def test_artefakt_traegt_das_verdikt() -> None:
    """Das Verdikt darf nicht gegen die eigenen Zahlen stehen (E-085)."""
    if not ARTEFAKT.exists():
        pytest.skip("H-086 noch nicht gelaufen")
    d = json.loads(ARTEFAKT.read_text(encoding="utf-8"))
    kf = d["aufspaltung"]["krisenfreie_fenster"]
    traegt = d["verdikt"]["traegt_ohne_krise"]
    assert traegt == (kf["vorsprung_pp"] is not None and kf["vorsprung_pp"] > 0)
    # Der gemessene Befund: in krisenfreien Fenstern gewinnt der Filter nichts.
    assert kf["n"] > 0, "ohne krisenfreie Fenster waere der Lauf sinnlos"
    assert traegt is False
    assert kf["gewonnen"] == 0
