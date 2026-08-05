"""P13c — Misst der Trendfilter einen Mechanismus oder zwei Ereignisse?

DIE FRAGE, DIE NACH P13/P13b OFFEN BLEIBT
-----------------------------------------
Der Filter besteht die Zielfunktion über ein breites Fensterband, über alle
drei Trend-Definitionen und auch mit einem Handelstag Verzögerung. Das
schließt „gefundener Parameter" aus. Es schließt **nicht** aus, dass der
gesamte Vorsprung aus zwei Ereignissen stammt.

Der Suchzeitraum ist 1995-01 bis 2016-12. Er enthält genau zwei Bärenmärkte
mit mehr als 40 % Rückgang: 2000–2002 und 2007–2009. Ein rollierendes
10-Jahres-Fenster, das in diesem Zeitraum beginnt, startet zwischen 1995 und
2006 — und trifft damit zwangsläufig mindestens einen der beiden. Ob es
überhaupt ein Fenster ohne Krise gibt, ist keine Meinung, sondern zählbar.
Genau das zählt dieses Modul.

Wenn die krisenfreie Gruppe leer ist, ist das der Befund: die Stichprobe kann
„Trendfolge wirkt" nicht von „Trendfolge hat diese beiden Abstürze umgangen"
unterscheiden. Die effektive Stichprobe für den Mechanismus wären dann zwei
Ereignisse, nicht 144 Fenster — und die 144 Fenster überlappen ohnehin
massiv, jeder Handelstag steckt in bis zu 120 von ihnen.

WELCHE KONFIGURATION HIER UNTERSUCHT WIRD — UND WARUM DIESE
-----------------------------------------------------------
`preis > SMA200`. **A priori gewählt**, nicht aus dem Ergebnis: das ist der
Lehrbuchparameter, den jede Darstellung der Trendfolge nennt, und er stand
schon in P4s Kontrollblock. Die beste Zelle des Rasters zu zerlegen wäre eine
zweite Auswahl auf denselben Daten; die Zerlegung des a-priori-Parameters ist
es nicht.

TRIAL-ZÄHLER
------------
Steigt **nicht**. Hier wird keine Konfiguration gesucht oder verglichen,
sondern eine bereits gezählte zerlegt (E-090). Die Fenstergruppierung nach
Benchmark-Rückgang ist eine Eigenschaft von SPY allein und hängt von keinem
Kandidaten ab.
"""

from __future__ import annotations

import json
import statistics
import warnings
from pathlib import Path

import pandas as pd  # noqa: F401

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import run_buy_and_hold  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.p5_gate_robustheit import FENSTER, gate_preis_ueber_sma  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

#: Ab welchem Benchmark-Rueckgang ein Fenster als Krisenfenster gilt. 30 % ist
#: eine Setzung, deshalb wird unten die ganze Verteilung mit ausgegeben — die
#: Aussage darf nicht an der Schwelle haengen. Zum Vergleich: der DD-Deckel der
#: Zielfunktion liegt bei 35 %, Buy-and-Hold erreicht im Suchfenster 55 %.
KRISEN_DD = -0.30

#: A priori, nicht aus dem Raster gewaehlt (siehe Modul-Docstring).
FENSTER_APRIORI = 200


def gruppiere(fenster) -> dict:
    """Fenster nach Benchmark-Rueckgang trennen und beide Gruppen beziffern."""
    krise = [f for f in fenster if f.benchmark_maxdd <= KRISEN_DD]
    ruhig = [f for f in fenster if f.benchmark_maxdd > KRISEN_DD]

    def kennzahlen(gruppe: list) -> dict:
        if not gruppe:
            # Fail-loud statt 0.0: eine leere Gruppe ist der interessante
            # Fall und darf nicht wie eine gemessene Null aussehen (E-103).
            return {
                "n": 0,
                "median_kandidat": None,
                "median_benchmark": None,
                "median_vorsprung_pp": None,
                "gewonnen": None,
            }
        k = statistics.median(f.kandidat_faktor for f in gruppe)
        b = statistics.median(f.benchmark_faktor for f in gruppe)
        return {
            "n": len(gruppe),
            "median_kandidat": k,
            "median_benchmark": b,
            "median_vorsprung_pp": (k - b) * 100.0,
            "gewonnen": sum(
                1 for f in gruppe if f.kandidat_faktor > f.benchmark_faktor
            ),
        }

    return {
        "krisenfenster": kennzahlen(krise),
        "ruhige_fenster": kennzahlen(ruhig),
        "schwelle": KRISEN_DD,
        "benchmark_dd_verteilung": {
            "schlimmster": min((f.benchmark_maxdd for f in fenster), default=None),
            "mildester": max((f.benchmark_maxdd for f in fenster), default=None),
            "median": statistics.median(f.benchmark_maxdd for f in fenster)
            if fenster
            else None,
        },
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print(
        f"Zerlegung von preis>SMA{FENSTER_APRIORI} (a priori). "
        f"Kein Trial-Increment — keine Suche.\n",
        flush=True,
    )

    # Warmlaufspanne des Fensterrasters. Gehoert ins Artefakt, weil der Befund
    # sie nennt und eine getippte Zahl dort schon einmal falsch war: "15 Monate"
    # war der ABSOLUTE Warmlauf bei Fenster 320, etikettiert als DIFFERENZ zu
    # Fenster 100 (F-auditor-2).
    reihe = d.close["SPY"].dropna()
    warm = {
        f: reihe.rolling(f).mean().first_valid_index()
        for f in (min(FENSTER), max(FENSTER))
    }
    spanne_tage = (warm[max(FENSTER)] - warm[min(FENSTER)]).days
    ergebnis: dict = {
        "fenster_apriori": FENSTER_APRIORI,
        "schwelle": KRISEN_DD,
        "warmlauf": {
            "fenster_klein": min(FENSTER),
            "fenster_gross": max(FENSTER),
            "erster_gueltiger_klein": str(warm[min(FENSTER)].date()),
            "erster_gueltiger_gross": str(warm[max(FENSTER)].date()),
            "differenz_tage": spanne_tage,
            "differenz_monate": round(spanne_tage / 30.44, 1),
        },
    }
    w = ergebnis["warmlauf"]
    print(
        f"Warmlauf: Fenster {w['fenster_klein']} ab {w['erster_gueltiger_klein']}, "
        f"Fenster {w['fenster_gross']} ab {w['erster_gueltiger_gross']} "
        f"-> Differenz {w['differenz_monate']} Monate\n",
        flush=True,
    )
    for welt, name, kwargs in [("ZERO", "ZERO", {}), ("PRIVAT_DE", "PRIVAT_DE", {})]:
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        gate = gate_preis_ueber_sma(d.close, FENSTER_APRIORI)
        r = run_buy_and_hold(d, make_regime(name, **kwargs), risk_off_gate=gate)
        a = auswerten(r.equity_netto, bench.equity_netto, label=f"{welt}/apriori")
        g = gruppiere(a.fenster)
        g["n_fenster"] = a.n_fenster
        ergebnis[welt] = g

        print(f"=== {welt} ===", flush=True)
        v = g["benchmark_dd_verteilung"]
        print(
            f"  Benchmark-MaxDD je Fenster: schlimmster {v['schlimmster']:.1%} | "
            f"median {v['median']:.1%} | mildester {v['mildester']:.1%}"
        )
        for gruppe, titel in [
            ("krisenfenster", f"mit Rueckgang <= {KRISEN_DD:.0%}"),
            ("ruhige_fenster", f"ohne Rueckgang <= {KRISEN_DD:.0%}"),
        ]:
            k = g[gruppe]
            if k["n"] == 0:
                print(f"  {titel:<28} 0 Fenster — GRUPPE LEER")
                continue
            print(
                f"  {titel:<28} {k['n']:>3} Fenster | Kandidat "
                f"{k['median_kandidat']:.3f}x vs Benchmark "
                f"{k['median_benchmark']:.3f}x | Vorsprung "
                f"{k['median_vorsprung_pp']:+.1f} pp | gewonnen "
                f"{k['gewonnen']}/{k['n']}"
            )
        print(flush=True)

    # Artefakt als LETZTE Anweisung (E-116).
    (OUT / "p13c_ereignisabhaengigkeit.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"-> {OUT / 'p13c_ereignisabhaengigkeit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
