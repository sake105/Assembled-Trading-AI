"""Wie viele rollierende 10-Jahres-Fenster sind krisenfrei — kurz und lang?

WARUM DIESES SKRIPT EXISTIERT
-----------------------------
Die Zahlen "338 von 1.080" und "0 von 144" standen nur im Docstring von
`pull_gratis_quellen.py`. Kein committeter Code erzeugte sie — derselbe
Zustand, den E-125 fuer die Form-4-Abdeckung beanstandet, im selben Commit
(F-auditor-8). Eine Zahl, die kein Skript reproduziert, ist keine Messung.

WAS GEMESSEN WIRD
-----------------
Auf der CRSP-Marktreihe (Ken French, taeglich ab 1926) der Anteil rollierender
10-Jahres-Fenster ohne nennenswerten Rueckgang — einmal ueber die volle
Historie, einmal ueber das Suchfenster der Kampagne (1995–2016).

Das beantwortet die Frage, an der P13 prinzipiell gescheitert ist: dort war
KEIN einziges der 144 Fenster krisenfrei, "Trendfolge wirkt" liess sich also
nicht von "Trendfolge hat zwei Abstuerze umgangen" trennen.

WIE WEIT DAS TRAEGT
-------------------
Die krisenfreien Fenster ueberlappen monatlich und stammen aus wenigen
zusammenhaengenden Bloecken. Nicht-ueberlappend bleiben nur eine Handvoll.
Das Skript weist deshalb BEIDES aus — die Fensterzahl und die Zahl der
disjunkten Bloecke. Wer aus der grossen Zahl Signifikanz ableitet, wiederholt
E-078: die effektive Stichprobe ist die Zahl der unabhaengigen Ereignisse.

KEIN TRIAL
----------
Reine Beschreibung der Datenlage, keine Strategie, kein Kandidat (E-090).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.mandat2.metrics import (  # noqa: E402
    FENSTER_JAHRE,
    max_drawdown,
    rolling_windows,
)

GRATIS = Path(__file__).resolve().parent / "data_gratis"
ZIEL = Path(__file__).resolve().parent / "krisenfreie_fenster.json"

#: Ab welchem Rueckgang ein Fenster als Krisenfenster gilt. Setzung — deshalb
#: wird unten die ganze Verteilung mit ausgegeben, damit die Aussage nicht an
#: der Schwelle haengt.
KRISEN_DD = -0.30

#: Das Suchfenster der Kampagne. Der Vergleich mit der vollen Historie ist der
#: eigentliche Zweck des Skripts.
SUCHE = ("1995-01-01", "2016-12-30")


def auswerten(kurs: pd.Series, label: str) -> dict:
    fenster = list(
        rolling_windows(
            pd.DatetimeIndex(kurs.index), jahre=FENSTER_JAHRE, schritt_monate=1
        )
    )
    dds = [(a, max_drawdown(kurs.loc[a:b])) for a, b in fenster]
    ruhig = [a for a, d in dds if d > KRISEN_DD]

    # Disjunkte Bloecke: aufeinanderfolgende krisenfreie Fensterstarts, die
    # weniger als ein Fenster auseinanderliegen, gehoeren zur selben Episode.
    bloecke = 0
    letzter: pd.Timestamp | None = None
    for start in ruhig:
        if letzter is None or (start - letzter).days > FENSTER_JAHRE * 365:
            bloecke += 1
        letzter = start

    werte = [d for _, d in dds]
    return {
        "label": label,
        "von": str(kurs.index.min().date()),
        "bis": str(kurs.index.max().date()),
        "n_fenster": len(fenster),
        "krisenfrei": len(ruhig),
        "krisenfrei_anteil": round(len(ruhig) / len(fenster), 4) if fenster else None,
        "disjunkte_bloecke": bloecke,
        "mildester_rueckgang": round(max(werte), 4) if werte else None,
        "schlimmster_rueckgang": round(min(werte), 4) if werte else None,
        "schwelle": KRISEN_DD,
    }


def main() -> int:
    p = GRATIS / "fama_french_daily.parquet"
    if not p.exists():
        raise SystemExit(
            f"[ERROR] {p.name} fehlt — erst pull_gratis_quellen.py laufen lassen."
        )
    # Der Spaltenname sagt, was die Reihe ist: CRSP value-weighted, NICHT SPY.
    kurs = pd.read_parquet(p)["index_crsp_vw"]

    ergebnis = {
        "quelle": "Ken French Data Library (CRSP value-weighted), taeglich",
        "voll": auswerten(kurs, "1926-2026 (voll)"),
        "suchfenster": auswerten(kurs.loc[SUCHE[0] : SUCHE[1]], "1995-2016 (Suche)"),
        "einordnung": (
            "Die krisenfreien Fenster ueberlappen monatlich; massgeblich fuer "
            "die effektive Stichprobe ist 'disjunkte_bloecke', nicht "
            "'krisenfrei' (E-078)."
        ),
    }
    for schluessel in ("voll", "suchfenster"):
        e = ergebnis[schluessel]
        print(
            f"  {e['label']:<20} {e['n_fenster']:>5} Fenster | krisenfrei "
            f"{e['krisenfrei']:>4} ({e['krisenfrei_anteil']:.0%}) in "
            f"{e['disjunkte_bloecke']} Bloecken | mildester "
            f"{e['mildester_rueckgang']:.1%} | schlimmster "
            f"{e['schlimmster_rueckgang']:.1%}"
        )
    # Artefakt als LETZTE Anweisung (E-116).
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
