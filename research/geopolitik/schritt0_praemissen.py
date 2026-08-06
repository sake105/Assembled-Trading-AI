"""Schritt 0 — Tragen die Anker-Beispiele der Geopolitik-These?

AUFTRAG (Hans, 2026-08-06)
--------------------------
Handel auf schnelle, gerichtete Aussagen einzelner Akteure ("Trump: buy a
Dell" / Iran-Waffenruhe). VOR jedem Bau wird geprueft, ob die erinnerten
Anker-Beispiele echten Kursdaten standhalten — und was davon ab dem ersten
HANDELBAREN Kurs uebrig ist.

WAS DIESER SCHRITT NICHT IST
----------------------------
Kein Trial, keine Strategie, keine Ereignisstudie. Zwei erinnerte Beispiele
werden verifiziert und zerlegt. Ein systematisches, erinnerungsfreies
Ereignis-Universum ist Schritt 1 — und dessen Notwendigkeit ist gerade das
Ergebnis dieses Schritts: erinnert werden die Treffer, nicht die Nieten.

EXTERNE FAKTEN (Web-Recherche 2026-08-06, im Artefakt als Quellenangabe)
------------------------------------------------------------------------
* Dell: Trump kaufte lt. Ethics-Filing ~10.02.2026 fuer 1-5 Mio USD Dell;
  Kundgebung Rome/Georgia 19.02.2026 ("go out and buy a Dell computer");
  Wiederholung im Weissen Haus 08.05.2026; dritte Empfehlung Anfang Juli.
  Presse: Verdreifachung seit Februar "primaer AI-Server-Nachfrage".
* Iran: Waffenruhe-Ankuendigung ~1 Uhr nachts in der Nacht auf den
  24.06.2025 — also AUSSERHALB der Handelszeiten. Wenige Tage spaeter
  "ceasefire is over"-Aussage mit Gegenbewegung.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "research" / "mandat" / "data"
ZIEL = HIER / "schritt0_praemissen.json"


def kurse() -> tuple[pd.Series, pd.Series]:
    pv = pd.read_parquet(DATA / "prices_verdict.parquet")
    spy = pv[pv["symbol"] == "SPY"].set_index("timestamp")["close"]
    spy.index = spy.index.tz_localize(None)
    dell = pd.read_parquet(DATA / "_sc_close.parquet", columns=["DELL"])[
        "DELL"
    ].dropna()
    return spy.dropna(), dell


def schritt(reihe: pd.Series, a: str, b: str) -> float:
    return float(reihe.loc[b] / reihe.loc[a] - 1.0)


def main() -> int:
    spy, dell = kurse()

    iran = {
        "ereignis": "Waffenruhe-Ankuendigung ~01:00 nachts auf den 2025-06-24",
        "handelbarkeit": (
            "AUSSERHALB der Handelszeiten. Der Schluss-zu-Schluss-Move des "
            "Folgetags enthaelt die Eroeffnungsluecke; was davon nach der "
            "Eroeffnung noch uebrig war, ist mit Tagesdaten NICHT messbar "
            "(kein Open im Panel, kein Intraday fuer SPY auf Platte)."
        ),
        "vortag_23_06": round(schritt(spy, "2025-06-20", "2025-06-23"), 4),
        "ankuendigungsnacht_24_06": round(schritt(spy, "2025-06-23", "2025-06-24"), 4),
        "folgetag_25_06": round(schritt(spy, "2025-06-24", "2025-06-25"), 4),
        "befund": (
            "Richtung bestaetigt, Groesse relativiert: +1,10 % Schluss-zu-"
            "Schluss (Presse nannte bis 2,5 % — vermutlich ab Intraday-Tief "
            "oder inkl. Vortag, der schon +0,99 % lief). Folgetag +0,06 %: "
            "nach dem ersten handelbaren Kurs war praktisch nichts mehr da."
        ),
    }

    dell_befund = {
        "ereignisse": {
            "2026-02-19 Kundgebung": {
                "tag": round(schritt(dell, "2026-02-18", "2026-02-19"), 4),
                "folgetag": round(schritt(dell, "2026-02-19", "2026-02-20"), 4),
            },
            "2026-05-08 Weisses Haus": {
                "tag": round(schritt(dell, "2026-05-07", "2026-05-08"), 4),
                "folgetag": round(schritt(dell, "2026-05-08", "2026-05-11"), 4),
            },
        },
        "gesamt_feb_bis_panelende": round(schritt(dell, "2026-02-18", "2026-07-08"), 4),
        "befund": (
            "Die erinnerten +100-250 % existieren (+271 % Feb-Jul), sind aber "
            "NICHT der Ereigniseffekt: die Ereignistage summieren sich auf "
            "rund 30 der 271 pp, die Presse nennt AI-Server-Nachfrage als "
            "Haupttreiber. Der staerkste Einzeleffekt (08.05., +13,1 % am "
            "Tag) war ein Intraday-Move waehrend der Handelszeit — wer ihn "
            "nicht binnen Minuten fing, sondern zum Schluss kaufte, verlor "
            "am Folgetag -5,2 %. Das Ereignis vom 19.02. bewegte fast nichts "
            "(+1,9 %) — derselbe Satz, dreimal gesagt, wirkte voellig "
            "verschieden."
        ),
    }

    ergebnis = {
        "schritt": 0,
        "frage": "Tragen die Anker-Beispiele der Geopolitik-These?",
        "iran": iran,
        "dell": dell_befund,
        "gesamtbefund": (
            "Praemisse TEILWEISE bestaetigt. Die Ereignisse existieren und "
            "bewegen in die richtige Richtung. Aber (1) der handelbare Anteil "
            "ist um eine Groessenordnung kleiner als die erinnerte Zahl, "
            "(2) der groesste Teil der Dell-Rally ist fundamental, nicht "
            "ereignisgetrieben, (3) identische Aussagen wirken nicht "
            "reproduzierbar (Feb ~0, Mai +13 %), (4) es gibt scharfe "
            "Gegenbewegungen (Dell -5,2 % Folgetag; 'ceasefire over'). "
            "Konsequenz: Schritt 1 (erinnerungsfreies Ereignis-Universum) "
            "ist noetig, BEVOR irgendeine Strategie formuliert wird — die "
            "Erinnerung selektiert nachweislich die Treffer."
        ),
        "datenluecken": [
            "Kein Open-Kurs im Panel: Eroeffnungsluecke vs. handelbarer Rest "
            "nicht trennbar (Iran).",
            "Kein Intraday fuer DELL/SPY auf Platte: der +13-%-Tag ist in "
            "Minutenaufloesung nicht zerlegbar.",
        ],
        "quellen": [
            "Yahoo Finance / Barchart / t-online Berichte zu Trump-Dell "
            "(Feb/Mai/Jul 2026), Ethics-Filing-Berichterstattung",
            "NBC/Fox/Bloomberg zur Iran-Waffenruhe 23./24.06.2025",
        ],
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(ergebnis["iran"], indent=2, ensure_ascii=False)[:400])
    print(json.dumps(ergebnis["dell"], indent=2, ensure_ascii=False)[:400])
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
