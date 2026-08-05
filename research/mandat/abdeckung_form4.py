"""Wie viel des handelbaren Universums deckt der Form-4-Bestand ab?

WARUM DIESES SKRIPT EXISTIERT
-----------------------------
Die Zahl "6.146 von 8.876 (69,2 %)" stand in einer Commit-Message, ohne dass
irgendein committeter Code sie erzeugt haette. Eine Zahl, die kein Skript
reproduziert, ist keine Messung, sondern eine Erinnerung — sie wird spaeter
zitiert, obwohl niemand sie nachrechnen kann (E-125).

WAS DIE ZAHL WERT IST — UND WAS NICHT
--------------------------------------
Der Abgleich laeuft ueber den **rohen Ticker-String**, und der ist in BEIDE
Richtungen unbeschraenkt:

* **zu niedrig**, weil das Preispanel EODHD-Suffixe fuehrt (`AAC_old2`), die
  im SEC-Bestand nie so heissen;
* **zu hoch**, weil Ticker recycelt werden — Panel-Ticker X (Firma A,
  2003–2008) gegen SEC-Ticker X (Firma B, 2015) zaehlt als abgedeckt, obwohl
  es zwei verschiedene Unternehmen sind.

Das ist exakt Befund 7 (Ticker ist kein Schluessel), und zwar in genau dem
Modul, das den ersten Pull-Ansatz mit derselben Begruendung verworfen hat. Die
saubere Loesung braucht eine CIK-Bruecke fuer das Preispanel; die gibt es
nicht. Deshalb wird die Zahl **mit ihrer Fehlerrichtung** ausgewiesen und
NICHT als Deckungsaussage.

Belastbar ist dagegen die untere Schranke ueber die Emittenten-CIK: wie viele
verschiedene Unternehmen der Bestand ueberhaupt fuehrt.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).resolve().parent / "data"
DERA = DATA / "form4_dera"
GAP = DATA / "form4_gap_symbols.csv"
ZIEL = Path(__file__).resolve().parent / "abdeckung_form4.json"


def main() -> int:
    dateien = sorted(glob.glob(str(DERA / "*.parquet")))
    if not dateien or not GAP.exists():
        raise SystemExit(
            "[ERROR] Bestand oder Ziel-Liste fehlt — erst pull_form4_dera.py "
            "laufen lassen. (research/mandat/data/ ist gitignored.)"
        )
    df = pd.concat(
        [pd.read_parquet(p, columns=["symbol", "ISSUERCIK"]) for p in dateien],
        ignore_index=True,
    )
    ziel = set(pd.read_csv(GAP)["symbol"].astype(str).str.upper())
    haben = set(df["symbol"].dropna().unique())
    treffer = ziel & haben

    ergebnis = {
        "quelle": "research/mandat/data/form4_dera (SEC DERA, ab 2006Q1)",
        "emittenten_cik": int(df["ISSUERCIK"].nunique()),
        "ticker_im_bestand": int(len(haben)),
        "zeilen_ohne_ticker": int(df["symbol"].isna().sum()),
        "ziel_universum": len(ziel),
        "ziel_getroffen_roher_ticker": len(treffer),
        "ziel_quote_roher_ticker": round(len(treffer) / len(ziel), 4),
        "warnung": (
            "Der Abgleich ist ein ROHER TICKER-STRING-VERGLEICH und in beide "
            "Richtungen unbeschraenkt: zu niedrig durch EODHD-Suffixe "
            "(_old), zu hoch durch recycelte Ticker (Befund 7). KEINE "
            "Deckungsaussage. Belastbar ist nur 'emittenten_cik'."
        ),
    }
    # Artefakt als LETZTE Anweisung (E-116).
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for k, v in ergebnis.items():
        print(f"  {k}: {v}")
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
