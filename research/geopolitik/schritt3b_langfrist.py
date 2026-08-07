"""Welle 48b — Post-hoc-Horizonterweiterung (20/60/120/250 Tage).

Auf Auftrag, nach gefallenem Abbruchkriterium. +2 Trials, vorab gebucht.
Nutzt die Studien-Maschinerie aus schritt3 unveraendert (keine zweite
Wahrheit) — nur die Horizonte sind andere.

Ein Bestehen hier waere eine NEUE Hypothese ("Aussagen wirken verzoegert"),
kein Wiederaufleben von Welle 48 — so vorab registriert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.geopolitik import schritt3_ereignisstudie as s3  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402

ZIEL = HIER / "schritt3b_langfrist.json"
HORIZONTE_LANG = (20, 60, 120, 250)


def main() -> int:
    s3.HORIZONTE = HORIZONTE_LANG  # Messung, kein zweiter Pfad (E-102)
    ev = pd.read_parquet(s3.EREIGNISSE)
    ev["zeit_et"] = pd.to_datetime(ev["zeit_et"])
    kurse = s3.lade_kurse()
    print(
        "Trials kumuliert: "
        + str(TrialCounter().increment(2, label="Welle 48b Langfrist")),
        flush=True,
    )
    ergebnis = {
        "registriert": "Welle 48b, post-hoc auf Auftrag, +2 vorab",
        "hinweis_e078": (
            "250-Tage-Fenster desselben Tickers ueberlappen massiv; t-Werte "
            "sind nach oben verzerrt. Ein Bestehen waere eine NEUE Hypothese, "
            "kein Wiederaufleben von Welle 48."
        ),
        "A": s3.studie(ev[ev.regel == "A"], kurse, "A"),
        "B": s3.studie(ev[ev.regel == "B"], kurse, "B"),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for r in ("A", "B"):
        e = ergebnis[r]
        print(f"Regel {r}: {e['n_ereignisse']} Ereignisse")
        for hz, v in e["horizonte"].items():
            print(
                f"  {hz:>5}: mittel {v['mittel_pp']} pp (t={v['t']}, n={v['n']}) | "
                f"Kontrolle {v['kontrolle_mittel_pp']} pp | "
                f"{'PASS' if v['pass_t2'] else 'fail'}"
            )
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
