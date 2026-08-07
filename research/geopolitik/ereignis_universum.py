"""Schritt 1b — Regeln A/B auf das Vollarchiv anwenden (Welle 48, kein Trial).

Die Regeln und Wortlisten stehen in Welle 48 und sind dort VERBRAUCHT — hier
werden sie angewandt, nicht verbessert. Es wird KEIN Kurs angefasst: Ausgabe
ist das Ereignis-Universum (Zeitstempel ET, Ticker, Richtung), sonst nichts.

ZEITZONE: verifiziert ET — der Waffenruhe-Post (Status 31624) traegt im Archiv
"June 23, 2025, 6:02 PM" und ist extern auf 18:02 ET dokumentiert.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ARCHIV = HIER / "data" / "trumpstruth"
NAMEN = HIER / "namensliste.json"
ZIEL_EREIGNISSE = HIER / "data" / "ereignisse.parquet"
ZIEL_STATS = HIER / "ereignis_universum.json"

# --- Wortlisten: WOERTLICH aus Welle 48. Aenderung = neue Registrierung. ---
A_POS = {
    "buy",
    "invest",
    "great",
    "best",
    "tremendous",
    "incredible",
    "deal",
    "approved",
    "winner",
}
A_NEG = {
    "tariff",
    "tariffs",
    "sue",
    "suing",
    "investigate",
    "investigation",
    "boycott",
    "against",
    "fire",
    "fired",
    "crooked",
    "failing",
}
B_DEESK = ["ceasefire", "peace deal", "end the war", "truce", "agreement reached"]
B_ESK = [
    "strike",
    "strikes",
    "attack",
    "bombing",
    "tariff",
    "tariffs",
    "sanctions",
    "blockade",
]


def woerter(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def phrasen_treffer(text: str, phrasen: list[str]) -> bool:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return any(
        re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", t) for p in phrasen
    )


def main() -> int:
    fs = sorted(glob.glob(str(ARCHIV / "chunk_*.parquet")))
    if not fs:
        raise SystemExit("[ERROR] Archiv fehlt — erst pull_trumpstruth.py laufen.")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates("status_id").sort_values("status_id")
    df["zeit_et"] = pd.to_datetime(df["zeit_roh"], format="%B %d, %Y, %I:%M %p")
    df["text"] = df["text"].fillna("")

    namen = json.loads(NAMEN.read_text(encoding="utf-8"))["namen"]
    # Vorkompilierte Ganzwort-Muster je Suchname (case-insensitiv; Grossschrift
    # in Posts ist die Regel, nicht die Ausnahme).
    muster = {
        n: re.compile(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", re.I)
        for n in namen
    }

    ereignisse: list[dict] = []
    stats = {
        "posts": len(df),
        "mit_text": int((df["text"].str.len() > 0).sum()),
        "a_firmentreffer": 0,
        "a_mehrere_firmen": 0,
        "a_ohne_richtung": 0,
        "a_gemischt": 0,
        "a_ereignisse": 0,
        "b_treffer": 0,
        "b_beide_listen": 0,
        "b_ereignisse": 0,
    }

    for _, z in df.iterrows():
        text = z["text"]
        if not text:
            continue
        w = woerter(text)

        # --- Regel A: genau EIN Unternehmen + eindeutige Richtung
        getroffen = {t for n, t in namen.items() if muster[n].search(text)}
        if getroffen:
            stats["a_firmentreffer"] += 1
            if len(getroffen) > 1:
                stats["a_mehrere_firmen"] += 1
            else:
                pos, neg = bool(w & A_POS), bool(w & A_NEG)
                if pos and neg:
                    stats["a_gemischt"] += 1
                elif not pos and not neg:
                    stats["a_ohne_richtung"] += 1
                else:
                    stats["a_ereignisse"] += 1
                    ereignisse.append(
                        {
                            "status_id": int(z["status_id"]),
                            "zeit_et": z["zeit_et"],
                            "regel": "A",
                            "ticker": next(iter(getroffen)),
                            "richtung": 1 if pos else -1,
                        }
                    )

        # --- Regel B: Makro auf SPY
        deesk = phrasen_treffer(text, B_DEESK)
        esk = phrasen_treffer(text, B_ESK)
        if deesk or esk:
            stats["b_treffer"] += 1
            if deesk and esk:
                stats["b_beide_listen"] += 1
            else:
                stats["b_ereignisse"] += 1
                ereignisse.append(
                    {
                        "status_id": int(z["status_id"]),
                        "zeit_et": z["zeit_et"],
                        "regel": "B",
                        "ticker": "SPY",
                        "richtung": 1 if deesk else -1,
                    }
                )

    ev = pd.DataFrame(ereignisse)
    ev.to_parquet(ZIEL_EREIGNISSE, index=False)

    stats["zeitraum"] = [str(df["zeit_et"].min()), str(df["zeit_et"].max())]
    if len(ev):
        a = ev[ev.regel == "A"]
        b = ev[ev.regel == "B"]
        stats["a_richtung_pos"] = int((a.richtung == 1).sum())
        stats["a_richtung_neg"] = int((a.richtung == -1).sum())
        stats["a_ticker_unique"] = int(a.ticker.nunique())
        stats["a_top_ticker"] = a.ticker.value_counts().head(10).to_dict()
        stats["b_richtung_pos"] = int((b.richtung == 1).sum())
        stats["b_richtung_neg"] = int((b.richtung == -1).sum())
        stats["ereignisse_je_jahr"] = (
            ev.zeit_et.dt.year.value_counts().sort_index().to_dict()
        )
    ZIEL_STATS.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n-> {ZIEL_EREIGNISSE} ({len(ev)} Ereignisse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
