"""Odd-Lot-Tender Forward-Scanner — die operative Konsequenz aus H-084.

H-084 (Welle 44) hat den Mechanismus bestaetigt: Self-Tender mit
Odd-Lot-Klausel nehmen Positionen <= 99 Aktien ohne Proration an — 60,5 %
positive Captures, Median +3,7 %, aber Kapazitaet nur ~200-600 EUR/Jahr.
Verdikt damals: **kein Backtest-Feld, sondern Forward-Scanner + manuelle
Fallpruefung.** Das hier ist dieser Scanner.

QUELLE
------
EDGAR-Volltextsuche (efts.sec.gov, frei): SC-TO-I-Filings, deren Text
"odd lot" enthaelt. Ausgabe ist eine WATCHLIST fuer manuelle Pruefung —
KEINE Handelsempfehlung und kein Trade. Jeder Fall braucht den Blick ins
Filing: Tender-Preisspanne, Fristende, Odd-Lot-Klausel im Wortlaut.

BETRIEB
-------
Idempotent: bekannte Accessions werden uebersprungen. Gedacht fuer einen
woechentlichen Lauf (manuell oder Task-Scheduler — Registrierung dort ist
Operator-Entscheidung, nicht Teil dieses Skripts).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIEL = ROOT / "output" / "ops" / "oddlot_watchlist.json"
UA = "Assembled-Trading-AI (hans.oertel2@gmail.com)"
BASIS = "https://efts.sec.gov/LATEST/search-index"

#: Rueckblick je Lauf. Tender laufen typisch 20 Boersentage — 60 Tage
#: Rueckblick verpasst nichts und haelt die Liste klein.
RUECKBLICK_TAGE = 60


def suche(von: str, bis: str) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "q": '"odd lot"',
            "forms": "SC TO-I",
            "dateRange": "custom",
            "startdt": von,
            "enddt": bis,
        }
    )
    req = urllib.request.Request(f"{BASIS}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as h:  # noqa: S310
        d = json.loads(h.read().decode("utf-8"))
    aus = []
    for t in d.get("hits", {}).get("hits", []):
        s = t.get("_source", {})
        aus.append(
            {
                "accession": t.get("_id", "").split(":")[0],
                "firma": (s.get("display_names") or ["?"])[0],
                "eingereicht": s.get("file_date"),
                "ciks": s.get("ciks"),
                "datei": t.get("_id", "").split(":")[-1],
            }
        )
    return aus


def main() -> int:
    ZIEL.parent.mkdir(parents=True, exist_ok=True)
    bekannt: dict = {}
    if ZIEL.exists():
        bekannt = json.loads(ZIEL.read_text(encoding="utf-8")).get("faelle", {})

    bis = date.today()
    von = bis - timedelta(days=RUECKBLICK_TAGE)
    treffer = suche(von.isoformat(), bis.isoformat())
    time.sleep(0.5)  # SEC-Hoeflichkeit

    neu = 0
    for t in treffer:
        acc = t["accession"]
        if acc and acc not in bekannt:
            t["gefunden_am"] = bis.isoformat()
            t["geprueft"] = False  # manuelle Fallpruefung steht aus
            bekannt[acc] = t
            neu += 1

    ZIEL.write_text(
        json.dumps(
            {
                "hinweis": (
                    "Watchlist fuer MANUELLE Pruefung (H-084-Mechanismus, "
                    "Kapazitaet ~200-600 EUR/Jahr). Kein Trade ohne Blick ins "
                    "Filing: Preisspanne, Frist, Odd-Lot-Klausel im Wortlaut."
                ),
                "stand": bis.isoformat(),
                "faelle": bekannt,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    offen = sum(1 for f in bekannt.values() if not f.get("geprueft"))
    print(
        f"[OK] {len(treffer)} Treffer im Fenster, {neu} neu, "
        f"{offen} ungeprueft in der Watchlist -> {ZIEL}"
    )
    for acc, f in sorted(bekannt.items(), key=lambda x: x[1].get("eingereicht") or "")[
        -5:
    ]:
        marke = " " if f.get("geprueft") else "*"
        print(f"  {marke} {f.get('eingereicht')} | {f.get('firma')} | {acc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
