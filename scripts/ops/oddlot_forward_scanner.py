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
import os
import sys
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
    total = d.get("hits", {}).get("total", {}).get("value")
    geliefert = len(d.get("hits", {}).get("hits", []))
    if total is not None and total > geliefert:
        # EDGAR deckelt bei 100 Treffern. Ohne diesen Abgleich fiele der Rest
        # still weg und die Ausgabe bliebe "[OK] 100 Treffer" — dieselbe
        # Fail-Open-Richtung wie der Drossel-Pfad darunter (E-132).
        raise SystemExit(
            f"[ERROR] EDGAR meldet {total} Treffer, liefert aber nur "
            f"{geliefert} — Fenster verkleinern oder paginieren."
        )
    if "hits" not in d:
        # HTTP 200 mit Fehler-/Drossel-JSON sah sonst aus wie ein leerer
        # Markt — ein wochenlang gedrosselter Scanner meldete "[OK] 0
        # Treffer" (E-103). Ein ECHTES leeres hits-Array bleibt gueltig.
        raise SystemExit(
            f"[ERROR] EDGAR-Antwort ohne 'hits'-Feld — vermutlich gedrosselt "
            f"oder Schema geaendert: {str(d)[:200]}"
        )
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
    alter_stand: str | None = None
    if ZIEL.exists():
        alt = json.loads(ZIEL.read_text(encoding="utf-8"))
        bekannt = alt.get("faelle", {})
        alter_stand = alt.get("stand")

    bis = date.today()
    von = bis - timedelta(days=RUECKBLICK_TAGE)
    # "stand" wurde frueher geschrieben und nie gelesen — ein Aussetzer
    # laenger als das Rueckblickfenster verlor Filings dauerhaft und lautlos
    # (E-132). Jetzt: Fenster rueckwirkend ab dem letzten Stand oeffnen.
    if alter_stand:
        letzter = date.fromisoformat(alter_stand)
        if (bis - letzter).days > RUECKBLICK_TAGE:
            von = letzter
            print(
                f"[WARN] Scanner lief zuletzt {alter_stand} — Fenster "
                f"rueckwirkend geoeffnet ({von}..{bis}). Bei >100 Treffern "
                f"bricht der Deckel-Guard laut ab.",
                flush=True,
            )
    # In Scheiben <= RUECKBLICK_TAGE abfragen: ein rueckwirkend geoeffnetes
    # Fenster koennte sonst >100 Treffer haben -> Deckel-Guard -> Abbruch VOR
    # dem Write -> "stand" bliebe alt -> jeder Folgelauf braeche wieder ab
    # (selbstverstaerkend, Stage-3-Fund zu E-132). Scheiben halten jede
    # Einzelantwort unter dem Deckel (Basisrate ~18 Filings/60 Tage).
    treffer, gesehen_acc = [], set()
    start = von
    while start < bis:
        ende = min(start + timedelta(days=RUECKBLICK_TAGE), bis)
        for t in suche(start.isoformat(), ende.isoformat()):
            if t["accession"] not in gesehen_acc:  # Scheibengrenzen ueberlappen
                gesehen_acc.add(t["accession"])
                treffer.append(t)
        start = ende

    neu = 0
    for t in treffer:
        acc = t["accession"]
        if acc and acc not in bekannt:
            t["gefunden_am"] = bis.isoformat()
            t["geprueft"] = False  # manuelle Fallpruefung steht aus
            bekannt[acc] = t
            neu += 1

    # Atomar schreiben: die Watchlist ist gitignored und der EINZIGE
    # Speicherort der manuellen geprueft-Flags. Ein Abbruch mitten im Write
    # wuerde sie sonst irreversibel verstuemmeln.
    tmp = ZIEL.with_suffix(".tmp")
    tmp.write_text(
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
    os.replace(tmp, ZIEL)
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
