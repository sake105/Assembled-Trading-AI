"""Namensliste fuer Regel A (Welle 48) — fixiert VOR jeder Kursbetrachtung.

Regel A verlangt eine vorab festgelegte Abbildung Unternehmensname -> Ticker.
Diese Datei erzeugt sie aus der S&P-Mitgliederreihe 2022–2026 des
Verdict-Panels — mechanisch, ohne Blick auf irgendein Kursergebnis.

WARUM MECHANISCH UND NICHT KURATIERT
------------------------------------
Jede Handkuratierung ("Dell muss rein, das war doch das Beispiel") waere
genau die Erinnerungs-Selektion, die Schritt 1 ausschliessen soll. Die Liste
nimmt ALLE Mitglieder des Fensters und leitet die Suchnamen nach festen
Regeln aus dem Firmennamen ab. Dass dabei auch Namen entstehen, die nie in
einem Post auftauchen, ist gewollt — sie kosten nichts.

BEKANNTE GRENZEN (dokumentiert, nicht still)
--------------------------------------------
* Mehrdeutige Alltagswoerter als Firmenname (Apple, Target, Visa, Oracle...)
  erzeugen falsch-positive Treffer. Die Regel-A-Bedingung "genau EIN
  Unternehmen im Text" faengt einen Teil davon; der Rest ist Rauschen, das
  die Ereignisstudie gegen die Kontrollgruppe tragen muss.
* Nur S&P-Mitglieder: Aussagen ueber Nicht-Mitglieder (Truth Social selbst,
  Private, Auslaender) fallen durch. Bewusst — dort haben wir keine Kurse.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "research" / "mandat" / "data" / "sp500_historical_constituents.csv"
ZIEL = HIER / "namensliste.json"

#: Fenster der Mitgliedschaft — Truth Social existiert seit 2022-02.
VON, BIS = "2022-01-01", "2026-12-31"

#: Suffixe, die aus Firmennamen entfernt werden, bevor sie Suchnamen werden.
SUFFIXE = (
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "companies",
    "plc",
    "ltd",
    "ltd.",
    "group",
    "holdings",
    "holding",
    "technologies",
    "technology",
    "international",
    "&",
    "the",
)

#: Mindestlaenge eines Suchnamens. Einbuchstabige oder Zwei-Buchstaben-Reste
#: ("3M" ist die Ausnahme unten) treffen sonst alles.
MIN_LAENGE = 4

#: Handverdrahtete Ausnahmen NUR fuer Namen, die die mechanische Regel
#: zerstoert (Kurznamen unter MIN_LAENGE, die eindeutig sind). KEINE
#: inhaltliche Kuratierung.
AUSNAHMEN = {"3M": "MMM", "GE": None, "GM": "GM", "IBM": "IBM", "AMD": "AMD"}


def suchnamen(firmenname: str) -> list[str]:
    """Firmenname -> Liste plausibler Suchnamen (mechanisch)."""
    basis = firmenname.strip()
    # Klammerzusaetze und Klassen-Suffixe weg
    basis = re.sub(r"\(.*?\)", " ", basis)
    basis = re.sub(r"\bclass [a-c]\b", " ", basis, flags=re.I)
    woerter = [w for w in re.split(r"[\s,]+", basis) if w]
    # Suffixe vom Ende entfernen
    while woerter and woerter[-1].lower().rstrip(".") in {
        s.rstrip(".") for s in SUFFIXE
    }:
        woerter.pop()
    if not woerter:
        return []
    voll = " ".join(woerter)
    # REVISION 2026-08-07, VOR jedem Kurskontakt (Welle 48 Nachtrag):
    # Die erste Fassung erzeugte zusaetzlich das ERSTE WORT als Alias.
    # Ergebnis auf dem Vollarchiv: "America First" -> First Solar (357),
    # "Deep State" -> State Street (351), "Fake News" -> News Corp (318),
    # "the best" -> Best Buy (262), "Witch Hunt" -> J.B. Hunt (94).
    # Alltagswoerter als Alias sind keine Firmennennungen. Es bleibt NUR der
    # volle bereinigte Name; Einwort-Namen (Dell, Apple, Nvidia) bleiben,
    # weil der volle Name selbst ein Wort ist. Die Restambiguitaet echter
    # Einwort-Namen (Apple, Target) bleibt dokumentiertes Rauschen.
    aus = [voll]
    return [a for a in aus if len(a) >= MIN_LAENGE or a in AUSNAHMEN]


def main() -> int:
    if not CSV.exists():
        raise SystemExit(f"[ERROR] {CSV.name} fehlt.")
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    fenster = df[(df["date"] >= VON) & (df["date"] <= BIS)]
    ticker: set[str] = set()
    for zeile in fenster["tickers"]:
        ticker.update(t.strip() for t in str(zeile).split(",") if t.strip())

    # Namensquelle: der DERA-Bestand fuehrt ISSUERNAME + Ticker als offizielle
    # SEC-Angaben. Die Mitglieder-CSV hat nur Ticker — ein frueherer Entwurf
    # las eine nicht existierende companies-Spalte und lieferte still 4 Namen.
    import glob

    dera = sorted(glob.glob(str(ROOT / "research/mandat/data/form4_dera/*.parquet")))
    if not dera:
        raise SystemExit("[ERROR] DERA-Bestand fehlt — Namensquelle nicht verfuegbar.")
    frames = [
        pd.read_parquet(f, columns=["symbol", "ISSUERNAME", "filing_date"])
        for f in dera[-8:]  # juengste 8 Quartale: aktuelle Namen
    ]
    dn = pd.concat(frames, ignore_index=True).dropna(subset=["symbol", "ISSUERNAME"])
    # Juengster Name je Ticker (Firmen benennen sich um)
    dn = dn.sort_values("filing_date").drop_duplicates("symbol", keep="last")
    dn = dn[dn["symbol"].isin(ticker)]

    namen_map: dict[str, str] = {}
    kollision: set[str] = set()
    for _, z in dn.iterrows():
        for s in suchnamen(str(z["ISSUERNAME"]).title()):
            k = s.lower()
            if k in kollision:
                continue
            bekannt = {n.lower(): t for n, t in namen_map.items()}
            if k in bekannt and bekannt[k] != z["symbol"]:
                kollision.add(k)
                namen_map = {n: t for n, t in namen_map.items() if n.lower() != k}
            else:
                namen_map[s] = str(z["symbol"])
    # REVISION 2026-08-07 Teil 2, weiterhin VOR jedem Kurskontakt:
    # Suffix-Bereinigung reduziert manche Firmennamen auf Alltagswoerter
    # ("American International Group" -> "American": 1.086 Fake-Treffer;
    # "News Corporation" -> "News": 450; "Southern Company" -> "Southern").
    # Kursblindes Kriterium: Einwort-Namen, die zu den 500 haeufigsten
    # Woertern des ARCHIVS selbst gehoeren, sind Sprachgebrauch, keine
    # Firmennennung. Schwelle 500 hier einmalig fixiert. Das Kriterium nutzt
    # nur Posttexte, keine Kurse — es bleibt ergebnisblind.
    import glob as _glob
    import re as _re
    from collections import Counter

    chunks = sorted(_glob.glob(str(HIER / "data" / "trumpstruth" / "chunk_*.parquet")))
    if chunks:
        texte = pd.concat(
            [pd.read_parquet(c, columns=["text"]) for c in chunks], ignore_index=True
        )["text"].fillna("")
        haeufig = Counter()
        for t in texte:
            haeufig.update(_re.findall(r"[a-z']+", t.lower()))
        top500 = {w for w, _ in haeufig.most_common(500)}
        vorher = len(namen_map)
        entfernt = [n for n in namen_map if " " not in n and n.lower() in top500]
        namen_map = {n: t for n, t in namen_map.items() if n not in entfernt}
        print(
            f"Top-500-Filter: {vorher - len(namen_map)} Einwort-Namen entfernt "
            f"({', '.join(sorted(entfernt)[:8])}...)"
        )
    for k, v in AUSNAHMEN.items():
        if v:
            namen_map[k] = v

    ergebnis = {
        "fixiert": "2026-08-06, VOR jeder Kursbetrachtung (Welle 48)",
        "fenster": [VON, BIS],
        "n_ticker_im_fenster": len(ticker),
        "n_suchnamen": len(namen_map),
        "hinweis": (
            "Mechanisch aus der Mitgliederliste abgeleitet. Aenderungen an "
            "dieser Liste nach Beginn der Ereignisstudie sind eine NEUE "
            "Registrierung, keine Korrektur."
        ),
        "namen": dict(sorted(namen_map.items())),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(namen_map)} Suchnamen fuer {len(ticker)} Ticker -> {ZIEL.name}")
    beispiele = [k for k in ("Dell", "Nvidia", "Boeing", "Apple") if k in namen_map]
    print("Beispiele enthalten:", beispiele)
    return 0


if __name__ == "__main__":
    sys.exit(main())
