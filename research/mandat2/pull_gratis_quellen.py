"""Kostenlose Datenquellen ziehen, die im Repo noch fehlen.

WARUM DAS HIER STEHT
--------------------
Die Kampagne ist an zwei Stellen datenblockiert, und eine davon laesst sich
ohne einen einzigen Euro entschaerfen.

**Luecke 2 — zu wenige unabhaengige Baerenmaerkte.** P13 hat gezeigt: im
Suchfenster 1995–2016 ist KEIN einziges der 144 rollierenden 10-Jahres-Fenster
krisenfrei, das mildeste hat −47,5 % Benchmark-Rueckgang. Damit ist
"Trendfolge wirkt" nicht von "Trendfolge hat 2000–2002 und 2008 umgangen" zu
trennen. Die effektive Stichprobe fuer den Mechanismus sind zwei Ereignisse.

Ken French veroeffentlicht die **taegliche** US-Marktrendite seit **1926-07**
kostenlos. Das sind zusaetzlich 1929, 1937, 1973/74 und 1987 — vier weitere
unabhaengige Baerenmaerkte, und zwar aus CRSP-Daten, also konstruktionsbedingt
survivorship-frei. Kein Retail-Anbieter verkauft das guenstiger als gratis.

WAS HIER NICHT PASSIERT
-----------------------
Kein Test, keine Hypothese, kein Trial-Increment (E-090). Das hier beschafft
Daten und beschreibt, was sie sind — mehr nicht. Insbesondere ist der
Marktfaktor **nicht SPY**: er ist wertgewichtet ueber alle CRSP-Firmen (NYSE,
AMEX, NASDAQ), enthaelt also mehr und kleinere Namen. Wer ihn als Benchmark
benutzt, misst eine andere Groesse — das ist bei einem Vergleich
"Filter gegen kein Filter auf DEMSELBEN Basiswert" unproblematisch, bei einem
Vergleich gegen einen ETF nicht (E-079).

QUELLEN
-------
* **Ken French Data Library** (Dartmouth) — taegliche Faktoren ab 1926-07,
  akademischer Standard, aus CRSP.
* **Shiller** (Yale) — monatliche S&P-Kurse, Dividenden und Gewinne ab 1871.
  Grobkoerniger, dafuer noch laenger; taugt als unabhaengige Gegenprobe.
* **CBOE VIX-Historie** — ab 1990, fuer Regime-/Vol-Fragen.
* **fja05680/sp500** (GitHub) — historische S&P-500-Zusammensetzung ab 1996.
  Unabhaengige Gegenprobe zur eigenen Membership-Reihe, an der Befund 7
  (Ticker als Schluessel) haengt.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIEL = Path(__file__).resolve().parent / "data_gratis"

#: Ein echter User-Agent. Dartmouth und CBOE liefern anonymen Requests
#: gelegentlich 403; ein anonymer Bulk-Zugriff ist ausserdem unhoeflich.
UA = "Assembled-Trading-AI (Forschung, hans.oertel2@gmail.com)"

FF_TAEGLICH = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SP500_MEMBERS = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def hole(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def fama_french() -> dict:
    """Taegliche Marktrendite ab 1926 als Kursreihe.

    Die CSV hat einen mehrzeiligen Kopf und am Ende einen zweiten Block mit
    Jahresdaten. Beides wird abgeschnitten, indem nur Zeilen mit achtstelligem
    Datum uebernommen werden — robuster als eine feste Zeilenzahl, die sich mit
    jedem Update verschiebt.
    """
    roh = hole(FF_TAEGLICH)
    with zipfile.ZipFile(io.BytesIO(roh)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")

    zeilen = []
    for z_ in text.splitlines():
        teile = [t.strip() for t in z_.split(",")]
        if len(teile) >= 5 and teile[0].isdigit() and len(teile[0]) == 8:
            zeilen.append(teile[:5])
    if not zeilen:
        raise SystemExit("[ERROR] Fama-French: keine Datenzeilen erkannt")

    df = pd.DataFrame(zeilen, columns=["datum", "mkt_rf", "smb", "hml", "rf"])
    df["datum"] = pd.to_datetime(df["datum"], format="%Y%m%d")
    for c in ("mkt_rf", "smb", "hml", "rf"):
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df.dropna(subset=["mkt_rf", "rf"]).set_index("datum").sort_index()

    # Marktrendite = Ueberrendite + risikoloser Satz; daraus eine Indexreihe,
    # damit die bestehende Engine sie wie einen Kurs behandeln kann.
    df["mkt"] = df["mkt_rf"] + df["rf"]
    df["index"] = (1.0 + df["mkt"]).cumprod() * 100.0
    return {
        "datei": "fama_french_daily.parquet",
        "df": df,
        "von": str(df.index.min().date()),
        "bis": str(df.index.max().date()),
        "zeilen": len(df),
    }


def vix() -> dict:
    df = pd.read_csv(io.BytesIO(hole(VIX)))
    spalte = df.columns[0]
    df[spalte] = pd.to_datetime(df[spalte], errors="coerce")
    df = df.dropna(subset=[spalte]).set_index(spalte).sort_index()
    return {
        "datei": "vix_history.parquet",
        "df": df,
        "von": str(df.index.min().date()),
        "bis": str(df.index.max().date()),
        "zeilen": len(df),
    }


def sp500_mitglieder() -> dict:
    df = pd.read_csv(io.BytesIO(hole(SP500_MEMBERS, timeout=180)))
    spalte = df.columns[0]
    df[spalte] = pd.to_datetime(df[spalte], errors="coerce")
    df = df.dropna(subset=[spalte]).sort_values(spalte)
    return {
        "datei": "sp500_members_extern.parquet",
        "df": df,
        "von": str(df[spalte].min().date()),
        "bis": str(df[spalte].max().date()),
        "zeilen": len(df),
    }


QUELLEN = {
    "fama_french": (fama_french, "US-Marktfaktor taeglich ab 1926 (CRSP, Dartmouth)"),
    "vix": (vix, "CBOE VIX-Historie ab 1990"),
    "sp500_mitglieder": (sp500_mitglieder, "S&P-500-Zusammensetzung ab 1996 (extern)"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nur", nargs="*", choices=sorted(QUELLEN), default=None)
    args = ap.parse_args(argv)
    ZIEL.mkdir(parents=True, exist_ok=True)

    protokoll: dict[str, dict] = {}
    for name, (fn, beschreibung) in QUELLEN.items():
        if args.nur and name not in args.nur:
            continue
        print(f"[ZIEHE] {name}: {beschreibung}", flush=True)
        try:
            ergebnis = fn()
        except Exception as e:
            # Fail-loud je Quelle, aber nicht fail-fast: eine tote URL darf die
            # anderen nicht verhindern. Der Fehler steht im Protokoll, nicht
            # nur auf der Konsole (E-103).
            print(f"[FEHLER] {name}: {type(e).__name__}: {e}", flush=True)
            protokoll[name] = {"status": "FEHLER", "fehler": f"{type(e).__name__}: {e}"}
            continue
        df = ergebnis.pop("df")
        df.to_parquet(ZIEL / ergebnis["datei"])
        protokoll[name] = {"status": "OK", "beschreibung": beschreibung, **ergebnis}
        print(
            f"[OK] {name}: {ergebnis['zeilen']} Zeilen "
            f"{ergebnis['von']}..{ergebnis['bis']} -> {ergebnis['datei']}",
            flush=True,
        )

    # Artefakt als LETZTE Anweisung (E-116).
    (ZIEL / "_protokoll.json").write_text(
        json.dumps(protokoll, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> {ZIEL / '_protokoll.json'}", flush=True)
    fehler = [k for k, v in protokoll.items() if v["status"] != "OK"]
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
