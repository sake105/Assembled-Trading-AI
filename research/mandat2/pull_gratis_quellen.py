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

WIE WEIT DIE ZAHL 338 TRAEGT — UND WIE WEIT NICHT
--------------------------------------------------
Gemessen auf dieser Reihe: ueber 1926–2026 sind **338 von 1.080** rollierenden
10-Jahres-Fenstern krisenfrei (Rueckgang schwaecher als 30 %), im Suchfenster
1995–2016 dagegen **0 von 144**. Beides unabhaengig nachgerechnet.

Die 338 sind aber **monatlich ueberlappend** und stammen aus nur vier
zusammenhaengenden Bloecken (grob 1940–1960, 1974–1977, 1987–1991,
2008–2010). Nicht-ueberlappend bleiben rund **fuenf** krisenfreie Fenster.

Was die Zahl also traegt: dass 1995–2016 ein Sonderfall ist und die lange
Reihe ueberhaupt krisenfreie Perioden enthaelt. Was sie NICHT traegt:
"338 unabhaengige Belege". Wer daraus Signifikanz ableitet, wiederholt
E-078 — die effektive Stichprobe ist die Zahl der unabhaengigen Ereignisse,
nicht die der Fenster.

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


def parse_ff_text(text: str) -> pd.DataFrame:
    """Rohtext der Fama-French-CSV in eine Kursreihe uebersetzen.

    Als reine Funktion herausgezogen, weil die Tests sonst nur die fertigen
    Parquets pruefen und der Parser selbst ungetestet bliebe — eine Aenderung
    hier waere dann regressionsfrei moeglich, solange niemand neu zieht
    (Stage-1-Befund: sechs von neun Code-Mutationen ueberlebten aus genau
    diesem Grund).
    """
    zeilen = []
    for zeile in text.splitlines():
        teile = [t.strip() for t in zeile.split(",")]
        if len(teile) >= 5 and teile[0].isdigit() and len(teile[0]) == 8:
            zeilen.append(teile[:5])
    if not zeilen:
        raise SystemExit("[ERROR] Fama-French: keine Datenzeilen erkannt")

    df = pd.DataFrame(zeilen, columns=["datum", "mkt_rf", "smb", "hml", "rf"])
    df["datum"] = pd.to_datetime(df["datum"], format="%Y%m%d")
    # Die Quelle liefert PROZENT. Ohne die Division waere jede Tagesrendite um
    # Faktor 100 zu gross — die Kursreihe saehe monoton steigend trotzdem
    # plausibel aus.
    for c in ("mkt_rf", "smb", "hml", "rf"):
        werte = pd.to_numeric(df[c], errors="coerce")
        # Die Quelle kodiert Fehlwerte als -99.99 / -999. Nach der Division
        # waere daraus eine Tagesrendite von -99,99 % geworden, die jedes
        # dropna passiert. Aktuell 0 Vorkommen — der Guard kostet nichts und
        # faengt es, falls die Quelle je nachliefert (F-senior-9).
        df[c] = werte.mask(werte <= -99.0) / 100.0
    df = df.dropna(subset=["mkt_rf", "rf"]).set_index("datum").sort_index()

    # Marktrendite = Ueberrendite + risikoloser Satz; daraus eine Indexreihe,
    # damit die bestehende Engine sie wie einen Kurs behandeln kann.
    df["mkt"] = df["mkt_rf"] + df["rf"]
    # Name als Guard (F-senior-7): ein Konsument, der das Parquet liest, sieht
    # den Docstring nie. "index" haette wie ein Kursindex ausgesehen und waere
    # gegen einen ETF gestellt worden — das ist genau E-079. Der Name sagt
    # jetzt, was es ist: CRSP, value-weighted, NICHT SPY.
    df["index_crsp_vw"] = (1.0 + df["mkt"]).cumprod() * 100.0
    return df


def fama_french() -> dict:
    """Taegliche Marktrendite ab 1926 als Kursreihe.

    Die CSV hat einen mehrzeiligen Kopf, der abgeschnitten wird, indem nur
    Zeilen mit achtstelligem Ziffern-Datum uebernommen werden — robuster als
    eine feste Zeilenzahl, die sich mit jedem Update verschiebt.

    KORREKTUR (Stage-1-Review): Ein frueherer Kommentar behauptete hier
    zusaetzlich einen angehaengten JAHRES-Block. Das ist fuer diese Datei
    **falsch** — nachgemessen liefern lockerer und strenger Filter exakt
    dieselben 26.274 Zeilen. Der Jahresblock steckt in der MONATLICHEN
    Variante der Quelle, nicht in der taeglichen. Der Filter bleibt trotzdem
    streng: er kostet nichts und schuetzt, falls die Quelle das Format
    angleicht.
    """
    roh = hole(FF_TAEGLICH)
    with zipfile.ZipFile(io.BytesIO(roh)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    df = parse_ff_text(text)
    return {
        "datei": "fama_french_daily.parquet",
        "df": df,
        "von": str(df.index.min().date()),
        "bis": str(df.index.max().date()),
        "zeilen": len(df),
    }


def datumsspalte(df: pd.DataFrame, erwartet: tuple[str, ...]) -> str:
    """Datumsspalte namentlich suchen, positionell nur als Rueckfall.

    Rein positionell (`df.columns[0]`) wird bei einer Spaltenumordnung der
    Quelle ALLES zu NaT, der Frame ist leer, und `to_parquet` schreibt ihn
    trotzdem — ein stiller Totalausfall, der als Erfolg protokolliert wird
    (F-senior-8, dieselbe Fail-Open-Richtung wie E-103).
    """
    for name in erwartet:
        if name in df.columns:
            return name
    return df.columns[0]


def vix() -> dict:
    df = pd.read_csv(io.BytesIO(hole(VIX)))
    spalte = datumsspalte(df, ("DATE", "Date", "date"))
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
    spalte = datumsspalte(df, ("date", "Date", "DATE"))
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
        if ergebnis["zeilen"] == 0:
            # Sonst meldet das Protokoll "OK, 0 Zeilen" und main() gibt 0
            # zurueck — ein Totalausfall als Erfolg (F-senior-8).
            print(f"[FEHLER] {name}: leeres Ergebnis", flush=True)
            protokoll[name] = {"status": "FEHLER", "fehler": "0 Zeilen geparst"}
            continue
        df.to_parquet(ZIEL / ergebnis["datei"])
        protokoll[name] = {"status": "OK", "beschreibung": beschreibung, **ergebnis}
        if name == "fama_french":
            protokoll[name]["benchmark_warnung"] = (
                "index_crsp_vw ist der CRSP-VALUE-WEIGHTED Gesamtmarkt (NYSE, "
                "AMEX, NASDAQ) — NICHT SPY. Gegen einen ETF gestellt misst man "
                "die Universums- und Gewichtungsdifferenz mit (E-079). Fuer "
                "'Filter gegen kein Filter auf demselben Basiswert' sauber."
            )
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
