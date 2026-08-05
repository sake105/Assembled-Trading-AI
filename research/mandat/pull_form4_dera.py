"""Form-4-Vollbestand aus den SEC-DERA-Quartalsdatensaetzen — survivorship-frei.

DIE LETZTE INSIDER-LUECKE, UND WARUM SIE OFFEN BLIEB
----------------------------------------------------
H-053 sollte die §4.6.1-These testen: Insider-Information ist in kleinen,
unbeobachteten Firmen am groessten. Der Vorbehalt im Registry macht das Verdikt
wertlos fuer genau diese These:

    "nur 723 Symbole Form-4 (S&P-Historie, NICHT echtes Small-Cap-Universum
     -> §4.6.1-These 'kleine Firmen' ungetestet)"

Die Preisseite lag breit vor (15.101 Namen inkl. Delisted), die Signalseite
nicht. Getestet wurde nochmal S&P.

DER ERSTE ANLAUF WAR FALSCH GEBAUT
-----------------------------------
Der naheliegende Weg — Ticker in CIK aufloesen, dann je CIK den
Submissions-Feed ziehen — hat einen toedlichen Konstruktionsfehler:
`company_tickers.json` der SEC enthaelt **nur aktuell gelistete** Firmen.
Gemessen an unserem Ziel: von 8.876 handelbaren Namen ohne Form-4 loesten sich
**3.490 auf (39,3 %)** — und von 653 ausdruecklich delisteten Varianten
**null**. Ein Pull auf dieser Basis haette Insider-Signale ausschliesslich fuer
Ueberlebende gezogen: Survivorship auf der Signalseite, dieselbe Klasse wie
Befund 7 (Ticker als Schluessel). Der Ansatz wurde deshalb verworfen, nicht
repariert.

WAS DIESES MODUL STATTDESSEN TUT
--------------------------------
Die SEC veroeffentlicht die Form 3/4/5 seit **2006Q1** als strukturierte
Quartalsdatensaetze (DERA, "Insider Transactions Data Sets"): ~15 MB je
Quartal, mit `ISSUERTRADINGSYMBOL` direkt im SUBMISSION-File. Damit
* entfaellt jede Ticker-CIK-Aufloesung — der Emittent steht im Datensatz,
* ist der Bestand **universumsunabhaengig**: er enthaelt auch Firmen, die
  spaeter verschwunden sind, weil nicht ueber eine Liste heutiger Namen
  ausgewaehlt wird,
* kostet der Vollbestand ~81 Downloads statt Millionen Einzelabrufe.

WIE WEIT DIE SURVIVORSHIP-FREIHEIT REICHT — UND WO SIE ENDET
------------------------------------------------------------
Nur fuer `as_of >= 2006-01-01`. Form 4 ist seit 2003-06-30 elektronisch
pflichtig; die Filings 2003-06 bis 2005-12 existieren auf EDGAR, fehlen diesem
Bestand aber, weil die DERA-Reihe erst 2006Q1 beginnt.

Gemessen am eigenen Preispanel (`prices_verdict.parquet`, 1.167 Symbole,
1995–2026): **39 Symbole (3,3 %) haben keinen einzigen Kurs ab 2006-01-01** und
koennen strukturell nie ein Form-4-Signal bekommen — genau die
Vor-2006-Ausscheider. Das ist Survivorship auf der Signalseite, kleiner als
beim verworfenen `company_tickers.json`-Ansatz, aber nicht abwesend. Ein
frueherer Entwurf schrieb hier "jede jemals eingereichte Form 4" — das war
falsch (F-senior-4).

PIT — UND WO DIESE QUELLE GROEBER IST
-------------------------------------
Der bestehende Ingester setzt `available_at` auf die **ACCEPTANCE-DATETIME**
(UTC, minutengenau). Die DERA-Datensaetze fuehren nur `FILING_DATE`
(Tagesaufloesung). Das ist im Repo als dokumentierter Fallback vermerkt
("filing_date = FILED-AS-OF date (gating fallback)"), aber es ist groeber:
eine nach Handelsschluss angenommene Meldung waere bei naiver Behandlung noch
am selben Tag "verfuegbar".

Deshalb wird `available_at` hier **konservativ auf den Folgetag** gesetzt.
Lieber einen Tag Signal verschenken als einen Tag Lookahead einbauen — die
Richtung des Fehlers ist damit gegen die Strategie, nicht fuer sie.

`transaction_date` bleibt unangetastet und wird NIE fuer Verfuegbarkeit
benutzt (E-038).

TRANSAKTIONSCODES
-----------------
Nur `P` (Open-Market-Kauf) und `S` (Open-Market-Verkauf) gelten als gerichtet.
Alles andere — Zuteilungen, Ausuebungen, Schenkungen, Steuereinbehalt — wird
als `unknown` gefuehrt und NICHT zu einem Richtungssignal umgedeutet.

Die Klassifikation wird dafuer aus dem Core-Ingester **importiert**
(`classify_transaction_code`), nicht nachgebaut. Der erste Entwurf behauptete
an dieser Stelle "dieselbe Konvention wie im bestehenden Ingester" und mappte
trotzdem auf `{"buy","sell"}`, waehrend der bestehende Bestand `{"P","S"}`
unter demselben Spaltennamen fuehrt. Sechs vorhandene Konsumenten filtern hart
auf `"P"` — sie haetten auf diesem Bestand still **null Zeilen** geliefert, und
ein leeres Ergebnis ist im Research nicht von einem echten Null-Befund zu
unterscheiden (E-123).

VERHAELTNIS ZUM BESTEHENDEN `form4_broad`-BESTAND
--------------------------------------------------
`transaction_type` ist jetzt angeglichen (beide `{"P","S","unknown"}`), die
uebrigen Spaltennamen sind es NICHT: dort `shares`, `price`, `issuer_cik`,
`transaction_code`, hier `trans_shares`, `trans_pricepershare`, `ISSUERCIK`,
`TRANS_CODE`. Ein Store-Tausch scheitert deshalb mit **KeyError** — laut, nicht
still. Das ist der Grund, warum die Konsolidierung ein Follow-up bleiben darf
und keine Voraussetzung ist; die Behauptung "die Vertraege sind angeglichen"
galt nur fuer eine Spalte (F-auditor-7).

KEIN TRIAL
----------
Reine Datenbeschaffung (E-090). Ob die §4.6.1-Patrone erneut angefasst wird,
ist eine separate Entscheidung — das Feld gilt nach H-031/H-053 als
verschossen. Diese Daten machen den Registry-Vorbehalt ueberpruefbar; sie
oeffnen das Feld nicht automatisch wieder.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Die Klassifikation wird IMPORTIERT, nicht nachgebaut. Der erste Entwurf
# mappte auf {"buy","sell"}, der bestehende Bestand fuehrt {"P","S"} unter
# demselben Spaltennamen — sechs vorhandene Konsumenten filtern hart auf "P"
# und haetten auf diesem Bestand still null Zeilen geliefert (E-123).
from src.assembled_core.data.edgar_form4_ingest import (  # noqa: E402
    classify_transaction_code,
)

DATA = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA / "form4_dera"
BASIS = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{jahr}q{quartal}_form345.zip"
)
UA = "Assembled-Trading-AI hans.oertel2@gmail.com"

#: Erstes verfuegbares Quartal — per Direktabruf geprueft: 2005 und frueher
#: liefern 404, 2006Q1 liefert 200.
START_JAHR, START_QUARTAL = 2006, 1

#: Code fuer den Open-Market-Kauf, wie ihn `classify_transaction_code` liefert.
#: Als Konstante, weil das Literal im Laufprotokoll nach dem Umstieg auf die
#: Core-Klassifikation stehen blieb und still 0 zaehlte — alle 81 Quartale
#: meldeten "0 Kaeufe", waehrend der Bestand 1,28 Mio Kaufzeilen enthielt.
#: Genau die Silent-Zero-Klasse aus E-123, eine Schicht tiefer.
KAUF_CODE = "P"

#: SEC-Fair-Access: hoeflicher Abstand zwischen Downloads. Die Dateien sind
#: gross, ein aggressiver Takt bringt hier nichts und riskiert eine Sperre.
ABSTAND_S = 1.0

SUB_SPALTEN = [
    "ACCESSION_NUMBER",
    "FILING_DATE",
    "PERIOD_OF_REPORT",
    "DOCUMENT_TYPE",
    "ISSUERCIK",
    "ISSUERNAME",
    "ISSUERTRADINGSYMBOL",
]
TRANS_SPALTEN = [
    "ACCESSION_NUMBER",
    # Primaerschluessel der Transaktionstabelle. Ohne ihn ist der Fan-out des
    # one-to-many-Merges mit den Meldepflichtigen NICHT rueckgaengig zu machen:
    # 2,87 % der Filings haben mehrere Owner. Storeweit gemessen blaeht das die
    # Zeilenzahl um 17,3 % und die Stueckzahlsumme um 37,8 % auf; bei den
    # KAUFzeilen sind es sogar 52,7 %. drop_duplicates ueber die Fachspalten
    # waere kein Ausweg — es kollabiert echte Mehrfachausfuehrungen mit
    # (ueberkorrigiert auf 52,1 %). Nur der Schluessel macht es umkehrbar (E-124).
    "NONDERIV_TRANS_SK",
    "TRANS_DATE",
    "TRANS_CODE",
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
    "TRANS_ACQUIRED_DISP_CD",
    "DIRECT_INDIRECT_OWNERSHIP",
]
OWNER_SPALTEN = [
    "ACCESSION_NUMBER",
    "RPTOWNERCIK",
    "RPTOWNERNAME",
    "RPTOWNER_RELATIONSHIP",
    "RPTOWNER_TITLE",
]


def quartale(bis_jahr: int, bis_quartal: int) -> list[tuple[int, int]]:
    aus = []
    j, q = START_JAHR, START_QUARTAL
    while (j, q) <= (bis_jahr, bis_quartal):
        aus.append((j, q))
        q += 1
        if q > 4:
            j, q = j + 1, 1
    return aus


def lade_quartal(jahr: int, quartal: int) -> pd.DataFrame | None:
    """Ein Quartal laden und auf gerichtete Transaktionen reduzieren.

    Gibt None zurueck, wenn das Quartal (noch) nicht veroeffentlicht ist — das
    ist ein normaler Zustand am aktuellen Rand, kein Fehler.
    """
    url = BASIS.format(jahr=jahr, quartal=quartal)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as h:  # noqa: S310
            roh = h.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

    z = zipfile.ZipFile(io.BytesIO(roh))

    def lies(name: str, spalten: list[str]) -> pd.DataFrame:
        df = pd.read_csv(
            io.BytesIO(z.read(name)), sep="\t", dtype=str, low_memory=False
        )
        fehlend = [c for c in spalten if c not in df.columns]
        if fehlend:
            # Fail-loud: ein stilles Weglassen wuerde die Spalte spaeter als
            # leer erscheinen lassen, und niemand saehe den Unterschied.
            raise SystemExit(
                f"[ERROR] {jahr}Q{quartal}/{name}: Spalten fehlen {fehlend} — "
                f"Schema geaendert, Pull anhalten."
            )
        return df[spalten]

    return aufbereiten(
        lies("SUBMISSION.tsv", SUB_SPALTEN),
        lies("NONDERIV_TRANS.tsv", TRANS_SPALTEN),
        lies("REPORTINGOWNER.tsv", OWNER_SPALTEN),
        jahr,
        quartal,
    )


def aufbereiten(
    sub: pd.DataFrame,
    trans: pd.DataFrame,
    owner: pd.DataFrame,
    jahr: int,
    quartal: int,
) -> pd.DataFrame:
    """Die drei TSV-Tabellen zu gerichteten Transaktionen verbinden.

    Als reine Funktion herausgezogen: die Tests pruefen sonst nur die fertigen
    Parquets, nicht den Code, der sie erzeugt (Stage-1-Befund).
    """
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]

    df = trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
    # Ein Filing kann mehrere Meldepflichtige haben (Cluster-Kaeufe!). Die
    # Zeilen werden deshalb NICHT dedupliziert — die Zahl der Insider je Titel
    # ist genau das Signal, um das es in H-053 ging.
    df = df.merge(owner, on="ACCESSION_NUMBER", how="left")

    # Format gepinnt: '31-MAR-2006'. Monat alphabetisch, also keine
    # Tag/Monat-Verwechslung moeglich — aber explizit ist schneller und
    # ueberlebt einen Formatwechsel der Quelle als NaT statt als Fehlparse.
    df["filing_date"] = pd.to_datetime(
        df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"
    )
    df["transaction_date"] = pd.to_datetime(
        df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"
    )
    # Konservativ: Verfuegbarkeit erst am Folgetag (siehe Modul-Docstring).
    # UTC-lokalisiert, weil der Core-Ingester (form4_rows_to_dataframe) das
    # ebenfalls tut. Naiv gegen tz-aware unter demselben Spaltennamen ergaebe
    # bei einem concat eine object-Spalte und einen stillen Objektvergleich.
    df["available_at"] = (df["filing_date"] + pd.Timedelta(days=1)).dt.tz_localize(
        "UTC"
    )
    # Ein Transaktionsdatum NACH dem Meldedatum ist unmoeglich — man meldet
    # nach dem Handel. In 2006Q1 stehen Transaktionsdaten von 1982 bis 2020;
    # das sind Tippfehler in den Filings. Sie werden NICHT still entfernt
    # (das waere ein unsichtbarer Eingriff), sondern markiert und gezaehlt.
    # Wer sie ungefiltert benutzt, baut sich ein Lookahead ein.
    df["datum_plausibel"] = (
        df["transaction_date"].notna()
        & df["filing_date"].notna()
        & (df["transaction_date"] <= df["filing_date"])
        & (df["transaction_date"] >= df["filing_date"] - pd.Timedelta(days=3 * 365))
    )
    df["transaction_type"] = df["TRANS_CODE"].map(classify_transaction_code)
    for c in ("TRANS_SHARES", "TRANS_PRICEPERSHARE"):
        df[c.lower()] = pd.to_numeric(df[c], errors="coerce")
    # `astype(str)` macht aus fehlenden Werten die Strings "nan"/"None" — die
    # sehen wie Ticker aus, werden als Symbole mitgezaehlt und wuerden bei
    # jedem Join Zehntausende Zeilen unter einem Phantom-Ticker sammeln.
    # Gemessen ueber alle 81 Quartale: 49.271 Zeilen (NONE 36.772, NAN 12.458,
    # NA 32, N/A 9). Fehlend bleibt hier fehlend; ISSUERCIK traegt den Fall.
    sym = df["ISSUERTRADINGSYMBOL"].astype("string").str.strip().str.upper()
    df["symbol"] = sym.mask(sym.isin(["NAN", "NONE", "NA", "N/A", ""]))
    # Berichtigungen (4/A, storeweit 2,20 %) wiederholen in der Regel den vollen
    # Transaktionssatz — dieselbe oekonomische Transaktion steht dann zweimal
    # im Bestand, und die Berichtigung kann in einem SPAETEREN Quartal liegen.
    # Hier wird markiert, nicht gefiltert: ein Filter waere ein stiller Eingriff.
    df["ist_berichtigung"] = df["DOCUMENT_TYPE"].eq("4/A")
    # Welche Verfuegbarkeitsdefinition in dieser Zeile steckt. Der Core-Ingester
    # schreibt unter demselben Spaltennamen die ACCEPTANCE-Minute (tz-aware).
    # Ohne diese Spalte kann ein Konsument, der beide Bestaende mischt, die
    # beiden Zeitachsen nicht auseinanderhalten (F-senior-5).
    df["available_at_basis"] = "filing_date+1d"
    df["quartal"] = f"{jahr}Q{quartal}"
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    heute = pd.Timestamp.utcnow()
    ap.add_argument("--bis-jahr", type=int, default=int(heute.year))
    ap.add_argument("--bis-quartal", type=int, default=int((heute.month - 1) // 3 + 1))
    ap.add_argument("--max-quartale", type=int, default=0, help="0 = alle offenen")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alle = quartale(args.bis_jahr, args.bis_quartal)
    offen = [(j, q) for j, q in alle if not (OUT_DIR / f"{j}q{q}.parquet").exists()]
    print(
        f"[START] {len(alle)} Quartale ab {START_JAHR}Q{START_QUARTAL}, "
        f"{len(offen)} offen",
        flush=True,
    )
    if args.max_quartale:
        offen = offen[: args.max_quartale]

    zeilen = 0
    fehlend: list[str] = []
    unplausibel_gesamt: list[int] = []
    for j, q in offen:
        t0 = time.monotonic()
        df = lade_quartal(j, q)
        if df is None:
            fehlend.append(f"{j}Q{q}")
            print(f"[SKIP] {j}Q{q}: noch nicht veroeffentlicht (404)", flush=True)
            continue
        df.to_parquet(OUT_DIR / f"{j}q{q}.parquet", index=False)
        zeilen += len(df)
        kauf = df[df["transaction_type"] == KAUF_CODE]
        kaeufe = len(kauf)
        kauf_txn = int(kauf["NONDERIV_TRANS_SK"].nunique())
        unplausibel = int((~df["datum_plausibel"]).sum())
        unplausibel_gesamt.append(unplausibel)
        print(
            f"[OK] {j}Q{q}: {len(df):>7} Zeilen | {df['symbol'].nunique():>5} Symbole "
            f"| {kaeufe:>6} Kaufzeilen ({kauf_txn} Txn) "
            f"| {unplausibel:>5} Datum unplausibel "
            f"| {time.monotonic() - t0:.0f}s",
            flush=True,
        )
        time.sleep(ABSTAND_S)

    fertig = sorted(OUT_DIR.glob("*.parquet"))
    # Artefakt als LETZTE Anweisung (E-116).
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(
            {
                "quelle": "SEC DERA Insider Transactions Data Sets (Form 3/4/5)",
                "erstes_quartal": f"{START_JAHR}Q{START_QUARTAL}",
                "quartale_vorhanden": len(fertig),
                "quartale_nicht_veroeffentlicht": fehlend,
                "zeilen_in_diesem_lauf": zeilen,
                "pit": (
                    "available_at = FILING_DATE + 1 Tag (konservativ; DERA fuehrt "
                    "keine ACCEPTANCE-DATETIME). transaction_date immutabel."
                ),
                "codes": "nur P=buy / S=sell gerichtet, Rest 'unknown'",
                "datum_unplausibel_in_diesem_lauf": sum(unplausibel_gesamt),
                "datum_plausibel_regel": (
                    "transaction_date <= filing_date und nicht aelter als 3 Jahre; "
                    "markiert statt entfernt — Zeilen mit datum_plausibel=False "
                    "duerfen NICHT in ein Signal eingehen (Lookahead)"
                ),
                "survivorship": (
                    "survivorship-frei fuer as_of >= 2006-01-01; davor KEINE "
                    "Abdeckung (Form 4 ist seit 2003-06-30 elektronisch "
                    "pflichtig, DERA beginnt 2006Q1). Vor-2006-Delistings "
                    "bekommen strukturell nie ein Signal — im eigenen "
                    "Preispanel betrifft das 39 von 1.167 Symbolen (3,3 %)."
                ),
                "transaction_type": (
                    "aus classify_transaction_code() des Core-Ingesters — "
                    "Werte {'P','S','unknown'} wie im bestehenden Bestand"
                ),
                "available_at_basis": (
                    "filing_date+1d (UTC). Der Core-Ingester schreibt unter "
                    "demselben Spaltennamen die ACCEPTANCE-Minute; wer beide "
                    "Bestaende mischt, MUSS auf available_at_basis gruppieren "
                    "und darf die groebere Definition nur nach oben runden."
                ),
                "berichtigungen": (
                    "4/A koexistiert mit dem Original (storeweit 2,20 %) und ist als "
                    "ist_berichtigung markiert, NICHT gefiltert. Aufloesung "
                    "muss quartalsuebergreifend erfolgen."
                ),
                "fan_out": (
                    "Transaktionen sind je Meldepflichtigem dupliziert "
                    "(Cluster-Signal). Zaehlungen ueber RPTOWNERCIK.nunique(); "
                    "Stueck- und Wertsummen ERST nach "
                    "drop_duplicates('NONDERIV_TRANS_SK'). Storeweit gemessen "
                    "sind roh: Zeilen +17,3 %, Stueckzahl +37,8 %, KAUFzeilen "
                    "+52,7 % (1.278.080 Zeilen gegen 837.272 verschiedene "
                    "Kauftransaktionen)."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[FERTIG] {len(fertig)} Quartale auf Platte -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
