"""Tests fuer die kostenlosen Datenquellen und den DERA-Form-4-Bestand.

Schwerpunkt liegt auf den Stellen, an denen ein stiller Parse- oder
Klassifikationsfehler ein Ergebnis verfaelschen wuerde, ohne dass irgendetwas
kaputt aussieht:

* der Fama-French-Parser (ein falsch abgeschnittener Kopf/Fuss liefert
  plausible, aber falsche Renditen),
* die Plausibilitaetsregel fuer Transaktionsdaten (ein Datum nach dem
  Meldedatum ist ein Lookahead),
* die Quartalsfolge (ein Off-by-one laesst still ein Quartal aus).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.mandat.pull_form4_dera import (
    START_JAHR,
    START_QUARTAL,
    aufbereiten,
    quartale,
)
from research.mandat2.pull_gratis_quellen import datumsspalte, parse_ff_text
from src.assembled_core.data.edgar_form4_ingest import classify_transaction_code

GRATIS = Path(__file__).resolve().parents[1] / "research/mandat2/data_gratis"
DERA = Path(__file__).resolve().parents[1] / "research/mandat/data/form4_dera"


# ------------------------------------------------------------- Quartalsfolge
def test_quartalsfolge_beginnt_bei_der_ersten_verfuegbaren_periode() -> None:
    """2005 und frueher liefert die SEC 404 — per Direktabruf geprueft."""
    q = quartale(2006, 4)
    assert q[0] == (START_JAHR, START_QUARTAL) == (2006, 1)
    assert q == [(2006, 1), (2006, 2), (2006, 3), (2006, 4)]


def test_quartalsfolge_zaehlt_ueber_jahresgrenzen() -> None:
    q = quartale(2007, 2)
    assert q[-3:] == [(2006, 4), (2007, 1), (2007, 2)]
    assert len(q) == 6


def test_quartalsfolge_ist_leer_vor_dem_start() -> None:
    """Sonst zoege ein Tippfehler im Jahr stillschweigend nichts."""
    assert quartale(2005, 4) == []


# ------------------------------------------- Parser gegen synthetischen Text
# Die folgenden Tests rufen den CODE auf, nicht die committeten Parquets. Ohne
# sie sind Parser-Aenderungen regressionsfrei moeglich, solange niemand neu
# zieht — Stage 1 wies das nach: sechs von neun Code-Mutationen ueberlebten.
FF_ROH = """This file was created by CMPT_ME_BEDECILES using the 202606 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,0.10,-0.24,-0.28,0.009
19260702,0.45,-0.32,-0.08,0.009
19291028,-12.35,0.30,0.60,0.010

  Copyright 2026 Kenneth R. French
"""


def test_ff_parser_schneidet_den_kopf_ab() -> None:
    df = parse_ff_text(FF_ROH)
    assert len(df) == 3
    assert list(df.index.strftime("%Y%m%d")) == ["19260701", "19260702", "19291028"]


def test_ff_parser_rechnet_prozent_in_dezimal() -> None:
    """Ohne die Division waere jede Rendite um Faktor 100 zu gross."""
    df = parse_ff_text(FF_ROH)
    assert df["mkt_rf"].iloc[0] == pytest.approx(0.0010)
    assert df["rf"].iloc[0] == pytest.approx(0.00009)
    assert df["mkt"].iloc[2] == pytest.approx(-0.1235 + 0.0001)


def test_ff_parser_baut_den_index_aus_der_gesamtrendite() -> None:
    """`mkt_rf` statt `mkt` waere die Ueberrendite — eine andere Groesse."""
    df = parse_ff_text(FF_ROH)
    erwartet = (1.0 + df["mkt"]).cumprod() * 100.0
    assert (df["index_crsp_vw"] - erwartet).abs().max() < 1e-12
    assert df["index_crsp_vw"].iloc[0] != pytest.approx(
        ((1.0 + df["mkt_rf"]).cumprod() * 100.0).iloc[0]
    )


def test_ff_parser_ignoriert_zeilen_ohne_achtstelliges_datum() -> None:
    """Schutz, falls die Quelle je einen Jahresblock anhaengt.

    Fuer die TAEGLICHE Datei tut sie das nachweislich NICHT (lockerer und
    strenger Filter liefern dieselben 26.274 Zeilen) — der Test sichert die
    Regel, nicht eine Eigenschaft der heutigen Datei.
    """
    mit_jahresblock = FF_ROH + "\n Annual Factors:\n1926,9.85,-6.20,4.30,3.27\n"
    assert len(parse_ff_text(mit_jahresblock)) == 3


def test_ff_parser_meldet_leeren_input_laut() -> None:
    with pytest.raises(SystemExit):
        parse_ff_text("nur ein Kopf, keine Daten\n")


# ------------------------------------------- DERA-Aufbereitung synthetisch
def _dera_tabellen(
    trans_datum: str, filing_datum: str, code: str = "P", symbol: str = "ABC"
):
    sub = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "FILING_DATE": [filing_datum],
            "PERIOD_OF_REPORT": [filing_datum],
            "DOCUMENT_TYPE": ["4"],
            "ISSUERCIK": ["12345"],
            "ISSUERNAME": ["Beispiel AG"],
            "ISSUERTRADINGSYMBOL": [symbol],
        }
    )
    trans = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "NONDERIV_TRANS_SK": ["sk1"],
            "TRANS_DATE": [trans_datum],
            "TRANS_CODE": [code],
            "TRANS_SHARES": ["100"],
            "TRANS_PRICEPERSHARE": ["10.5"],
            "TRANS_ACQUIRED_DISP_CD": ["A"],
            "DIRECT_INDIRECT_OWNERSHIP": ["D"],
        }
    )
    owner = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["a1"],
            "RPTOWNERCIK": ["999"],
            "RPTOWNERNAME": ["Muster, Max"],
            "RPTOWNER_RELATIONSHIP": ["Officer"],
            "RPTOWNER_TITLE": ["CFO"],
        }
    )
    return sub, trans, owner


def test_taggleiche_meldung_bleibt_plausibel() -> None:
    """Der Gleichheitsfall traegt 7,7 % aller Zeilen (Stage-1-Messung).

    `<` statt `<=` wuerde in 2006Q1 allein 13.574 von 176.561 Zeilen still als
    unplausibel markieren — und zwar genau das frischeste Signal.
    """
    df = aufbereiten(*_dera_tabellen("15-MAR-2006", "15-MAR-2006"), 2006, 1)
    assert bool(df["datum_plausibel"].iloc[0]) is True


def test_meldung_vor_dem_handel_ist_unplausibel() -> None:
    """Ein Handel NACH der Meldung ist unmoeglich — und waere ein Lookahead."""
    df = aufbereiten(*_dera_tabellen("16-MAR-2006", "15-MAR-2006"), 2006, 1)
    assert bool(df["datum_plausibel"].iloc[0]) is False


def test_uralte_transaktion_ist_unplausibel() -> None:
    """Die 3-Jahres-Untergrenze traegt 228 der 273 Markierungen in 2006Q1.

    Faellt sie weg, ist die Regel faktisch abgeschaltet.
    """
    df = aufbereiten(*_dera_tabellen("15-MAR-1990", "15-MAR-2006"), 2006, 1)
    assert bool(df["datum_plausibel"].iloc[0]) is False


def test_verfuegbarkeit_liegt_einen_tag_nach_der_meldung() -> None:
    df = aufbereiten(*_dera_tabellen("14-MAR-2006", "15-MAR-2006"), 2006, 1)
    assert df["available_at"].iloc[0] == pd.Timestamp("2006-03-16", tz="UTC")


def test_fehlender_ticker_wird_nicht_zum_symbol_nan() -> None:
    """`astype(str)` macht aus fehlenden Werten die Strings 'nan'/'None'.

    Ueber alle 81 Quartale betraf das 49.271 Zeilen; sie wurden als Symbole
    mitgezaehlt und wuerden bei jedem Join unter einem Phantom-Ticker landen.
    """
    for platzhalter in (None, "nan", "None", "N/A", ""):
        sub, trans, owner = _dera_tabellen(
            "14-MAR-2006", "15-MAR-2006", symbol=platzhalter
        )
        df = aufbereiten(sub, trans, owner, 2006, 1)
        assert pd.isna(df["symbol"].iloc[0]), f"{platzhalter!r} wurde zum Ticker"
        assert df["ISSUERCIK"].iloc[0] == "12345", "CIK traegt den Fall weiter"


def test_nur_form4_bleibt_uebrig() -> None:
    sub, trans, owner = _dera_tabellen("14-MAR-2006", "15-MAR-2006")
    sub.loc[0, "DOCUMENT_TYPE"] = "3"
    assert len(aufbereiten(sub, trans, owner, 2006, 1)) == 0


def test_mehrere_meldepflichtige_werden_nicht_dedupliziert() -> None:
    """Cluster-Kaeufe sind das Signal aus H-053 — Zeilen duerfen nicht kollabieren."""
    sub, trans, owner = _dera_tabellen("14-MAR-2006", "15-MAR-2006")
    owner = pd.concat([owner, owner.assign(RPTOWNERCIK="888")], ignore_index=True)
    assert len(aufbereiten(sub, trans, owner, 2006, 1)) == 2


def test_transaktionscodes_kommen_aus_dem_core_ingester() -> None:
    """E-123: gleicher Spaltenname, andere Wertemenge = zweite Wahrheit.

    Der erste Entwurf mappte auf {"buy","sell"}, waehrend der bestehende
    Bestand {"P","S"} fuehrt. Sechs Konsumenten filtern hart auf "P" und
    haetten still null Zeilen geliefert — ein leeres Ergebnis ist im Research
    nicht von einem echten Null-Befund zu unterscheiden.
    """
    for code in ("P", "S", "A", "M", "G", "F"):
        sub, trans, owner = _dera_tabellen("14-MAR-2006", "15-MAR-2006", code=code)
        df = aufbereiten(sub, trans, owner, 2006, 1)
        assert df["transaction_type"].iloc[0] == classify_transaction_code(code)
    assert classify_transaction_code("P") == "P", "keine Umbenennung nach 'buy'"


def test_verfuegbarkeit_ist_utc_wie_im_core_bestand() -> None:
    """Naiv gegen tz-aware unter demselben Spaltennamen ergibt beim concat
    eine object-Spalte und einen stillen Objektvergleich (F-senior-5)."""
    df = aufbereiten(*_dera_tabellen("14-MAR-2006", "15-MAR-2006"), 2006, 1)
    assert str(df["available_at"].dtype) == "datetime64[ns, UTC]"
    assert df["available_at_basis"].iloc[0] == "filing_date+1d"


def test_primaerschluessel_macht_den_fanout_rueckgaengig() -> None:
    """Ohne NONDERIV_TRANS_SK ist die Aufblaehung nicht reversibel (E-124).

    Deduplizieren ueber die Fachspalten wuerde echte Mehrfachausfuehrungen
    mitkollabieren — deshalb der Schluessel.
    """
    sub, trans, owner = _dera_tabellen("14-MAR-2006", "15-MAR-2006")
    owner = pd.concat([owner, owner.assign(RPTOWNERCIK="888")], ignore_index=True)
    df = aufbereiten(sub, trans, owner, 2006, 1)
    assert len(df) == 2, "Cluster-Signal bleibt erhalten"
    assert len(df.drop_duplicates("NONDERIV_TRANS_SK")) == 1, "Summen entblaehbar"


def test_berichtigungen_werden_markiert_nicht_gefiltert() -> None:
    """Eine 4/A wiederholt den Transaktionssatz — Filtern waere ein stiller
    Eingriff, Verschweigen eine Doppelzaehlung (F-senior-3)."""
    sub, trans, owner = _dera_tabellen("14-MAR-2006", "15-MAR-2006")
    sub.loc[0, "DOCUMENT_TYPE"] = "4/A"
    df = aufbereiten(sub, trans, owner, 2006, 1)
    assert len(df) == 1
    assert bool(df["ist_berichtigung"].iloc[0]) is True


# ------------------------------------------------- Datumsspalte / Fehlwerte
def test_datumsspalte_wird_namentlich_gefunden() -> None:
    df = pd.DataFrame({"OPEN": [1.0], "DATE": ["2020-01-02"]})
    assert datumsspalte(df, ("DATE", "Date")) == "DATE", "nicht positionell"


def test_datumsspalte_faellt_auf_die_erste_zurueck() -> None:
    df = pd.DataFrame({"zeitpunkt": ["2020-01-02"], "wert": [1.0]})
    assert datumsspalte(df, ("DATE", "Date")) == "zeitpunkt"


def test_ff_parser_faengt_den_fehlwert_sentinel() -> None:
    """-99.99 waere nach der Division eine Tagesrendite von -99,99 % und
    wuerde jedes dropna passieren (F-senior-9)."""
    text = ",Mkt-RF,SMB,HML,RF\n19260701,-99.99,0.10,0.10,0.009\n19260702,0.45,0.1,0.1,0.009\n"
    df = parse_ff_text(text)
    assert len(df) == 1, "die Sentinel-Zeile faellt raus"
    assert df.index[0] == pd.Timestamp("1926-07-02")


# ------------------------------------------------- Fama-French-Kursreihe
@pytest.fixture(scope="module")
def ff() -> pd.DataFrame:
    p = GRATIS / "fama_french_daily.parquet"
    if not p.exists():
        pytest.skip("Fama-French noch nicht gezogen")
    return pd.read_parquet(p)


def ff_spalten() -> list[str]:
    p = GRATIS / "fama_french_daily.parquet"
    if not p.exists():
        pytest.skip("Fama-French noch nicht gezogen")
    return list(pd.read_parquet(p).columns)


def test_ff_deckt_ein_jahrhundert_ab(ff: pd.DataFrame) -> None:
    """Der ganze Zweck der Quelle: mehr als die zwei Baerenmaerkte von P13."""
    assert ff.index.min() <= pd.Timestamp("1926-07-31")
    assert ff.index.max() >= pd.Timestamp("2020-01-01")
    assert len(ff) > 20_000


def test_ff_renditen_sind_dezimal_nicht_prozent(ff: pd.DataFrame) -> None:
    """Die Quelle liefert Prozent. Wer die Division vergisst, bekommt eine
    Kursreihe, die um Groessenordnungen falsch ist — und sie sieht monoton
    steigend trotzdem plausibel aus."""
    assert ff["mkt"].abs().max() < 0.5, "Tagesrendite ueber 50 % = nicht dividiert"
    assert ff["mkt"].std() < 0.05


def test_ff_kein_kopf_oder_fussblock_durchgerutscht(ff: pd.DataFrame) -> None:
    """Die CSV haengt hinter den Tagesdaten einen JAHRES-Block an.

    Rutscht der durch, stehen Jahresrenditen als Tagesrenditen in der Reihe —
    einzeln unauffaellig, in der Summe zerstoerend.
    """
    assert ff.index.is_monotonic_increasing
    assert not ff.index.has_duplicates
    luecken = ff.index.to_series().diff().dropna()
    assert luecken.max() < pd.Timedelta(days=30), "Sprung > 1 Monat = Blockwechsel"


def test_ff_index_ist_kumulierte_marktrendite(ff: pd.DataFrame) -> None:
    erwartet = (1.0 + ff["mkt"]).cumprod() * 100.0
    assert (ff["index_crsp_vw"] - erwartet).abs().max() < 1e-6
    assert ff["index_crsp_vw"].iloc[0] > 0


def test_spaltenname_warnt_vor_der_benchmark_verwechslung() -> None:
    """Der Name IST der Guard (F-senior-7): ein Konsument, der das Parquet
    liest, sieht den Docstring nie. "index" haette wie ein Kursindex
    ausgesehen und waere gegen einen ETF gestellt worden (E-079)."""
    assert "index" not in ff_spalten(), "generischer Name lockt zum ETF-Vergleich"
    assert "index_crsp_vw" in ff_spalten()


def test_ff_marktrendite_ist_ueberrendite_plus_zins(ff: pd.DataFrame) -> None:
    assert (ff["mkt"] - (ff["mkt_rf"] + ff["rf"])).abs().max() < 1e-12


def test_ff_enthaelt_die_bekannten_crashs(ff: pd.DataFrame) -> None:
    """Gegenprobe gegen eine plausibel aussehende, aber falsche Reihe."""
    for tag, schwelle in (("1929-10-28", -0.10), ("1987-10-19", -0.15)):
        r = ff.loc[pd.Timestamp(tag), "mkt"]
        assert r < schwelle, f"{tag}: {r:.3f} — Crash fehlt in der Reihe"


# ------------------------------------------------------ DERA-Plausibilitaet
@pytest.fixture(scope="module")
def dera() -> pd.DataFrame:
    p = DERA / "2006q1.parquet"
    if not p.exists():
        pytest.skip("DERA-Bestand noch nicht gezogen")
    return pd.read_parquet(p)


def test_dera_markiert_unmoegliche_transaktionsdaten(dera: pd.DataFrame) -> None:
    """Ein Handel NACH der Meldung ist unmoeglich — und waere ein Lookahead.

    Die Zeilen werden markiert, nicht entfernt: ein stilles Loeschen waere ein
    unsichtbarer Eingriff in die Datenbasis.
    """
    kaputt = dera[dera["transaction_date"] > dera["filing_date"]]
    assert len(kaputt) > 0, "2006Q1 enthaelt nachweislich solche Faelle"
    assert not kaputt["datum_plausibel"].any()


def test_dera_verfuegbarkeit_liegt_nach_dem_meldedatum(dera: pd.DataFrame) -> None:
    """Konservativ, weil DERA keine ACCEPTANCE-DATETIME fuehrt."""
    d = dera.dropna(subset=["filing_date"])
    # available_at ist UTC-lokalisiert (dtype-kompatibel zum Core-Bestand),
    # filing_date naiv — fuer den Vergleich muss man das gleichziehen.
    assert (d["available_at"] > d["filing_date"].dt.tz_localize("UTC")).all()


def test_dera_klassifiziert_nur_open_market_gerichtet(dera: pd.DataFrame) -> None:
    """Zuteilungen und Ausuebungen duerfen nicht zu Kaeufen umgedeutet werden."""
    assert set(dera.loc[dera["TRANS_CODE"] == "P", "transaction_type"]) == {"P"}
    assert set(dera.loc[dera["TRANS_CODE"] == "S", "transaction_type"]) == {"S"}
    andere = dera[~dera["TRANS_CODE"].isin(["P", "S"])]
    if len(andere):
        assert set(andere["transaction_type"]) == {"unknown"}


def test_dera_enthaelt_den_emittenten_als_stabilen_schluessel(
    dera: pd.DataFrame,
) -> None:
    """Befund 7: der Ticker ist kein Schluessel. ISSUERCIK ist einer."""
    assert dera["ISSUERCIK"].notna().all()
    assert dera["ISSUERCIK"].nunique() > 1000
