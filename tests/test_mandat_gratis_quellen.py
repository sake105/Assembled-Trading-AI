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

from research.mandat.pull_form4_dera import START_JAHR, START_QUARTAL, quartale

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


# ------------------------------------------------- Fama-French-Kursreihe
@pytest.fixture(scope="module")
def ff() -> pd.DataFrame:
    p = GRATIS / "fama_french_daily.parquet"
    if not p.exists():
        pytest.skip("Fama-French noch nicht gezogen")
    return pd.read_parquet(p)


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
    assert (ff["index"] - erwartet).abs().max() < 1e-6
    assert ff["index"].iloc[0] > 0


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
    assert (d["available_at"] > d["filing_date"]).all()


def test_dera_klassifiziert_nur_open_market_gerichtet(dera: pd.DataFrame) -> None:
    """Zuteilungen und Ausuebungen duerfen nicht zu Kaeufen umgedeutet werden."""
    assert set(dera.loc[dera["TRANS_CODE"] == "P", "transaction_type"]) == {"buy"}
    assert set(dera.loc[dera["TRANS_CODE"] == "S", "transaction_type"]) == {"sell"}
    andere = dera[~dera["TRANS_CODE"].isin(["P", "S"])]
    if len(andere):
        assert set(andere["transaction_type"]) == {"unknown"}


def test_dera_enthaelt_den_emittenten_als_stabilen_schluessel(
    dera: pd.DataFrame,
) -> None:
    """Befund 7: der Ticker ist kein Schluessel. ISSUERCIK ist einer."""
    assert dera["ISSUERCIK"].notna().all()
    assert dera["ISSUERCIK"].nunique() > 1000
