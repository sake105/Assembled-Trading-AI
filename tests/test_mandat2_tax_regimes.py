"""Gate fuer Mandat II Phase 1: Steuerregime korrekt und rueckwaertskompatibel.

Der wichtigste Test ist ``test_privat_de_reproduziert_mandat_i_exakt``: wenn
``PRIVAT_DE`` nicht bit-genau dasselbe tut wie Mandat I, ist JEDER Vergleich
zwischen Mandat I und II wertlos — dann misst man den Umbau, nicht die
Strategie. Alle uebrigen Tests pinnen die steuerlichen Asymmetrien, auf denen
die GmbH-Hypothese beruht.
"""

from __future__ import annotations

import random

import pytest

from research.mandat2.portfolio import Portfolio
from research.mandat2.tax_regimes import (
    PRIVAT_SATZ,
    AssetClass,
    PrivatDE,
    VvGmbH,
    VvGmbHMitAusschuettung,
    ZeroTax,
    gmbh_gesamtsatz,
    make_regime,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------
# 1. Rueckwaertskompatibilitaet: PRIVAT_DE == Mandat I
# --------------------------------------------------------------------------
def test_privat_de_reproduziert_mandat_i_exakt():
    """Zufaellige Handelsfolge durch BEIDE Implementierungen, Endzustand identisch.

    Verglichen werden Cash, Steuer, Kosten und die Lots — nicht nur der
    Endwert, damit sich keine kompensierenden Fehler verstecken koennen.

    UMFANG, ehrlich (F-senior-3): geprueft wird der TRADE-Pfad. Dividenden und
    terminal_liquidation liegen in Mandat I nicht in TaxedPortfolio, sondern in
    verdict_engine.run_backtest — sie sind hier strukturell nicht pruefbar. Bei
    den Dividenden weicht Mandat II ausserdem BEWUSST ab (E-068).
    """
    from research.mandat.h011_kandidat_a import TaxedPortfolio

    alt = TaxedPortfolio(100_000.0)
    neu = Portfolio(100_000.0, PrivatDE())

    rng = random.Random(42)
    symbols = ["AAA", "BBB", "CCC"]
    # Jahreswechsel mitlaufen lassen: der Sparerpauschbetrag ist der einzige
    # jahresabhaengige Teil und genau dort waere ein Off-by-one unsichtbar.
    import pandas as pd

    dates = pd.date_range("2019-06-01", periods=400, freq="7D")

    for i, d in enumerate(dates):
        alt.set_date(d)
        neu.set_date(d)
        sym = symbols[i % len(symbols)]
        px = 50.0 + 40.0 * rng.random()
        if rng.random() < 0.55:
            notional = 1_000.0 + 9_000.0 * rng.random()
            alt.buy(sym, notional, px)
            neu.buy(sym, notional, px)
        else:
            q = alt.qty(sym) * rng.random()
            alt.sell(sym, q, px)
            neu.sell(sym, q, px)

    # Exakte Gleichheit, nicht approx: beide fuehren dieselben Operationen in
    # derselben Reihenfolge aus, also ist Bit-Gleichheit erreichbar — und
    # "bit-genau" soll auch geprueft werden, wenn es behauptet wird.
    assert neu.cash == alt.cash
    assert neu.tax_paid == alt.tax_paid
    assert neu.costs_paid == alt.costs_paid
    assert neu.regime.loss_pot == alt.loss_pot
    assert set(neu.lots) == set(alt.lots)
    for sym in alt.lots:
        # Lots sind [[qty, px], ...] — verschachtelt, also Feld fuer Feld.
        assert len(neu.lots[sym]) == len(alt.lots[sym])
        for (nq, npx), (aq, apx) in zip(neu.lots[sym], alt.lots[sym]):
            assert nq == aq
            assert npx == apx
    # Und der Endwert, den ein Backtest tatsaechlich ablesen wuerde:
    prices = pd.Series({s: 70.0 for s in symbols})
    assert neu.value(prices) == pytest.approx(alt.value(prices), abs=1e-9)


# --------------------------------------------------------------------------
# 2. ZERO — die Referenzwelt
# --------------------------------------------------------------------------
def test_zero_zahlt_nirgends_steuer():
    r = ZeroTax()
    assert r.on_realized_gain(10_000.0) == 0.0
    assert r.on_dividend(5_000.0) == 0.0
    assert r.on_terminal(1_000_000.0, 100_000.0) == 0.0


# --------------------------------------------------------------------------
# 3. Die drei GmbH-Asymmetrien — das inhaltliche Herz von Mandat II
# --------------------------------------------------------------------------
def test_gmbh_kursgewinn_ist_faktisch_steuerfrei():
    """§8b Abs. 2 + Abs. 3 S. 1: nur 5 % gelten als nichtabziehbare BA."""
    r = VvGmbH(hebesatz=4.00)
    tax = r.on_realized_gain(100_000.0)
    erwartet = 100_000.0 * 0.05 * gmbh_gesamtsatz(4.00)
    assert tax == pytest.approx(erwartet)
    # Groessenordnung explizit gepinnt: ~1,5 %, nicht ~26 %.
    assert 0.014 < tax / 100_000.0 < 0.016


def test_gmbh_verluste_bringen_keinen_steuervorteil():
    """§8b Abs. 3 S. 3 — der Preis der Steuerfreiheit.

    In PRIVAT_DE senkt ein Verlust die spaetere Steuerlast (Verlusttopf); in
    der GmbH ist er steuerlich schlicht weg. Eine Strategie mit vielen kleinen
    Verlusten und wenigen grossen Gewinnen wird dadurch relativ schlechter
    behandelt, als der 1,5-%-Satz allein vermuten laesst.
    """
    r = VvGmbH()
    assert r.on_realized_gain(-50_000.0) == 0.0
    assert r.verluste_nicht_abziehbar == pytest.approx(50_000.0)
    # Kein Verlusttopf: der naechste Gewinn wird voll besteuert.
    assert r.on_realized_gain(50_000.0) > 0.0

    p = PrivatDE()
    assert p.on_realized_gain(-50_000.0) == 0.0
    assert p.on_realized_gain(50_000.0) == 0.0  # vollstaendig verrechnet


def test_gmbh_streubesitz_dividende_ist_teurer_als_privat():
    """§8b Abs. 4: unter 10 % Beteiligung volle Steuerpflicht.

    Das ist die Richtung, in der die GmbH SCHLECHTER ist — wichtig, damit die
    Kampagne die GmbH nicht pauschal als Gewinner behandelt.
    """
    gmbh = VvGmbH()
    privat = PrivatDE()
    assert gmbh.on_dividend(10_000.0, AssetClass.AKTIE) > privat.on_dividend(
        10_000.0, AssetClass.AKTIE
    )


def test_kst_und_gewst_schachtel_sind_zwei_verschiedene_schwellen():
    """§8b Abs. 4 KStG greift ab 10 %, §9 Nr. 2a GewStG erst ab 15 %.

    Dazwischen ist die Dividende koerperschaftsteuerlich fast frei, aber
    gewerbesteuerlich VOLL steuerpflichtig. Eine einzige Flagge fuer beide
    Schwellen haette dort ~1,5 % statt ~14 % gerechnet (F-senior-6).
    """
    streu = VvGmbH()  # < 10 %
    zwischen = VvGmbH(kst_schachtel=True)  # 10-15 %
    voll = VvGmbH(kst_schachtel=True, gewst_schachtel=True)  # >= 15 %

    d = 10_000.0
    assert (
        voll.on_dividend(d, AssetClass.AKTIE)
        < zwischen.on_dividend(d, AssetClass.AKTIE)
        < streu.on_dividend(d, AssetClass.AKTIE)
    )
    # Im Zwischenbereich bleibt die GewSt in voller Hoehe stehen.
    assert zwischen.dividenden_satz > 0.13


def test_gmbh_hebesatz_wirkt():
    niedrig = VvGmbH(hebesatz=2.00)
    hoch = VvGmbH(hebesatz=5.00)
    assert niedrig.on_realized_gain(100_000.0) < hoch.on_realized_gain(100_000.0)


def test_gmbh_fixkosten_werden_ausgewiesen():
    """Rechtsformkosten (Buchfuehrung, Abschluss, Berater) — bei 100k
    Startkapital groesser als der gesamte Steuervorteil (F-senior-5)."""
    ohne = VvGmbH()
    mit = VvGmbH(fixkosten_pa=3_500.0)
    assert ohne.annual_fixed_costs() == 0.0
    assert mit.annual_fixed_costs() == 3_500.0
    # Groessenordnungs-Beleg fuer die Warnung: 10 Jahre Fixkosten gegen den
    # Steuervorteil auf einen kompletten Portfolio-Umschlag mit +50 % Gewinn.
    steuervorteil = 50_000.0 * (PRIVAT_SATZ - mit.kursgewinn_satz)
    assert 10 * mit.annual_fixed_costs() > 0.5 * steuervorteil


# --------------------------------------------------------------------------
# 4. Ausschuettungsebene
# --------------------------------------------------------------------------
def test_ausschuettung_besteuert_nur_den_zuwachs():
    r = VvGmbHMitAusschuettung()
    # Kein Zuwachs -> keine Ausschuettungssteuer.
    assert r.on_terminal(100_000.0, 100_000.0) == 0.0
    assert r.on_terminal(80_000.0, 100_000.0) == 0.0
    tax = r.on_terminal(300_000.0, 100_000.0)
    assert tax == pytest.approx(200_000.0 * PRIVAT_SATZ)


def test_ausschuettung_ist_nie_besser_als_thesaurierend():
    """Sanity: die Kontrollrechnung muss unter der Hauptrechnung liegen."""
    the = VvGmbH()
    aus = VvGmbHMitAusschuettung()
    assert aus.on_realized_gain(10_000.0) == pytest.approx(
        the.on_realized_gain(10_000.0)
    )
    assert aus.on_terminal(500_000.0, 100_000.0) > the.on_terminal(500_000.0, 100_000.0)


def test_settle_terminal_zieht_nur_am_ende_ab():
    """Die Equity-Kurve waehrend des Laufs darf die Ausschuettung NICHT sehen."""
    import pandas as pd

    p = Portfolio(100_000.0, VvGmbHMitAusschuettung())
    p.buy("AAA", 50_000.0, 100.0)
    laufend = p.value(pd.Series({"AAA": 200.0}))
    netto = p.settle_terminal(laufend)
    assert netto < laufend
    assert p.tax_terminal > 0.0


# --------------------------------------------------------------------------
# 5. Turnover-Oekonomie: die eigentliche Mandat-II-Hypothese
# --------------------------------------------------------------------------
def test_haeufiges_umschichten_kostet_in_der_gmbh_ein_vielfaches_weniger():
    """Dieselbe Handelsfolge, vier Regime — der Steuerunterschied ist der Punkt.

    In Mandat I starben Momentum-/Rotationsstrategien an genau dieser Zahl.
    Wenn der Faktor hier nicht gross ist, ist die GmbH-Hypothese tot, bevor
    ein einziger Backtest laeuft.
    """
    import pandas as pd

    def lauf(regime) -> float:
        p = Portfolio(100_000.0, regime)
        d = pd.Timestamp("2020-01-06")
        px = 100.0
        for _ in range(40):  # 40x kompletter Umschlag mit je +5 % Gewinn
            p.set_date(d)
            p.buy("AAA", p.cash, px)
            px *= 1.05
            p.sell("AAA", p.qty("AAA"), px)
            d += pd.Timedelta(days=9)
        return p.tax_paid

    steuer_privat = lauf(PrivatDE())
    steuer_gmbh = lauf(VvGmbH())
    steuer_zero = lauf(ZeroTax())

    assert steuer_zero == 0.0
    assert steuer_gmbh > 0.0
    # Der Faktor ist das Argument der ganzen Phase: ~26,4 % vs ~1,5 %.
    assert steuer_privat / steuer_gmbh > 10.0


# --------------------------------------------------------------------------
# 6. Factory
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["ZERO", "PRIVAT_DE", "GMBH_THESAURIEREND", "GMBH_AUSSCHUETTUNG"]
)
def test_factory_liefert_frische_instanzen(name):
    a = make_regime(name)
    b = make_regime(name)
    assert a is not b  # Zustand darf sich zwischen Laeufen nicht vererben
    assert a.name == name


def test_factory_lehnt_unbekanntes_regime_ab():
    with pytest.raises(ValueError, match="Unbekanntes Steuerregime"):
        make_regime("SCHWEIZ")


# --------------------------------------------------------------------------
# 7. END-TO-END: effektiver Steuersatz pro Dividenden-Euro
#
# Diese Tests haetten den BLOCKER gefangen, den die isolierten Regime-Tests
# oben durchgelassen haben (E-068): das Regime-Objekt war korrekt, das
# PORTFOLIO-Verhalten invertiert. Lehre: Ergebnisgroessen messen, nicht nur
# Komponenten unit-testen.
# --------------------------------------------------------------------------
def _dividenden_lauf(regime, jahre: int = 10, div_quote: float = 0.02) -> tuple:
    """Kauf, N Jahre Dividende, Endverkauf. Kein Kursgewinn, keine Kosten.

    Rueckgabe: (effektiver Satz auf den Dividenden-Euro, Brutto-Dividenden).
    """
    import pandas as pd

    p = Portfolio(100_000.0, regime, cost_bps=0.0)
    px = 100.0
    p.set_date(pd.Timestamp("2010-01-04"))
    p.buy("AAA", 100_000.0, px)
    brutto = 0.0
    for j in range(jahre):
        d = pd.Timestamp(f"{2010 + j}-06-01")
        p.set_date(d)
        per_share = px * div_quote
        brutto += p.qty("AAA") * per_share
        p.book_dividend("AAA", per_share)
        # Total-Return-Panel: die Bruttodividende steckt im Kurspfad.
        px += per_share
    p.set_date(pd.Timestamp(f"{2010 + jahre}-01-04"))
    p.sell("AAA", p.qty("AAA"), px)
    return p.tax_paid / brutto, brutto


def test_dividende_wird_genau_einmal_besteuert_privat():
    """PRIVAT_DE: effektiv 26,375 %, nicht 52,75 %.

    Ohne die Basis-Anhebung in book_dividend wurde derselbe Euro zweimal
    getroffen — am Ex-Tag und noch einmal im Veraeusserungsgewinn.
    """
    satz, _ = _dividenden_lauf(PrivatDE(pauschbetrag=0.0))
    assert satz == pytest.approx(PRIVAT_SATZ, rel=1e-6)


def test_dividende_wird_genau_einmal_besteuert_gmbh():
    """GmbH: effektiv 29,825 % (Streubesitz), nicht 31,3 %."""
    r = VvGmbH()
    satz, _ = _dividenden_lauf(r)
    assert satz == pytest.approx(r.dividenden_satz, rel=1e-6)


def test_gmbh_dividende_bleibt_auch_end_to_end_teurer_als_privat():
    """Die Kernasymmetrie, auf Portfolio-Ebene gemessen.

    Genau hier war das Vorzeichen vorher gedreht: das Modell machte
    GmbH-Dividenden 41 % BILLIGER, fachlich sind sie 13 % TEURER. Ein gruener
    Unittest auf dem Regime-Objekt reichte nicht, um das zu sehen.
    """
    satz_privat, _ = _dividenden_lauf(PrivatDE(pauschbetrag=0.0))
    satz_gmbh, _ = _dividenden_lauf(VvGmbH())
    assert satz_gmbh > satz_privat
    assert satz_gmbh / satz_privat == pytest.approx(29.825 / 26.375, rel=0.01)


def test_basis_anhebung_veraendert_den_kursgewinn_nicht_die_bewertung():
    """Die Anhebung darf NUR die Steuerbasis bewegen, nicht den Marktwert."""
    import pandas as pd

    p = Portfolio(100_000.0, ZeroTax(), cost_bps=0.0)
    p.set_date(pd.Timestamp("2020-01-02"))
    p.buy("AAA", 100_000.0, 100.0)
    vor = p.value(pd.Series({"AAA": 100.0}))
    p.book_dividend("AAA", 2.0)
    nach = p.value(pd.Series({"AAA": 100.0}))
    assert nach == pytest.approx(vor)  # ZeroTax: kein Cash-Abfluss
    assert p.lots["AAA"][0][1] == pytest.approx(102.0)  # Basis angehoben


# --------------------------------------------------------------------------
# 8. Instrumentenklasse — der Benchmark ist ein FONDS, keine Aktie
# --------------------------------------------------------------------------
def test_etf_benchmark_zahlt_in_der_gmbh_deutlich_mehr_als_einzelaktien():
    """§20 InvStG (Teilfreistellung 80 % KSt / 40 % GewSt) statt §8b KStG.

    Mit demselben Satz auf beiden Seiten bekaeme der Einzelaktien-Kandidat
    rund 10 Prozentpunkte gegenueber SPY geschenkt — ein PASS waere dann ein
    Rechtsform-Artefakt statt Alpha (F-senior-2).
    """
    r = VvGmbH()
    aktie = r.on_realized_gain(100_000.0, AssetClass.AKTIE)
    fonds = r.on_realized_gain(100_000.0, AssetClass.FONDS)
    assert fonds > aktie
    assert fonds / 100_000.0 == pytest.approx(0.11565, abs=1e-4)
    assert aktie / 100_000.0 == pytest.approx(0.014913, abs=1e-5)
    assert (fonds - aktie) / 100_000.0 > 0.09  # ~10 pp geschenkter Vorteil


def test_etf_im_privatregime_entspricht_mandat_i_etf_tax():
    """Teilfreistellung 30 % -> 18,4625 %; Mandat I nutzte ETF_TAX = 0.185."""
    r = PrivatDE(pauschbetrag=0.0)
    tax = r.on_realized_gain(100_000.0, AssetClass.FONDS)
    assert tax / 100_000.0 == pytest.approx(0.184625, abs=1e-6)
    assert abs(tax / 100_000.0 - 0.185) < 0.001  # Mandat-I-Naeherung


def test_default_assetklasse_ist_aktie():
    """Rueckwaertskompatibilitaet: alte Aufrufe ohne Klasse bleiben §8b/Aktie."""
    r = VvGmbH()
    assert r.on_realized_gain(1_000.0) == r.on_realized_gain(1_000.0, AssetClass.AKTIE)


# --------------------------------------------------------------------------
# 9. Deterministische Randfaelle, die die Zufallsfolge NICHT trifft
#    (F-senior-8: gemessen 0 Treffer fuer Cash-Knappheit und qty-Clamp)
# --------------------------------------------------------------------------
def test_kauf_ueber_verfuegbarem_cash_wird_gedeckelt():
    p = Portfolio(1_000.0, ZeroTax())
    p.buy("AAA", 999_999.0, 10.0)
    assert p.cash >= 0.0
    assert p.qty("AAA") > 0.0
    assert p.value(__import__("pandas").Series({"AAA": 10.0})) == pytest.approx(
        1_000.0 - p.costs_paid
    )


def test_verkauf_ueber_bestand_wird_gedeckelt():
    p = Portfolio(10_000.0, ZeroTax(), cost_bps=0.0)
    p.buy("AAA", 10_000.0, 100.0)
    q = p.qty("AAA")
    p.sell("AAA", q * 5, 100.0)  # deutlich mehr als vorhanden
    assert p.qty("AAA") == 0.0
    assert p.cash == pytest.approx(10_000.0)


def test_sparerpauschbetrag_wird_jedes_jahr_neu_gewaehrt():
    """Zwei Jahre mit je > 1.000 EUR Gewinn — Reset explizit, nicht zufaellig."""
    import pandas as pd

    r = PrivatDE()
    p = Portfolio(100_000.0, r, cost_bps=0.0)
    for jahr in (2020, 2021):
        p.set_date(pd.Timestamp(f"{jahr}-03-02"))
        p.buy("AAA", 50_000.0, 100.0)
        p.sell("AAA", p.qty("AAA"), 110.0)  # +5.000 EUR Gewinn
    # Zwei Jahre -> zweimal 1.000 EUR steuerfrei.
    erwartet = 2 * (5_000.0 - 1_000.0) * PRIVAT_SATZ
    assert p.tax_paid == pytest.approx(erwartet, rel=1e-9)


# --------------------------------------------------------------------------
# 10. Fixkosten und End-Liquidation muessen WIRKEN, nicht nur existieren
# --------------------------------------------------------------------------
def test_fixkosten_fliessen_wirklich_ab():
    """Ein Regler, der nichts bewegt, ist gefaehrlicher als sein Fehlen.

    Vorher gab annual_fixed_costs() nur den Konstruktorwert zurueck — niemand
    belastete ihn. Die Doku sagte gleichzeitig, fuer die Frage „GmbH oder
    privat?" muesse er gesetzt werden (F-auditor-3).
    """
    import pandas as pd

    p = Portfolio(100_000.0, VvGmbH(fixkosten_pa=3_500.0))
    for jahr in range(2010, 2020):  # 9 Jahreswechsel
        p.set_date(pd.Timestamp(f"{jahr}-01-04"))
    assert p.fixed_costs_paid == pytest.approx(9 * 3_500.0)
    assert p.cash == pytest.approx(100_000.0 - 9 * 3_500.0)


def test_kein_abzug_im_ersten_jahr_und_nicht_ohne_parameter():
    import pandas as pd

    p = Portfolio(100_000.0, VvGmbH(fixkosten_pa=3_500.0))
    p.set_date(pd.Timestamp("2010-01-04"))
    p.set_date(pd.Timestamp("2010-12-30"))  # gleiches Jahr
    assert p.fixed_costs_paid == 0.0

    ohne = Portfolio(100_000.0, VvGmbH())
    for jahr in (2010, 2011, 2012):
        ohne.set_date(pd.Timestamp(f"{jahr}-01-04"))
    assert ohne.fixed_costs_paid == 0.0
    assert ohne.cash == 100_000.0


def test_end_liquidation_realisiert_die_steuer_regimegerecht():
    """Ohne sie traegt das Endvermoegen unversteuerte Buchgewinne.

    Genau dieser mark-to-market-Fehler drehte in Mandat I das Vorzeichen des
    Kernbefunds (BUG 2: H-032 low-div mtm 2,006 Mio vs postliq 1,590 Mio vs
    ETF 1,610 Mio) — F-auditor-4.
    """
    import pandas as pd

    preise = pd.Series({"AAA": 200.0})

    def endwert(regime) -> tuple[float, float]:
        p = Portfolio(100_000.0, regime, cost_bps=0.0)
        p.set_date(pd.Timestamp("2010-01-04"))
        p.buy("AAA", 100_000.0, 100.0)
        mtm = p.value(preise)
        p.liquidate_all(preise)
        return mtm, p.value(preise)

    mtm_privat, liq_privat = endwert(PrivatDE(pauschbetrag=0.0))
    mtm_gmbh, liq_gmbh = endwert(VvGmbH())

    assert mtm_privat == pytest.approx(mtm_gmbh)  # vor Steuer identisch
    assert liq_privat < mtm_privat  # Steuer wird faellig
    # Und der Unterschied ist genau der Regimeunterschied — nicht ein
    # Nebeneffekt: 26,375 % gegen 1,49 % auf 100.000 EUR Buchgewinn.
    assert (mtm_privat - liq_privat) == pytest.approx(100_000.0 * PRIVAT_SATZ)
    assert (mtm_gmbh - liq_gmbh) == pytest.approx(100_000.0 * VvGmbH().kursgewinn_satz)
    assert liq_gmbh > liq_privat


def test_end_liquidation_leert_das_depot():
    import pandas as pd

    p = Portfolio(100_000.0, ZeroTax(), cost_bps=0.0)
    p.set_date(pd.Timestamp("2010-01-04"))
    p.buy("AAA", 50_000.0, 100.0)
    p.buy("BBB", 50_000.0, 50.0)
    p.liquidate_all(pd.Series({"AAA": 110.0, "BBB": 55.0}))
    assert p.lots == {}
