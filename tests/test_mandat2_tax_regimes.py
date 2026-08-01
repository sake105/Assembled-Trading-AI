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

    assert neu.cash == pytest.approx(alt.cash, abs=1e-9)
    assert neu.tax_paid == pytest.approx(alt.tax_paid, abs=1e-9)
    assert neu.costs_paid == pytest.approx(alt.costs_paid, abs=1e-9)
    assert neu.regime.loss_pot == pytest.approx(alt.loss_pot, abs=1e-9)
    assert set(neu.lots) == set(alt.lots)
    for sym in alt.lots:
        # Lots sind [[qty, px], ...] — verschachtelt, also Feld fuer Feld.
        assert len(neu.lots[sym]) == len(alt.lots[sym])
        for (nq, npx), (aq, apx) in zip(neu.lots[sym], alt.lots[sym]):
            assert nq == pytest.approx(aq, abs=1e-9)
            assert npx == pytest.approx(apx, abs=1e-9)
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
    gmbh = VvGmbH(beteiligung_ueber_10pct=False)
    privat = PrivatDE()
    assert gmbh.on_dividend(10_000.0) > privat.on_dividend(10_000.0)

    # Ab 10 % Beteiligung kippt es (fuer uns theoretisch, aber der Vertrag
    # soll stimmen).
    gross = VvGmbH(beteiligung_ueber_10pct=True)
    assert gross.on_dividend(10_000.0) < privat.on_dividend(10_000.0)


def test_gmbh_hebesatz_wirkt():
    niedrig = VvGmbH(hebesatz=2.00)
    hoch = VvGmbH(hebesatz=5.00)
    assert niedrig.on_realized_gain(100_000.0) < hoch.on_realized_gain(100_000.0)


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
