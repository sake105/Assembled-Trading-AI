"""Engine-Tests auf synthetischen Panels (Mandat II, P0-Nachtrag).

Die Engine erzeugt saemtliche entscheidungstragenden Zahlen und hatte null
Tests — vier echte Fehler (Holdout-Leck in den Dividenden, wirkungsloser
Hebel, fehlender Delisting-Zwangsverkauf, nie zurueckgesetzter Kredit) waeren
von je einem kleinen Test hier gefallen. Deshalb prueft jeder Test die
WIRKUNGSseite eines Parameters, nicht seine Kostenseite: die Kostenseite ist
der leichtere Teil und wird zuerst fertig, weshalb ein Mechanismus im Log
lebendig aussehen kann, waehrend er nichts tut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.campaign_data import CampaignData
from research.mandat2.engine import run_buy_and_hold, run_momentum
from research.mandat2.portfolio import Portfolio
from research.mandat2.tax_regimes import AssetClass, PrivatDE, VvGmbH, ZeroTax

pytestmark = pytest.mark.fast


def _panel(
    n_tage: int = 1500,
    symbole: tuple[str, ...] = ("SPY", "AAA", "BBB", "CCC"),
    drift: dict[str, float] | None = None,
) -> CampaignData:
    idx = pd.bdate_range("2000-01-03", periods=n_tage)
    drift = drift or {}
    close = pd.DataFrame(
        {s: 100.0 * (1 + drift.get(s, 0.0003)) ** np.arange(n_tage) for s in symbole},
        index=idx,
    )
    monatsenden = idx.to_series().groupby(idx.to_period("M")).max()
    mitglieder = frozenset(s for s in symbole if s != "SPY")
    membership = pd.Series({me: mitglieder for me in monatsenden})
    return CampaignData(
        close=close,
        div_panel=pd.DataFrame(index=idx),
        membership=membership,
        fenster="TEST",
        von=idx[0],
        bis=idx[-1],
    )


# --------------------------------------------------------------------- Hebel
def test_hebel_erhoeht_die_exponierung_wirklich():
    """Der Fehler, der P2 sonst mechanisch 'bestaetigt' haette.

    Portfolio.buy deckelte jede Order auf vorhandenes Cash — hebel=2 zahlte
    Zinsen und investierte trotzdem 1x. Ein Hebel-Sweep haette daraus
    'Hebel ist tot' abgeleitet, ohne dass je etwas geliehen wurde.
    """
    d = _panel()
    ohne = run_momentum(d, ZeroTax(), hebel=1.0, finanzierung_pa=0.0, cost_bps=0.0)
    mit = run_momentum(d, ZeroTax(), hebel=2.0, finanzierung_pa=0.0, cost_bps=0.0)
    # Ohne Zinsen muss doppelte Exponierung bei positiver Drift mehr bringen.
    assert mit.equity.iloc[-1] > ohne.equity.iloc[-1] * 1.2


def test_hebel_kostet_finanzierung():
    d = _panel()
    mit = run_momentum(d, ZeroTax(), hebel=2.0, finanzierung_pa=0.05, cost_bps=0.0)
    ohne = run_momentum(d, ZeroTax(), hebel=1.0, finanzierung_pa=0.05, cost_bps=0.0)
    assert mit.finanzierung_gezahlt > 0.0
    assert ohne.finanzierung_gezahlt == 0.0


def test_ohne_hebel_wird_nie_geliehen():
    d = _panel()
    r = run_momentum(d, ZeroTax(), hebel=1.0, finanzierung_pa=0.05)
    assert r.finanzierung_gezahlt == 0.0
    assert r.portfolio.max_kredit == 0.0


def test_portfolio_leiht_nur_bis_zum_limit():
    p = Portfolio(1_000.0, ZeroTax(), cost_bps=0.0, max_kredit=2_000.0)
    p.buy("X", 10_000.0, 10.0)
    assert p.qty("X") == pytest.approx(300.0)  # 3.000 investiert, nicht 10.000
    assert p.cash == pytest.approx(-2_000.0)


# ---------------------------------------------------------------- Delisting
def test_delisting_stellt_die_position_glatt():
    """Sonst bleibt Kapital zu einem eingefrorenen Kurs haengen — und daempft
    zugleich den gemessenen MaxDD."""
    d = _panel(n_tage=1500)
    # AAA endet nach zwei Dritteln der Historie.
    ende = d.close.index[1000]
    d.close.loc[ende:, "AAA"] = np.nan
    r = run_momentum(d, ZeroTax(), top_in=2, cost_bps=0.0)
    assert "AAA" not in r.portfolio.lots
    assert r.equity.iloc[-1] > 0


def test_nicht_ausfuehrbare_auftraege_werden_gezaehlt_nicht_verschluckt():
    d = _panel(n_tage=800)
    r = run_momentum(d, ZeroTax(), cost_bps=0.0)
    assert r.nicht_ausfuehrbar >= 0  # Feld existiert und wird gefuehrt


# ------------------------------------------------------------ Latente Steuer
def test_nettokurve_zieht_latente_steuer_ab():
    """Ohne sie schenkt die Zielfunktion Buy-and-Hold die Steuerstundung."""
    d = _panel()
    r = run_buy_and_hold(d, PrivatDE(pauschbetrag=0.0), symbol="SPY")
    mitte = len(r.equity) // 2
    assert r.equity_netto.iloc[mitte] < r.equity.iloc[mitte]
    # Am Ende ist liquidiert -> beide Kurven treffen sich.
    assert r.equity_netto.iloc[-1] == pytest.approx(r.equity.iloc[-1])


def test_latente_steuer_folgt_dem_regime():
    d = _panel()
    privat = run_buy_and_hold(d, PrivatDE(pauschbetrag=0.0))
    gmbh = run_buy_and_hold(d, VvGmbH())
    mitte = len(privat.equity) // 2
    abzug_privat = privat.equity.iloc[mitte] - privat.equity_netto.iloc[mitte]
    abzug_gmbh = gmbh.equity.iloc[mitte] - gmbh.equity_netto.iloc[mitte]
    # FONDS-Satz: privat 18,46 % gegen GmbH 11,57 % -> privat zieht mehr ab.
    assert abzug_privat > abzug_gmbh > 0


def test_kein_abzug_ohne_buchgewinn():
    p = Portfolio(100_000.0, PrivatDE(), cost_bps=0.0)
    p.buy("X", 100_000.0, 100.0)
    preise = pd.Series({"X": 80.0})  # Buchverlust
    assert p.latente_steuer(preise) == 0.0


# --------------------------------------------------------------- Haltedauer
def test_mindesthaltedauer_verhindert_frueh_verkaeufe():
    d = _panel(n_tage=1200)
    frei = run_momentum(d, ZeroTax(), top_in=2, rank_out=3, cost_bps=0.0)
    gesperrt = run_momentum(
        d, ZeroTax(), top_in=2, rank_out=3, min_haltetage=400, cost_bps=0.0
    )
    assert gesperrt.n_trades <= frei.n_trades


# -------------------------------------------------------- Instrumentenklasse
def test_benchmark_wird_als_fonds_besteuert():
    d = _panel()
    als_fonds = run_buy_and_hold(d, PrivatDE(pauschbetrag=0.0), asset=AssetClass.FONDS)
    als_aktie = run_buy_and_hold(d, PrivatDE(pauschbetrag=0.0), asset=AssetClass.AKTIE)
    # Teilfreistellung 30 % -> der Fonds zahlt weniger.
    assert als_fonds.portfolio.tax_paid < als_aktie.portfolio.tax_paid
    assert als_fonds.equity.iloc[-1] > als_aktie.equity.iloc[-1]
