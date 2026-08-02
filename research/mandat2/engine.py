"""Backtest-Engine mit Steuerregime, Haltedauer und Hebel (Mandat II, P0).

Gegenueber der Mandat-I-Engine (``research/mandat/verdict_engine.py``) neu:

* **Steuerregime als Parameter** statt fest verdrahtet — inkl. Instrumenten-
  klasse, damit der SPY-Benchmark als FONDS gerechnet wird und nicht als
  Aktie (E-069).
* **Mindesthaltedauer** als expliziter Parameter (Stunden bis Jahre, hier auf
  Tagesraster: 0 = frei, 365 = ein Jahr) — Mandat I testete Haltedauern nur
  punktuell.
* **Hebel mit Finanzierungskosten.** Ein Hebel ohne Zinskosten ist eine
  Fantasie: er verwandelt jede positive Driftserie in einen Gewinner. Die
  Finanzierung laeuft taeglich auf die geliehene Summe.
* **End-Liquidation** am Ende jedes Laufs, damit das Endvermoegen keine
  unversteuerten Buchgewinne enthaelt (der mark-to-market-Fehler, der in
  Mandat I das Vorzeichen des Kernbefunds drehte).

Bewusst NICHT enthalten (und deshalb kein stiller Vorteil): Margin Calls,
Intraday-Ausfuehrung, Leerverkaeufe, Slippage jenseits der Kosten-bps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.mandat2.campaign_data import CampaignData
from research.mandat2.portfolio import COST_BPS_DEFAULT, Portfolio
from research.mandat2.tax_regimes import AssetClass, TaxRegime

START_KAPITAL = 100_000.0
#: Broker-Finanzierungssatz p. a. auf die geliehene Summe. Konservativ
#: gewaehlt: wer Hebel testet, soll ihn bezahlen.
FINANZIERUNG_PA_DEFAULT = 0.045


@dataclass
class LaufErgebnis:
    equity: pd.Series  # Mark-to-market (Diagnose)
    #: Equity ABZUEGLICH der latenten Steuer auf offene Buchgewinne. DAS ist
    #: die Kurve, auf der die Zielfunktion misst. Ohne sie traegt der
    #: umschichtende Kandidat seine Steuer laufend, der Buy-and-Hold-Benchmark
    #: nie — und bekaeme die gesamte Steuerstundung geschenkt (E-071).
    equity_netto: pd.Series
    portfolio: Portfolio
    label: str
    n_trades: int
    finanzierung_gezahlt: float
    nicht_ausfuehrbar: int = 0

    def kurz(self) -> str:
        p = self.portfolio
        return (
            f"{self.label}: End {self.equity.iloc[-1]:,.0f} | "
            f"Steuer {p.tax_paid:,.0f} (Kurs {p.tax_on_gains:,.0f} / "
            f"Div {p.tax_on_dividends:,.0f}) | Kosten {p.costs_paid:,.0f} | "
            f"Fixkosten {p.fixed_costs_paid:,.0f} | "
            f"Finanzierung {self.finanzierung_gezahlt:,.0f} | Trades {self.n_trades}"
        )


def _monatsenden(idx: pd.DatetimeIndex) -> set[pd.Timestamp]:
    return set(idx.to_series().groupby(idx.to_period("M")).max())


def run_buy_and_hold(
    data: CampaignData,
    regime: TaxRegime,
    *,
    symbol: str = "SPY",
    asset: AssetClass = AssetClass.FONDS,
    risk_off_gate: pd.Series | None = None,
    startkapital: float = START_KAPITAL,
    label: str | None = None,
) -> LaufErgebnis:
    """Der Benchmark. ``asset=FONDS``, weil SPY ein Fonds ist (§20 InvStG).

    ``risk_off_gate`` erlaubt denselben Trendfilter wie bei run_strategy —
    gebraucht fuer die Kontrollfrage "schafft der reine Index MIT Gate den
    DD-Deckel?". Das ueber run_strategy zu bauen war ein Fehler: SPY ist ein
    ETF und steht NICHT in der S&P-Mitgliederliste, die dortige Auswahl
    verwarf ihn stillschweigend und zog eine beliebige Aktie.
    """
    close = data.close[symbol].dropna()
    pf = Portfolio(startkapital, regime, asset=asset)
    equity: list[tuple[pd.Timestamp, float]] = []
    equity_netto: list[tuple[pd.Timestamp, float]] = []
    monatsenden = _monatsenden(pd.DatetimeIndex(close.index))
    for t in close.index:
        pf.set_date(t)
        px = float(close.loc[t])
        risk_on = True
        if risk_off_gate is not None:
            g = risk_off_gate.get(t)
            risk_on = bool(g) if g is not None and np.isfinite(float(g)) else True
        # Gate nur an Rebalance-Tagen auswerten, wie bei run_strategy.
        if risk_off_gate is not None and t in monatsenden and px > 0:
            if risk_on and pf.qty(symbol) <= 0:
                pf.buy(symbol, pf.cash, px)
            elif not risk_on and pf.qty(symbol) > 0:
                pf.sell(symbol, pf.qty(symbol), px)
        elif risk_off_gate is None and pf.qty(symbol) <= 0 and px > 0:
            pf.buy(symbol, pf.cash, px)
        div = data.div_panel.get(symbol)
        if div is not None and t in div.index and pf.qty(symbol) > 0:
            d = div.loc[t]
            if np.isfinite(d) and d > 0:
                pf.book_dividend(symbol, float(d))
        preise = pd.Series({symbol: px})
        equity.append((t, pf.value(preise)))
        equity_netto.append((t, pf.wert_nach_latenter_steuer(preise)))
    letzte = pd.Series({symbol: float(close.iloc[-1])})
    pf.liquidate_all(letzte)
    e = pd.Series(dict(equity)).sort_index()
    e.iloc[-1] = pf.value(letzte)
    en = pd.Series(dict(equity_netto)).sort_index()
    en.iloc[-1] = pf.value(letzte)
    return LaufErgebnis(
        equity=e,
        equity_netto=en,
        portfolio=pf,
        label=label or f"BuyHold({symbol}, {regime.name})",
        n_trades=1,
        finanzierung_gezahlt=0.0,
    )


def momentum_score(close: pd.DataFrame) -> pd.DataFrame:
    """12-1-Momentum: Rendite von t-252 bis t-21 (letzter Monat ausgelassen)."""
    return close.shift(21) / close.shift(252) - 1.0


def zufalls_score(close: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Kontrollgruppe: reines Rauschen statt Signal.

    Der entscheidende Test fuer jeden Auswahl-Befund. Wenn eine
    Zufallsauswahl mit DERSELBEN Haltedisziplin denselben Vorsprung liefert,
    misst der Backtest nicht die Auswahl, sondern die Haltedauer bzw. den
    Turnover — und der vermeintliche Alpha-Befund ist ein Artefakt.

    Deterministisch ueber den Seed; jede Zeile ein eigener Zug, damit die
    Auswahl bei jedem Rebalance neu wuerfelt statt einmal festzustehen.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal(close.shape), index=close.index, columns=close.columns
    )


def run_strategy(
    data: CampaignData,
    regime: TaxRegime,
    *,
    score: pd.DataFrame | None = None,
    top_in: int = 20,
    rank_out: int = 60,
    min_haltetage: int = 0,
    hebel: float = 1.0,
    finanzierung_pa: float = FINANZIERUNG_PA_DEFAULT,
    cost_bps: float = COST_BPS_DEFAULT,
    asset: AssetClass = AssetClass.AKTIE,
    risk_off_gate: pd.Series | None = None,
    startkapital: float = START_KAPITAL,
    label: str | None = None,
) -> LaufErgebnis:
    """Monatliche Auswahl nach einem beliebigen Score, mit Haltedisziplin.

    Args:
        score: (Tag x Symbol), hoeher = besser. None -> 12-1-Momentum.
        top_in: gekauft wird, wer im Rang <= top_in liegt.
        rank_out: verkauft wird erst, wenn der Rang > rank_out faellt
            (Turnover-Bremse aus Mandat I H-012).
        min_haltetage: Position wird nicht vor Ablauf verkauft — auch dann
            nicht, wenn das Signal es wollte. 0 = keine Sperre.
        hebel: 1.0 = ungehebelt. > 1.0 investiert entsprechend mehr und
            zahlt ``finanzierung_pa`` taeglich auf die geliehene Summe.
        risk_off_gate: (Tag -> bool). True = investiert, False = alles in Cash.
            Wird am Rebalance-Tag gelesen. Der einzige Hebel im Modell, der
            den Drawdown ueberhaupt erreichen kann.
        startkapital: Entscheidend fuer die GmbH-Frage — laufende
            Rechtsformkosten sind ein FIXbetrag, ihre relative Last haengt
            also am Kapital. Bei 100.000 EUR sind 3.500 EUR/J 3,5 % p. a.,
            bei 1 Mio nur 0,35 %.
    """
    close = data.close
    idx = pd.DatetimeIndex(close.index)
    mom = momentum_score(close) if score is None else score.reindex_like(close)
    monatsenden = _monatsenden(idx)
    membership = data.membership

    pf = Portfolio(startkapital, regime, cost_bps=cost_bps, asset=asset)
    close_ff = close.ffill()
    # Delisting-Zwangsverkauf (aus Mandat I uebernommen, hier vorher FEHLEND):
    # 208 von 1.037 Symbolen enden vor dem Panelende. Ohne diesen Schritt
    # blieb Kapital in toten Namen zu einem eingefrorenen Kurs haengen und
    # daempfte zugleich den gemessenen MaxDD (F-senior-6). Genau darauf beruft
    # sich die Truncation-Hygiene in campaign_data.
    last_valid = close.apply(lambda c: c.last_valid_index())
    global_last = idx[-1]
    nicht_ausfuehrbar = 0
    kaufdatum: dict[str, pd.Timestamp] = {}
    equity: list[tuple[pd.Timestamp, float]] = []
    equity_netto: list[tuple[pd.Timestamp, float]] = []
    pending: list[tuple[str, str, float]] = []
    n_trades = 0
    finanzierung = 0.0
    tagessatz = finanzierung_pa / 252.0
    geliehen = 0.0

    for t in idx:
        pf.set_date(t)
        px_t = close.loc[t]

        # Finanzierung auf die geliehene Summe, taeglich.
        if geliehen > 0:
            zins = geliehen * tagessatz
            pf.cash -= zins
            finanzierung += zins

        for aktion, sym, betrag in pending:
            px = px_t.get(sym, np.nan)
            if not np.isfinite(px) or px <= 0:
                # Kein Kurs heute: fuer Verkaeufe auf den letzten gueltigen
                # ausweichen, statt den Auftrag still zu verschlucken.
                lv = last_valid.get(sym)
                if aktion == "sell_all" and lv is not None and lv < t:
                    px = close.at[lv, sym]
                if not np.isfinite(px) or px <= 0:
                    nicht_ausfuehrbar += 1
                    continue
            if aktion == "sell_all":
                q = pf.qty(sym)
                if q > 0:
                    pf.sell(sym, q, float(px))
                    kaufdatum.pop(sym, None)
                    n_trades += 1
            elif aktion == "trade_to":
                cur = pf.qty(sym) * px
                delta = betrag - cur
                if delta > 1.0:
                    pf.buy(sym, delta, float(px))
                    kaufdatum.setdefault(sym, t)
                    n_trades += 1
                elif delta < -1.0:
                    kd = kaufdatum.get(sym)
                    if (
                        min_haltetage > 0
                        and kd is not None
                        and (t - kd).days < min_haltetage
                    ):
                        continue  # Haltedauer gilt auch fuer Teilverkaeufe
                    pf.sell(sym, -delta / px, float(px))
                    n_trades += 1
                    if pf.qty(sym) <= 0:
                        kaufdatum.pop(sym, None)
        pending = []

        # Delisting: Historie endete und es ist kein blosser Datenausfall.
        for sym in list(pf.lots):
            lv = last_valid.get(sym)
            if lv is not None and lv < t and lv < global_last - pd.Timedelta(days=10):
                pending.append(("sell_all", sym, 0.0))

        # Dividenden
        if t in data.div_panel.index:
            zeile = data.div_panel.loc[t]
            for sym in list(pf.lots):
                d = zeile.get(sym, np.nan)
                if np.isfinite(d) and d > 0:
                    pf.book_dividend(sym, float(d))

        ff_t = close_ff.loc[t]
        wert = pf.value(ff_t)
        equity.append((t, wert))
        equity_netto.append((t, pf.wert_nach_latenter_steuer(ff_t)))

        if t not in monatsenden:
            continue

        # ---------------------------------------------------- Rebalance
        risk_on = True
        if risk_off_gate is not None:
            g = risk_off_gate.get(t)
            risk_on = bool(g) if g is not None and np.isfinite(float(g)) else True

        # Reset VOR den Frueh-Continues: sonst behalten geliehen und
        # max_kredit auf den Pfaden 'keine Mitgliederliste' und 'keine Scores'
        # den Vorwert — dieselbe Klasse, die schon einmal als behoben galt
        # (F-senior-7).
        geliehen = 0.0
        pf.max_kredit = 0.0
        if not risk_on:
            # Risk-off: alles glattstellen, nichts Neues kaufen.
            for sym in list(pf.lots):
                pending.append(("sell_all", sym, 0.0))
            continue

        mitglieder = membership.get(t)
        if mitglieder is None:
            continue
        kandidaten = sorted(set(mitglieder) & set(close.columns))
        scores = mom.loc[t, kandidaten].dropna()
        if scores.empty:
            continue
        # nlargest statt rank(): 'average' bei Gleichstaenden liefert mal
        # mehr, mal weniger als top_in Namen und veraendert damit still die
        # Positionsgroesse (F-senior-15). Deterministisch ueber sortierten Index.
        scores = scores.sort_index()
        rang = scores.rank(ascending=False, method="first")
        ziel = sorted(scores.nlargest(top_in, keep="first").index)

        for sym in list(pf.lots):
            if sym in ziel:
                continue
            r = rang.get(sym, np.inf)
            if r <= rank_out:
                continue  # Turnover-Bremse: noch gut genug
            if min_haltetage > 0:
                kd = kaufdatum.get(sym)
                if kd is not None and (t - kd).days < min_haltetage:
                    continue  # Haltedauer laeuft noch
            pending.append(("sell_all", sym, 0.0))

        if ziel:
            investiert = wert * hebel
            geliehen = max(investiert - wert, 0.0)
            pf.max_kredit = geliehen
            je = investiert / len(ziel)
            for sym in ziel:
                pending.append(("trade_to", sym, je))

    letzte = close_ff.iloc[-1]
    pf.liquidate_all(letzte)
    e = pd.Series(dict(equity)).sort_index()
    e.iloc[-1] = pf.value(letzte)
    en = pd.Series(dict(equity_netto)).sort_index()
    en.iloc[-1] = pf.value(letzte)
    return LaufErgebnis(
        equity=e,
        equity_netto=en,
        portfolio=pf,
        label=label
        or (
            f"Strat(top{top_in}/out{rank_out}, hold{min_haltetage}d, "
            f"x{hebel}, {regime.name})"
        ),
        n_trades=n_trades,
        finanzierung_gezahlt=finanzierung,
        nicht_ausfuehrbar=nicht_ausfuehrbar,
    )


def run_momentum(data: CampaignData, regime: TaxRegime, **kwargs) -> LaufErgebnis:
    """Rueckwaertskompatibler Name: 12-1-Momentum ueber run_strategy."""
    kwargs.setdefault("label", None)
    return run_strategy(data, regime, score=None, **kwargs)


def sma_gate(close: pd.DataFrame, symbol: str = "SPY", fenster: int = 200) -> pd.Series:
    """Klassisches Trendfilter-Gate: investiert, solange der Index ueber seiner SMA liegt.

    Der einfachste Mechanismus, der den Drawdown ueberhaupt adressiert — und
    bewusst der einfachste, damit ein Erfolg nicht sofort nach Overfitting
    aussieht. PIT-sauber: die SMA am Tag t nutzt nur Kurse bis t.
    """
    s = close[symbol]
    return (s > s.rolling(fenster).mean()).astype(float)
