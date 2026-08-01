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
    equity: pd.Series
    portfolio: Portfolio
    label: str
    n_trades: int
    finanzierung_gezahlt: float

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
    label: str | None = None,
) -> LaufErgebnis:
    """Der Benchmark. ``asset=FONDS``, weil SPY ein Fonds ist (§20 InvStG)."""
    close = data.close[symbol].dropna()
    pf = Portfolio(START_KAPITAL, regime, asset=asset)
    equity: list[tuple[pd.Timestamp, float]] = []
    gekauft = False
    for t in close.index:
        pf.set_date(t)
        px = float(close.loc[t])
        if not gekauft and px > 0:
            pf.buy(symbol, pf.cash, px)
            gekauft = True
        div = data.div_panel.get(symbol)
        if div is not None and t in div.index:
            d = div.loc[t]
            if np.isfinite(d) and d > 0:
                pf.book_dividend(symbol, float(d))
        equity.append((t, pf.value(pd.Series({symbol: px}))))
    letzte = pd.Series({symbol: float(close.iloc[-1])})
    pf.liquidate_all(letzte)
    e = pd.Series(dict(equity)).sort_index()
    e.iloc[-1] = pf.value(letzte)
    return LaufErgebnis(
        equity=e,
        portfolio=pf,
        label=label or f"BuyHold({symbol}, {regime.name})",
        n_trades=1,
        finanzierung_gezahlt=0.0,
    )


def run_momentum(
    data: CampaignData,
    regime: TaxRegime,
    *,
    top_in: int = 20,
    rank_out: int = 60,
    min_haltetage: int = 0,
    hebel: float = 1.0,
    finanzierung_pa: float = FINANZIERUNG_PA_DEFAULT,
    cost_bps: float = COST_BPS_DEFAULT,
    asset: AssetClass = AssetClass.AKTIE,
    label: str | None = None,
) -> LaufErgebnis:
    """12-1-Momentum, monatlich, mit Haltedauer- und Hebelparameter.

    Args:
        top_in: gekauft wird, wer im Rang <= top_in liegt.
        rank_out: verkauft wird erst, wenn der Rang > rank_out faellt
            (Turnover-Bremse aus Mandat I H-012).
        min_haltetage: Position wird nicht vor Ablauf verkauft — auch dann
            nicht, wenn das Signal es wollte. 0 = keine Sperre.
        hebel: 1.0 = ungehebelt. > 1.0 investiert entsprechend mehr und
            zahlt ``finanzierung_pa`` taeglich auf die geliehene Summe.
    """
    close = data.close
    idx = pd.DatetimeIndex(close.index)
    mom = close.shift(21) / close.shift(252) - 1.0
    monatsenden = _monatsenden(idx)
    membership = data.membership

    pf = Portfolio(START_KAPITAL, regime, cost_bps=cost_bps, asset=asset)
    close_ff = close.ffill()
    kaufdatum: dict[str, pd.Timestamp] = {}
    equity: list[tuple[pd.Timestamp, float]] = []
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
                    pf.sell(sym, -delta / px, float(px))
                    n_trades += 1
        pending = []

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

        if t not in monatsenden:
            continue

        # ---------------------------------------------------- Rebalance
        mitglieder = membership.get(t)
        if mitglieder is None:
            continue
        kandidaten = sorted(set(mitglieder) & set(close.columns))
        scores = mom.loc[t, kandidaten].dropna()
        if scores.empty:
            continue
        rang = scores.rank(ascending=False)
        ziel = [s for s in sorted(rang.index) if rang[s] <= top_in]

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
            je = investiert / len(ziel)
            for sym in ziel:
                pending.append(("trade_to", sym, je))
            geliehen = max(investiert - wert, 0.0)

    letzte = close_ff.iloc[-1]
    pf.liquidate_all(letzte)
    e = pd.Series(dict(equity)).sort_index()
    e.iloc[-1] = pf.value(letzte)
    return LaufErgebnis(
        equity=e,
        portfolio=pf,
        label=label
        or f"Mom(top{top_in}/out{rank_out}, hold{min_haltetage}d, x{hebel}, {regime.name})",
        n_trades=n_trades,
        finanzierung_gezahlt=finanzierung,
    )
