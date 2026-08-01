"""Regime-agnostisches FIFO-Portfolio (Mandat II, Phase 0).

Struktur bewusst 1:1 uebernommen von ``research/mandat/h011_kandidat_a.py``
(``TaxedPortfolio``) — dieselbe FIFO-Mechanik, dieselbe Kostenlogik, dieselbe
Reihenfolge der Operationen. Der EINZIGE Unterschied: die Steuerentscheidung
wandert in ein ``TaxRegime``-Objekt.

AUTORITAET (F-senior-9): Ab 2026-08-01 ist DIESE Datei die autoritative
Portfolio-/Steuermechanik fuer neue Arbeit. ``research/mandat/h011_kandidat_a.
TaxedPortfolio`` gilt als EINGEFROREN — sie wird nur noch als Vergleichsanker
fuer den Regressionstest gelesen, nicht mehr weiterentwickelt.

Umfang der Rueckwaertskompatibilitaet, ehrlich abgegrenzt (F-senior-3):
``tests/test_mandat2_tax_regimes.py`` prueft den TRADE-Pfad (FIFO, Kosten,
Verlusttopf, Sparerpauschbetrag) bit-genau gegen Mandat I. NICHT geprueft und
strukturell nicht pruefbar gegen ``TaxedPortfolio``, weil dort gar nicht
vorhanden: Dividenden (liegen in ``verdict_engine.run_backtest``) und
``terminal_liquidation``. Bei den Dividenden weicht Mandat II ausserdem
BEWUSST ab — siehe ``book_dividend`` und E-068.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.mandat2.tax_regimes import AssetClass, TaxRegime

# Mandat I: h011_kandidat_a.COST_BPS = 10.0 (einseitig, in bps auf das Notional).
# NICHT raten — der Regressionstest hat eine Fehlannahme von 5.0 sofort gefangen.
COST_BPS_DEFAULT = 10.0


class Portfolio:
    """FIFO-Lots + austauschbares Steuerregime."""

    def __init__(
        self,
        cash: float,
        regime: TaxRegime,
        cost_bps: float = COST_BPS_DEFAULT,
        asset: AssetClass = AssetClass.AKTIE,
    ) -> None:
        # Instrumentenklasse des Portfolios. Entscheidend fuer den Benchmark:
        # SPY ist ein FONDS (§20 InvStG), kein AKTIE-Anteil (§8b KStG) — mit
        # demselben Satz auf beiden Seiten bekaeme der Einzelaktien-Kandidat
        # ~10 pp geschenkt (F-senior-2).
        self.asset = asset
        self.initial_capital = cash
        self.cash = cash
        self.regime = regime
        self.cost_bps = cost_bps
        self.lots: dict[str, list[list[float]]] = {}  # sym -> [[qty, px], ...]
        self.tax_paid = 0.0
        self.costs_paid = 0.0
        # Diagnose: getrennt ausweisen, weil die Regime sich genau hier
        # unterscheiden (GmbH: Dividende teuer, Kursgewinn billig).
        self.tax_on_gains = 0.0
        self.tax_on_dividends = 0.0
        self.tax_terminal = 0.0
        self.fixed_costs_paid = 0.0
        self._last_year: int | None = None

    # ------------------------------------------------------------- Kalender
    def set_date(self, date) -> None:
        """Jahreswechsel: Freibetraege scharf, Rechtsformkosten belasten.

        Die Fixkosten muessen hier WIRKLICH abfliessen. Ein Parameter, den man
        setzen kann und der nichts veraendert, ist gefaehrlicher als sein
        Fehlen — er suggeriert, die Frage „GmbH oder privat?" sei beantwortet
        (F-auditor-3).
        """
        jahr = date.year
        if self._last_year is not None and jahr != self._last_year:
            kosten = self.regime.annual_fixed_costs()
            if kosten:
                self.cash -= kosten
                self.fixed_costs_paid += kosten
        self._last_year = jahr
        self.regime.new_year(jahr)

    # ------------------------------------------------------------ Positionen
    def qty(self, sym: str) -> float:
        return sum(q for q, _ in self.lots.get(sym, []))

    def buy(self, sym: str, notional: float, px: float) -> None:
        if notional <= 0 or px <= 0:
            return
        cost = notional * self.cost_bps / 1e4
        spend = min(notional, max(self.cash - cost, 0.0))
        if spend <= 0:
            return
        cost = spend * self.cost_bps / 1e4
        q = (spend - cost) / px
        self.cash -= spend
        self.costs_paid += cost
        self.lots.setdefault(sym, []).append([q, px])

    def sell(self, sym: str, qty: float, px: float) -> None:
        lots = self.lots.get(sym, [])
        qty = min(qty, sum(q for q, _ in lots))
        if qty <= 0 or px <= 0:
            return
        proceeds = qty * px
        cost = proceeds * self.cost_bps / 1e4
        gain = 0.0
        rest = qty
        while rest > 1e-12 and lots:
            lq, lpx = lots[0]
            take = min(rest, lq)
            gain += take * (px - lpx)
            lq -= take
            rest -= take
            if lq <= 1e-12:
                lots.pop(0)
            else:
                lots[0][0] = lq
        gain -= cost  # Transaktionskosten mindern den steuerpflichtigen Gewinn
        tax = self.regime.on_realized_gain(gain, self.asset)
        self.cash += proceeds - cost - tax
        self.tax_paid += tax
        self.tax_on_gains += tax
        self.costs_paid += cost
        if not lots and sym in self.lots:
            del self.lots[sym]

    # ------------------------------------------------------------ Dividenden
    def book_dividend(
        self, sym: str, per_share: float, asset: AssetClass | None = None
    ) -> None:
        """Dividende genau EINMAL besteuern.

        Der Preis-Panel ist total-return-adjustiert, die Bruttodividende steckt
        also bereits im Kurspfad. Mandat I entnahm daher nur die Steuer und
        liess die Lot-Basis unveraendert — mit der Folge, dass derselbe
        Dividenden-Euro ZWEIMAL besteuert wurde: einmal am Ex-Tag und ein
        zweites Mal beim Verkauf, weil er im Veraeusserungsgewinn wieder
        auftauchte.

        Bei EINEM Steuersatz war das common-mode und weitgehend folgenlos. In
        Mandat II ist der Satz ein Parameter, und die Verzerrung skaliert mit
        dem KURSGEWINN-Satz: gemessen 52,75 % effektiv bei PRIVAT_DE gegen
        31,32 % bei der GmbH — das DREHT das Vorzeichen der Kernasymmetrie
        (fachlich sind GmbH-Dividenden mit 29,83 % teurer als 26,375 %, nicht
        billiger). Der Fix hebt die Lot-Basis um die Bruttodividende an, damit
        der Dividendenanteil nicht erneut im Kursgewinn landet.

        Konsequenz, bewusst: Mandat-II-Laeufe MIT Dividenden reproduzieren die
        Mandat-I-Zahlen nicht mehr. Das ist eine dokumentierte Korrektur, keine
        Regression — siehe E-068.
        """
        if per_share <= 0:
            return
        lots = self.lots.get(sym)
        if not lots:
            return
        gross = self.qty(sym) * per_share
        if gross <= 0:
            return
        tax = self.regime.on_dividend(gross, asset or self.asset)
        self.cash -= tax
        self.tax_paid += tax
        self.tax_on_dividends += tax
        # Basis-Anhebung: pro Stueck genau die Bruttodividende, in denselben
        # Einheiten wie der Kurs (dieselbe Groesse, die oben mit qty
        # multipliziert wurde).
        for lot in lots:
            lot[1] += per_share

    # ------------------------------------------------------------- Bewertung
    def value(self, prices: pd.Series) -> float:
        v = self.cash
        for sym, lots in self.lots.items():
            px = prices.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px
        return v

    def liquidate_all(self, prices: pd.Series) -> None:
        """Alle Lots zum Fensterende verkaufen — Steuer wird real faellig.

        Ohne diesen Schritt traegt das Endvermoegen unversteuerte Buchgewinne
        (mark-to-market). Genau das ist der Fehler, den Mandat I als BUG 2
        gefuehrt hat und der dort das Vorzeichen des Kernbefunds drehte:
        H-032 low-div mtm 2.005.510 gegen postliq 1.589.963 gegen ETF
        1.610.149 — mark-to-market machte den Verlierer zum Gewinner.

        Regimegerecht ohne Sonderfall: der Verkauf laeuft durch
        ``on_realized_gain``, also privat 26,375 %, GmbH-Aktie ~1,49 %,
        Fonds 18,46 % bzw. 11,57 %.
        """
        for sym in list(self.lots):
            px = prices.get(sym, np.nan)
            if np.isfinite(px) and px > 0:
                self.sell(sym, self.qty(sym), float(px))

    def settle_terminal(self, final_value: float) -> float:
        """Schlussbesteuerung (nur GMBH_AUSSCHUETTUNG) — gibt das Nettovermoegen zurueck.

        Getrennt von ``value()``, damit die Equity-Kurve waehrend des Laufs
        unveraendert bleibt und nur der Endwert die Ausschuettungsebene traegt.
        """
        tax = self.regime.on_terminal(final_value, self.initial_capital)
        self.tax_terminal = tax
        self.tax_paid += tax
        return final_value - tax
