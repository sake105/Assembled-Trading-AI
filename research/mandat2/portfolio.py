"""Regime-agnostisches FIFO-Portfolio (Mandat II, Phase 0).

Struktur bewusst 1:1 uebernommen von ``research/mandat/h011_kandidat_a.py``
(``TaxedPortfolio``) — dieselbe FIFO-Mechanik, dieselbe Kostenlogik, dieselbe
Reihenfolge der Operationen. Der EINZIGE Unterschied: die Steuerentscheidung
wandert in ein ``TaxRegime``-Objekt.

Warum so konservativ: nur wenn ``PRIVAT_DE`` die Mandat-I-Zahlen exakt
reproduziert, sind Mandat-II-Ergebnisse mit Mandat I vergleichbar. Der
Regressionstest in ``test_regime_regression.py`` ist das Gate fuer Phase 1 —
laeuft er nicht gruen, ist jeder Vergleich wertlos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.mandat2.tax_regimes import TaxRegime

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
    ) -> None:
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

    # ------------------------------------------------------------- Kalender
    def set_date(self, date) -> None:
        self.regime.new_year(date.year)

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
        tax = self.regime.on_realized_gain(gain)
        self.cash += proceeds - cost - tax
        self.tax_paid += tax
        self.tax_on_gains += tax
        self.costs_paid += cost
        if not lots and sym in self.lots:
            del self.lots[sym]

    # ------------------------------------------------------------ Dividenden
    def book_dividend(self, sym: str, per_share: float) -> None:
        """Nur die Steuer wird als Cash entnommen.

        Wie in Mandat I: der Preis-Panel ist bereits total-return-adjustiert,
        die Bruttodividende steckt also im Kurspfad. Wir ziehen nur die Steuer
        ab — das approximiert Netto-Reinvestition.
        """
        if per_share <= 0:
            return
        gross = self.qty(sym) * per_share
        if gross <= 0:
            return
        tax = self.regime.on_dividend(gross)
        self.cash -= tax
        self.tax_paid += tax
        self.tax_on_dividends += tax

    # ------------------------------------------------------------- Bewertung
    def value(self, prices: pd.Series) -> float:
        v = self.cash
        for sym, lots in self.lots.items():
            px = prices.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px
        return v

    def settle_terminal(self, final_value: float) -> float:
        """Schlussbesteuerung (nur GMBH_AUSSCHUETTUNG) — gibt das Nettovermoegen zurueck.

        Getrennt von ``value()``, damit die Equity-Kurve waehrend des Laufs
        unveraendert bleibt und nur der Endwert die Ausschuettungsebene traegt.
        """
        tax = self.regime.on_terminal(final_value, self.initial_capital)
        self.tax_terminal = tax
        self.tax_paid += tax
        return final_value - tax
