# -*- coding: utf-8 -*-
"""Deutsches Privat-Steuerregime fuer die operative Steuer-Sicht (Plan 5.5).

PORT (2026-08-17, Audit-Plan 5.5, nach Operator-Deny-Lift) aus
``research/mandat2/tax_regimes.py`` — die dort FESTGESCHRIEBENE Methodik
(Memory: „NICHT wieder aufmachen") wird 1:1 uebernommen, nicht neu designt:

- Satz 26,375 % (Abgeltung 25 % + SolZ 5,5 % darauf), KEINE Kirchensteuer.
- Sparerpauschbetrag 1.000 EUR/Jahr, Reset bei ``new_year``.
- Aktien-Verlusttopf: ueberjaehrig persistent (= Verlustvortrag).
- Reihenfolge je realisiertem Gewinn: Verlusttopf -> Pauschbetrag -> Satz.
- Realisierungsbasiert (``on_terminal`` = 0, laufend besteuert).

NICHT portiert: VvGmbH-Regime (Forschungsfrage GmbH-vs-privat, fuer die
Anlage-KAP-Sicht des Piloten irrelevant). ``latent_rate``/``latente_last``
sind mitportiert, damit die Klasse zur Quelle bitgleich bleibt
(Paritaetstest ``tests/test_tax_regime_de_parity.py``) — im Betrieb werden
sie nicht aufgerufen (Zielfunktions-Konstrukt der Forschungskampagne).

Das Research-Original bleibt unangetastet und ist durch
``tests/test_mandat2_tax_regimes.py`` eingefroren; DIESE Datei ist die
operative Wahrheit fuer accounting/compliance.
"""

from __future__ import annotations

from enum import Enum

ABGELTUNG = 0.25
SOLZ_AUF_ABGELTUNG = 0.055
PRIVAT_SATZ = ABGELTUNG * (1 + SOLZ_AUF_ABGELTUNG)  # 0.26375
SPARERPAUSCHBETRAG = 1000.0

# §20 InvStG Teilfreistellung fuer Aktienfonds (>= 51 % Kapitalbeteiligung)
TFS_AKTIENFONDS_PRIVAT = 0.30


class AssetClass(Enum):
    """Wovon der Satz abhaengt — nicht nur das Regime entscheidet."""

    AKTIE = "aktie"  # Anteil an einer Kapitalgesellschaft
    FONDS = "fonds"  # Investmentfonds/ETF (§20 InvStG)
    DERIVAT = "derivat"  # Termingeschaeft


class ZeroTax:
    """Referenzwelt ohne jede Steuer (Brutto-Sicht fuer Vergleiche/Tests)."""

    name = "ZERO"

    def new_year(self, year: int) -> None:
        return None

    def on_realized_gain(
        self, gain: float, asset: AssetClass = AssetClass.AKTIE
    ) -> float:
        return 0.0

    def on_dividend(self, gross: float, asset: AssetClass = AssetClass.AKTIE) -> float:
        return 0.0

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0

    def annual_fixed_costs(self) -> float:
        return 0.0

    def latent_rate(self, asset: AssetClass = AssetClass.AKTIE) -> float:
        return 0.0

    def latente_last(
        self, gewinne: float, verluste: float, asset: AssetClass = AssetClass.AKTIE
    ) -> float:
        return 0.0


class PrivatDE:
    """Deutscher Privatanleger — Mandat-I-Verhalten fuer Einzelaktien.

    Reihenfolge (Verlusttopf -> Pauschbetrag -> Steuer) bewusst 1:1 aus
    Mandat I uebernommen, damit der Regressionstest greifen kann.
    """

    name = "PRIVAT_DE"

    def __init__(
        self,
        satz: float = PRIVAT_SATZ,
        pauschbetrag: float = SPARERPAUSCHBETRAG,
    ) -> None:
        self.satz = satz
        self.pauschbetrag_annual = pauschbetrag
        self.pauschbetrag_left = 0.0
        self.loss_pot = 0.0
        self._cur_year: int | None = None

    def _satz(self, asset: AssetClass) -> float:
        if asset is AssetClass.FONDS:
            # Teilfreistellung 30 % -> 18,4625 %
            return self.satz * (1 - TFS_AKTIENFONDS_PRIVAT)
        return self.satz

    def new_year(self, year: int) -> None:
        if year != self._cur_year:
            self._cur_year = year
            self.pauschbetrag_left = self.pauschbetrag_annual

    def on_realized_gain(
        self, gain: float, asset: AssetClass = AssetClass.AKTIE
    ) -> float:
        if gain < 0:
            self.loss_pot += -gain
            return 0.0
        offset = min(gain, self.loss_pot)
        self.loss_pot -= offset
        taxable = gain - offset
        used = min(taxable, self.pauschbetrag_left)
        self.pauschbetrag_left -= used
        return (taxable - used) * self._satz(asset)

    def on_dividend(self, gross: float, asset: AssetClass = AssetClass.AKTIE) -> float:
        # Wie Mandat I: Dividenden laufen NICHT gegen den Aktien-Verlusttopf
        # und NICHT gegen den Pauschbetrag (bewusst konservativ).
        return max(gross, 0.0) * self._satz(asset)

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0  # laufend besteuert

    def annual_fixed_costs(self) -> float:
        return 0.0

    def latent_rate(self, asset: AssetClass = AssetClass.AKTIE) -> float:
        return self._satz(asset)

    def latente_last(
        self, gewinne: float, verluste: float, asset: AssetClass = AssetClass.AKTIE
    ) -> float:
        """Verluste verrechnen UND den bestehenden Verlusttopf anrechnen."""
        netto = gewinne - verluste - self.loss_pot
        if netto <= 0:
            return 0.0
        return netto * self._satz(asset)
