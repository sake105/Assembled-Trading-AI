"""Steuerregime als austauschbare Objekte (Mandat II, Phase 0).

Mandat I hatte den deutschen Privatanleger-Satz fest in ``TaxedPortfolio``
verdrahtet (``research/mandat/h011_kandidat_a.py``). Mandat II vergleicht vier
Welten, also muss das Regime ein Parameter werden.

Vertrag jedes Regimes
---------------------
Ein Regime ist ZUSTANDSBEHAFTET (Verlusttopf, Jahresfreibetrag) und wird pro
Backtest-Lauf frisch instanziiert. Es entscheidet ausschliesslich ueber Steuer —
Kosten, Lots und Cash bleiben Sache des Portfolios.

    regime.new_year(2020)              -> Jahreswechsel (Freibetraege scharf)
    tax = regime.on_realized_gain(g)   -> Steuer auf einen realisierten Gewinn
    tax = regime.on_dividend(brutto)   -> Steuer auf eine Bruttodividende
    tax = regime.on_terminal(v, e)     -> Schlussbesteuerung (Ausschuettung)

``on_realized_gain`` bekommt den Gewinn NACH Abzug der Transaktionskosten
(so wie Mandat I es rechnete) und gibt die faellige Steuer zurueck; Verluste
(negativer Gewinn) geben 0 zurueck und werden regime-intern verbucht.

Steuerliche Grundlagen (Stand 2026, ohne Kirchensteuer)
-------------------------------------------------------
PRIVAT_DE  §20 EStG: 25 % + 5,5 % SolZ = 26,375 %, FIFO, Aktien-Verlusttopf
           (nur mit Aktiengewinnen verrechenbar), Sparerpauschbetrag 1.000 €/J.

GMBH       §8b Abs. 2 KStG: Gewinne aus der Veraeusserung von Anteilen an
           Kapitalgesellschaften bleiben ausser Ansatz; §8b Abs. 3 S. 1 behandelt
           5 % davon als nichtabziehbare Betriebsausgabe -> effektiv
           5 % x Gesamtsatz.
           §8b Abs. 3 S. 3: Gewinnminderungen (Veraeusserungsverluste) sind
           NICHT abziehbar -> kein Verlusttopf, kein Steuervorteil aus Verlusten.
           §8b Abs. 4: Bezuege aus Streubesitz (< 10 % Beteiligung zu Beginn des
           Kalenderjahres) sind VOLL steuerpflichtig. Ein Aktienportfolio ist
           immer Streubesitz -> Dividenden zum vollen Satz.
           Gesamtsatz = KSt 15 % + SolZ 0,825 % + GewSt (3,5 % x Hebesatz).

WICHTIG: Das ist die gaengige Lesart, keine Steuerberatung. Fuer den VERGLEICH
von Strategien untereinander ist die Modellierung belastbar; bevor daraus eine
Strukturentscheidung wird, gehoert sie fachlich geprueft.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# --------------------------------------------------------------- Konstanten
KST = 0.15  # Koerperschaftsteuer
SOLZ_AUF_KST = 0.055  # Solidaritaetszuschlag, bemessen auf die KSt
GEWST_MESSZAHL = 0.035  # Steuermesszahl §11 Abs. 2 GewStG
HEBESATZ_DEFAULT = 4.00  # 400 % — grober Mittelwert deutscher Grossstaedte

ABGELTUNG = 0.25
SOLZ_AUF_ABGELTUNG = 0.055
PRIVAT_SATZ = ABGELTUNG * (1 + SOLZ_AUF_ABGELTUNG)  # 0.26375
SPARERPAUSCHBETRAG = 1000.0

# §8b Abs. 3 S. 1 / Abs. 5: pauschal 5 % gelten als nichtabziehbare Betriebsausgabe
NICHTABZIEHBAR_QUOTE = 0.05


def gmbh_gesamtsatz(hebesatz: float = HEBESATZ_DEFAULT) -> float:
    """KSt + SolZ + GewSt als ein Satz."""
    return KST * (1 + SOLZ_AUF_KST) + GEWST_MESSZAHL * hebesatz


@runtime_checkable
class TaxRegime(Protocol):
    """Vertrag, den jedes Steuerregime erfuellt."""

    name: str

    def new_year(self, year: int) -> None: ...
    def on_realized_gain(self, gain: float) -> float: ...
    def on_dividend(self, gross: float) -> float: ...
    def on_terminal(self, final_value: float, initial_capital: float) -> float: ...


class ZeroTax:
    """Referenzwelt ohne jede Steuer.

    Zweck: die Frage von Mandat I, Kernbefund 1, sauber isolieren — existiert
    ueberhaupt BRUTTO-Alpha? Wenn ein Kandidat hier nicht gewinnt, kann kein
    Steuerregime ihn retten.
    """

    name = "ZERO"

    def new_year(self, year: int) -> None:
        return None

    def on_realized_gain(self, gain: float) -> float:
        return 0.0

    def on_dividend(self, gross: float) -> float:
        return 0.0

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0


class PrivatDE:
    """Deutscher Privatanleger — identisch zu Mandat I.

    Bewusst 1:1 nachgebaut (inkl. Reihenfolge Verlusttopf -> Pauschbetrag ->
    Steuer), damit Mandat-II-Zahlen gegen Mandat I regressionsgeprueft werden
    koennen. Jede Abweichung hier wuerde alle Vergleiche entwerten.
    """

    name = "PRIVAT_DE"

    def __init__(
        self,
        satz: float = PRIVAT_SATZ,
        pauschbetrag: float = SPARERPAUSCHBETRAG,
        dividenden_satz: float | None = None,
    ) -> None:
        self.satz = satz
        # Dividenden laufen in Mandat I NICHT gegen den Aktien-Verlusttopf und
        # NICHT gegen den Pauschbetrag (bewusst konservativ) — uebernommen.
        self.dividenden_satz = satz if dividenden_satz is None else dividenden_satz
        self.pauschbetrag_annual = pauschbetrag
        self.pauschbetrag_left = 0.0
        self.loss_pot = 0.0
        self._cur_year: int | None = None

    def new_year(self, year: int) -> None:
        if year != self._cur_year:
            self._cur_year = year
            self.pauschbetrag_left = self.pauschbetrag_annual

    def on_realized_gain(self, gain: float) -> float:
        if gain < 0:
            self.loss_pot += -gain
            return 0.0
        offset = min(gain, self.loss_pot)
        self.loss_pot -= offset
        taxable = gain - offset
        used = min(taxable, self.pauschbetrag_left)
        self.pauschbetrag_left -= used
        return (taxable - used) * self.satz

    def on_dividend(self, gross: float) -> float:
        return max(gross, 0.0) * self.dividenden_satz

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0  # laufend besteuert, keine Schlussebene


class VvGmbH:
    """Vermoegensverwaltende GmbH, thesaurierend (Kapital bleibt in der GmbH).

    Die drei Asymmetrien gegenueber PRIVAT_DE, die diese Welt interessant
    machen — und die eine davon, die sie schlechter macht:

      + Kursgewinn effektiv ~1,5 % statt 26,375 %  -> Turnover kostet fast nichts
      - Veraeusserungsverluste NICHT abziehbar     -> kein Verlusttopf
      - Streubesitz-Dividenden VOLL steuerpflichtig ~29,8 % statt 26,375 %

    Erwartete Folge fuer die Strategiewahl: Momentum/Rotation gewinnt relativ,
    Dividendenstrategien verlieren relativ. Genau das ist die Hypothese.
    """

    name = "GMBH_THESAURIEREND"

    def __init__(
        self,
        hebesatz: float = HEBESATZ_DEFAULT,
        beteiligung_ueber_10pct: bool = False,
    ) -> None:
        self.gesamtsatz = gmbh_gesamtsatz(hebesatz)
        self.kursgewinn_satz = NICHTABZIEHBAR_QUOTE * self.gesamtsatz
        # §8b Abs. 4: unter 10 % Beteiligung volle Steuerpflicht der Bezuege.
        # Ein diversifiziertes Aktienportfolio ist immer Streubesitz.
        self.dividenden_satz = (
            NICHTABZIEHBAR_QUOTE * self.gesamtsatz
            if beteiligung_ueber_10pct
            else self.gesamtsatz
        )
        self.verluste_nicht_abziehbar = 0.0  # nur Buchhaltung/Diagnose

    def new_year(self, year: int) -> None:
        return None

    def on_realized_gain(self, gain: float) -> float:
        if gain < 0:
            # §8b Abs. 3 S. 3: Gewinnminderung bleibt ausser Ansatz. Kein
            # Verlusttopf, kein Vortrag, kein Steuervorteil — nur merken.
            self.verluste_nicht_abziehbar += -gain
            return 0.0
        return gain * self.kursgewinn_satz

    def on_dividend(self, gross: float) -> float:
        return max(gross, 0.0) * self.dividenden_satz

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0  # thesaurierend: keine Entnahme modelliert


class VvGmbHMitAusschuettung(VvGmbH):
    """GmbH + Durchschau bis ins Privatvermoegen.

    Kontrollrechnung zur thesaurierenden Variante: am Ende wird alles ueber der
    urspruenglichen Einlage ausgeschuettet und beim Gesellschafter mit
    Abgeltungsteuer belegt. Beantwortet „wie viel davon kann Hans wirklich
    ausgeben?" statt „wie viel steht in der GmbH?".

    Vereinfachung, bewusst und benannt: eine einmalige Schlussausschuettung
    statt laufender Entnahmen. Das ist die guenstigste Variante fuer die GmbH
    (maximale Stundung) — die Zahl ist also eine OBERgrenze des Nettovermoegens,
    keine Punktprognose.
    """

    name = "GMBH_AUSSCHUETTUNG"

    def __init__(
        self,
        hebesatz: float = HEBESATZ_DEFAULT,
        beteiligung_ueber_10pct: bool = False,
        ausschuettung_satz: float = PRIVAT_SATZ,
    ) -> None:
        super().__init__(hebesatz, beteiligung_ueber_10pct)
        self.ausschuettung_satz = ausschuettung_satz

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        ausschuettbar = max(final_value - initial_capital, 0.0)
        return ausschuettbar * self.ausschuettung_satz


REGIMES: dict[str, type] = {
    "ZERO": ZeroTax,
    "PRIVAT_DE": PrivatDE,
    "GMBH_THESAURIEREND": VvGmbH,
    "GMBH_AUSSCHUETTUNG": VvGmbHMitAusschuettung,
}


def make_regime(name: str, **kwargs) -> TaxRegime:
    """Frische Regime-Instanz. Pro Backtest-Lauf genau eine (Zustand!)."""
    try:
        cls = REGIMES[name]
    except KeyError:
        raise ValueError(
            f"Unbekanntes Steuerregime {name!r}. Bekannt: {sorted(REGIMES)}"
        ) from None
    return cls(**kwargs)
