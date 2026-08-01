"""Steuerregime als austauschbare Objekte (Mandat II, Phase 0).

Mandat I hatte den deutschen Privatanleger-Satz fest in ``TaxedPortfolio``
verdrahtet (``research/mandat/h011_kandidat_a.py``). Mandat II vergleicht vier
Welten, also muss das Regime ein Parameter werden.

Vertrag jedes Regimes
---------------------
Ein Regime ist ZUSTANDSBEHAFTET (Verlusttopf, Jahresfreibetrag, Fixkosten) und
wird pro Backtest-Lauf frisch instanziiert. Es entscheidet ausschliesslich ueber
Steuer — Kosten, Lots und Cash bleiben Sache des Portfolios.

    regime.new_year(2020)                    -> Jahreswechsel
    tax = regime.on_realized_gain(g, klasse) -> Steuer auf realisierten Gewinn
    tax = regime.on_dividend(brutto, klasse) -> Steuer auf Bruttoausschuettung
    tax = regime.on_terminal(v, e)           -> Schlussbesteuerung
    kosten = regime.annual_fixed_costs()     -> laufende Rechtsformkosten

INSTRUMENTENKLASSE (F-senior-2, 2026-08-01)
-------------------------------------------
Der Satz haengt NICHT nur vom Regime ab, sondern auch davon, WAS gehalten wird.
Das ist kein Detail: §8b KStG gilt fuer Anteile an Kapitalgesellschaften, ein
Aktien-ETF faellt dagegen unter §20 InvStG. Wuerde man beiden Seiten denselben
Satz geben, bekaeme der Einzelaktien-Kandidat rund 10 Prozentpunkte
Steuervorteil gegenueber dem SPY-Benchmark geschenkt — ein PASS waere dann ein
Rechtsform-Artefakt, kein Alpha. Mandat I hatte diese Unterscheidung
(``ETF_TAX = 0.185``); sie darf hier nicht verlorengehen.

Steuerliche Grundlagen (Stand 2026, ohne Kirchensteuer)
-------------------------------------------------------
PRIVAT_DE  §20 EStG: 25 % + 5,5 % SolZ = 26,375 %, FIFO, Aktien-Verlusttopf,
           Sparerpauschbetrag 1.000 €/J. Aktienfonds: Teilfreistellung 30 %
           (§20 Abs. 1 InvStG) -> 18,4625 % (= Mandat I ``ETF_TAX``).

GMBH       §8b Abs. 2 KStG: Veraeusserungsgewinne aus Anteilen bleiben ausser
           Ansatz; Abs. 3 S. 1 behandelt 5 % als nichtabziehbare Betriebsausgabe
           -> effektiv 5 % x Gesamtsatz.
           Abs. 3 S. 3: VeraeusserungsVERLUSTE sind NICHT abziehbar.
           Abs. 4: Bezuege aus Streubesitz (< 10 %) sind VOLL steuerpflichtig.
           Aktienfonds: §20 InvStG Teilfreistellung 80 % KSt / 40 % GewSt.
           Gesamtsatz = KSt 15 % + SolZ 0,825 % + GewSt (3,5 % x Hebesatz).

BEWUSSTE VEREINFACHUNGEN (benannt, nicht verschwiegen)
------------------------------------------------------
- Vorabpauschale auf Fondsanteile ist NICHT modelliert. Sie belastet den
  ETF-Benchmark, ihr Weglassen macht den Benchmark also BESSER — konservativ
  in unsere Richtung, weil wir ihn schlagen wollen.
- Termingeschaefte: voller Satz, Verluste abziehbar. Die Sonderregeln
  (§15 Abs. 4 EStG) sind nicht abgebildet.
- Keine Steuerberatung. Fuer den VERGLEICH von Strategien belastbar; vor einer
  Strukturentscheidung fachlich pruefen lassen.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

# --------------------------------------------------------------- Konstanten
KST = 0.15
SOLZ_AUF_KST = 0.055
GEWST_MESSZAHL = 0.035  # §11 Abs. 2 GewStG
HEBESATZ_DEFAULT = 4.00  # 400 % — grober Mittelwert deutscher Grossstaedte

ABGELTUNG = 0.25
SOLZ_AUF_ABGELTUNG = 0.055
PRIVAT_SATZ = ABGELTUNG * (1 + SOLZ_AUF_ABGELTUNG)  # 0.26375
SPARERPAUSCHBETRAG = 1000.0

NICHTABZIEHBAR_QUOTE = 0.05  # §8b Abs. 3 S. 1 / Abs. 5

# §20 InvStG Teilfreistellung fuer Aktienfonds (>= 51 % Kapitalbeteiligung)
TFS_AKTIENFONDS_PRIVAT = 0.30
TFS_AKTIENFONDS_KST = 0.80
TFS_AKTIENFONDS_GEWST = 0.40


class AssetClass(Enum):
    """Wovon der Satz abhaengt — nicht nur das Regime entscheidet."""

    AKTIE = "aktie"  # Anteil an einer Kapitalgesellschaft (§8b KStG)
    FONDS = "fonds"  # Investmentfonds/ETF (§20 InvStG) — der SPY-Benchmark
    DERIVAT = "derivat"  # Termingeschaeft — weder §8b noch InvStG


def kst_mit_solz() -> float:
    return KST * (1 + SOLZ_AUF_KST)


def gewst(hebesatz: float = HEBESATZ_DEFAULT) -> float:
    return GEWST_MESSZAHL * hebesatz


def gmbh_gesamtsatz(hebesatz: float = HEBESATZ_DEFAULT) -> float:
    """KSt + SolZ + GewSt als ein Satz."""
    return kst_mit_solz() + gewst(hebesatz)


class TaxRegime(Protocol):
    """Vertrag, den jedes Steuerregime erfuellt.

    Bewusst NICHT ``runtime_checkable``: ein Protocol mit dem Datenmember
    ``name`` laesst ``issubclass()`` mit TypeError scheitern, und ``REGIMES``
    haelt Klassen — ein spaeterer Registry-Check waere genau der erste
    Aufrufer, der darueber stolpert (F-senior-10).
    """

    name: str

    #: ``latent_rate`` liefert den Grenzsatz fuer eine GEDACHTE Liquidation.
    #: Bewusste Naeherung: Verlusttopf und Sparerpauschbetrag bleiben
    #: unberuecksichtigt (sie wuerden die latente Last senken), und der Aufruf
    #: veraendert KEINEN Regimezustand. Gebraucht wird er, damit die
    #: Zielfunktion nicht auf einer Mark-to-market-Kurve misst (E-071).
    def new_year(self, year: int) -> None: ...
    def on_realized_gain(self, gain: float, asset: AssetClass) -> float: ...
    def on_dividend(self, gross: float, asset: AssetClass) -> float: ...
    def on_terminal(self, final_value: float, initial_capital: float) -> float: ...
    def annual_fixed_costs(self) -> float: ...
    def latent_rate(self, asset: AssetClass) -> float: ...


class ZeroTax:
    """Referenzwelt ohne jede Steuer.

    Zweck: Kernbefund 1 aus Mandat I sauber isolieren — existiert ueberhaupt
    BRUTTO-Alpha? Wenn ein Kandidat hier nicht gewinnt, kann kein Steuerregime
    ihn retten.
    """

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
            # Teilfreistellung 30 % -> 18,4625 % (= Mandat I ETF_TAX 0.185)
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


class VvGmbH:
    """Vermoegensverwaltende GmbH, thesaurierend (Kapital bleibt in der GmbH).

    Die Asymmetrien, um die es geht — zwei zugunsten, zwei zulasten:

      + Kursgewinn Aktie effektiv ~1,5 % statt 26,375 %  -> Turnover fast gratis
      - Veraeusserungsverluste NICHT abziehbar           -> kein Verlusttopf
      - Streubesitz-Dividenden ~29,8 % statt 26,375 %    -> Dividenden teurer
      - laufende Rechtsformkosten (s. u.)

    ``fixkosten_pa``: Buchfuehrung, Jahresabschluss/E-Bilanz, Steuerberater,
    IHK, Offenlegung. Realistisch 2.000-5.000 EUR p. a. Bei 100.000 EUR
    Startkapital sind das 2-5 % p. a. und damit GROESSER als der gesamte
    modellierte Steuervorteil — die Zahl darf nicht auf 0 stehenbleiben, wenn
    die Frage lautet „GmbH oder privat?" (F-senior-5). Default ist bewusst
    0.0, damit reine Strategie-gegen-Strategie-Vergleiche innerhalb eines
    Regimes unverzerrt bleiben; die Kampagne setzt sie explizit.
    """

    name = "GMBH_THESAURIEREND"

    def __init__(
        self,
        hebesatz: float = HEBESATZ_DEFAULT,
        kst_schachtel: bool = False,  # >= 10 %: §8b Abs. 4 KStG
        gewst_schachtel: bool = False,  # >= 15 %: §9 Nr. 2a GewStG
        fixkosten_pa: float = 0.0,
    ) -> None:
        self.hebesatz = hebesatz
        self.gesamtsatz = gmbh_gesamtsatz(hebesatz)
        self.kursgewinn_satz = NICHTABZIEHBAR_QUOTE * self.gesamtsatz
        # Fondsanteile: Teilfreistellung getrennt fuer KSt und GewSt.
        self.fonds_satz = (1 - TFS_AKTIENFONDS_KST) * kst_mit_solz() + (
            1 - TFS_AKTIENFONDS_GEWST
        ) * gewst(hebesatz)
        # Dividenden: zwei SEPARATE Schwellen, nicht eine (F-senior-6).
        # 10 % befreit koerperschaftsteuerlich zu 95 %, 15 % zusaetzlich
        # gewerbesteuerlich. Dazwischen: KSt fast frei, GewSt voll.
        kst_teil = (
            NICHTABZIEHBAR_QUOTE * kst_mit_solz() if kst_schachtel else kst_mit_solz()
        )
        gewst_teil = (
            NICHTABZIEHBAR_QUOTE * gewst(hebesatz)
            if gewst_schachtel
            else gewst(hebesatz)
        )
        self.dividenden_satz = kst_teil + gewst_teil
        self.fixkosten_pa = fixkosten_pa
        self.verluste_nicht_abziehbar = 0.0  # Diagnose

    def new_year(self, year: int) -> None:
        return None

    def annual_fixed_costs(self) -> float:
        return self.fixkosten_pa

    def on_realized_gain(
        self, gain: float, asset: AssetClass = AssetClass.AKTIE
    ) -> float:
        if asset is AssetClass.FONDS:
            # InvStG kennt keine §8b-Verlustsperre: Verluste wirken normal.
            return max(gain, 0.0) * self.fonds_satz
        if asset is AssetClass.DERIVAT:
            return max(gain, 0.0) * self.gesamtsatz
        if gain < 0:
            # §8b Abs. 3 S. 3: Gewinnminderung bleibt ausser Ansatz. Kein
            # Verlusttopf, kein Vortrag, kein Steuervorteil — nur merken.
            self.verluste_nicht_abziehbar += -gain
            return 0.0
        return gain * self.kursgewinn_satz

    def on_dividend(self, gross: float, asset: AssetClass = AssetClass.AKTIE) -> float:
        if gross <= 0:
            return 0.0
        if asset is AssetClass.FONDS:
            return gross * self.fonds_satz
        return gross * self.dividenden_satz

    def latent_rate(self, asset: AssetClass = AssetClass.AKTIE) -> float:
        if asset is AssetClass.FONDS:
            return self.fonds_satz
        if asset is AssetClass.DERIVAT:
            return self.gesamtsatz
        return self.kursgewinn_satz

    def on_terminal(self, final_value: float, initial_capital: float) -> float:
        return 0.0  # thesaurierend: keine Entnahme modelliert


class VvGmbHMitAusschuettung(VvGmbH):
    """GmbH + Durchschau bis ins Privatvermoegen.

    Kontrollrechnung: am Ende wird alles ueber der urspruenglichen Einlage
    ausgeschuettet und beim Gesellschafter besteuert. Beantwortet „wie viel
    kann Hans wirklich ausgeben?" statt „wie viel steht in der GmbH?".

    Vereinfachung und ihre Grenzen (F-senior-7): eine einmalige
    Schlussausschuettung statt laufender Entnahmen. Direktional eine
    OBERgrenze des Nettovermoegens, weil unrealisierte Gewinne bei laufender
    Entnahme frueher auf Koerperschaftsebene belastet wuerden. NICHT strikt,
    denn (a) ueber mehrere Jahre gestreckte Ausschuettungen nutzen den
    Sparerpauschbetrag mehrfach, und (b) laeuft der Zugriff ueber die
    LIQUIDATION der GmbH und haelt der Gesellschafter >= 1 %, greift §17 EStG
    Teileinkuenfteverfahren (60 % x persoenlichem Satz) statt Abgeltungsteuer
    — zahlenmaessig zufaellig fast gleich (0,6 x 44,3 % ~ 26,6 %), rechtlich
    ein anderer Tatbestand.
    """

    name = "GMBH_AUSSCHUETTUNG"

    def __init__(
        self,
        hebesatz: float = HEBESATZ_DEFAULT,
        kst_schachtel: bool = False,
        gewst_schachtel: bool = False,
        fixkosten_pa: float = 0.0,
        ausschuettung_satz: float = PRIVAT_SATZ,
    ) -> None:
        super().__init__(hebesatz, kst_schachtel, gewst_schachtel, fixkosten_pa)
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
