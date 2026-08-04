"""P12e — Sind die bisherigen Verdicts der Kampagne kontaminiert?

DIE ZWEI OFFENEN FRAGEN AUS P12d
--------------------------------
P12d hinterliess zwei Punkte, die ueber P12 hinausgehen und **alle** bisherigen
Verdicts betreffen:

1. **Korrumpierte Kursserien.** Neun Namen des Index-Universums tragen
   Vendor-Fehler (MEL, CIN, HPC, CFC, KRI, RX, SLR, CPWR, TIN — Serien mit zwei
   ineinander verschraenkten Preisskalen). Wurden sie in P1-P11 **gehalten**?
   Ein Name, dessen Kurs scheinbar um Faktor 18.000 springt, hat maximales
   Momentum und wuerde von jeder Rangfolge gewaehlt.

2. **Abdeckungsluecke.** Nur 84-96 % der Index-Mitglieder haben ueberhaupt eine
   Preisspalte, und die Fehlenden sind mit Index-Austritten angereichert. Die
   Zahlen standen bisher als Prosa im Befund-Generator — also ausgerechnet die
   Zahlen, die eine Einschraenkung tragen, waren die einzigen ohne Artefakt.

WIE HIER GEMESSEN WIRD
----------------------
Die Auswahl wird **nachgespielt**, nicht nachgebaut: dieselbe
``momentum_score``, dieselben ``_monatsenden``, dieselbe
``membership(t) & close.columns``-Schnittmenge und dasselbe
``nlargest(top_in)`` wie in ``engine.run_strategy``. Wuerde ich die Logik
paraphrasieren, koennte das Ergebnis am Detail vorbeigehen — und die Frage
lautet gerade, was die Engine TATSAECHLICH gewaehlt hat.

Gemessen wird ueber das volle SUCHfenster (1995-2016), nicht nur ueber das
P12-Fenster: P1-P11 liefen auf der ganzen Strecke.

WAS DIESER TEST NICHT KANN
--------------------------
Er sagt, OB und WANN ein korrumpierter Name gewaehlt worden waere. Er sagt
nicht, wie gross der Ergebniseffekt in den einzelnen Phasen war — dafuer
muessten alle Phasen mit bereinigtem Panel neu laufen. Ist die Schnittmenge
leer, eruebrigt sich das; ist sie es nicht, ist die Neurechnung faellig und
dieser Befund benennt sie als solche.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import _monatsenden, momentum_score  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402
from research.mandat2.p12d_survivorship_schranke import (  # noqa: E402
    korruptions_spannen,
)

OUT = Path(__file__).resolve().parent / "results"
TOP_IN = 20  # Parametrisierung der Kampagne (P2-Gewinner)
# momentum_score = close.shift(MOM_LAG) / close.shift(MOM_FENSTER) — in
# HANDELSTAGEN. Hier gespiegelt, damit die Kanal-B-Rechnung nicht still an einer
# Fensteraenderung in der Engine vorbeilaeuft (Stage-1-Finding F-test-4).
MOM_LAG, MOM_FENSTER = 21, 252


def abdeckung_je_termin(membership: pd.Series, spalten: set[str]) -> list[dict]:
    """Anteil der Index-Mitglieder mit Preisspalte, je Monatsende.

    Diese Zahlen trugen im Befund eine Einschraenkung und standen trotzdem
    nur als Prosa im Generator — die einzige Stelle des Abschnitts ohne
    Artefakt (Stage-3-Finding F-auditor-4). Hier werden sie erzeugt.
    """
    aus = []
    for t, mitglieder in membership.items():
        n = len(mitglieder)
        if not n:
            continue
        aus.append(
            {
                "termin": f"{t:%Y-%m-%d}",
                "n_mitglieder": n,
                "n_mit_spalte": len(set(mitglieder) & spalten),
                "abdeckung": len(set(mitglieder) & spalten) / n,
            }
        )
    return aus


def austritts_anreicherung(membership: pd.Series, spalten: set[str]) -> dict:
    """Sind die Namen OHNE Preisspalte ueberdurchschnittlich ausgeschieden?

    Wenn ja, ist die Abdeckungsluecke nicht neutral, sondern korreliert mit
    genau dem Ereignis, das ein survivorship-freier Test abbilden soll.
    """
    erst, letzt = membership.index[0], membership.index[-1]
    am_ende = set(membership.loc[letzt])
    start = set(membership.loc[erst])
    mit = start & spalten
    ohne = start - spalten
    return {
        "referenz_start": f"{erst:%Y-%m-%d}",
        "referenz_ende": f"{letzt:%Y-%m-%d}",
        "n_mit_spalte": len(mit),
        "n_ohne_spalte": len(ohne),
        "ueberlebensquote_mit_spalte": len(mit & am_ende) / len(mit) if mit else None,
        "ueberlebensquote_ohne_spalte": len(ohne & am_ende) / len(ohne)
        if ohne
        else None,
    }


def gewaehlte_namen(data, top_in: int = TOP_IN) -> dict[str, list[str]]:
    """Spielt die Auswahl von ``engine.run_strategy`` nach — Termin fuer Termin.

    Bewusst mit den Funktionen der Engine selbst (``momentum_score``,
    ``_monatsenden``), nicht paraphrasiert: gefragt ist, was die Engine
    TATSAECHLICH waehlt, nicht was eine Nachbildung waehlen wuerde.
    """
    close = data.close
    idx = pd.DatetimeIndex(close.index)
    mom = momentum_score(close)
    aus: dict[str, list[str]] = {}
    for t in _monatsenden(idx):
        mitglieder = data.membership.get(t)
        if mitglieder is None:
            continue
        kandidaten = sorted(set(mitglieder) & set(close.columns))
        scores = mom.loc[t, kandidaten].dropna()
        if scores.empty:
            continue
        scores = scores.sort_index()
        aus[f"{t:%Y-%m-%d}"] = sorted(scores.nlargest(top_in, keep="first").index)
    return aus


def gehaltene_namen(data, **kw) -> tuple[dict[str, set[str]], "pd.Series"]:
    """Die TATSAECHLICH gehaltenen Namen je Handelstag — aus der echten Engine.

    WARUM NICHT AUS DER AUSWAHL ABGELEITET (Stage-1-Finding F-test-1, BLOCKER)
    -------------------------------------------------------------------------
    Die erste Fassung schloss vom letzten Top-20-Auswahltermin auf das Halten:
    „letzter Termin > 31 Tage her -> nicht mehr im Bestand". Das ist falsch. Die
    Engine verkauft nicht beim Verlassen der Top-20, sondern erst bei
    ``rang > rank_out`` (Default 60) bzw. nach ``min_haltetage`` — die Haltemenge
    ist echt groesser als die Auswahlmenge.

    Was der Proxy verfehlte, ist nicht akademisch: **GPS lag am 1996-12-20 im
    Bestand**, dem Tag seines Vendor-Fehlers. Die Portfolio-Tagesrendite betrug
    dort **+12,36 %** — der zweitgroesste Einzeltag der gesamten 21 Jahre, rein
    aus einem Datenfehler. Der Proxy stufte den Fall als „81 Tage vor dem Glitch,
    Halteperiode vorbei" ein.

    Statt die Halte-Logik nachzubauen (und dabei erneut abzuweichen), wird hier
    die echte ``run_strategy`` gefahren und ``Portfolio.set_date`` umschlossen,
    um den Bestand je Handelstag mitzuschreiben. Kein Eingriff in die Engine.
    """
    from research.mandat2 import engine as eng
    from research.mandat2.portfolio import Portfolio

    protokoll: dict[str, set[str]] = {}
    original = Portfolio.set_date

    def aufzeichnend(self, t):  # noqa: ANN001
        original(self, t)
        protokoll[f"{t:%Y-%m-%d}"] = set(self.lots)

    Portfolio.set_date = aufzeichnend
    try:
        lauf = eng.run_strategy(data, make_regime("ZERO"), **kw)
    finally:
        Portfolio.set_date = original
    return protokoll, lauf.equity


def kanal_halten(
    glitches: dict[str, dict],
    bestand: dict[str, set[str]],
    rendite: "pd.Series",
) -> dict[str, dict]:
    """Kanal A — an welchen korrupten Tagen lag der Name im BESTAND?

    ``bestand`` kommt aus ``gehaltene_namen`` und damit aus der echten Engine.
    Ein Proxy aus der Auswahlmenge unterschaetzt die Haltedauer und lieferte
    die falsche Entwarnung „keiner ueber den Halte-Kanal" (E-102).
    """
    if not bestand:
        raise SystemExit(
            "Bestandsprotokoll leer — die Instrumentierung hat nicht gegriffen."
        )
    aus: dict[str, dict] = {}
    for sym, g in glitches.items():
        tage = [t for t in g["uebergaenge"] if sym in bestand.get(t, ())]
        if not tage:
            continue
        wirkung = {}
        for t in tage:
            ts = pd.Timestamp(t, tz="UTC")
            if ts not in rendite.index:
                # Fail-loud: ein fehlender Renditewert zu einem GEHALTENEN
                # korrupten Tag wuerde als 0,0 % gerendert und damit als
                # Entwarnung gelesen. Die Messung darf nicht in die beruhigende
                # Richtung ausfallen (E-103).
                raise SystemExit(
                    f"{sym}: keine Portfolio-Rendite fuer den gehaltenen "
                    f"korrupten Tag {t} — Verdrahtung pruefen, nicht ignorieren."
                )
            wirkung[t] = float(rendite.loc[ts])
        # Vorzeichen BEHALTEN: der Extremwert kann ein Verlusttag sein. Eine
        # frueher gespeicherte abs()-Groesse wurde danach mit "+" formatiert und
        # als "Gewinn" beschrieben (F-senior-4).
        extrem = max(wirkung.values(), key=abs, default=0.0)
        # Rang und Beitrag MESSEN, nicht behaupten: eine fruehere Fassung
        # schrieb "zweitgroesster Einzeltag der 21 Jahre" und "rein aus einem
        # Datenfehler" ins permanente Fehler-Log, ohne beides je zu rechnen
        # (Stage-2-Finding F-senior-9).
        extrem_tag = max(wirkung, key=lambda k: abs(wirkung[k]))
        alle = rendite.dropna()
        rang = int((alle > wirkung[extrem_tag]).sum()) + 1
        aus[sym] = {
            "tage": tage,
            "n_tage": len(tage),
            "portfolio_tagesrendite": wirkung,
            "groesste_wirkung": extrem,
            "groesste_wirkung_betrag": abs(extrem),
            "groesste_wirkung_tag": extrem_tag,
            "rang_unter_allen_tagen": rang,
            "n_handelstage": int(len(alle)),
        }
    return aus


def _skala_id(tag: "pd.Timestamp", spannen: list) -> int:
    """Welche Preisskala gilt an diesem Tag? 0 = Basisskala, sonst Spannen-Nr."""
    for k, (a, b) in enumerate(spannen, start=1):
        if a <= tag < b:
            return k
    return 0


def kanal_auswahl(
    spannen_je_symbol: dict[str, dict],
    gewaehlt: dict[str, list[str]],
    idx: "pd.DatetimeIndex",
) -> dict[str, list[str]]:
    """Kanal B — welche Auswahltermine hatten einen kontaminierten Score?

    DIE RICHTIGE BEDINGUNG (und warum beide frueheren Fassungen daneben lagen)
    -------------------------------------------------------------------------
    ``momentum_score`` ist ``close.shift(MOM_LAG) / close.shift(MOM_FENSTER)``,
    also ein Quotient aus GENAU ZWEI Stuetzstellen. Daraus folgt:

    * Es gibt kein „Fenster", in dem ein Fehlertag den Score beruehrt — nur die
      beiden Beine zaehlen. Meine erste Fassung markierte jeden Termin im
      Intervall nach einem Fehlertag; bei einem Ein-Tages-Spike sind das
      ~230 Termine zu viel.
    * Umgekehrt genuegt es nicht, „ein Bein liegt auf einer falschen Skala" zu
      pruefen: liegen BEIDE Beine auf DERSELBEN falschen Skala, kuerzt sich der
      Faktor heraus und der Score ist korrekt.

    Richtig ist deshalb: **der Score ist kontaminiert, wenn die beiden Beine auf
    VERSCHIEDENEN Skalen liegen.** Das deckt beide Morphologien in einer
    Bedingung ab — den dauerhaften Niveaubruch (ein Bein davor, eins danach)
    ebenso wie den Einzelspike (ein Bein faellt genau auf den Spike).
    """
    aus: dict[str, list[str]] = {}
    pos_von_datum = {d: k for k, d in enumerate(idx)}
    for sym, info in spannen_je_symbol.items():
        spannen = [
            (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
            for a, b in info["spannen"]
        ]
        if not spannen:
            continue
        betroffen: list[str] = []
        for termin, namen in gewaehlt.items():
            if sym not in namen:
                continue
            p = pos_von_datum.get(pd.Timestamp(termin, tz="UTC"))
            if p is None or p < MOM_FENSTER:
                continue
            bein_kurz = idx[p - MOM_LAG]
            bein_lang = idx[p - MOM_FENSTER]
            if _skala_id(bein_kurz, spannen) != _skala_id(bein_lang, spannen):
                betroffen.append(termin)
        if betroffen:
            aus[sym] = sorted(betroffen)
    return aus


def main() -> int:
    OUT.mkdir(exist_ok=True)
    regen = "--regen" in sys.argv
    d = load_campaign()
    spalten = set(d.close.columns)

    if regen:
        print("[SKIP] Trial-Zaehler: --regen (Wiederholungslauf)", flush=True)
    else:
        print(
            f"Trials kumuliert: "
            f"{TrialCounter().increment(1, label='P12e Panel-Hygiene')}\n",
            flush=True,
        )

    # ---- Frage 1: sind die Preisfehler in eine Rendite eingegangen? ----
    # Glitch-Erkennung ueber das VOLLE Suchfenster — P1-P11 liefen auf der
    # ganzen Strecke, nicht nur auf dem P12-Fenster.
    alle_je_mitglied: set[str] = set()
    for mitglieder in d.membership:
        alle_je_mitglied |= set(mitglieder)
    glitches = korruptions_spannen(d.close, alle_je_mitglied & spalten)
    uebergangstage = sum(len(g["uebergaenge"]) for g in glitches.values())
    tage_falsch = sum(g["n_tage_falsch"] for g in glitches.values())
    print(
        f"Korrumpierte Namen im vollen Suchfenster: {len(glitches)}\n"
        f"  {uebergangstage} Uebergangstage (dort ist die TAGESRENDITE verzerrt)\n"
        f"  {tage_falsch} Tage auf falscher Preisskala (dort ist nur das NIVEAU\n"
        "  falsch — fuer Renditen folgenlos, solange Vortag und Tag dieselbe\n"
        "  Skala teilen; relevant nur fuer die Momentum-Beine)"
    )
    for sym, g in sorted(glitches.items(), key=lambda kv: -kv[1]["n_tage_falsch"])[:5]:
        print(
            f"  {sym:<7}{g['n_tage_falsch']:>5} Tage falsch | Spannen {g['spannen'][:2]}"
        )

    # ---- Kanal A: HALTEN ueber einen korrupten Tag ----
    # Aus der ECHTEN Engine, nicht aus der Auswahl abgeleitet. Der frueher
    # benutzte Proxy (letzter Top-20-Termin <= 31 Tage her) unterschaetzt die
    # Haltedauer massiv, weil die Engine erst bei rang > rank_out verkauft.
    bestand, equity = gehaltene_namen(d, top_in=TOP_IN)
    rendite = equity.pct_change(fill_method=None)
    halte_kanal = kanal_halten(glitches, bestand, rendite)

    idx = pd.DatetimeIndex(d.close.index)
    gewaehlt = gewaehlte_namen(d)
    auswahl_kanal = kanal_auswahl(glitches, gewaehlt, idx)

    plaetze_gesamt = sum(len(v) for v in gewaehlt.values())
    plaetze_auswahl = sum(len(v) for v in auswahl_kanal.values())

    print(f"\nAuswahltermine: {len(gewaehlt)} | vergebene Plaetze: {plaetze_gesamt}")
    print(
        f"\nKANAL A (ueber einen korrupten Tag GEHALTEN): {len(halte_kanal)} Namen, "
        f"{sum(v['n_tage'] for v in halte_kanal.values())} Handelstage"
    )
    for sym, v in sorted(halte_kanal.items()):
        print(
            f"  {sym:<7}{v['n_tage']:>3} Tage | groesste Portfolio-Tagesrendite an "
            f"einem dieser Tage: {v['groesste_wirkung']:+.2%}"
        )
    print(
        f"\nKANAL B (im kontaminierten Momentum-Fenster GEWAEHLT): "
        f"{len(auswahl_kanal)} Namen, {plaetze_auswahl} von {plaetze_gesamt} "
        f"Plaetzen ({plaetze_auswahl / plaetze_gesamt:.2%})"
    )
    for sym, termine in sorted(auswahl_kanal.items()):
        print(f"  {sym:<7}{len(termine):>3} Termine: {termine[0]} .. {termine[-1]}")

    # ---- Frage 2: Abdeckung ----
    abd = abdeckung_je_termin(d.membership, spalten)
    anr = austritts_anreicherung(d.membership, spalten)
    quoten = [a["abdeckung"] for a in abd]
    print(
        f"\nAbdeckung der Index-Mitglieder: min {min(quoten):.1%} | "
        f"Median {sorted(quoten)[len(quoten) // 2]:.1%} | max {max(quoten):.1%}"
    )
    if anr["ueberlebensquote_ohne_spalte"] is not None:
        faktor = (
            anr["ueberlebensquote_mit_spalte"] / anr["ueberlebensquote_ohne_spalte"]
            if anr["ueberlebensquote_ohne_spalte"] > 0
            else float("inf")
        )
        anr["anreicherungsfaktor"] = faktor
        print(
            f"Ueberlebensquote {anr['referenz_start']} -> {anr['referenz_ende']}: "
            f"mit Spalte {anr['ueberlebensquote_mit_spalte']:.1%}, "
            f"ohne {anr['ueberlebensquote_ohne_spalte']:.1%} "
            f"-> Faktor {faktor:.1f}x"
        )

    ergebnis = {
        "fenster": d.fenster,
        "top_in": TOP_IN,
        "momentum_fenster_handelstage": [MOM_LAG, MOM_FENSTER],
        "n_auswahltermine": len(gewaehlt),
        "auswahlplaetze_gesamt": plaetze_gesamt,
        "korrumpierte_namen": glitches,
        "uebergangstage_gesamt": uebergangstage,
        "tage_auf_falscher_skala_gesamt": tage_falsch,
        "halte_kanal": halte_kanal,
        "auswahl_kanal": auswahl_kanal,
        "auswahlplaetze_kanal_b": plaetze_auswahl,
        "anteil_plaetze_kanal_b": plaetze_auswahl / plaetze_gesamt
        if plaetze_gesamt
        else 0.0,
        "kontaminiert": bool(halte_kanal or auswahl_kanal),
        # Vollstaendig, weil genau das die tragende Einschraenkung ist: Steuern
        # und Kosten veraendern Cash und damit Positionsgroessen und den
        # Bestandspfad, aus dem Kanal A kommt (F-senior-7). Ein Provenienzfeld,
        # das die Provenienz nicht dokumentiert, ist wertlos.
        "gemessene_konfiguration": {
            "score": "momentum_score (12-1)",
            "top_in": TOP_IN,
            "rank_out": 60,
            "min_haltetage": 0,
            "hebel": 1.0,
            "regime": "ZERO (keine Steuern)",
            "cost_bps": "engine-Default",
        },
        "abdeckung": {
            "min": min(quoten),
            "median": sorted(quoten)[len(quoten) // 2],
            "max": max(quoten),
            "je_termin": abd,
        },
        "austritts_anreicherung": anr,
    }
    (OUT / "p12e_panel_hygiene.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12e_panel_hygiene.json'}")

    print("\n" + "=" * 76)
    if halte_kanal or auswahl_kanal:
        groesste = max(
            (v["groesste_wirkung"] for v in halte_kanal.values()), default=0.0
        )
        print("BEFUND: Die bisherigen Verdicts sind BERUEHRT — ueber BEIDE Kanaele.")
        print(
            f"        Halte-Kanal: {len(halte_kanal)} Namen an "
            f"{sum(v['n_tage'] for v in halte_kanal.values())} korrupten Handelstagen; "
            f"groesste"
        )
        print(f"        Portfolio-Tagesrendite an einem solchen Tag: {groesste:+.2%}.")
        print(
            f"        Auswahl-Kanal: {plaetze_auswahl} von {plaetze_gesamt} Plaetzen "
            f"({plaetze_auswahl / plaetze_gesamt:.2%})."
        )
        print("        Eine fruehere Fassung meldete 'keiner ueber den Halte-Kanal' —")
        print("        das beruhte auf einem Proxy aus der AUSWAHL statt aus dem")
        print("        Bestand und war falsch (E-102).")
        print("        Ob das ein Verdikt dreht, ist NICHT gesagt: dafuer braeuchte es")
        print("        einen Lauf aller Phasen mit bereinigtem Panel.")
    else:
        print("BEFUND: Kein korrumpierter Name wurde ueber einen korrupten Tag")
        print("        gehalten oder im kontaminierten Momentum-Fenster gewaehlt.")
    print(
        "\nGELTUNGSBEREICH: gemessen fuer EINE Konfiguration (12-1-Momentum,\n"
        "top20, ungegatet). Phasen mit anderem Score, anderem top_in oder einem\n"
        "risk_off_gate waehlen andere Namen — dafuer gilt dieser Befund nicht."
    )
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
