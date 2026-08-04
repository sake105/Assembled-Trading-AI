"""Trennt recycelte Ticker in zwei Spalten — eine je Unternehmen.

DAS PROBLEM
-----------
P12h hat 29 Panel-Spalten mit einer Lücke von mindestens 500 Handelstagen
gefunden. Das Muster: die Historie
von Firma A, eine mehrjährige Lücke, dann Firma B unter demselben Symbol. Zwei
Schäden entstehen daraus:

1. **Der Kurssprung am Wiedereinstieg.** Die Engine bewertet eine in A gekaufte
   Position erstmals mit dem Kurs von B (CGP: −3,49 % Portfolio-Tagesrendite).
2. **Der Ausfall der Delisting-Hygiene** — der größere Schaden. Der
   Zwangsverkauf prüft ``last_valid < t``; bei einem recycelten Ticker liegt
   ``last_valid`` am Ende der Serie von B, die Bedingung ist nie erfüllt. CGP
   lief 3.264 Handelstage (13 Jahre) im Bestand weiter, ohne dass die Firma noch
   existierte.

WARUM TRENNEN UND NICHT ABSCHNEIDEN
-----------------------------------
Naheliegend wäre, die Serie nach dem letzten Kurs von A abzuschneiden. Das wäre
konservativ, entfernt aber Firma B vollständig aus dem Universum — auch dann,
wenn sie dort legitim stand. Namen zu entfernen, weil ihre Daten unbequem sind,
ist genau die Fehlerklasse, die diese Kampagne als E-079 registriert hat.

Stattdessen bekommt jede Firma ihre eigene Spalte: ``SYM`` behält die Historie
bis zum letzten echten Kurs von A, ``SYM#2`` trägt die Serie ab dem
Wiedereinstieg. Die Index-Mitgliedschaft wandert nach demselben Schnitt mit —
Termine vor dem Bruch bleiben bei ``SYM``, spätere gehen an ``SYM#2``.

Wirkung: ``last_valid`` für ``SYM`` liegt jetzt am letzten echten Kurs, der
Zwangsverkauf greift, und kein Kurssprung verbindet mehr zwei Unternehmen.

WAS DAMIT NICHT BEHOBEN WIRD
----------------------------
Die Zuordnung ist eine **Vermutung aus der Lückenlänge**, keine Auskunft des
Datenlieferanten. Wer es genau wissen will, braucht die Symbol-Change-Historie
(EODHD ``get_symbol_change_history``). Kurze Unterbrechungen unterhalb der
Schwelle bleiben unentdeckt, und ein Symbol, das dreimal vergeben wurde, wird
nur einmal getrennt.
"""

from __future__ import annotations

import pandas as pd

#: TRENNZEICHEN der Folgespalten (``SYM#2``, ``SYM#3``, …). Ein ``#`` kommt in
#: echten Tickern nicht vor — damit ist keine erzeugte Spalte je mit einem
#: echten Symbol verwechselbar. Früher stand hier ``"#2"``, und der Code schnitt
#: sich mit ``SUFFIX[0]`` das Zeichen heraus; wer die Konstante geändert hätte,
#: hätte still falsche Namen erzeugt (Stage-2-Finding F-senior-12).
SUFFIX = "#"

#: Ab welcher Lückenlänge (Handelstage) wird getrennt?
#:
#: HERLEITUNG — und warum nicht 120 (Stage-eigener Fund)
#: -----------------------------------------------------
#: Der Detektor in P12h findet *unterbrochene* Serien. Die haben mindestens
#: zwei Ursachen, und die erste Fassung dieser Trennung warf sie zusammen:
#:
#: * **Ticker-Recycling** — Firma A endet, Firma B übernimmt das Symbol.
#: * **Vendor-Datenlöcher** — die Firma existiert durchgehend, der Feed hat
#:   Lücken. CCE (Coca-Cola Enterprises) hat sechs davon, jährlich im November.
#:
#: Bei 120 Handelstagen hätte die Trennung CCE in sieben Stücke zerlegt — eine
#: „Reparatur", die mehr kaputt macht als der Fehler (E-107).
#:
#: Gemessen über alle 57 Lücken ≥ 60 Handelstage im Panel, Anteil mit Faktor
#: zwischen 0,5 und 2,0 (also „Kurs setzt ungefähr fort"):
#:
#: ===========  =====  ==================
#: Lücke (Tage)      n  Faktor nahe 1
#: ===========  =====  ==================
#: 60–120           2  100 %
#: 120–250         14   86 %
#: 250–500         11   36 %
#: 500–1000        12   42 %
#: ≥ 1000          18   17 %
#: ===========  =====  ==================
#:
#: Unter 250 Tagen dominieren Datenlöcher, ab 500 der Firmenwechsel. Die
#: Schwelle liegt bei 500 (zwei Jahre): so lange pausiert keine fortbestehende
#: Aktie, und alle drei in P12h gemessenen Schadensfälle (CGP, NGH, NVLS)
#: liegen darüber. Der Bereich 120–500 bleibt **unangetastet** und wird als
#: eigenes, ungelöstes Problem ausgewiesen — nicht stillschweigend mitbehandelt.
#:
#: Die belastbare Alternative wäre die Symbol-Change-Historie des Anbieters.
#: Sie beginnt bei EODHD erst 2022-07-22 und deckt das Suchfenster nicht ab
#: (geprüft, nicht vermutet).
MIN_LUECKE = 500


def segmente(
    close: pd.DataFrame, sym: str, min_luecke: int = MIN_LUECKE
) -> list[pd.Timestamp]:
    """Alle Schnittpunkte einer Spalte — nicht nur den groessten.

    Ein Symbol kann mehr als zweimal vergeben worden sein: WLL hat bei der
    Modulschwelle drei Segmente (Schnitte 2008-12-17 und 2014-12-16). Die
    erste Fassung schnitt nur an der laengsten Luecke und liess die uebrigen
    stehen — der Fail-Closed-Waechter fing das, statt es still durchzulassen.
    (Aufgefallen war es damals an RYC, das bei der niedrigeren Schwelle 120
    drei Segmente hatte; bei 500 sind es zwei.)
    """
    ser = close[sym].dropna()
    if len(ser) < 2:
        return []
    pos = pd.Series(close.index.get_indexer(ser.index))
    luecken = pos.diff()
    return [ser.index[i] for i in luecken.index[luecken >= min_luecke]]


def trenne(
    close: pd.DataFrame,
    membership: pd.Series,
    treffer: list[dict],
    div_panel: pd.DataFrame | None = None,
    min_luecke: int = MIN_LUECKE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, dict]:
    """Zerlegt jede betroffene Spalte in eine Spalte je Unternehmen.

    ``treffer`` ist die Ausgabe von ``p12h_ticker_recycling.unterbrechungen``
    und dient nur als Kandidatenliste — die Schnittpunkte werden hier neu und
    vollstaendig bestimmt.

    Rückgabe: (close, membership, div_panel, Protokoll).
    """
    aus = close.copy()
    div = None if div_panel is None else div_panel.copy()
    protokoll: dict[str, dict] = {}

    for x in treffer:
        sym = x["symbol"]
        if sym not in aus.columns:
            continue
        schnitte = segmente(close, sym, min_luecke)
        if not schnitte:
            continue
        grenzen = [*schnitte, None]
        neue: list[dict] = []
        for k, start in enumerate(grenzen[:-1], start=2):
            ende = grenzen[k - 1]
            maske = aus.index >= start
            if ende is not None:
                maske &= aus.index < ende
            if not maske.any():
                continue
            neu = f"{sym}{SUFFIX}{k}"
            aus[neu] = close[sym].where(maske)
            if div is not None and sym in div.columns:
                dm = (div.index >= start) & (
                    (div.index < ende) if ende is not None else True
                )
                div[neu] = div_panel[sym].where(dm)
            neue.append(
                {
                    "spalte": neu,
                    "ab": f"{start:%Y-%m-%d}",
                    "n_punkte": int(aus[neu].notna().sum()),
                }
            )
        # Das erste Segment behaelt das Originalsymbol.
        ab_erstem_schnitt = aus.index >= schnitte[0]
        aus.loc[ab_erstem_schnitt, sym] = float("nan")
        if div is not None and sym in div.columns:
            div.loc[div.index >= schnitte[0], sym] = float("nan")
        # Der KURSFAKTOR je Schnitt gehoert ins Protokoll, nicht nur in die
        # Herleitung der Schwelle (Stage-2-Finding F-senior-3). Die Schwelle
        # wurde aus zwei Achsen gewonnen — Lueckenlaenge UND Faktor —, entscheidet
        # aber nur nach einer. Wer nach einem Merkmal trennt, uebernimmt die
        # Fehlerrate des anderen; sie muss dann wenigstens sichtbar sein.
        ser_roh = close[sym].dropna()
        faktoren = []
        for s in schnitte:
            i = ser_roh.index.get_loc(s)
            vor, nach = float(ser_roh.iloc[i - 1]), float(ser_roh.iloc[i])
            faktoren.append(nach / vor if vor else None)
        protokoll[sym] = {
            "n_segmente": len(neue) + 1,
            "schnitte": [f"{s:%Y-%m-%d}" for s in schnitte],
            "faktoren": faktoren,
            # „Kurs setzt fort" nach derselben Definition wie in der Herleitung
            # der Schwelle. True heisst: dieser Schnitt ist ein wahrscheinlicher
            # FEHLTREFFER — die Trennung erzeugt dort ein fabriziertes Delisting.
            "faktor_nahe_eins": [
                bool(f is not None and 0.5 <= f <= 2.0) for f in faktoren
            ],
            "n_punkte_erstes": int(aus[sym].notna().sum()),
            "weitere": neue,
        }

    # Die Mitgliedschaft muss mitwandern, sonst waehlt die Engine `SYM` zu
    # Terminen, an denen dort nur noch NaN steht — der Name faellt dann still
    # aus dem Universum statt als naechste Firma weiterzulaufen.
    umbenannt = {}
    for t, namen in membership.items():
        neu_namen = set(namen)
        tag = f"{t:%Y-%m-%d}"
        for sym, info in protokoll.items():
            if sym not in neu_namen:
                continue
            passend = [w for w in info["weitere"] if tag >= w["ab"]]
            if passend:
                neu_namen.discard(sym)
                neu_namen.add(passend[-1]["spalte"])
        umbenannt[t] = frozenset(neu_namen)
    m_neu = pd.Series(umbenannt)

    # FAIL-CLOSED: nach der Trennung darf keine erzeugte Spalte mehr eine Luecke
    # der Groessenordnung tragen, die den Fall ueberhaupt definiert hat.
    for sym, info in protokoll.items():
        for spalte in [sym, *(w["spalte"] for w in info["weitere"])]:
            ser = aus[spalte].dropna()
            if len(ser) < 2:
                continue
            pos = pd.Series(aus.index.get_indexer(ser.index))
            if pos.diff().max() >= min_luecke:
                raise SystemExit(
                    f"Trennung hat {spalte} nicht sauber zerlegt — es bleibt "
                    f"eine Luecke von >= {min_luecke} Handelstagen."
                )
    return aus, m_neu, div, protokoll
