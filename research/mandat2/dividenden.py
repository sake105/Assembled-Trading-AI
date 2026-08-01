"""Dividenden auf die Skala des Preispanels bringen (Mandat II, P0-Nachtrag).

Das Problem
-----------
``prices_verdict.parquet`` ist TOTAL-RETURN-adjustiert und rueckwaerts
normalisiert (SPY 1995: 26,43 im Panel gegen real ~46 — Faktor ~1,74).
``dividends.parquet`` fuehrt dagegen NOMINALE Betraege je Stueck.

Wer den nominalen Betrag direkt auf das adjustierte Panel bucht, ueberzeichnet
die Dividende um genau ``raw/adj``: gemessen SPY 1995 implizit 4,01 % statt
real 2,70 %, 2008 3,11 % statt 2,27 %. Unter EINEM Steuersatz war das
common-mode. In Mandat II ist der Dividendensatz ein Parameter, der sich vom
Kursgewinnsatz stark unterscheidet (GmbH 29,83 % gegen 1,49 %) — die
Ueberzeichnung trifft damit genau die Asymmetrie, die gemessen werden soll.
Dieselbe Falle wie E-068, eine Groesse weiter.

Die Rekonstruktion
------------------
Fuer eine total-return-adjustierte Reihe gilt zwischen zwei Handelstagen

    adj(t) / adj(t-1) = (raw(t) + d_t) / raw(t-1)

Nach ``raw(t-1)`` aufgeloest und rueckwaerts iteriert:

    raw(t-1) = (raw(t) + d_t) * adj(t-1) / adj(t)

Der Startpunkt ist bekannt, weil rueckwaerts normalisiert wird: am LETZTEN
Kurs der Reihe sind adjustierter und roher Kurs identisch. Damit laesst sich
der Rohpfad ohne zweite Datenquelle rekonstruieren und daraus die Dividende in
Panel-Einheiten bilden:

    d_panel(t) = d_nominal(t) * adj(t) / raw(t)

Grenzen, ausdruecklich
----------------------
* Splits sind in beiden Quellen bereits verarbeitet; eine Split-Anpassung
  findet hier NICHT statt.
* Bei delisteten Namen ist „letzter Kurs = roh" eine Annahme, keine Tatsache.
  Der Fehler bleibt lokal (er wirkt nur auf die Dividenden dieses Namens) und
  ist bei toten Namen klein, weil dort ohnehin kaum noch ausgeschuettet wird.
* Fehlt eine Dividende in der Quelle, faellt der Faktor fuer diesen Namen zu
  klein aus — die Steuer waere dann zu niedrig, nicht zu hoch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rohpfad(adj: pd.Series, divs: pd.Series) -> pd.Series:
    """Rohkurse aus adjustiertem Pfad + nominalen Dividenden rekonstruieren.

    Args:
        adj: adjustierte Kurse, aufsteigend, ohne NaN.
        divs: nominale Dividende je Stueck, indiziert auf Handelstage aus
            ``adj.index`` (0/fehlend = keine Ausschuettung).

    Returns:
        Rohkurs-Reihe auf demselben Index. Am letzten Punkt gleich ``adj``.
    """
    if adj.empty:
        return adj
    a = adj.astype(float).to_numpy()
    d = divs.reindex(adj.index).fillna(0.0).astype(float).to_numpy()
    raw = np.empty_like(a)
    raw[-1] = a[-1]
    for i in range(len(a) - 1, 0, -1):
        if a[i] <= 0 or not np.isfinite(a[i]):
            raw[i - 1] = a[i - 1]
            continue
        raw[i - 1] = (raw[i] + d[i]) * a[i - 1] / a[i]
    return pd.Series(raw, index=adj.index)


def auf_panel_skalieren(close: pd.DataFrame, div_panel: pd.DataFrame) -> pd.DataFrame:
    """Nominales Dividendenpanel -> Panel-Einheiten.

    Nur Symbole mit mindestens einer Dividende werden angefasst; alle anderen
    Spalten bleiben unveraendert (und damit 0/NaN).
    """
    out = div_panel.copy()
    for sym in div_panel.columns:
        if sym not in close.columns:
            continue
        d = div_panel[sym]
        if not (d.fillna(0.0) > 0).any():
            continue
        a = close[sym].dropna()
        if a.empty:
            continue
        raw = rohpfad(a, d)
        faktor = (a / raw).reindex(div_panel.index)
        out[sym] = d * faktor
    return out


def implizite_jahresrendite(
    close: pd.DataFrame, div_panel: pd.DataFrame, symbol: str
) -> pd.Series:
    """Diagnose: Jahressumme der Dividende / mittlerer Kurs.

    Der Sanity-Check fuer die Skalierung. Bei SPY muss das Ergebnis in der
    bekannten Bandbreite der SPY-Dividendenrendite liegen (grob 1,3-3,5 %);
    liegt es deutlich darueber, ist die Skala falsch.
    """
    a = close[symbol].dropna()
    d = div_panel.get(symbol)
    if d is None:
        return pd.Series(dtype=float)
    d = d.reindex(a.index).fillna(0.0)
    jahr = a.index.year
    return d.groupby(jahr).sum() / a.groupby(jahr).mean()
