"""Bereinigt die Skalenbrüche aus dem Kampagnen-Panel — minimal und offen.

WARUM NICHT DIE NAMEN AUSSCHLIESSEN
-----------------------------------
Naheliegend wäre, die 25 korrumpierten Namen zu entfernen. Das wäre aber eine
neue Auswahl mit Wissen aus der Zukunft — dieselbe Fehlerklasse, die dieser
ganze Strang untersucht (E-079, P12d). Wer Namen entfernt, weil ihre Daten
kaputt sind, entfernt überproportional Übernahme- und Delisting-Fälle: von den
25 Namen sind viele genau deshalb auffällig, weil sie eine Kapitalmaßnahme
hatten.

WAS STATTDESSEN PASSIERT
------------------------
Die Skalenbrüche werden **gespleißt**. Innerhalb einer Korruptionsspanne
``[a, b)`` liegen die Kurse um einen konstanten Faktor
``f = px[a] / px[a-1]`` daneben; geteilt durch ``f`` liegt die Spanne wieder
auf der Basisskala.

Wirkung, präzise:

* Die Rendite am Übergangstag ``a`` wird exakt **0** — wir ersetzen eine
  unbekannte Rendite durch „keine Bewegung". Das ist eine Setzung, keine
  Messung, und sie steht hier, damit sie nachlesbar ist.
* Am Rücksprungtag ``b`` wird sie **nicht** exakt 0, sondern
  ``(1 + r_b) · f − 1``. Der Detektor lässt eine Spanne schließen, wenn dieser
  Wert innerhalb von ``PAAR_TOLERANZ`` (15 %) um null liegt — die Bereinigung
  setzt dort also eine Rendite bis zu dieser Größe ein. Im echten Lauf reicht
  das von −4,50 % (ABC) bis **+14,87 % (WFT)**. Eine frühere Fassung dieses
  Docstrings behauptete „exakt 0" für beide Ränder; das war falsch
  (Stage-1-Finding F3).
* **Alle übrigen Renditen innerhalb der Spanne bleiben bis auf
  Maschinengenauigkeit erhalten** — sie waren nie falsch, weil sich ein
  konstanter Faktor im Quotienten herauskürzt. Nicht bitgleich: (x/f)/(y/f)
  ist in IEEE-754 nicht identisch zu x/y; der zugehörige Test prüft
  konsequenterweise mit rel=1e-15 (Stage-2-Finding F-senior-9).

Je Spanne werden also genau zwei Renditen angefasst, nicht Hunderte. Minimal
ist das in den **Renditen**, nicht in den Kursniveaus: gespleißt wird immer
der Teil *nach* dem Sprung, und bei einem Bruch am Anfang der Historie sind
das fast alle Tage (TWX 99 %, RHT 98 %). Für alles, was diese Kampagne rechnet
— Renditen und Dividenden je Kurseinheit —, ist das folgenlos, weil beide
skaleninvariant sind. Wer absolute Kursniveaus braucht, darf das bereinigte
Panel nicht dafür verwenden (Stage-1-Finding F5).

WAS DAMIT NICHT REPARIERT WIRD
------------------------------
Nur die Morphologie, die der Detektor sieht: Sprünge über +200 % bei
Vortagskursen über 1 USD. Dauerhafte Niveausprünge im Band 100–200 %
(AYE +170 %, TOY +155 %, HIG +102 %) bleiben unberührt — ob sie echt sind, ist
offen (P12d). Die Bereinigung ist eine Untergrenze, keine Garantie.
"""

from __future__ import annotations

import pandas as pd


def bereinige(
    close: pd.DataFrame,
    spannen_je_symbol: dict[str, dict],
    div_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Spleißt die Korruptionsspannen auf die Basisskala zurück.

    ``div_panel`` MUSS mitgegeben werden, wenn eines existiert: es ist ein aus
    ``close`` ABGELEITETES Feld in Panel-Einheiten (``d_nominal * adj/raw``).
    Wird nur ``close`` durch ``f`` geteilt, steigt die implizite
    Dividendenrendite in der Spanne um genau ``f`` — bei WIN gemessen von 26 %
    auf 274 % (Stage-1-Finding F-test-3). Der Vergleich zweier Panels waere dann
    unfair auf genau der Achse, um die es geht: Dividenden werden in der GmbH
    mit 29,83 % besteuert, Kursgewinne mit 1,49 %.

    Rueckgabe: (bereinigtes close, bereinigtes div_panel, Protokoll).
    """
    aus = close.copy()
    div = None if div_panel is None else div_panel.copy()
    protokoll: dict[str, list[dict]] = {}
    for sym, info in spannen_je_symbol.items():
        if info.get("unaufloesbar"):
            continue  # verschraenkte Skalen: melden, nicht raten
        if sym not in aus.columns:
            continue
        reihe = aus[sym]
        eintraege = []
        for a_s, b_s in info["spannen"]:
            a = pd.Timestamp(a_s, tz="UTC")
            b = pd.Timestamp(b_s, tz="UTC")
            pos = reihe.index.searchsorted(a)
            if pos == 0:
                continue  # kein Vortag -> kein Faktor bestimmbar
            vor = reihe.iloc[pos - 1]
            ab = reihe.iloc[pos]
            if not (pd.notna(vor) and pd.notna(ab)) or vor <= 0:
                continue
            f = float(ab / vor)
            if f <= 0:
                continue
            maske = (reihe.index >= a) & (reihe.index < b)
            reihe = reihe.where(~maske, reihe / f)
            if div is not None and sym in div.columns:
                dm = (div.index >= a) & (div.index < b)
                div.loc[dm, sym] = div.loc[dm, sym] / f
            eintraege.append(
                {"von": a_s, "bis": b_s, "faktor": f, "n_tage": int(maske.sum())}
            )
        if eintraege:
            aus[sym] = reihe
            protokoll[sym] = eintraege

    zahlen = gegenprobe(close, aus)
    if zahlen["neu_entstanden"]:
        raise SystemExit(
            f"Bereinigung hat {zahlen['neu_entstanden']} NEUE Ausreisser "
            f"(>+100 % oder <-50 %) erzeugt: {', '.join(zahlen['wo_neu'][:10])}"
            " — das ist schlimmer als der Fehler, gegen den bereinigt wird."
        )
    return aus, div, protokoll


def auffaellig(df: pd.DataFrame, referenz: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tage, an denen ein Kurs unmoeglich springt — in BEIDE Richtungen.

    Die Abwaertsschwelle ist nicht willkuerlich: eine Rueckkehr aus einem
    Sprung um Faktor 2 ist ein Fall um 50 %. Wer nur nach oben schaut, sieht
    die halbe Fehlerklasse nicht — und ein kuenstlicher -80-%-Tag faellt in
    einem Momentum-Backtest niemandem auf, er verschiebt nur still die
    Rangliste (F-test-4, E-107).

    WARUM ``referenz`` (Stage-1-Finding F1, BLOCKER)
    -----------------------------------------------
    Die 1-USD-Schwelle traegt eine fachliche Aussage: unter einem Dollar sind
    Verzehnfachungen real, dort ist ein Sprung kein Vendor-Fehler. Diese Frage
    beantwortet der Kurs, wie er WIRKLICH war — nicht der, den die Bereinigung
    daraus gemacht hat. Genau das war der Fehler: ``bereinige`` teilt durch
    ``f`` und schiebt Kurse damit systematisch unter die eigene Schwelle. Im
    echten Lauf fielen 379 Symbol-Tage in fuenf Namen so aus dem Blickfeld
    (RHT lag nie unter 3,02 USD und rutschte an 86 Tagen darunter). Der
    Waechter, der neue Fehler finden soll, wurde also von der Reparatur selbst
    blind gemacht — ein Fail-Open im Fail-Closed.

    ``referenz`` ist deshalb immer das ORIGINALPANEL.
    """
    basis = df if referenz is None else referenz
    r = df.pct_change(fill_method=None)
    return ((r > 1.0) | (r < -0.5)) & (basis.shift(1) > 1.0)


def gegenprobe(alt: pd.DataFrame, neu: pd.DataFrame) -> dict:
    """Was hat die Bereinigung beseitigt — und was hat sie ANGERICHTET?

    Die erste Fassung zaehlte nur, was verschwindet. Eine Reparatur, die ihre
    eigenen Nebenwirkungen nicht misst, ist eine zweite, unbeobachtete
    Datenquelle — und zwar genau dort, wo die Fragestellung entschieden wird
    (E-107). ``beseitigt`` und ``bleibt`` gehoeren zusammen in den Befund:
    eine Bereinigung, die 6 % der Auffaelligkeiten raeumt, ist eine
    Untergrenze und darf nicht als „Panel ist jetzt sauber" gelesen werden.
    """
    a, n = auffaellig(alt), auffaellig(neu, referenz=alt)
    entstanden = n & ~a
    wo = [
        f"{sym} {tag:%Y-%m-%d}"
        for sym in entstanden.columns[entstanden.any()]
        for tag in entstanden.index[entstanden[sym]]
    ]
    return {
        "auffaellig_original": int(a.to_numpy().sum()),
        "auffaellig_bereinigt": int(n.to_numpy().sum()),
        "beseitigt": int((a & ~n).to_numpy().sum()),
        "neu_entstanden": len(wo),
        "wo_neu": wo,
    }
