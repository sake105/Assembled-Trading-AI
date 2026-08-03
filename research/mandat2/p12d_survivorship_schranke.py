"""P12d — Wie stark ist das Intraday-Universum survivorship-verzerrt?

DER ANLASS
----------
P12 lief auf Namen, die 2004-2016 durchgehend im Index waren — eine Auswahl
mit Wissen aus der Zukunft. Der Versuch, das durch ein Point-in-Time-Universum
zu heilen, scheiterte an der Datenquelle: der EODHD-**Intraday**-Endpunkt fuehrt
keine delisteten Ticker. Gemessen an 22 ausgeschiedenen Namen betrug die
Trefferquote **18 %** gegen 92 % bei Ueberlebenden; jeder Name, der wirklich vom
Markt verschwand (MER 2008, NXTL 2005, EOP 2007, RBK 2006), liefert null Bars.

Das **Tages**panel enthaelt die Toten dagegen vollstaendig — dort ist die Groesse
der Verzerrung messbar. Genau das passiert hier: nicht die Verzerrung beheben
(das geht mit dieser Quelle nicht), sondern sie **beziffern**, damit der
P12-Befund seine eigene Unsicherheit kennt statt sie nur zu erwaehnen.

WAS HIER NICHT PASSIERT
-----------------------
Keine Strategie, kein Signal, keine Auswahl. Drei Kauf-und-Halten-Kurven ueber
drei Universen. Der Trial-Zaehler steigt trotzdem, weil Backtests laufen.

WICHTIG — die Tages-Engine der Kampagne ist bereits PIT-korrekt
---------------------------------------------------------------
``engine.run_strategy`` waehlt je Termin aus ``membership.get(t)``, also der
Index-Zusammensetzung ZU DIESEM Zeitpunkt, und haelt delistete Namen ueber
``last_valid`` bis zum letzten Kurs. Das PIT-Universum 2004 enthaelt
nachweislich Pleite-Ticker (EKDKQ = Eastman Kodak, MTLQQ = Motors Liquidation,
WNDXQ). **P1-P11 sind damit survivorship-frei** — betroffen ist ausschliesslich
der Intraday-Strang P12, dessen Universum durch die Datenverfuegbarkeit
vorselektiert ist.

ZWEI BEHANDLUNGEN DES DELISTINGS
--------------------------------
Was mit dem Erloes eines delisteten Namens geschieht, ist eine Annahme, keine
Messung — und sie wirkt in die Ergebnisrichtung:

* **halten** (konservativ fuer den Benchmark): der Erloes bleibt liegen und
  verzinst sich nicht. Senkt den Benchmark, macht ihn also LEICHTER schlagbar.
* **umschichten**: der Erloes wird gleichmaessig auf die ueberlebenden Namen
  verteilt. Hebt den Benchmark.

Beide werden gerechnet und als Spanne ausgewiesen. Eine einzelne Zahl waere
hier Scheinpraezision (E-078).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
INTRADAY = Path(__file__).resolve().parents[2] / "data" / "raw" / "intraday_1h"
# Exakt das P12-Fenster, damit die Zahlen vergleichbar sind.
VON, BIS = "2006-06-22", "2016-12-30"


GLITCH_SCHWELLE = 2.0  # +200 % an EINEM Tag bei Vortagskurs > 1 USD


def glitch_verdaechtig(px: pd.DataFrame, namen: set[str]) -> dict[str, dict]:
    """Namen mit unmoeglichen Tagesspruengen — Vendor-Fehler, keine Renditen.

    ``campaign_data`` truncatet bereits Mikropreis-Artefakte (Vortag < 1 USD,
    Sprung > 100 %). Diese Regel greift eine Klasse darueber nicht: MEL springt
    2014-11-12 von 7,73 auf 141.630, CIN 2010-12-29 von 3,13 auf 1.774, HPC
    2014-12-04 von 56,61 auf 5.300. Beim Rendite-Produkt eskalieren solche
    Spitzen; im ersten Lauf lief das PIT-Universum dadurch auf 10^74.

    Wird hier NICHT stillschweigend korrigiert, sondern gemeldet: die Namen
    kommen ins Ergebnis-Artefakt, damit die Bereinigung nachvollziehbar ist.
    """
    sp = [s for s in sorted(namen) if s in px.columns and px[s].notna().any()]
    t = px[sp]
    r = t.pct_change(fill_method=None).where(t.notna() & t.shift(1).notna())
    gross = (r > GLITCH_SCHWELLE) & (t.shift(1) > 1.0)
    aus = {}
    for sym in sp:
        if bool(gross[sym].any()):
            i = r[sym].idxmax()
            aus[sym] = {
                "zeitpunkt": f"{i:%Y-%m-%d}",
                "von": float(t[sym].shift(1).loc[i]),
                "auf": float(t[sym].loc[i]),
                "sprung": float(r[sym].loc[i]),
            }
    return aus


def buy_and_hold(
    px: pd.DataFrame, namen: set[str], *, umschichten: bool
) -> tuple[pd.Series, dict]:
    """Gleichgewichtet KAUFEN und HALTEN; Delistings am letzten Kurs verwerten.

    Beide Varianten sind Buy-and-Hold. Sie unterscheiden sich ausschliesslich
    darin, was mit dem Erloes eines delisteten Namens geschieht:

    * ``umschichten=False`` — der Erloes bleibt als totes Geld liegen.
    * ``umschichten=True``  — der Erloes wird am Delisting-Tag pro rata auf die
      noch lebenden Positionen verteilt.

    WAS HIER FALSCH WAR (Stage-1-Finding F-test-1, BLOCKER)
    -------------------------------------------------------
    Die erste Fassung rechnete ``(1 + r.mean(axis=1)).cumprod()``. Das ist ein
    TAEGLICH gleichgewichtet rebalanciertes Portfolio, kein Buy-and-Hold: es
    weicht auch dann ab, wenn ueberhaupt nichts delistet (synthetisch gemessen
    -9,1 % bei zwei gegenlaeufigen Titeln), und sein Rebalancing-Bonus waechst
    mit der Zahl der Namen — +2,5 % bei n=20, +36,6 % bei n=418.

    Genau diese Asymmetrie erzeugte die frueher berichtete Ueberhoehung von nur
    +0,14 % p. a. Sie war ein Artefakt des Rebalancings, kein Ergebnis, und
    trug eine Entwarnung, die sie nicht tragen konnte.

    Hier wird wertbasiert simuliert: Positionen bleiben liegen, Kapital wandert
    nur am Delisting-Tag.

    Rueckgabe
    ---------
    ``(kurve, diagnose)``. Die Diagnose zaehlt, WER nicht mitspielt. Stille
    Filterung waere hier besonders teuer, weil die Aussteiger der eigentliche
    Messgegenstand sind (F-test-3).
    """
    kandidaten = sorted(namen & set(px.columns))
    ohne_spalte = sorted(namen - set(px.columns))
    dabei = [s for s in kandidaten if pd.notna(px[s].iloc[0])]
    kein_startkurs = [s for s in kandidaten if s not in dabei]
    if not dabei:
        raise ValueError("leeres Universum")

    teil = px[dabei]
    kurse = teil.to_numpy(dtype=float)
    n = len(dabei)

    # Wann endet welche Position? Nur Titel, die VOR Fensterende auslaufen,
    # sind Delistings; die uebrigen leben bis zum Schluss.
    tod_an: dict[int, list[int]] = {}
    for k, sym in enumerate(dabei):
        letzt = teil[sym].last_valid_index()
        if letzt is not None and letzt < teil.index[-1]:
            tod_an.setdefault(teil.index.get_loc(letzt), []).append(k)

    pos = np.full(n, 1.0 / n)  # Positionswerte
    lebt = np.ones(n, dtype=bool)
    totes_geld = 0.0
    kurve = np.empty(len(teil))
    n_delistings = 0

    for t in range(len(teil)):
        if t > 0:
            p0, p1 = kurse[t - 1], kurse[t]
            gut = lebt & ~np.isnan(p0) & ~np.isnan(p1)
            pos[gut] *= p1[gut] / p0[gut]
        kurve[t] = pos[lebt].sum() + totes_geld

        gestorben = tod_an.get(t)
        if not gestorben:
            continue
        erloes = float(pos[gestorben].sum())
        pos[gestorben] = 0.0
        lebt[list(gestorben)] = False
        n_delistings += len(gestorben)
        if erloes <= 0:
            continue
        summe = float(pos[lebt].sum())
        if umschichten and lebt.any() and summe > 0:
            pos[lebt] += erloes * (pos[lebt] / summe)
        else:
            totes_geld += erloes

    diagnose = {
        "n_dabei": n,
        "n_ohne_preisspalte": len(ohne_spalte),
        "n_ohne_startkurs": len(kein_startkurs),
        "ohne_startkurs": kein_startkurs[:20],
        "n_delistings_im_fenster": n_delistings,
    }
    return pd.Series(kurve, index=teil.index), diagnose


def kennzahlen(k: pd.Series, jahre: float) -> dict:
    k = k.dropna()
    lauf = k.cummax()
    return {
        "endwert": float(k.iloc[-1]),
        "cagr": float(k.iloc[-1] ** (1.0 / jahre) - 1.0),
        "maxdd": float((k / lauf - 1.0).min()),
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    regen = "--regen" in sys.argv
    d = load_campaign()
    m = d.membership
    px = d.close.loc[VON:BIS]
    jahre = (px.index[-1] - px.index[0]).days / 365.25

    if regen:
        print("[SKIP] Trial-Zaehler: --regen (Wiederholungslauf)", flush=True)
    else:
        print(
            f"Trials kumuliert: "
            f"{TrialCounter().increment(6, label='P12d Survivorship-Schranke')}\n",
            flush=True,
        )

    start = m.index[m.index >= "2004-01-01"][0]
    maske = (m.index >= "2004-01-01") & (m.index <= "2016-12-31")
    durchgehend = set.intersection(*[set(s) for s in m.loc[maske]])
    pit = set(m.loc[start])
    # NICHT vom Plattenstand ablesen: der laufende Pull legt waehrend der
    # Messung Dateien nach, das Universum waere ein bewegliches Ziel und die
    # Zahl morgen eine andere. Fixiert auf das Universum, das P12 wirklich
    # gerechnet hat — es steht in dessen Ergebnis-Artefakt.
    p12 = OUT / "p12_intraday_haltedauer.json"
    if not p12.exists():
        raise SystemExit("results/p12_intraday_haltedauer.json fehlt")
    intraday = set(json.loads(p12.read_text(encoding="utf-8"))["universum"])

    # Beleg, dass das PIT-Universum wirklich Tote enthaelt — sonst misst dieser
    # Test nichts und wuerde es nicht merken.
    tote_marker = {"EKDKQ", "MTLQQ", "WNDXQ", "ENRNQ", "AAMRQ", "NRTLQ"}
    tote_im_pit = sorted(tote_marker & pit)

    universen = {
        "intraday_p12": intraday,
        "durchgehend_2004_2016": durchgehend,
        "pit_2004": pit,
    }
    # Datenqualitaet ZUERST — sonst misst der Rest Vendor-Fehler.
    glitches = glitch_verdaechtig(px, pit)
    sauber = {k: (v - set(glitches)) for k, v in universen.items()}
    print(f"Glitch-verdaechtige Namen im PIT-Universum: {len(glitches)}")
    for sym, g in sorted(glitches.items(), key=lambda kv: -kv[1]["sprung"])[:5]:
        print(
            f"  {sym:<7}{g['sprung']:>14,.0%}  {g['zeitpunkt']}  "
            f"{g['von']:.2f} -> {g['auf']:.2f}"
        )
    print()

    zeilen = []
    print(f"Fenster {VON}..{BIS} ({jahre:.2f} Jahre)")
    print(
        f"{'Universum':<26}{'n':>5}{'halten':>12}{'umschichten':>14}{'CAGR halten':>14}"
    )
    for name, namen in sauber.items():
        k_h, diag = buy_and_hold(px, namen, umschichten=False)
        k_u, _ = buy_and_hold(px, namen, umschichten=True)
        a_h, a_u = kennzahlen(k_h, jahre), kennzahlen(k_u, jahre)
        for wie, a in (("halten", a_h), ("umschichten", a_u)):
            if not (0.05 < a["endwert"] < 100.0):
                raise SystemExit(
                    f"{name}/{wie}: Endwert {a['endwert']:.3g} ausserhalb jeder "
                    "Plausibilitaet — Rechenfehler, kein Ergebnis."
                )
        n = diag["n_dabei"]
        zeilen.append(
            {
                "universum": name,
                "n": n,
                "halten": a_h,
                "umschichten": a_u,
                "diagnose": diag,
            }
        )
        print(
            f"{name:<26}{n:>5}{a_h['endwert']:>11.3f}x{a_u['endwert']:>13.3f}x"
            f"{a_h['cagr']:>13.2%}"
        )

    spy = kennzahlen(buy_and_hold(px, {"SPY"}, umschichten=False)[0], jahre)
    print(f"{'SPY (Referenz)':<26}{1:>5}{spy['endwert']:>11.3f}x")

    def hol(u: str) -> dict:
        return next(z for z in zeilen if z["universum"] == u)

    intra_, pit_ = hol("intraday_p12"), hol("pit_2004")
    schranke = {
        "cagr_halten": intra_["halten"]["cagr"] - pit_["halten"]["cagr"],
        "cagr_umschichten": intra_["umschichten"]["cagr"] - pit_["umschichten"]["cagr"],
    }
    ergebnis = {
        "fenster": f"{VON}..{BIS}",
        "jahre": jahre,
        "tote_ticker_im_pit_universum": tote_im_pit,
        "glitch_schwelle": GLITCH_SCHWELLE,
        "ausgeschlossene_glitches": glitches,
        "zeilen": zeilen,
        "spy": spy,
        "ueberhoehung_cagr": schranke,
    }
    (OUT / "p12d_survivorship.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12d_survivorship.json'}")

    print("\n" + "=" * 76)
    print(f"Pleite-Ticker im PIT-Universum (Beleg): {tote_im_pit}")
    print(
        "UEBERHOEHUNG des P12-Benchmarks gegenueber dem survivorship-freien Universum:"
    )
    print(
        f"  {schranke['cagr_halten']:+.2%} p. a. (Delisting-Erloes gehalten)  |  "
        f"{schranke['cagr_umschichten']:+.2%} p. a. (umgeschichtet)"
    )
    print(
        "Die Tages-Engine (P1-P11) waehlt je Termin aus membership(t) und ist\n"
        "damit PIT-korrekt — betroffen ist ausschliesslich der Intraday-Strang."
    )
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
