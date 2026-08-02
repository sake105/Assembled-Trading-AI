"""P9 — Forensik: warum traegt `Preis > SMA` und die Alternativen nicht?

DIES IST KEINE SUCHE
--------------------
Es wird kein Kandidat ausgewaehlt, keine Variante bevorzugt, kein Parameter
optimiert. Deshalb wird der Trial-Zaehler NICHT erhoeht: was hier passiert, ist
das Zerlegen eines bereits gemessenen Ergebnisses, nicht das Erzeugen eines
neuen. (Wuerde daraus ein neuer Kandidat entstehen, muesste er wie jeder andere
gezaehlt werden.)

DIE OFFENE FRAGE AUS P5/P8
--------------------------
`Preis > SMA` besteht ueber ein zusammenhaengendes Fensterband (11-12 von 12).
`SMA steigt` und `Rendite > 0` bestehen nur lueckig (3-7 von 12). Die drei
Definitionen sind normalerweise stark korreliert — dass sie hier so
auseinanderlaufen, ist erklaerungsbeduerftig.

Und: die Zahl gerissener Fenster ist ueber alle 72 P5-Laeufe fast binaer
(0, 64 oder 69, nichts dazwischen). Das deutet auf EIN Ereignis, das entweder
gefangen wird oder nicht.

WAS HIER GEMESSEN WIRD
----------------------
1. Welches Ereignis erzeugt die 69 gerissenen Fenster? (Datumsbereich der
   betroffenen Fenster)
2. Wann genau steigt jede Definition in den beiden Baerenmaerkten aus und
   wieder ein? Wie viele Tage frueher oder spaeter?
3. Wie viel Zeit verbringt jede Definition ausserhalb des Marktes — und wie oft
   schaltet sie? (Whipsaw-Neigung, entscheidend fuer den COVID-Holdout)
4. Trifft eine Definition den Wiedereinstieg systematisch schlechter?
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_strategy  # noqa: E402
from research.mandat2.metrics import DD_DECKEL, auswerten, max_drawdown  # noqa: E402
from research.mandat2.p5_gate_robustheit import DEFINITIONEN  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
KRISEN = {
    "Dotcom 2000-2002": ("2000-01-01", "2003-06-30"),
    "Finanzkrise 2007-2009": ("2007-06-01", "2009-12-31"),
}


def schaltpunkte(gate: pd.Series, von: str, bis: str) -> list[tuple[str, str]]:
    """(Datum, 'raus'|'rein') im Zeitraum."""
    g = gate.loc[von:bis].dropna()
    wechsel = g.diff().fillna(0)
    return [
        (str(t.date()), "rein" if wechsel.loc[t] > 0 else "raus")
        for t in g.index[wechsel != 0]
    ]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    spy = d.close["SPY"]

    # ---------------------------------------------------------- 1. Welches Ereignis?
    bench = run_buy_and_hold(d, make_regime("ZERO"))
    ohne = run_strategy(d, make_regime("ZERO"), **PARAMS)
    a = auswerten(ohne.equity_netto, bench.equity_netto, label="ohne Gate")
    gerissen = a.gerissene_fenster
    print(
        f"\n=== 1. Welche Fenster reissen ohne Gate? ({len(gerissen)}/{a.n_fenster}) ==="
    )
    gemeinsam = None
    if gerissen:
        print(
            f"  erstes Fenster: {gerissen[0].start.date()} .. {gerissen[0].ende.date()}"
        )
        print(
            f"  letztes Fenster: {gerissen[-1].start.date()} .. {gerissen[-1].ende.date()}"
        )
        # Gemeinsamer Teilzeitraum ALLER gerissenen Fenster — existiert nur,
        # wenn sie sich ueberlappen. Bei 144/144 tun sie das NICHT (das
        # frueheste Fenster endet vor dem Beginn des spaetesten), und dann
        # ist max(start) > min(ende). Diesen Fall abfangen statt eine
        # sinnlose negative Spanne auszugeben.
        spaetester_start = max(f.start for f in gerissen)
        fruehestes_ende = min(f.ende for f in gerissen)
        if spaetester_start < fruehestes_ende:
            gemeinsam = (spaetester_start, fruehestes_ende)
            print(
                f"  ALLE gerissenen Fenster enthalten: "
                f"{spaetester_start.date()} .. {fruehestes_ende.date()}"
            )
            print(
                f"  MaxDD des Kandidaten dort: "
                f"{max_drawdown(ohne.equity_netto.loc[spaetester_start:fruehestes_ende]):.1%}"
            )
        else:
            print(
                "  KEIN gemeinsamer Teilzeitraum — die gerissenen Fenster "
                "ueberlappen sich nicht alle. Das ist der Fall, wenn ausnahmslos "
                "jedes Fenster reisst: es gibt dann nicht EIN Ereignis, sondern "
                "mindestens zwei getrennte."
            )

    # ---------------------------------------------------------- 2./3. Gate-Vergleich
    print("\n=== 2. Verhalten der drei Definitionen (Fenster 200) ===")
    zeilen = []
    for name, fn in DEFINITIONEN.items():
        gate = fn(d.close, 200)
        g = gate.dropna()
        anteil_drin = float(g.mean())
        wechsel = int((g.diff().fillna(0) != 0).sum())
        r = run_strategy(d, make_regime("ZERO"), risk_off_gate=gate, **PARAMS)
        av = auswerten(r.equity_netto, bench.equity_netto, label=name)
        zeilen.append(
            {
                "definition": name,
                "anteil_investiert": anteil_drin,
                "n_schaltungen": wechsel,
                "maxdd": av.schlimmster_maxdd,
                "median": av.median_kandidat,
                "gerissen": len(av.gerissene_fenster),
                "krisen": {},
            }
        )
        print(
            f"\n  {name}: {anteil_drin:.1%} der Zeit investiert, {wechsel} Schaltungen"
            f" | MaxDD {av.schlimmster_maxdd:.1%} | gerissen {len(av.gerissene_fenster)}"
        )
        for krise, (von, bis) in KRISEN.items():
            sp = schaltpunkte(gate, von, bis)
            zeilen[-1]["krisen"][krise] = sp
            raus = [t for t, art in sp if art == "raus"]
            rein = [t for t, art in sp if art == "rein"]
            # Wie tief war SPY schon, als das Gate zum ersten Mal rausging?
            tief = ""
            if raus:
                bis_raus = spy.loc[von : raus[0]]
                if len(bis_raus) > 1:
                    tief = f", SPY zum 1. Ausstieg schon {max_drawdown(bis_raus):.1%}"
            print(
                f"    {krise}: {len(sp)} Schaltungen, 1. raus {raus[0] if raus else '-'}"
                f", 1. rein {rein[0] if rein else '-'}{tief}"
            )

    (OUT / "p9_gate_forensik.json").write_text(
        json.dumps(
            {
                "dd_deckel": DD_DECKEL,
                "gerissene_fenster_ohne_gate": len(gerissen),
                "gemeinsamer_zeitraum": [
                    str(gemeinsam[0].date()),
                    str(gemeinsam[1].date()),
                ]
                if gemeinsam
                else None,
                "definitionen": zeilen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT / 'p9_gate_forensik.json'}")

    # ------------------------------------------- 3. Die wirksame Stichprobe
    idx = pd.DatetimeIndex(d.close.index)
    monatsenden = sorted(set(idx.to_series().groupby(idx.to_period("M")).max()))
    print()
    print(f"=== 3. Das Gate wird NUR an {len(monatsenden)} Monatsenden gelesen ===")
    mat = {}
    for name, fn in DEFINITIONEN.items():
        g = fn(d.close, 200)
        me = g.reindex(monatsenden).dropna()
        mat[name] = me
        taeglich = int((g.dropna().diff().fillna(0) != 0).sum())
        wirksam = int((me.diff().fillna(0) != 0).sum())
        print(
            f"  {name:<12} {taeglich:>4} taegliche Flips -> nur {wirksam:>3} "
            f"WIRKSAME Regimewechsel ({me.mean():.1%} investiert)"
        )
    m = pd.DataFrame(mat).dropna()
    print()
    print("  Uebereinstimmung an den Rebalance-Terminen:")
    print(f"    alle drei gleich: {(m.nunique(axis=1) == 1).mean():.1%}")
    for a, b in (
        ("preis>sma", "sma steigt"),
        ("preis>sma", "rendite>0"),
        ("sma steigt", "rendite>0"),
    ):
        print(f"    {a} vs {b}: {(m[a] == m[b]).mean():.1%}")

    # ---------------------------------------------------------- 4. Synthese
    print("\n" + "=" * 72)
    best = max(zeilen, key=lambda z: -abs(z["maxdd"]))
    print("SYNTHESE")
    for z in sorted(zeilen, key=lambda z: z["maxdd"], reverse=True):
        print(
            f"  {z['definition']:<12} MaxDD {z['maxdd']:>7.1%} | "
            f"{z['anteil_investiert']:>5.1%} investiert | "
            f"{z['n_schaltungen']:>3} Schaltungen | Median {z['median']:.3f}"
        )
    print(f"\n  Geringster Drawdown: {best['definition']}")
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
