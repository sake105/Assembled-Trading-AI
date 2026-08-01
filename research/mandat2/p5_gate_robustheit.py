"""P5 — Ist das SMA200-Gate ein Mechanismus oder ein gefundener Parameter?

DIE FRAGE
---------
P4 fand: mit SMA200 besteht der Kandidat in allen drei Steuerwelten. Mit
SMA100 reissen 64 von 144 Fenstern, mit SMA300 in der GmbH-Welt 61. Drei Werte
getestet, der mittlere gewinnt — genau das Muster eines gefundenen Parameters.

Ein MECHANISMUS muss zwei Tests bestehen:

1. **Fenster-Band statt Punkt.** Ueber ein feines Raster (100…320) darf das
   Ergebnis schwanken, aber es muss ein zusammenhaengendes Band geben, in dem
   es funktioniert. Funktioniert nur 200 und die Nachbarn nicht, ist es
   Rauschen.
2. **Andere Trend-Definitionen.** Wenn nur „Preis > SMA" funktioniert, aber
   weder „SMA steigt" noch „12-Monats-Rendite > 0", dann traegt nicht die
   Trendfolge, sondern die eine Formel.

Beide Tests sind billig und beide koennen den P4-Kandidaten toeten. Sie kommen
deshalb VOR DSR/PBO und weit vor dem Holdout.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_strategy  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
FENSTER = list(range(100, 321, 20))  # 12 Werte
WELTEN = [("ZERO", "ZERO", {}), ("PRIVAT_DE", "PRIVAT_DE", {})]


def gate_preis_ueber_sma(close: pd.DataFrame, fenster: int) -> pd.Series:
    s = close["SPY"]
    return (s > s.rolling(fenster).mean()).astype(float)


def gate_sma_steigt(close: pd.DataFrame, fenster: int) -> pd.Series:
    """Trend ueber die Steigung statt ueber die Lage."""
    m = close["SPY"].rolling(fenster).mean()
    return (m > m.shift(21)).astype(float)


def gate_rendite_positiv(close: pd.DataFrame, fenster: int) -> pd.Series:
    """Absolute Momentum: Rendite ueber das Fenster > 0."""
    s = close["SPY"]
    return (s > s.shift(fenster)).astype(float)


DEFINITIONEN = {
    "preis>sma": gate_preis_ueber_sma,
    "sma steigt": gate_sma_steigt,
    "rendite>0": gate_rendite_positiv,
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n = len(FENSTER) * len(DEFINITIONEN) * len(WELTEN)
    print(
        f"Trials kumuliert: {TrialCounter().increment(n, label='P5 Gate-Robustheit')}\n",
        flush=True,
    )

    zeilen = []
    for welt, name, kwargs in WELTEN:
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        for defname, fn in DEFINITIONEN.items():
            print(f"=== {welt} | {defname} ===", flush=True)
            for f in FENSTER:
                gate = fn(d.close, f)
                r = run_strategy(
                    d, make_regime(name, **kwargs), risk_off_gate=gate, **PARAMS
                )
                a = auswerten(
                    r.equity_netto, bench.equity_netto, label=f"{welt}/{defname}/{f}"
                )
                zeilen.append(
                    {
                        "welt": welt,
                        "definition": defname,
                        "fenster": f,
                        "endwert": float(r.equity.iloc[-1]),
                        "median_kandidat": a.median_kandidat,
                        "median_benchmark": a.median_benchmark,
                        "schlimmster_maxdd": a.schlimmster_maxdd,
                        "gerissene_fenster": len(a.gerissene_fenster),
                        "bestanden": a.bestanden,
                    }
                )
                print(
                    f"  {f:>4}: DD {a.schlimmster_maxdd:>7.1%} | gerissen "
                    f"{len(a.gerissene_fenster):>3}/144 | Median {a.median_kandidat:.3f}"
                    f" | {'BESTANDEN' if a.bestanden else '-'}",
                    flush=True,
                )

    (OUT / "p5_gate_robustheit.json").write_text(
        json.dumps({"params": PARAMS, "zeilen": zeilen}, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("BAND-ANALYSE: wie viele der 12 Fenster bestehen je Definition/Welt?")
    for welt, _, _ in WELTEN:
        for defname in DEFINITIONEN:
            g = [z for z in zeilen if z["welt"] == welt and z["definition"] == defname]
            ok = [z["fenster"] for z in g if z["bestanden"]]
            zus = ""
            if ok:
                luecken = [b - a for a, b in zip(ok, ok[1:])]
                zus = (
                    " (zusammenhaengend)"
                    if all(x == 20 for x in luecken)
                    else " (LUECKIG)"
                )
            print(f"  {welt:<10} {defname:<12}: {len(ok):>2}/12 bestehen {ok}{zus}")
    print()
    ges = sum(1 for z in zeilen if z["bestanden"])
    print(f"Insgesamt bestanden: {ges} von {len(zeilen)}")
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
