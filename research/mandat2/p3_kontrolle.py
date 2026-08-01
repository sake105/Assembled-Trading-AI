"""P3 — Kontrolltest: ist der P2-Befund Momentum-Alpha oder Turnover-Artefakt?

DIE FRAGE
---------
P2 fand: mit ``hold730 / out200`` schlaegt 12-1-Momentum den Index in allen
Steuerwelten (ZERO Median 2,737 gegen 1,948). Aber ``out200`` heisst bei ~500
Namen faktisch „nie auf Rang verkaufen" — die Strategie ist dann kaum noch
Momentum, sondern **Buy-and-Hold einer vor zwei Jahren getroffenen Auswahl**.

Wenn eine ZUFAELLIGE Auswahl mit derselben Haltedisziplin denselben Vorsprung
liefert, misst der Backtest nicht die Auswahl, sondern das Halten — und der
Alpha-Befund ist ein Artefakt. Das ist der billigste und haerteste Test, den
es fuer diesen Befund gibt, und er muss VOR jeder weiteren Optimierung kommen.

AUFBAU
------
Identische Mechanik, identische Parameter, identisches Universum. Einziger
Unterschied: der Score ist Rauschen statt Momentum. 20 Seeds ergeben eine
Verteilung statt eines Punktes — ein einzelner Zufallslauf koennte durch
Glueck gewinnen.

Entscheidungsregel, VOR dem Lauf festgelegt:
* Liegt der Momentum-Median INNERHALB der Zufallsverteilung (< 95. Perzentil),
  ist der P2-Befund ein Turnover-Artefakt. Verdikt: kein Alpha.
* Liegt er darueber, ueberlebt Momentum diesen Test — mehr nicht; DSR/PBO und
  Holdout stehen weiterhin aus.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import (  # noqa: E402
    run_buy_and_hold,
    run_strategy,
    sma_gate,
    zufalls_score,
)
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
import os  # noqa: E402

N_SEEDS = 20
#: Die Gewinnerkombination aus P2 — unveraendert uebernommen.
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
WELT = "ZERO"  # dort war der Momentum-Vorsprung am groessten
#: Zweiter Durchgang MIT dem SMA200-Gate aus P4. Eigene Regel aus P3:
#: jeder Auswahl-Befund braucht die Zufallskontrolle, bevor er zaehlt — der
#: gegatete Befund ist ein neuer Befund und braucht sie erneut.
MIT_GATE = os.environ.get("MANDAT2_GATE", "") == "1"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    zaehler = TrialCounter()
    print(
        f"Trials kumuliert: {zaehler.increment(N_SEEDS, label='P3 Zufallskontrolle')}\n",
        flush=True,
    )

    gate = sma_gate(d.close, fenster=200) if MIT_GATE else None
    suffix = " (MIT SMA200-Gate)" if MIT_GATE else ""
    print(f"Variante:{suffix or ' ohne Gate'}\n", flush=True)
    bench = run_buy_and_hold(d, make_regime(WELT))
    mom = run_strategy(d, make_regime(WELT), score=None, risk_off_gate=gate, **PARAMS)
    a_mom = auswerten(mom.equity_netto, bench.equity_netto, label="Momentum")
    print(
        f"[BENCH]    {bench.equity.iloc[-1]:>12,.0f}  Median {a_mom.median_benchmark:.3f}"
    )
    print(
        f"[MOMENTUM] {mom.equity.iloc[-1]:>12,.0f}  Median {a_mom.median_kandidat:.3f}"
        f"  DD {a_mom.schlimmster_maxdd:.1%}  Trades {mom.n_trades}\n",
        flush=True,
    )

    zufall = []
    for seed in range(N_SEEDS):
        r = run_strategy(
            d,
            make_regime(WELT),
            score=zufalls_score(d.close, seed),
            risk_off_gate=gate,
            **PARAMS,
        )
        a = auswerten(r.equity_netto, bench.equity_netto, label=f"Zufall#{seed}")
        zufall.append(
            {
                "seed": seed,
                "endwert": float(r.equity.iloc[-1]),
                "median_kandidat": a.median_kandidat,
                "schlimmster_maxdd": a.schlimmster_maxdd,
                "n_trades": r.n_trades,
                "anteil_fenster_geschlagen": a.anteil_fenster_geschlagen,
            }
        )
        print(
            f"  Zufall#{seed:>2}: End {r.equity.iloc[-1]:>11,.0f} | "
            f"Median {a.median_kandidat:.3f} | DD {a.schlimmster_maxdd:>7.1%} | "
            f"Trades {r.n_trades}",
            flush=True,
        )

    medians = np.array([z["median_kandidat"] for z in zufall])
    p95 = float(np.percentile(medians, 95))
    perzentil_mom = float((medians < a_mom.median_kandidat).mean() * 100)
    artefakt = a_mom.median_kandidat <= p95

    ergebnis = {
        "welt": WELT,
        "mit_gate": MIT_GATE,
        "params": PARAMS,
        "momentum": {
            "endwert": float(mom.equity.iloc[-1]),
            "median": a_mom.median_kandidat,
            "maxdd": a_mom.schlimmster_maxdd,
            "n_trades": mom.n_trades,
        },
        "benchmark_median": a_mom.median_benchmark,
        "zufall": zufall,
        "zufall_median_mittel": float(medians.mean()),
        "zufall_median_p95": p95,
        "zufall_median_max": float(medians.max()),
        "momentum_perzentil_in_zufallsverteilung": perzentil_mom,
        "verdikt_turnover_artefakt": bool(artefakt),
    }
    (OUT / ("p3_kontrolle_gate.json" if MIT_GATE else "p3_kontrolle.json")).write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70, flush=True)
    print(f"Benchmark (SPY)          Median {a_mom.median_benchmark:.3f}")
    print(f"Momentum                 Median {a_mom.median_kandidat:.3f}")
    print(
        f"Zufall ({N_SEEDS} Seeds)        Median-Mittel {medians.mean():.3f} | "
        f"P95 {p95:.3f} | Max {medians.max():.3f}"
    )
    print(f"Momentum liegt im {perzentil_mom:.0f}. Perzentil der Zufallsverteilung")
    print()
    if artefakt:
        print("VERDIKT: TURNOVER-ARTEFAKT. Momentum liegt innerhalb dessen, was")
        print("         eine Zufallsauswahl mit derselben Haltedisziplin liefert.")
        print("         Der P2-Befund ist KEIN Auswahl-Alpha.")
    else:
        print("VERDIKT: Momentum ueberlebt den Kontrolltest (ueber P95 der")
        print("         Zufallsverteilung). Offen bleiben DSR/PBO und der Holdout.")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
