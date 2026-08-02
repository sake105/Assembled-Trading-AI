"""P11 — Ist der P3-Zufallsbefund nur der Gleichgewichtungs-Effekt?

DIE OFFENE NOTIZ AUS P3
-----------------------
P3 hat gezeigt: 20 ZUFAELLIG gezogene S&P-Namen, lange gehalten, erreichen
Median 2,69 gegen SPY 1,95. Ich hatte das damals als „bekannter
Gleichgewichtungs-Effekt" abgetan und notiert, es muesse „sauber gegen einen
gleichgewichteten Index-Benchmark statt gegen SPY" geprueft werden.

Genau das passiert hier. Denn wenn der Vorsprung verschwindet, sobald man
gegen einen gleichgewichteten Index misst, dann war der ganze P2/P3-Komplex
eine Messung gegen den falschen Massstab — und zwar von Anfang an.

DER RICHTIGE MASSSTAB
---------------------
SPY ist KAPITALgewichtet. Unsere Kandidaten sind GLEICHgewichtet. Der
Unterschied ist eine bekannte Faktorexposition (kleinere Namen hoeher
gewichtet), kein Alpha. Ein fairer Vergleich braucht deshalb einen
gleichgewichteten Benchmark aus demselben Universum.

Gebaut wird er als das, was er ist: alle Index-Mitglieder, gleichgewichtet,
monatlich rebalanciert, mit denselben Kosten und derselben Steuerwelt. Also
`run_strategy` mit einem konstanten Score — dann waehlt der Rang-Mechanismus
schlicht die ersten N Mitglieder, und mit einem hinreichend grossen `top_in`
sind das alle.

DIES IST KEINE ALPHA-SUCHE. Es wird ein Massstab korrigiert.
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
PARAMS = dict(rank_out=200, min_haltetage=730, hebel=1.0)
N_SEEDS = 8
WELT = "ZERO"


def gleichgewicht_score(close: pd.DataFrame) -> pd.DataFrame:
    """Konstanter Score -> der Rang entscheidet nicht, alle sind gleich."""
    return pd.DataFrame(1.0, index=close.index, columns=close.columns)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print(
        f"Trials kumuliert: {TrialCounter().increment(4 + 2 * N_SEEDS, label='P11 EW-Benchmark')}\n"
    )

    spy = run_buy_and_hold(d, make_regime(WELT))
    print(f"[SPY kapitalgewichtet]  End {spy.equity.iloc[-1]:>12,.0f}")

    # Gleichgewichteter Index: moeglichst breit, damit es wirklich "der Index"
    # ist und keine Auswahl.
    ew = run_strategy(
        d, make_regime(WELT), score=gleichgewicht_score(d.close), top_in=500, **PARAMS
    )
    a_ew = auswerten(ew.equity_netto, spy.equity_netto, label="EW-Index")
    print(
        f"[EW-Index breit]        End {ew.equity.iloc[-1]:>12,.0f} | "
        f"Median {a_ew.median_kandidat:.3f} vs SPY {a_ew.median_benchmark:.3f} | "
        f"DD {a_ew.schlimmster_maxdd:.1%}"
    )

    ergebnis = {
        "spy_end": float(spy.equity.iloc[-1]),
        "ew_index_end": float(ew.equity.iloc[-1]),
        "ew_index_median_vs_spy": a_ew.median_kandidat,
        "spy_median": a_ew.median_benchmark,
        "ew_index_maxdd": a_ew.schlimmster_maxdd,
        "zufall": [],
    }

    # Zufallsauswahl gegen BEIDE Massstaebe
    print("\n=== 20 Zufallsnamen gegen beide Massstaebe ===")
    print(f"  {'Seed':<6}{'Median vs SPY':>15}{'Median vs EW':>15}{'MaxDD':>9}")
    gegen_spy, gegen_ew = [], []
    for seed in range(N_SEEDS):
        r = run_strategy(
            d,
            make_regime(WELT),
            score=zufalls_score(d.close, seed),
            top_in=20,
            **PARAMS,
        )
        a_spy = auswerten(r.equity_netto, spy.equity_netto, label=f"z{seed}/SPY")
        a_ew2 = auswerten(r.equity_netto, ew.equity_netto, label=f"z{seed}/EW")
        gegen_spy.append(a_spy.median_kandidat / a_spy.median_benchmark)
        gegen_ew.append(a_ew2.median_kandidat / a_ew2.median_benchmark)
        ergebnis["zufall"].append(
            {
                "seed": seed,
                "median": a_spy.median_kandidat,
                "verhaeltnis_zu_spy": gegen_spy[-1],
                "verhaeltnis_zu_ew": gegen_ew[-1],
                "maxdd": a_spy.schlimmster_maxdd,
            }
        )
        print(
            f"  {seed:<6}{gegen_spy[-1]:>14.2f}x{gegen_ew[-1]:>14.2f}x"
            f"{a_spy.schlimmster_maxdd:>9.1%}"
        )

    ergebnis["mittel_verhaeltnis_spy"] = float(np.mean(gegen_spy))
    ergebnis["mittel_verhaeltnis_ew"] = float(np.mean(gegen_ew))

    # Und der Momentum-Kandidat mit Gate gegen den EW-Index
    kand = run_strategy(
        d,
        make_regime(WELT),
        top_in=20,
        risk_off_gate=sma_gate(d.close, fenster=140),
        **PARAMS,
    )
    a_k_spy = auswerten(kand.equity_netto, spy.equity_netto, label="Kandidat/SPY")
    a_k_ew = auswerten(kand.equity_netto, ew.equity_netto, label="Kandidat/EW")
    ergebnis["kandidat"] = {
        "verhaeltnis_zu_spy": a_k_spy.median_kandidat / a_k_spy.median_benchmark,
        "verhaeltnis_zu_ew": a_k_ew.median_kandidat / a_k_ew.median_benchmark,
    }

    (OUT / "p11_gleichgewicht.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p11_gleichgewicht.json'}")

    print("\n" + "=" * 70)
    print("Zufallsauswahl, Median-Verhaeltnis zum jeweiligen Massstab:")
    print(f"  gegen SPY (kapitalgewichtet): {np.mean(gegen_spy):.2f}x")
    print(f"  gegen EW-Index               : {np.mean(gegen_ew):.2f}x")
    print()
    print("Momentum-Kandidat mit Gate:")
    print(
        f"  gegen SPY                    : {ergebnis['kandidat']['verhaeltnis_zu_spy']:.2f}x"
    )
    print(
        f"  gegen EW-Index               : {ergebnis['kandidat']['verhaeltnis_zu_ew']:.2f}x"
    )
    print()
    if np.mean(gegen_ew) < 1.05:
        print("BEFUND: Der Zufallsvorsprung verschwindet gegen den richtigen")
        print("        Massstab. Der P3-Effekt WAR die Gleichgewichtung.")
    else:
        print("BEFUND: Der Zufallsvorsprung bleibt auch gegen den EW-Index —")
        print("        dann ist es nicht (nur) die Gewichtung.")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
