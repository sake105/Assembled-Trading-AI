"""P7 — DSR und PBO: ueberlebt der Kandidat die Mehrfachtest-Korrektur?

WOZU
----
Der P4/P5-Kandidat (20 Namen, hold730/out200, SMA-Gate) besteht die
Zielfunktion in-sample. Bei einem Trial-Zaehler von 2.124 sagt das wenig: die
BESTE von vielen Konfigurationen ist per Konstruktion nach oben verzerrt.

Zwei Korrekturen, beide aus Bailey/López de Prado:

* **DSR** (Deflated Sharpe Ratio) — wie wahrscheinlich ist der beobachtete
  Sharpe, wenn man beruecksichtigt, dass er der beste aus N Versuchen ist?
  Genutzt wird die bestehende Implementierung
  ``src/assembled_core/qa/deflated_sharpe.py`` (keine dritte Wahrheit) mit der
  EMPIRISCHEN Varianz der Sharpes ueber die Familie statt der IID-Naeherung.
* **PBO** (Probability of Backtest Overfitting, CSCV) — wie oft landet der
  in-sample beste Kandidat out-of-sample unter dem Median? Genutzt wird
  ``research/mandat/h011_kandidat_a.cscv_pbo`` unveraendert.

EHRLICHE WAHL DES N
-------------------
DSR bekommt N = 2.124, den KUMULIERTEN Zaehler ueber beide Mandate. Das ist
die konservative Wahl: man koennte argumentieren, nur die 37 Varianten dieser
Familie zaehlten. Aber der Kandidat ist aus einer Suche entstanden, die auf
Mandat I aufsetzt und dessen verworfene Richtungen kennt — dieses Vorwissen
ist Teil der Selektion. Beide Zahlen werden ausgewiesen.

Die PBO-Matrix enthaelt GENAU die Konfigurationen, ueber die gesucht wurde
(3 Trend-Definitionen x 12 Fenster + ungegatet). Eine kleinere Matrix wuerde
PBO kuenstlich druecken.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.mandat.h011_kandidat_a import cscv_pbo  # noqa: E402
from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_strategy  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.p5_gate_robustheit import DEFINITIONEN, FENSTER  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
WELT = "ZERO"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n_familie = len(DEFINITIONEN) * len(FENSTER) + 1
    n_gesamt = TrialCounter().total()
    print(f"Familie: {n_familie} Varianten | Trial-Zaehler kumuliert: {n_gesamt}\n")

    bench = run_buy_and_hold(d, make_regime(WELT))
    kurven: dict[str, pd.Series] = {}
    kennzahlen: dict[str, dict] = {}

    varianten = [("ohne Gate", None, None)]
    for defname, fn in DEFINITIONEN.items():
        for f in FENSTER:
            varianten.append((f"{defname}/{f}", fn, f))

    for name, fn, f in varianten:
        gate = None if fn is None else fn(d.close, f)
        r = run_strategy(d, make_regime(WELT), risk_off_gate=gate, **PARAMS)
        a = auswerten(r.equity_netto, bench.equity_netto, label=name)
        kurven[name] = r.equity_netto.pct_change().dropna()
        kennzahlen[name] = {
            "median": a.median_kandidat,
            "maxdd": a.schlimmster_maxdd,
            "bestanden": a.bestanden,
            "endwert": float(r.equity.iloc[-1]),
        }
        print(
            f"  {name:<16} Median {a.median_kandidat:>6.3f} | DD "
            f"{a.schlimmster_maxdd:>7.1%} | {'BESTANDEN' if a.bestanden else '-'}",
            flush=True,
        )

    rm = pd.DataFrame(kurven).dropna()
    print(f"\nRenditematrix: {rm.shape[0]} Tage x {rm.shape[1]} Varianten")

    # --- PBO ueber die gesamte Familie
    pbo = cscv_pbo(rm, n_blocks=8)
    print(f"\nPBO (CSCV, 8 Bloecke, C(8,4)=70 Splits): {pbo:.1%}")

    # --- DSR fuer den in-sample Gewinner (nach Median-Faktor, dem Zielmass)
    bestanden = {k: v for k, v in kennzahlen.items() if v["bestanden"]}
    gewinner = max(bestanden or kennzahlen, key=lambda k: kennzahlen[k]["median"])
    sharpes = rm.apply(lambda x: x.mean() / x.std() if x.std() > 0 else np.nan)
    var_emp = float(sharpes.var(ddof=1))
    print(f"\nGewinner (nach Zielmass): {gewinner}")
    print(f"Empirische Varianz der Sharpes ueber die Familie: {var_emp:.3e}")

    ergebnis = {
        "pbo": pbo,
        "gewinner": gewinner,
        "varianz_sharpes": var_emp,
        "n_familie": n_familie,
        "n_kumuliert": n_gesamt,
        "kennzahlen": kennzahlen,
        "dsr": {},
    }
    for label, n in (("familie", n_familie), ("kumuliert", n_gesamt)):
        res = deflated_sharpe(rm[gewinner], n_trials=n, variance_across_trials=var_emp)
        ergebnis["dsr"][label] = {
            "n_trials": n,
            "sharpe_observed": float(res.sharpe_observed),
            "sharpe_threshold": float(res.sharpe_threshold),
            "dsr_probability": float(res.deflated_sharpe_probability),
            "passes_5pct": bool(res.passes_5pct),
        }
        print(
            f"  DSR (N={n:>5}): Sharpe {res.sharpe_observed:.4f} gegen Schwelle "
            f"{res.sharpe_threshold:.4f} -> p = "
            f"{res.deflated_sharpe_probability:.4f} "
            f"{'BESTANDEN' if res.passes_5pct else 'DURCHGEFALLEN'}"
        )

    (OUT / "p7_dsr_pbo.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p7_dsr_pbo.json'}")

    print("\n" + "=" * 68)
    dsr_ok = ergebnis["dsr"]["kumuliert"]["passes_5pct"]
    pbo_ok = pbo < 0.5
    print(f"DSR mit kumuliertem N: {'BESTANDEN' if dsr_ok else 'DURCHGEFALLEN'}")
    print(
        f"PBO < 50 %:            {'BESTANDEN' if pbo_ok else 'DURCHGEFALLEN'} ({pbo:.1%})"
    )
    if dsr_ok and pbo_ok:
        print("\n-> Kandidat ist reif fuer den EINEN Holdout-Schuss.")
    else:
        print("\n-> Kandidat ist NICHT reif fuer den Holdout. Kein Schuss.")
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
