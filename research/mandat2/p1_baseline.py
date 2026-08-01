"""P1-Erstlauf: Momentum gegen SPY unter allen vier Regimen (Mandat II).

Beantwortet die Frage, mit der Hans das Mandat neu geoeffnet hat: kippt das
Urteil, wenn die deutsche Privatanleger-Steuer nicht mehr die Bremse ist?

WICHTIG: laeuft ausschliesslich auf dem SUCH-Fenster (bis 2016-12-31). Der
Holdout wird hier nicht angefasst.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_momentum  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

REGIME = ["ZERO", "PRIVAT_DE", "GMBH_THESAURIEREND", "GMBH_AUSSCHUETTUNG"]
OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    ergebnisse = {}

    for name in REGIME:
        bench = run_buy_and_hold(d, make_regime(name))
        kand = run_momentum(d, make_regime(name), top_in=20, rank_out=60)
        # Ausschuettungsebene erst am Ende, damit die Kurve sie nicht sieht.
        if name == "GMBH_AUSSCHUETTUNG":
            bench.equity.iloc[-1] = bench.portfolio.settle_terminal(
                float(bench.equity.iloc[-1])
            )
            kand.equity.iloc[-1] = kand.portfolio.settle_terminal(
                float(kand.equity.iloc[-1])
            )
        a = auswerten(kand.equity, bench.equity, label=f"Momentum vs SPY [{name}]")
        print(f"\n=== {name} ===", flush=True)
        print("  " + bench.kurz(), flush=True)
        print("  " + kand.kurz(), flush=True)
        print("  " + a.bericht(), flush=True)
        ergebnisse[name] = {
            "benchmark_end": float(bench.equity.iloc[-1]),
            "kandidat_end": float(kand.equity.iloc[-1]),
            "n_fenster": a.n_fenster,
            "median_kandidat": a.median_kandidat,
            "median_benchmark": a.median_benchmark,
            "anteil_fenster_geschlagen": a.anteil_fenster_geschlagen,
            "schlimmster_maxdd": a.schlimmster_maxdd,
            "deckel_eingehalten": a.deckel_eingehalten,
            "bestanden": a.bestanden,
            "steuer_kandidat": kand.portfolio.tax_paid,
            "steuer_benchmark": bench.portfolio.tax_paid,
            "trades_kandidat": kand.n_trades,
        }

    (OUT / "p1_baseline.json").write_text(
        json.dumps(ergebnisse, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p1_baseline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
