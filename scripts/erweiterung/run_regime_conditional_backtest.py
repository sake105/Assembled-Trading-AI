#!/usr/bin/env python
"""Regime-Conditional Allocator angewendet auf Expanded-Universe-Equity.

Liest die vom Expanded-Backtest erzeugte Equity-CSV, leitet aus dem
Equal-Weight-Benchmark ein Stress/Calm-Regime ab und schaltet zwischen
``benchmark_equal_weight`` (calm) und ``momentum_12_1_LongOnly`` (stress).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.regime_conditional_allocator import (  # noqa: E402
    RegimeConfig,
    allocate_regime_conditional,
    regime_metrics,
)


def main():
    eq_csv = Path("output/erweiterung_expanded_universe_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found")
        return 1
    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    # Calm-Returns: equal-weight; Stress-Returns: momentum_12_1_LongOnly
    bench_eq = df["benchmark_equal_weight"]
    fac_eq = df["momentum_12_1_LongOnly"]
    bench_ret = bench_eq.pct_change()
    fac_ret = fac_eq.pct_change()

    for thr in (0.05, 0.08, 0.10, 0.12):
        out = allocate_regime_conditional(
            bench_ret, fac_ret, RegimeConfig(drawdown_threshold=thr, smoothing_days=3)
        )
        metrics = regime_metrics(out)
        agg = all_metrics(out["allocated_return"].dropna())

        regime_share = (out["regime"].dropna() == "stress").mean()
        print(f"\n=== Threshold = {thr:.2f} ===")
        print(f"  stress-share={regime_share:.1%}")
        print(
            f"  Overall: AnnRet={agg.get('annualized_return', 0):+.2%} "
            f"Sharpe={agg.get('sharpe', 0):+.3f} "
            f"MDD={agg.get('max_drawdown', 0):+.2%}"
        )
        for label in ("calm", "stress"):
            m = metrics.get(label, {})
            if not m or m.get("ann_return") is None:
                continue
            print(
                f"  In {label}: AnnRet={m['ann_return']:+.2%} "
                f"Sharpe={(m.get('sharpe') or 0):+.3f} "
                f"MDD={(m.get('max_dd') or 0):+.2%} "
                f"days={m['n_days']}"
            )

    # Baseline: pure benchmark vs pure factor-tilt
    print("\n=== Reference (no switching) ===")
    bench_metrics = all_metrics(bench_ret.dropna())
    fac_metrics = all_metrics(fac_ret.dropna())
    print(
        f"  Pure Equal-Weight  : AnnRet={bench_metrics['annualized_return']:+.2%} "
        f"Sharpe={bench_metrics['sharpe']:+.3f} MDD={bench_metrics['max_drawdown']:+.2%}"
    )
    print(
        f"  Pure Mom-12/1-LO    : AnnRet={fac_metrics['annualized_return']:+.2%} "
        f"Sharpe={fac_metrics['sharpe']:+.3f} MDD={fac_metrics['max_drawdown']:+.2%}"
    )

    # Save the 0.08 variant as canonical result
    out_canonical = allocate_regime_conditional(
        bench_ret, fac_ret, RegimeConfig(drawdown_threshold=0.08, smoothing_days=3)
    )
    out_canonical["equity"] = (
        1 + out_canonical["allocated_return"].fillna(0)
    ).cumprod()
    out_canonical.to_csv("output/erweiterung_regime_conditional_equity.csv")
    print(
        "\nSaved canonical (thr=0.08) -> output/erweiterung_regime_conditional_equity.csv"
    )

    # JSON summary
    final = {
        "thresholds_tested": [0.05, 0.08, 0.10, 0.12],
        "canonical_threshold": 0.08,
        "reference": {
            "equal_weight": {
                k: (
                    float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else v
                )
                for k, v in bench_metrics.items()
            },
            "momentum_12_1_LongOnly": {
                k: (
                    float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else v
                )
                for k, v in fac_metrics.items()
            },
        },
    }
    Path("output/erweiterung_regime_conditional_summary.json").write_text(
        json.dumps(final, indent=2, default=str)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
