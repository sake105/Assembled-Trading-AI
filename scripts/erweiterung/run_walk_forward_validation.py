#!/usr/bin/env python
"""Walk-Forward Out-of-Sample-Validierung des Regime-Switching.

Idee
----
Bevor wir den Drawdown-Trigger-Switch als "validated" deklarieren, müssen wir
ihn streng out-of-sample testen. Walk-Forward:

1. Rolling 5y-Train-Window
2. Threshold-Optimierung auf Calmar-Ratio in Train
3. Apply zu nächstem 1y-Test-Window (out-of-sample)
4. Sammele Test-Returns zu einer OOS-Curve

Vergleicht OOS-Curve gegen:
- Fixed-Threshold (0.08, naive)
- Pure Mom-12/1
- Pure Equal-Weight
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)
from erweiterung.robustness.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    concat_oos_returns,
    walk_forward_threshold_search,
)


def main():
    # Lade die 19y-Equity-Curves
    eq_csv = Path("output/erweiterung_long_history_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found — run run_long_history_backtest.py first.")
        return 1

    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    bench = df["benchmark_equal_weight"].pct_change().dropna()
    fac = df["momentum_12_1_LongOnly"].pct_change().dropna()

    # Threshold-Grid
    grid = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

    print(f"Walk-Forward 5y train -> 1y test, grid={grid}")
    cfg = WalkForwardConfig(train_days=1260, test_days=252, step_days=252)
    wf = walk_forward_threshold_search(bench, fac, threshold_grid=grid, config=cfg)
    print(f"\n{len(wf)} walk-forward windows")
    print(
        wf[
            [
                "window_idx",
                "train_start",
                "test_start",
                "best_threshold",
                "train_obj",
                "test_ann_return",
                "test_sharpe",
                "test_max_dd",
            ]
        ].to_string(index=False)
    )

    # OOS-Curve
    oos = concat_oos_returns(bench, fac, wf)
    print(f"\nOOS series: {len(oos)} days, {oos.index.min()} -> {oos.index.max()}")

    # Fixed-threshold = 0.08 (naive, was wir vorher hatten)
    aligned_bf = pd.concat({"b": bench, "f": fac}, axis=1).dropna()
    bench_a = aligned_bf["b"]
    fac_a = aligned_bf["f"]
    eq_bench = (1 + bench_a.fillna(0)).cumprod()
    rolling_max = eq_bench.rolling(60, min_periods=1).max()
    dd = (1 - eq_bench / rolling_max).abs()
    fixed_regime_lag = pd.Series(
        np.where(dd > 0.08, "stress", "calm"), index=bench_a.index
    ).shift(1)
    fixed_alloc = pd.Series(
        np.where(fixed_regime_lag == "stress", fac_a, bench_a), index=bench_a.index
    ).dropna()
    # Trim to OOS-range for fair comparison
    fixed_alloc_oos = fixed_alloc.loc[oos.index.min() : oos.index.max()]
    bench_oos = bench.loc[oos.index.min() : oos.index.max()]
    fac_oos = fac.loc[oos.index.min() : oos.index.max()]

    print("\n" + "=" * 100)
    print("WALK-FORWARD OOS RESULTS (full OOS-period)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    for name, ret in [
        ("Pure Equal-Weight (OOS-period)", bench_oos),
        ("Pure Mom-12/1 LO (OOS-period)", fac_oos),
        ("Fixed-thr=0.08 Switch", fixed_alloc_oos),
        ("Walk-Forward OOS Switch", oos),
    ]:
        m = all_metrics(ret.dropna())
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Hansen-SPA: Walk-Forward-OOS vs Pure-Mom-12/1
    aligned = pd.concat(
        {"oos": oos, "fixed": fixed_alloc_oos, "pure_mom": fac_oos, "ew": bench_oos},
        axis=1,
    ).dropna()
    if len(aligned) > 200:
        baseline_pure = aligned["pure_mom"]
        challengers = aligned.drop(columns=["pure_mom"])
        excess = challengers.subtract(baseline_pure, axis=0)
        wrc = whites_reality_check(excess, n_bootstrap=2000, seed=42)
        spa = hansen_spa_test(excess, n_bootstrap=2000, seed=42)
        print(
            f"\nReality-Check vs Pure-Mom (OOS): best={wrc['best_strategy']} p={wrc['p_value']:.4f}"
        )
        print(
            f"Hansen-SPA vs Pure-Mom (OOS)  : best={spa['best_strategy']} p={spa['p_value']:.4f}"
        )

    # Per-window analysis
    print("\n" + "=" * 100)
    print("PER-WINDOW: Train-Calmar -> Test-Calmar Stability")
    print("=" * 100)
    wf_with_test_calmar = wf.copy()
    wf_with_test_calmar["test_calmar"] = wf_with_test_calmar.apply(
        lambda r: (
            r["test_ann_return"] / abs(r["test_max_dd"]) if r["test_max_dd"] != 0 else 0
        ),
        axis=1,
    )
    # Correlation Train-Obj vs Test-Calmar
    if len(wf_with_test_calmar) > 3:
        corr = wf_with_test_calmar["train_obj"].corr(wf_with_test_calmar["test_calmar"])
        print(f"Train-Calmar <-> Test-Calmar correlation: {corr:+.3f}")
        if corr > 0.3:
            print("  -> Training-Performance ist prädiktiv (>0.3)")
        elif corr < -0.1:
            print("  -> Training-Performance ist anti-prädiktiv (Overfit-Risk)")
        else:
            print("  -> Training-Performance hat keine starke prädiktive Kraft")

    # Save
    Path("output/erweiterung_walk_forward_oos_equity.csv").write_text(
        pd.DataFrame(
            {"oos_return": oos, "oos_equity": (1 + oos.fillna(0)).cumprod()}
        ).to_csv()
    )
    wf.to_csv("output/erweiterung_walk_forward_windows.csv", index=False)
    Path("output/erweiterung_walk_forward_summary.json").write_text(
        json.dumps(
            {
                "n_windows": int(len(wf)),
                "oos_days": int(len(oos)),
                "oos_metrics": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer))
                        else v
                    )
                    for k, v in all_metrics(oos).items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))
                },
                "threshold_distribution": wf["best_threshold"].value_counts().to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_walk_forward_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
