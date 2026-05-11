#!/usr/bin/env python
"""Vol-Targeting Backtest + Walk-Forward-Vergleich gegen Pure Long-Only & Switching.

Hypothese
---------
Vol-Targeting (Moreira/Muir 2017) sollte robuster sein als binäres Regime-
Switching — und sollte im Walk-Forward OOS einen statistisch nachweisbaren
Edge zeigen.

Test
----
1. Apply Vol-Targeting auf Pure-Mom-12/1-LO (Long-History 19y).
2. Walk-Forward Out-of-Sample: in jedem 5y-Train wird das beste Target-Vol
   gewählt (auf Calmar), dann auf 1y-Test angewandt.
3. Vergleich mit Pure-Mom, Pure-Equal-Weight, Switching.
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
from erweiterung.strategies.volatility_targeting import (  # noqa: E402
    VolTargetConfig,
    apply_vol_targeting,
)


def main():
    eq_csv = Path("output/erweiterung_long_history_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found.")
        return 1

    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    bench = df["benchmark_equal_weight"].pct_change().dropna()
    fac = df["momentum_12_1_LongOnly"].pct_change().dropna()

    # In-Sample-Test: verschiedene Target-Vols
    print("\n" + "=" * 100)
    print("IN-SAMPLE: Vol-Targeting auf Pure-Mom-12/1-LO (Long-History 19y)")
    print("=" * 100)
    print(
        f"{'Target-Vol':<12} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metrics_by_target = {}
    for tv in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        out = apply_vol_targeting(fac, VolTargetConfig(target_vol_annual=tv))
        scaled = out["scaled_return"].dropna()
        m = all_metrics(scaled)
        metrics_by_target[tv] = m
        print(
            f"  {tv:<10.2f} "
            f"{m['annualized_return']:>+8.2%} "
            f"{m['sharpe']:>+7.3f} "
            f"{m['sortino']:>+8.3f} "
            f"{m['calmar']:>+7.3f} "
            f"{m['max_drawdown']:>+7.2%}"
        )

    # Reference
    print("\nReference:")
    bench_m = all_metrics(bench)
    fac_m = all_metrics(fac)
    print(
        f"  Pure Equal-Weight       AnnRet={bench_m['annualized_return']:+.2%} "
        f"Sharpe={bench_m['sharpe']:+.3f} MDD={bench_m['max_drawdown']:+.2%}"
    )
    print(
        f"  Pure Mom-12/1 LO        AnnRet={fac_m['annualized_return']:+.2%} "
        f"Sharpe={fac_m['sharpe']:+.3f} MDD={fac_m['max_drawdown']:+.2%}"
    )

    # ===== Walk-Forward Out-of-Sample =====
    print("\n" + "=" * 100)
    print("WALK-FORWARD OOS: Target-Vol-Optimierung")
    print("=" * 100)
    train_days = 1260
    test_days = 252
    step = 252
    grid = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    aligned = pd.concat({"fac": fac}, axis=1).dropna()
    n = len(aligned)
    rows = []
    chunks = []
    start = 0
    while start + train_days + test_days <= n:
        train_idx = aligned.index[start : start + train_days]
        test_idx = aligned.index[start + train_days : start + train_days + test_days]
        train_fac = aligned.loc[train_idx, "fac"]
        # Combine train+test for trailing vol warmup
        combined = aligned.loc[
            aligned.index[max(0, start) : start + train_days + test_days], "fac"
        ]
        # find best target_vol via calmar in train
        best_tv = grid[0]
        best_calmar = -np.inf
        for tv in grid:
            out = apply_vol_targeting(train_fac, VolTargetConfig(target_vol_annual=tv))
            scaled = out["scaled_return"].dropna()
            if scaled.empty:
                continue
            ann_ret = (1 + scaled).prod() ** (252 / len(scaled)) - 1
            eq = (1 + scaled).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            calmar = ann_ret / abs(dd) if dd != 0 else -np.inf
            if calmar > best_calmar:
                best_calmar = calmar
                best_tv = tv

        # apply best_tv on combined, take test slice
        out_full = apply_vol_targeting(
            combined, VolTargetConfig(target_vol_annual=best_tv)
        )
        test_returns = out_full["scaled_return"].loc[test_idx].dropna()
        chunks.append(test_returns)

        t_ann = (1 + test_returns).prod() ** (252 / max(len(test_returns), 1)) - 1
        t_vol = test_returns.std() * np.sqrt(252)
        t_dd = (
            (1 + test_returns).cumprod() / (1 + test_returns).cumprod().cummax() - 1
        ).min()
        rows.append(
            {
                "window_idx": len(rows),
                "train_start": train_idx[0],
                "test_start": test_idx[0],
                "best_target_vol": best_tv,
                "train_calmar": best_calmar,
                "test_ann_return": t_ann,
                "test_realized_vol": t_vol,
                "test_max_dd": t_dd,
                "test_sharpe": t_ann / t_vol if t_vol > 0 else 0,
            }
        )
        start += step

    wf_df = pd.DataFrame(rows)
    print(f"\n{len(wf_df)} Windows:")
    print(
        wf_df[
            [
                "window_idx",
                "test_start",
                "best_target_vol",
                "train_calmar",
                "test_ann_return",
                "test_realized_vol",
                "test_sharpe",
                "test_max_dd",
            ]
        ].to_string(index=False)
    )

    oos = pd.concat(chunks).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]

    # OOS-Performance
    print("\n" + "=" * 100)
    print("OOS COMPARISON (vol-targeted vs others, common period)")
    print("=" * 100)
    bench_oos = bench.loc[oos.index.min() : oos.index.max()]
    fac_oos = fac.loc[oos.index.min() : oos.index.max()]
    wf_switch_csv = Path("output/erweiterung_walk_forward_oos_equity.csv")
    switch_oos = None
    if wf_switch_csv.exists():
        s = pd.read_csv(wf_switch_csv, index_col=0)
        s.index = pd.to_datetime(s.index, utc=True)
        switch_oos = s["oos_return"]

    candidates = {
        "Pure Equal-Weight (OOS)": bench_oos,
        "Pure Mom-12/1 LO (OOS)": fac_oos,
        "Vol-Targeted Mom (OOS)": oos,
    }
    if switch_oos is not None:
        candidates["Walk-Forward Switch (OOS)"] = switch_oos.loc[
            oos.index.min() : oos.index.max()
        ]

    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metric_dump = {}
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metric_dump[name] = m
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Reality-Check vs Pure-Mom
    align = pd.concat({k: v for k, v in candidates.items()}, axis=1).dropna()
    if "Pure Mom-12/1 LO (OOS)" in align.columns and len(align) > 200:
        baseline = align["Pure Mom-12/1 LO (OOS)"]
        excess = align.drop(columns=["Pure Mom-12/1 LO (OOS)"]).subtract(
            baseline, axis=0
        )
        wrc = whites_reality_check(excess, n_bootstrap=2000, seed=42)
        spa = hansen_spa_test(excess, n_bootstrap=2000, seed=42)
        print(
            f"\nReality-Check vs Pure-Mom (OOS): best={wrc['best_strategy']}  p={wrc['p_value']:.4f}"
        )
        print(
            f"Hansen-SPA vs Pure-Mom (OOS)  : best={spa['best_strategy']}  p={spa['p_value']:.4f}"
        )

    # Save
    out_csv = pd.DataFrame(
        {"oos_return": oos, "oos_equity": (1 + oos.fillna(0)).cumprod()}
    )
    out_csv.to_csv("output/erweiterung_vol_target_oos_equity.csv")
    Path("output/erweiterung_vol_target_summary.json").write_text(
        json.dumps(
            {
                "in_sample_metrics_by_target_vol": {
                    str(tv): {
                        k: (
                            float(v)
                            if isinstance(v, (int, float, np.floating, np.integer))
                            else v
                        )
                        for k, v in m.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))
                    }
                    for tv, m in metrics_by_target.items()
                },
                "walk_forward_windows": wf_df.to_dict("records"),
                "oos_metrics": {
                    name: {
                        k: (
                            float(v)
                            if isinstance(v, (int, float, np.floating, np.integer))
                            else v
                        )
                        for k, v in m.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))
                    }
                    for name, m in metric_dump.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_vol_target_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
