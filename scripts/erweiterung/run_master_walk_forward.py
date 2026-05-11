#!/usr/bin/env python
"""Walk-Forward Out-of-Sample-Validierung für Master-Allocator.

Setup
-----
- Common-Period: 2021-2026 (1316 Tage Master-Returns)
- Train: 2 Jahre (504 Tage)
- Test: 6 Monate (126 Tage)
- Step: 6 Monate
- Hyperparameter-Grid: sa_weight ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}
- Optimierung auf Calmar-Ratio im Train-Window
- Apply gefittete Wahl auf Test-Window

Vergleich gegen In-Sample-Default (sa_weight=0.7) und 60/40 Classic.
Calmar-Bootstrap der OOS-Curve gegen Pure-Master-In-Sample und 60/40.

Was diese Validierung klärt
---------------------------
- Ist die 70/30-Wahl OOS-stabil oder zerlegt sie bei realer
  Hyperparameter-Optimierung?
- Schlägt der adaptive Mix das fest-gewählte 70/30 OOS?
- Bleibt p=0.97 vs 60/40 OOS bestehen?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402


def _calmar(ret: pd.Series) -> float:
    if ret.empty or ret.std() == 0:
        return -np.inf
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann / abs(dd) if dd != 0 else -np.inf


def main():
    eq_csv = Path("output/erweiterung_master_allocation_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found. Run run_master_allocation.py first.")
        return 1

    df = pd.read_csv(eq_csv)
    first = df.columns[0]
    df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    sa_ret = df["SingleAsset_VolTarget"].pct_change().dropna()
    xa_ret = df["CrossAsset_Hybrid"].pct_change().dropna()
    bench_60_40 = df["60_40_Classic"].pct_change().dropna()
    pure_mom = df["Pure_Mom_12_1_LO"].pct_change().dropna()

    aligned = pd.concat(
        {"sa": sa_ret, "xa": xa_ret, "bench": bench_60_40, "mom": pure_mom}, axis=1
    ).dropna()
    print(
        f"Aligned panel: {len(aligned)} days, "
        f"{aligned.index.min()} -> {aligned.index.max()}"
    )

    # Walk-Forward Parameters
    train_days = 504  # 2 years
    test_days = 126  # 6 months
    step = 126
    grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    rows = []
    oos_chunks: list[pd.Series] = []
    fixed_70_30_chunks: list[pd.Series] = []
    start = 0
    while start + train_days + test_days <= len(aligned):
        train_idx = aligned.index[start : start + train_days]
        test_idx = aligned.index[start + train_days : start + train_days + test_days]
        train_sa = aligned.loc[train_idx, "sa"]
        train_xa = aligned.loc[train_idx, "xa"]
        test_sa = aligned.loc[test_idx, "sa"]
        test_xa = aligned.loc[test_idx, "xa"]

        # Find best sa_weight on Train via Calmar
        best_w = grid[0]
        best_obj = -np.inf
        train_objs = {}
        for w in grid:
            mix = w * train_sa + (1 - w) * train_xa
            obj = _calmar(mix)
            train_objs[w] = obj
            if obj > best_obj:
                best_obj = obj
                best_w = w

        # Apply best_w to Test
        test_mix = best_w * test_sa + (1 - best_w) * test_xa
        oos_chunks.append(test_mix)

        # Fixed 70/30
        fixed_mix = 0.70 * test_sa + 0.30 * test_xa
        fixed_70_30_chunks.append(fixed_mix)

        t_ann = (1 + test_mix).prod() ** (252 / max(len(test_mix), 1)) - 1
        t_vol = test_mix.std() * np.sqrt(252)
        t_dd = ((1 + test_mix).cumprod() / (1 + test_mix).cumprod().cummax() - 1).min()
        rows.append(
            {
                "window_idx": len(rows),
                "train_start": train_idx[0],
                "test_start": test_idx[0],
                "best_sa_weight": float(best_w),
                "train_calmar": float(best_obj),
                "test_ann_return": float(t_ann),
                "test_realized_vol": float(t_vol),
                "test_max_dd": float(t_dd),
                "test_sharpe": float(t_ann / t_vol) if t_vol > 0 else 0,
            }
        )
        start += step

    wf = pd.DataFrame(rows)
    print(f"\n{len(wf)} Walk-Forward Windows:")
    print(
        wf[
            [
                "window_idx",
                "train_start",
                "test_start",
                "best_sa_weight",
                "train_calmar",
                "test_ann_return",
                "test_sharpe",
                "test_max_dd",
            ]
        ].to_string(index=False)
    )

    oos = pd.concat(oos_chunks).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]

    fixed_oos = pd.concat(fixed_70_30_chunks).sort_index()
    fixed_oos = fixed_oos[~fixed_oos.index.duplicated(keep="first")]

    print(f"\nOOS series: {len(oos)} days, {oos.index.min()} -> {oos.index.max()}")

    bench_oos = bench_60_40.loc[oos.index.min() : oos.index.max()]
    mom_oos = pure_mom.loc[oos.index.min() : oos.index.max()]

    print("\n" + "=" * 100)
    print("WALK-FORWARD OOS RESULTS (Master-Allocator)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    candidates = {
        "60_40_Classic (OOS)": bench_oos,
        "Pure_Mom_12_1_LO (OOS)": mom_oos,
        "Fixed_70_30_Master (OOS)": fixed_oos,
        "Adaptive_Master (WF-OOS)": oos,
    }
    metrics_dump = {}
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Calmar-Bootstrap-Tests
    print("\n" + "=" * 100)
    print("OOS CALMAR-BOOTSTRAP vs 60/40 Classic")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    baseline = bench_oos
    for name, ret in candidates.items():
        if "60_40" in name:
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(),
            baseline.dropna(),
            n_bootstrap=2000,
            avg_block_size=20,
            seed=42,
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci:>22} "
            f"{p_gt:>6.3f}"
        )

    print("\n" + "=" * 100)
    print("OOS CALMAR-BOOTSTRAP vs Fixed_70_30_Master")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    for name, ret in candidates.items():
        if "Fixed_70_30" in name:
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(),
            fixed_oos.dropna(),
            n_bootstrap=2000,
            avg_block_size=20,
            seed=42,
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci:>22} "
            f"{p_gt:>6.3f}"
        )

    # Train→Test-Stabilität: Korrelation Train-Calmar ↔ Test-Calmar
    print("\n" + "=" * 100)
    print("TRAIN -> TEST STABILITAET")
    print("=" * 100)
    wf["test_calmar"] = wf.apply(
        lambda r: (
            r["test_ann_return"] / abs(r["test_max_dd"]) if r["test_max_dd"] != 0 else 0
        ),
        axis=1,
    )
    if len(wf) > 3:
        corr = wf["train_calmar"].corr(wf["test_calmar"])
        print(f"Train-Calmar <-> Test-Calmar correlation: {corr:+.3f}")
        if corr > 0.3:
            print("  -> Training prädiktiv (positiv)")
        elif corr < -0.1:
            print("  -> Training anti-prädiktiv (Overfit-Warnung)")
        else:
            print("  -> Schwacher Train-Test-Zusammenhang")
    print(f"\nGewählte SA-Weights pro Window: {wf['best_sa_weight'].tolist()}")
    print(f"Verteilung: {wf['best_sa_weight'].value_counts().to_dict()}")

    # Save
    out_csv = pd.DataFrame(
        {"oos_return": oos, "oos_equity": (1 + oos.fillna(0)).cumprod()}
    )
    out_csv.to_csv("output/erweiterung_master_walk_forward_oos.csv")
    wf.to_csv("output/erweiterung_master_walk_forward_windows.csv", index=False)
    Path("output/erweiterung_master_walk_forward_summary.json").write_text(
        json.dumps(
            {
                "n_windows": int(len(wf)),
                "oos_days": int(len(oos)),
                "train_test_corr": (
                    float(wf["train_calmar"].corr(wf["test_calmar"]))
                    if len(wf) > 3
                    else None
                ),
                "weight_distribution": wf["best_sa_weight"].value_counts().to_dict(),
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
                    for name, m in metrics_dump.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_walk_forward_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
