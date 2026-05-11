#!/usr/bin/env python
"""Walk-Forward OOS für Multi-Factor-Vol-Target.

Pipeline
--------
1. Für jedes Train-Window: bestimme optimalen Combiner (EW/InvVol/HRP)
   und Target-Vol via Calmar-Maximization.
2. Wende auf nächstes Test-Window an (OOS).
3. Vergleiche mit Pure-Mom, Equal-Weight, Single-VolTarget.
4. Calmar-Bootstrap auf OOS-Ergebnis.
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
from erweiterung.strategies.multi_factor_vol_target import (  # noqa: E402
    MultiFactorVolTargetConfig,
    combine_factors,
)
from erweiterung.strategies.volatility_targeting import (  # noqa: E402
    VolTargetConfig,
    apply_vol_targeting,
)


def _calmar(ret: pd.Series) -> float:
    if ret.empty or ret.std() == 0:
        return -np.inf
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann / abs(dd) if dd != 0 else -np.inf


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

    factor_returns = {
        "mom_12_1": df["momentum_12_1_LongOnly"].pct_change().dropna(),
        "res_mom": df["residual_momentum_LongOnly"].pct_change().dropna(),
        "low_vol": df["low_vol_LongOnly"].pct_change().dropna(),
    }
    bench_ret = df["benchmark_equal_weight"].pct_change().dropna()
    pure_mom = factor_returns["mom_12_1"]

    # Align all factors on common index
    aligned = pd.concat(factor_returns, axis=1).dropna()
    n = len(aligned)
    train_days = 1260
    test_days = 252
    step = 252

    # Walk-Forward
    rows = []
    chunks: list[pd.Series] = []
    target_grid = [0.08, 0.10, 0.12, 0.15, 0.18]
    combiner_grid = ["equal_weight", "inverse_vol", "hrp"]

    start = 0
    while start + train_days + test_days <= n:
        train_idx = aligned.index[start : start + train_days]
        test_idx = aligned.index[start + train_days : start + train_days + test_days]
        train_f = {k: v.loc[train_idx] for k, v in factor_returns.items()}
        combined_period = aligned.iloc[max(0, start) : start + train_days + test_days]
        combined_f = {k: combined_period[k] for k in factor_returns}

        # Grid-Suche
        best_obj = -np.inf
        best_target = target_grid[0]
        best_combiner = combiner_grid[0]
        for tv in target_grid:
            for comb in combiner_grid:
                cfg = MultiFactorVolTargetConfig(
                    target_vol_annual=tv, combiner=comb, smoothing_window=5
                )
                out = combine_factors(train_f, cfg)
                ret = out["combined"].dropna()
                obj = _calmar(ret)
                if obj > best_obj:
                    best_obj = obj
                    best_target = tv
                    best_combiner = comb

        # Apply auf full period inkl. warmup
        cfg_best = MultiFactorVolTargetConfig(
            target_vol_annual=best_target, combiner=best_combiner
        )
        out_full = combine_factors(combined_f, cfg_best)
        test_ret = out_full["combined"].loc[test_idx].dropna()
        chunks.append(test_ret)

        t_ann = (1 + test_ret).prod() ** (252 / max(len(test_ret), 1)) - 1
        t_vol = test_ret.std() * np.sqrt(252)
        t_dd = ((1 + test_ret).cumprod() / (1 + test_ret).cumprod().cummax() - 1).min()
        rows.append(
            {
                "window_idx": len(rows),
                "train_start": train_idx[0],
                "test_start": test_idx[0],
                "best_target_vol": best_target,
                "best_combiner": best_combiner,
                "train_calmar": best_obj,
                "test_ann_return": t_ann,
                "test_realized_vol": t_vol,
                "test_sharpe": t_ann / t_vol if t_vol > 0 else 0,
                "test_max_dd": t_dd,
            }
        )
        start += step

    wf = pd.DataFrame(rows)
    print(f"\n{len(wf)} Walk-Forward Windows:")
    print(
        wf[
            [
                "window_idx",
                "test_start",
                "best_target_vol",
                "best_combiner",
                "train_calmar",
                "test_ann_return",
                "test_sharpe",
                "test_max_dd",
            ]
        ].to_string(index=False)
    )

    oos = pd.concat(chunks).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]

    # OOS-Vergleich
    bench_oos = bench_ret.loc[oos.index.min() : oos.index.max()]
    pure_mom_oos = pure_mom.loc[oos.index.min() : oos.index.max()]

    # Single-VolTarget für Vergleich (in-sample auf gleichem Range)
    single_vt = apply_vol_targeting(pure_mom, VolTargetConfig(target_vol_annual=0.12))
    single_vt_oos = (
        single_vt["scaled_return"].loc[oos.index.min() : oos.index.max()].dropna()
    )

    # Walk-Forward-Switch zum Vergleich
    wf_switch_csv = Path("output/erweiterung_walk_forward_oos_equity.csv")
    switch_oos = None
    if wf_switch_csv.exists():
        s = pd.read_csv(wf_switch_csv, index_col=0)
        s.index = pd.to_datetime(s.index, utc=True)
        switch_oos = s["oos_return"].loc[oos.index.min() : oos.index.max()]

    candidates: dict[str, pd.Series] = {
        "Pure Equal-Weight (OOS)": bench_oos,
        "Pure Mom-12/1 LO (OOS)": pure_mom_oos,
        "Single-VolTarget Mom (OOS)": single_vt_oos,
        "MultiFac VolTarget OOS": oos,
    }
    if switch_oos is not None:
        candidates["Walk-Forward Switch (OOS)"] = switch_oos

    print("\n" + "=" * 100)
    print("OOS PERFORMANCE COMPARISON")
    print("=" * 100)
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

    # Calmar-Bootstrap vs Pure Mom
    print("\n" + "=" * 100)
    print("OOS CALMAR-BOOTSTRAP vs Pure Mom-12/1 LO")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    baseline_oos = pure_mom_oos
    for name, ret in candidates.items():
        if "Pure Mom-12/1" in name:
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(), baseline_oos, n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci_str = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci_str:>22} "
            f"{p_gt:>6.3f}"
        )

    # Calmar-Bootstrap vs Pure Equal-Weight
    print("\n" + "=" * 100)
    print("OOS CALMAR-BOOTSTRAP vs Pure Equal-Weight")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    for name, ret in candidates.items():
        if "Equal-Weight" in name:
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(), bench_oos, n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci_str = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci_str:>22} "
            f"{p_gt:>6.3f}"
        )

    # Threshold-Verteilung
    print("\nCombiner-Verteilung im Walk-Forward:")
    print(wf.groupby(["best_combiner", "best_target_vol"]).size().to_string())

    # Save
    pd.DataFrame(
        {"oos_return": oos, "oos_equity": (1 + oos.fillna(0)).cumprod()}
    ).to_csv("output/erweiterung_multi_factor_walk_forward_oos.csv")
    wf.to_csv("output/erweiterung_multi_factor_walk_forward_windows.csv", index=False)
    Path("output/erweiterung_multi_factor_walk_forward_summary.json").write_text(
        json.dumps(
            {
                "n_windows": int(len(wf)),
                "oos_days": int(len(oos)),
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
    print("\nSaved -> output/erweiterung_multi_factor_walk_forward_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
