#!/usr/bin/env python
"""Master-Allocation: kombiniere beste Single-Asset-VolTarget mit Cross-Asset-Hybrid.

Hypothese
---------
Die zwei besten Bausteine aus der bisherigen Forschung sind:

1. **Single-Asset Vol-Target Mom-12/1 LO** (22 Mega-Caps, 19y):
   Sharpe 1.46, MDD -15%. Aber: Single-Asset-Equity-Konzentration.

2. **Cross-Asset Hybrid VT+Mom** (11 ETFs, 5.3y):
   Sharpe 1.01, MDD -17%, Sortino 1.32. Echte Asset-Diversifikation.

Ein 50/50-Mix sollte:
- Equity-Faktor-Exposure behalten (Mom-Premie)
- Cross-Asset-Diversifikation hinzufügen
- MDD durch Korrelations-Mismatch weiter reduzieren

Test
----
Calmar-Bootstrap vs aller Einzel-Strategien. Common period nötig.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.altdata.yfinance_cache_loader import load_universe_panel  # noqa: E402
from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.volatility_targeting import (  # noqa: E402
    VolTargetConfig,
    apply_vol_targeting,
)

CROSS_ASSET_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "TLT",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
]


def main():
    cache_dir = "data/cache/yfinance"

    # Lade Cross-Asset
    panel = load_universe_panel(
        cache_dir,
        CROSS_ASSET_UNIVERSE,
        require_min_rows=200,
        skip_missing=False,
    )
    wide = panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    xa_rets = wide.pct_change().dropna()

    # Cross-Asset Hybrid: 50% VolTarget-EW + 50% XAsset-Mom-Top5
    ew = xa_rets.mean(axis=1)
    vt = apply_vol_targeting(ew, VolTargetConfig(target_vol_annual=0.10))
    vt_ew = vt["scaled_return"]

    # XAsset-Mom-Top5
    mom_12_1 = xa_rets.rolling(252, min_periods=200).apply(
        lambda x: (1 + x[:-21]).prod() - 1 if len(x) > 21 else np.nan, raw=False
    )
    daily_idx = xa_rets.index
    monthly_rebal = daily_idx[daily_idx.is_month_end | (daily_idx == daily_idx[-1])]
    cmom_returns = pd.Series(0.0, index=daily_idx)
    cur_weights = pd.Series(0.0, index=xa_rets.columns)
    for d in daily_idx:
        if d in monthly_rebal:
            mom_today = mom_12_1.loc[d].dropna()
            if len(mom_today) >= 5:
                top5 = mom_today.nlargest(5).index
                cur_weights = pd.Series(0.0, index=xa_rets.columns)
                cur_weights[top5] = 1.0 / 5.0
        if cur_weights.sum() > 0:
            cmom_returns.loc[d] = (xa_rets.loc[d] * cur_weights).sum()

    aligned_xa = pd.concat({"vt": vt_ew, "mom": cmom_returns}, axis=1).dropna()
    xa_hybrid = 0.5 * aligned_xa["vt"] + 0.5 * aligned_xa["mom"]

    # Lade Single-Asset Equity (22 Mega-Caps, long-history)
    eq_csv = Path("output/erweiterung_long_history_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found.")
        return 1
    eqdf = pd.read_csv(eq_csv)
    eqdf = eqdf.rename(columns={eqdf.columns[0]: "date"})
    eqdf["date"] = pd.to_datetime(eqdf["date"], utc=True)
    eqdf = eqdf.set_index("date").sort_index()
    pure_mom_19y = eqdf["momentum_12_1_LongOnly"].pct_change().dropna()
    eq_ew_19y = eqdf["benchmark_equal_weight"].pct_change().dropna()

    # Single-Asset VolTarget
    sa_vt = apply_vol_targeting(pure_mom_19y, VolTargetConfig(target_vol_annual=0.12))[
        "scaled_return"
    ]

    # 60/40 Classic
    classic_60_40 = 0.60 * xa_rets["SPY"] + 0.40 * xa_rets["AGG"]

    # Master-Allocations: verschiedene Mischungsverhältnisse
    aligned_master = pd.concat(
        {
            "sa_vt": sa_vt,
            "xa_hybrid": xa_hybrid,
            "pure_mom_19y": pure_mom_19y,
            "60_40": classic_60_40,
            "eq_ew": eq_ew_19y,
        },
        axis=1,
    ).dropna()
    print(
        f"Common period: {aligned_master.index.min()} -> {aligned_master.index.max()} "
        f"({len(aligned_master)} days)"
    )

    candidates: dict[str, pd.Series] = {
        "60_40_Classic": aligned_master["60_40"],
        "Pure_Mom_12_1_LO": aligned_master["pure_mom_19y"],
        "Equity_EW": aligned_master["eq_ew"],
        "SingleAsset_VolTarget": aligned_master["sa_vt"],
        "CrossAsset_Hybrid": aligned_master["xa_hybrid"],
    }

    # Master-Mixes
    for ratio_label, w_sa in [
        ("Master_70_30", 0.70),
        ("Master_50_50", 0.50),
        ("Master_30_70", 0.30),
    ]:
        candidates[ratio_label] = (
            w_sa * aligned_master["sa_vt"] + (1 - w_sa) * aligned_master["xa_hybrid"]
        )

    # Performance
    print("\n" + "=" * 100)
    print("MASTER ALLOCATION PERFORMANCE (common period)")
    print("=" * 100)
    print(
        f"{'Strategy':<28} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metrics_dump = {}
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<26} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Calmar-Bootstrap vs 60/40 + Pure-Mom
    for baseline_name in ("60_40_Classic", "Pure_Mom_12_1_LO"):
        print(f"\n{'=' * 100}")
        print(f"CALMAR-BOOTSTRAP vs {baseline_name}")
        print("=" * 100)
        print(
            f"{'Challenger':<28} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
        )
        print("-" * 100)
        baseline = candidates[baseline_name]
        for name, ret in candidates.items():
            if name == baseline_name:
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
                f"  {name:<26} "
                f"{out['observed_diff']:>+8.3f} "
                f"{out['mean_diff']:>+9.3f} "
                f"{ci:>22} "
                f"{p_gt:>6.3f}"
            )

    # Korrelations-Matrix der Bausteine
    print("\n" + "=" * 100)
    print("CORRELATION MATRIX (master building blocks)")
    print("=" * 100)
    corr_df = pd.DataFrame(
        {
            "SA_VolTarget": aligned_master["sa_vt"],
            "XA_Hybrid": aligned_master["xa_hybrid"],
            "Pure_Mom_LO": aligned_master["pure_mom_19y"],
            "60_40": aligned_master["60_40"],
        }
    ).corr()
    print(corr_df.round(3).to_string())

    # Sub-Period
    print("\n" + "=" * 100)
    print("SUB-PERIOD: Inflation_2022 (key stress test)")
    print("=" * 100)
    inf_start = pd.Timestamp("2022-01-01", tz="UTC")
    inf_end = pd.Timestamp("2022-12-31", tz="UTC")
    print(f"{'Strategy':<28} {'AnnRet':>9} {'Sharpe':>8} {'MDD':>8} {'Days':>5}")
    for name, ret in candidates.items():
        sub = ret[(ret.index >= inf_start) & (ret.index <= inf_end)].dropna()
        if len(sub) < 30:
            continue
        m = all_metrics(sub)
        print(
            f"  {name:<26} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%} "
            f"{len(sub):>5d}"
        )

    # Save
    eq_csv_out = pd.DataFrame(
        {k: (1 + v.fillna(0)).cumprod() for k, v in candidates.items()}
    )
    eq_csv_out.to_csv("output/erweiterung_master_allocation_equity.csv")
    Path("output/erweiterung_master_allocation_summary.json").write_text(
        json.dumps(
            {
                "common_period_days": int(len(aligned_master)),
                "metrics": {
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
                "correlation_matrix": corr_df.to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_allocation_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
