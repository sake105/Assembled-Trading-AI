#!/usr/bin/env python
"""Master V3 — Adaptive SA-Weight gesteuert durch Intermarket-Macro-Stress.

Idee
----
Master V2 hat statisches 70/30-Mix. V3 macht den Mix **adaptiv**:
- Bei niedrigem Macro-Stress (Risk-On): erhöhe SA-Equity-Weight (z.B. 80/20)
- Bei hohem Macro-Stress: reduziere SA-Weight (z.B. 50/50)

Stress-Indikator: erweiterung.strategies.intermarket_macro_factors.macro_stress_composite_score
Gemessen aus den 11 Cross-Asset-ETFs (TLT/SPY/HYG/AGG/GLD).

Test: ist adaptiv signifikant besser als statisch 70/30?
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
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.strategies.intermarket_macro_factors import (  # noqa: E402
    build_intermarket_panel,
    macro_stress_composite_score,
)
from erweiterung.strategies.master_allocator import (  # noqa: E402
    cross_asset_hybrid,
    vol_target_single_asset,
)


def _cs_long_only(panel, signal_col, quantile=0.3):
    out = panel.copy().sort_values(["symbol", "date"])
    out["sig_lag"] = out.groupby("symbol", group_keys=False)[signal_col].shift(1)
    by_d = out.groupby("date")["sig_lag"]
    out["sig_pct"] = by_d.rank(pct=True)
    out["position"] = 0.0
    out.loc[out["sig_pct"] >= 1 - quantile, "position"] = 1.0
    n_long = out.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    long_mask = out["position"] > 0
    out.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    out["pnl"] = out["position"] * out["return"]
    return out


def main():
    # === Equity Factor (Mom-12/1) ===
    eq_panel = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})
    eq_panel["date"] = pd.to_datetime(eq_panel["date"], utc=True)
    eq_panel = eq_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq_panel["return"] = eq_panel.groupby("symbol")["close"].pct_change()
    mom = momentum_12_1(eq_panel[["date", "symbol", "close"]])
    eq_panel = eq_panel.set_index(["date", "symbol"])
    eq_panel["mom_12_1"] = mom.reindex(eq_panel.index)
    eq_panel = eq_panel.reset_index()
    eq_factor = _cs_long_only(
        eq_panel.dropna(subset=["mom_12_1"]), "mom_12_1", quantile=0.3
    )
    eq_factor_ret = eq_factor.groupby("date").agg(pnl=("pnl", "sum"))["pnl"]
    eq_factor_ret.index = pd.to_datetime(eq_factor_ret.index, utc=True)

    # === Cross-Asset ===
    xa_frames = []
    for sym in [
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
    ]:
        p = Path("data/cache/yfinance_long") / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = sym
        xa_frames.append(df)
    xa_panel = pd.concat(xa_frames, ignore_index=True)
    xa_panel["date"] = pd.to_datetime(xa_panel["date"], utc=True)
    xa_wide = xa_panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    xa_rets = xa_wide.pct_change().dropna()

    # === Components ===
    sa_vt = vol_target_single_asset(eq_factor_ret)
    xa_hyb = cross_asset_hybrid(xa_rets)

    # === Intermarket Macro Stress ===
    intermarket = build_intermarket_panel(xa_wide)
    stress_score = macro_stress_composite_score(intermarket, percentile_window=252)
    print(
        f"Stress-Score: {stress_score.notna().sum()} valid days, "
        f"mean={stress_score.mean():.3f}, max={stress_score.max():.3f}"
    )

    # === Adaptive sa_weight ===
    # Stress=0 -> sa_weight=0.85, Stress=1 -> sa_weight=0.50
    # Linear interpolation
    sa_weight_adaptive = 0.85 - 0.35 * stress_score.fillna(0.5)
    sa_weight_adaptive = sa_weight_adaptive.shift(1)  # t-1 lag

    aligned = pd.concat(
        {
            "sa": sa_vt,
            "xa": xa_hyb,
            "w": sa_weight_adaptive,
        },
        axis=1,
    ).dropna()
    print(
        f"Aligned: {len(aligned)} days, "
        f"sa_weight range: [{aligned['w'].min():.2f}, {aligned['w'].max():.2f}], "
        f"mean: {aligned['w'].mean():.2f}"
    )

    master_v1 = 0.70 * aligned["sa"] + 0.30 * aligned["xa"]
    master_v3 = aligned["w"] * aligned["sa"] + (1 - aligned["w"]) * aligned["xa"]

    # 60/40
    classic = (0.60 * xa_rets["SPY"] + 0.40 * xa_rets["AGG"]).loc[
        master_v1.index.min() : master_v1.index.max()
    ]

    print("\n" + "=" * 100)
    print("MASTER V3 ADAPTIVE vs V1 STATIC vs 60/40")
    print("=" * 100)
    print(
        f"{'Strategy':<36} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    candidates = {
        "60_40_Classic": classic,
        "Master_V1 (static 70/30)": master_v1,
        "Master_V3 (adaptive 50-85%)": master_v3,
    }
    metrics_dump = {}
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<34} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    print("\nCalmar-Bootstrap V3 vs V1:")
    out = calmar_diff_bootstrap(
        master_v3.dropna(),
        master_v1.dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(f"  obs_diff={out['observed_diff']:+.3f}, 95% CI {ci}, p(>0)={p_gt:.3f}")

    print("\nCalmar-Bootstrap V3 vs 60/40:")
    out = calmar_diff_bootstrap(
        master_v3.dropna(),
        classic.dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(f"  obs_diff={out['observed_diff']:+.3f}, 95% CI {ci}, p(>0)={p_gt:.3f}")

    # Per-Regime-Performance
    print("\n" + "=" * 100)
    print("PER-STRESS-LEVEL Performance")
    print("=" * 100)
    stress_lag = stress_score.shift(1).reindex(master_v3.index)
    stress_bin = pd.cut(
        stress_lag,
        bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["low", "med-low", "med-high", "high"],
    )
    for level in ["low", "med-low", "med-high", "high"]:
        mask = stress_bin == level
        sub_v1 = master_v1.loc[mask].dropna()
        sub_v3 = master_v3.loc[mask].dropna()
        if len(sub_v1) < 30 or len(sub_v3) < 30:
            continue
        m1 = all_metrics(sub_v1)
        m3 = all_metrics(sub_v3)
        print(
            f"  Stress={level}: V1 Sharpe={m1.get('sharpe', 0):+.3f}, "
            f"V3 Sharpe={m3.get('sharpe', 0):+.3f}, days={len(sub_v3)}"
        )

    # Save
    out_df = pd.DataFrame(
        {
            "master_v1": master_v1,
            "master_v3_adaptive": master_v3,
            "adaptive_sa_weight": aligned["w"],
            "stress_score": stress_score.reindex(master_v3.index),
        }
    )
    out_df.to_csv("output/erweiterung_master_v3_adaptive_equity.csv")
    Path("output/erweiterung_master_v3_summary.json").write_text(
        json.dumps(
            {
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
                }
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_v3_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
