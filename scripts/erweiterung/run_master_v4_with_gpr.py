#!/usr/bin/env python
"""Master V4 = Master V1 + Caldara-Iacoviello GPR-Overlay (echte Multi-Decade-Daten).

Test
----
1. Pure Master V1 (70/30, Mom-12/1) — Baseline
2. Master V4 = V1 × GPR-Exposure-Multiplier (PAUSE/ACTIVE/WATCH/COOLDOWN)
3. Vergleich auf 19y mit echten GPR-Daten (1900-2026, monthly ffilled daily)

Calmar-Bootstrap: V4 vs V1, V4 vs 60/40.
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
from erweiterung.risk.gpr_overlay import (  # noqa: E402
    GPROverlayPolicy,
    apply_gpr_overlay,
    build_daily_gpr_overlay_series,
)
from erweiterung.strategies.cross_section_helpers import (  # noqa: E402
    cs_long_only_wide,
    long_format_to_wide,
)
from erweiterung.strategies.master_allocator import (  # noqa: E402
    MasterAllocator,
    MasterAllocatorConfig,
)


def _build_eq_factor():
    eq_panel = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})
    eq_panel["date"] = pd.to_datetime(eq_panel["date"], utc=True)
    eq_panel = eq_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq_panel["return"] = eq_panel.groupby("symbol")["close"].pct_change()
    mom = momentum_12_1(eq_panel[["date", "symbol", "close"]])
    eq_panel = eq_panel.set_index(["date", "symbol"])
    eq_panel["mom_12_1"] = mom.reindex(eq_panel.index)
    eq_panel = eq_panel.reset_index().dropna(subset=["mom_12_1", "return"])

    mom_wide = long_format_to_wide(eq_panel[["date", "symbol", "mom_12_1"]], "mom_12_1")
    ret_wide = long_format_to_wide(eq_panel[["date", "symbol", "return"]], "return").fillna(0)
    pnl, _ = cs_long_only_wide(mom_wide, ret_wide, quantile=0.3, lag_days=1)
    return pnl.dropna()


def _load_xa():
    frames = []
    for sym in ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "GLD", "SLV", "DBC"]:
        p = Path("data/cache/yfinance_long") / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = sym
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    wide = panel.pivot_table(index="date", columns="symbol", values="close").sort_index()
    return wide.pct_change().dropna()


def main():
    print("=" * 100)
    print("MASTER V4 — Master V1 + Caldara-Iacoviello GPR-Overlay")
    print("=" * 100)

    eq_factor_ret = _build_eq_factor()
    xa_rets = _load_xa()
    print(f"Equity-Faktor: {len(eq_factor_ret)} days")
    print(f"Cross-Asset:   {len(xa_rets)} days")

    # Master V1
    alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out = alloc.allocate(eq_factor_ret, xa_rets)
    master_v1 = out["master_return"].dropna()
    print(f"Master V1: {len(master_v1)} days")

    # GPR-Overlay
    print("\nGPR-Overlay-Distribution (state-shares):")
    overlay = build_daily_gpr_overlay_series(
        master_v1.index, GPROverlayPolicy(enabled=True)
    )
    state_counts = overlay["state_hint"].value_counts(normalize=True)
    for state, pct in state_counts.items():
        print(f"  {state:<10} {pct:.1%}")

    # Master V4: V1 × GPR-Multiplier (t-1 lag)
    pl_out = apply_gpr_overlay(master_v1, GPROverlayPolicy(enabled=True))
    master_v4 = pl_out["hedged_return"]
    print(f"\nMaster V4: {len(master_v4)} days, multiplier-stats:")
    print(f"  median={pl_out['exposure_multiplier'].median():.3f}")
    print(f"  min={pl_out['exposure_multiplier'].min():.3f}")
    print(f"  share<1.0: {(pl_out['exposure_multiplier'] < 1.0).mean():.1%}")

    # 60/40
    classic = (0.60 * xa_rets["SPY"] + 0.40 * xa_rets["AGG"]).loc[
        master_v1.index.min():master_v1.index.max()
    ]

    # Performance
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}")
    print("-" * 100)
    candidates = {
        "60_40_Classic": classic,
        "Master_V1 (no overlay)": master_v1,
        "Master_V4 (V1 + GPR)": master_v4,
    }
    metrics_dump = {}
    for name, r in candidates.items():
        m = all_metrics(r.dropna())
        metrics_dump[name] = m
        print(f"  {name:<30} "
              f"{m.get('annualized_return', 0):>+8.2%} "
              f"{m.get('sharpe', 0):>+7.3f} "
              f"{m.get('sortino', 0):>+8.3f} "
              f"{m.get('calmar', 0):>+7.3f} "
              f"{m.get('max_drawdown', 0):>+7.2%}")

    # Calmar-Bootstrap
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP")
    print("=" * 100)
    out_v4_vs_v1 = calmar_diff_bootstrap(master_v4.dropna(), master_v1.dropna(),
                                        n_bootstrap=2000, avg_block_size=20, seed=42)
    out_v4_vs_60 = calmar_diff_bootstrap(master_v4.dropna(), classic.dropna(),
                                        n_bootstrap=2000, avg_block_size=20, seed=42)
    out_v1_vs_60 = calmar_diff_bootstrap(master_v1.dropna(), classic.dropna(),
                                        n_bootstrap=2000, avg_block_size=20, seed=42)
    for label, o in [("V1 vs 60/40", out_v1_vs_60), ("V4 vs 60/40", out_v4_vs_60),
                     ("V4 vs V1", out_v4_vs_v1)]:
        if "error" in o:
            continue
        p_gt = 1.0 - o["p_value_one_sided_greater"]
        ci = f"[{o['ci_low_2.5']:+.2f}, {o['ci_high_97.5']:+.2f}]"
        print(f"  {label:<20} obs_diff={o['observed_diff']:+.3f}, 95% CI {ci}, p(>0)={p_gt:.3f}")

    # Sub-Period Analysis (GFC, 9/11 etc.)
    print("\n" + "=" * 100)
    print("SUB-PERIODS")
    print("=" * 100)
    periods = [
        ("Sept_2001 (9/11)", "2001-09-01", "2002-03-31"),
        ("GFC_2008", "2008-09-01", "2009-06-30"),
        ("COVID_2020", "2020-02-15", "2020-05-31"),
        ("Ukraine_2022", "2022-02-15", "2022-12-31"),
    ]
    print(f"{'Period':<22} {'V1 AnnRet':>11} {'V4 AnnRet':>11} {'V1 MDD':>9} {'V4 MDD':>9} {'V4-V1 dMDD':>11}")
    print("-" * 100)
    for label, start, end in periods:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        sub_v1 = master_v1.loc[s:e].dropna()
        sub_v4 = master_v4.loc[s:e].dropna()
        if len(sub_v1) < 10:
            continue
        m1 = all_metrics(sub_v1)
        m4 = all_metrics(sub_v4)
        delta_mdd = m4["max_drawdown"] - m1["max_drawdown"]
        print(f"  {label:<20} "
              f"{m1.get('annualized_return', 0):>+10.2%} "
              f"{m4.get('annualized_return', 0):>+10.2%} "
              f"{m1.get('max_drawdown', 0):>+8.2%} "
              f"{m4.get('max_drawdown', 0):>+8.2%} "
              f"{delta_mdd:>+10.2%}")

    # Save
    out_df = pd.DataFrame({
        "master_v1": master_v1,
        "master_v4_with_gpr": master_v4,
        "gpr_multiplier": pl_out["exposure_multiplier"],
        "gpr_state": pl_out["state_hint"],
    })
    out_df.to_csv("output/erweiterung_master_v4_gpr_equity.csv")
    Path("output/erweiterung_master_v4_gpr_summary.json").write_text(
        json.dumps(
            {"metrics": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                            for kk, vv in m.items() if not isinstance(vv, (pd.Series, pd.DataFrame))}
                         for k, m in metrics_dump.items()},
             "gpr_state_distribution": state_counts.to_dict()},
            indent=2, default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_v4_gpr_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
