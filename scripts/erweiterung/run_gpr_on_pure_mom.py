#!/usr/bin/env python
"""Test GPR-Overlay auf NICHT-Vol-Targeted Pure-Mom-12/1-LO.

Hypothese: Mainline-Style Strategies haben KEIN Vol-Target eingebaut. Auf
solche sollte GPR-Overlay echten Edge bringen (Reduktion in Tail-Events).

Test: Pure-Mom-12/1 vs Pure-Mom + GPR-Overlay auf 19y.
"""

from __future__ import annotations

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
)
from erweiterung.strategies.cross_section_helpers import (  # noqa: E402
    cs_long_only_wide,
    long_format_to_wide,
)


def main():
    print("=" * 100)
    print("GPR-OVERLAY auf Pure-Mom-12/1-LO (NICHT-Vol-Targeted, Mainline-Style)")
    print("=" * 100)

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

    mom_wide = long_format_to_wide(
        eq_panel[["date", "symbol", "mom_12_1"]], "mom_12_1"
    )
    ret_wide = long_format_to_wide(
        eq_panel[["date", "symbol", "return"]], "return"
    ).fillna(0)
    pure_mom_ret, _ = cs_long_only_wide(mom_wide, ret_wide, quantile=0.3, lag_days=1)
    pure_mom_ret = pure_mom_ret.dropna()
    print(f"Pure-Mom: {len(pure_mom_ret)} days")

    # Apply GPR-Overlay
    pl_out = apply_gpr_overlay(pure_mom_ret, GPROverlayPolicy(enabled=True))
    pure_mom_with_gpr = pl_out["hedged_return"]
    print(f"GPR multiplier stats: median={pl_out['exposure_multiplier'].median():.3f}, "
          f"share<1.0={(pl_out['exposure_multiplier'] < 1.0).mean():.1%}")

    # Comparison
    print("\n" + "=" * 100)
    print("PURE-MOM vs PURE-MOM + GPR-OVERLAY")
    print("=" * 100)
    print(f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}")
    print("-" * 100)
    for label, r in [("Pure-Mom-12/1 (no overlay)", pure_mom_ret),
                     ("Pure-Mom + GPR-Overlay", pure_mom_with_gpr)]:
        m = all_metrics(r.dropna())
        print(f"  {label:<30} "
              f"{m.get('annualized_return', 0):>+8.2%} "
              f"{m.get('sharpe', 0):>+7.3f} "
              f"{m.get('sortino', 0):>+8.3f} "
              f"{m.get('calmar', 0):>+7.3f} "
              f"{m.get('max_drawdown', 0):>+7.2%}")

    # Calmar-Bootstrap GPR vs no-overlay
    out = calmar_diff_bootstrap(
        pure_mom_with_gpr.dropna(), pure_mom_ret.dropna(),
        n_bootstrap=2000, avg_block_size=20, seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(f"\nCalmar-Bootstrap (GPR vs no-overlay): obs_diff={out['observed_diff']:+.3f}, "
              f"95% CI {ci}, p(>0)={p_gt:.3f}")

    # Sub-period
    print("\n" + "=" * 100)
    print("SUB-PERIODS (Tail-Events)")
    print("=" * 100)
    periods = [
        ("Sept_2001 (9/11)", "2001-09-01", "2002-03-31"),
        ("GFC_2008", "2008-09-01", "2009-06-30"),
        ("COVID_2020", "2020-02-15", "2020-05-31"),
        ("Ukraine_2022", "2022-02-15", "2022-12-31"),
    ]
    print(f"{'Period':<22} {'PM AnnRet':>11} {'PM+GPR':>10} {'PM MDD':>9} {'PM+GPR MDD':>11} {'dMDD':>9}")
    print("-" * 100)
    for label, start, end in periods:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        sub_pm = pure_mom_ret.loc[s:e].dropna()
        sub_gpr = pure_mom_with_gpr.loc[s:e].dropna()
        if len(sub_pm) < 10:
            continue
        m1 = all_metrics(sub_pm)
        m2 = all_metrics(sub_gpr)
        delta_mdd = m2["max_drawdown"] - m1["max_drawdown"]
        print(f"  {label:<20} "
              f"{m1.get('annualized_return', 0):>+10.2%} "
              f"{m2.get('annualized_return', 0):>+9.2%} "
              f"{m1.get('max_drawdown', 0):>+8.2%} "
              f"{m2.get('max_drawdown', 0):>+10.2%} "
              f"{delta_mdd:>+8.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
