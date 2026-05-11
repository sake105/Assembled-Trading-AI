#!/usr/bin/env python
"""Test Composite-Geo-Stress-Overlay (GPR + GDELT) auf Pure-Mom-12/1-LO.

Hypothese: Composite besser als GPR-Solo, weil:
- GPR fängt Threat-Rhetorik (slow)
- GDELT fängt Conflict-Events (fast)
- Tone-Inversion fängt News-Sentiment-Crashes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.risk.geo_stress_composite import (  # noqa: E402
    GeoStressPolicy,
    apply_geo_stress_overlay,
)
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
    print("COMPOSITE GEO-STRESS-OVERLAY (GPR + GDELT) auf Pure-Mom-12/1-LO")
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

    mom_wide = long_format_to_wide(eq_panel[["date", "symbol", "mom_12_1"]], "mom_12_1")
    ret_wide = long_format_to_wide(
        eq_panel[["date", "symbol", "return"]], "return"
    ).fillna(0)
    pure_mom_ret, _ = cs_long_only_wide(mom_wide, ret_wide, quantile=0.3, lag_days=1)
    pure_mom_ret = pure_mom_ret.dropna()
    print(
        f"Pure-Mom returns: {len(pure_mom_ret)} days, "
        f"{pure_mom_ret.index.min().date()} to {pure_mom_ret.index.max().date()}"
    )

    # Restrict to GDELT coverage (2013+)
    pure_mom_ret_2013 = pure_mom_ret.loc[
        pure_mom_ret.index >= pd.Timestamp("2013-04-15", tz="UTC")
    ]
    print(f"Pure-Mom GDELT-window: {len(pure_mom_ret_2013)} days")

    # Apply GPR-only (full 19y window)
    pl_gpr = apply_gpr_overlay(pure_mom_ret, GPROverlayPolicy(enabled=True))
    pm_gpr = pl_gpr["hedged_return"]

    # Apply Composite (GDELT window only)
    pl_comp = apply_geo_stress_overlay(pure_mom_ret_2013, GeoStressPolicy(enabled=True))
    pm_comp = pl_comp["hedged_return"]
    state_dist = pl_comp["state_series"].value_counts()
    print(f"Composite state distribution: {dict(state_dist)}")

    # Restrict GPR to same window for apples-to-apples
    pm_gpr_2013 = pm_gpr.loc[pure_mom_ret_2013.index]
    pm_base_2013 = pure_mom_ret_2013

    print("\n" + "=" * 100)
    print("METRICS (2013-04-15 onwards, GDELT-covered window)")
    print("=" * 100)
    header = (
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} "
        f"{'Sortino':>9} {'Calmar':>8} {'MDD':>9}"
    )
    print(header)
    print("-" * 100)
    for label, r in [
        ("Pure-Mom (no overlay)", pm_base_2013),
        ("Pure-Mom + GPR-Solo", pm_gpr_2013),
        ("Pure-Mom + Composite", pm_comp),
    ]:
        m = all_metrics(r.dropna())
        print(
            f"  {label:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+8.2%}"
        )

    # Calmar-Bootstrap: Composite vs no-overlay
    out_comp = calmar_diff_bootstrap(
        pm_comp.dropna(),
        pm_base_2013.dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out_comp:
        p_gt = 1.0 - out_comp["p_value_one_sided_greater"]
        ci = f"[{out_comp['ci_low_2.5']:+.2f}, {out_comp['ci_high_97.5']:+.2f}]"
        print(
            f"\nComposite vs no-overlay: obs_diff={out_comp['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Calmar-Bootstrap: Composite vs GPR-Solo
    out_vs_gpr = calmar_diff_bootstrap(
        pm_comp.dropna(),
        pm_gpr_2013.dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out_vs_gpr:
        p_gt = 1.0 - out_vs_gpr["p_value_one_sided_greater"]
        ci = f"[{out_vs_gpr['ci_low_2.5']:+.2f}, {out_vs_gpr['ci_high_97.5']:+.2f}]"
        print(
            f"Composite vs GPR-Solo:    obs_diff={out_vs_gpr['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Sub-period analysis
    print("\n" + "=" * 100)
    print("SUB-PERIODS")
    print("=" * 100)
    periods = [
        ("Ukraine_2022", "2022-02-01", "2022-12-31"),
        ("Inflation_2022H2", "2022-07-01", "2022-12-31"),
        ("Inauguration_2026", "2026-01-01", "2026-04-30"),
    ]
    print(
        f"{'Period':<22} {'PM':>10} {'PM+GPR':>10} {'PM+Comp':>11} "
        f"{'MDD':>9} {'GPR_MDD':>10} {'Comp_MDD':>10}"
    )
    print("-" * 100)
    for label, start, end in periods:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        sub_pm = pm_base_2013.loc[s:e].dropna()
        sub_gpr = pm_gpr_2013.loc[s:e].dropna()
        sub_comp = pm_comp.loc[s:e].dropna()
        if len(sub_pm) < 10:
            continue
        m1 = all_metrics(sub_pm)
        m2 = all_metrics(sub_gpr)
        m3 = all_metrics(sub_comp)
        print(
            f"  {label:<20} "
            f"{m1.get('annualized_return', 0):>+9.2%} "
            f"{m2.get('annualized_return', 0):>+9.2%} "
            f"{m3.get('annualized_return', 0):>+10.2%} "
            f"{m1.get('max_drawdown', 0):>+8.2%} "
            f"{m2.get('max_drawdown', 0):>+9.2%} "
            f"{m3.get('max_drawdown', 0):>+9.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
