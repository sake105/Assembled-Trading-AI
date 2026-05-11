#!/usr/bin/env python
"""Grid-Search: GDELT-Resolution × Composite-Performance.

Vergleicht monthly-only, monthly+biweekly, monthly+biweekly+weekly auf Master.
Honest test ob höhere GDELT-Resolution hilft.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.live.live_decision_engine import (  # noqa: E402
    LiveDecisionEngine,
    LiveEngineConfig,
)
from erweiterung.risk.geo_stress_composite import (  # noqa: E402
    GeoStressPolicy,
    compute_monthly_composite,
    expand_composite_to_daily,
)

XA_SYMBOLS = ["ACWI", "AGG", "ARKK"]


def _load_panels():
    panel = pd.read_parquet("data/sample/master_universe_panel.parquet")
    panel = panel.rename(columns={"timestamp": "date"})
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel = panel.sort_values(["symbol", "date"])
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    panel = panel.dropna(subset=["return"])
    xa = (
        panel[panel["symbol"].isin(XA_SYMBOLS)]
        .pivot(index="date", columns="symbol", values="return")
        .fillna(0)
    )
    eq = (
        panel[~panel["symbol"].isin(XA_SYMBOLS)]
        .pivot(index="date", columns="symbol", values="return")
        .fillna(0)
    )
    common = eq.index.intersection(xa.index)
    return eq.reindex(common), xa.reindex(common)


def _walk_forward(eq, xa, config, geo_overlay=None, bootstrap_days=504):
    engine = LiveDecisionEngine(config)
    engine.bootstrap_from_history(eq.iloc[:bootstrap_days], xa.iloc[:bootstrap_days])
    if geo_overlay is not None:
        engine.attach_geo_overlay(geo_overlay)
    records = []
    for i in range(bootstrap_days, len(eq)):
        date = eq.index[i]
        engine.update_with_new_day(date, eq.iloc[i], xa.iloc[i])
        out = engine.decide_next()
        eq_top = out["eq_top_weights"]
        top_syms = eq_top[eq_top > 0].index
        eq_factor_ret = eq.iloc[i].reindex(top_syms).fillna(0).mean()
        xa_ew_ret = xa.iloc[i].mean()
        pnl = (
            config.sa_weight * out["sa_leverage"] * eq_factor_ret
            + (1 - config.sa_weight) * out["xa_ew_leverage"] * xa_ew_ret
        )
        records.append(
            {"date": date, "pnl": pnl, "geo_mult": out.get("geo_multiplier", 1.0)}
        )
    return pd.DataFrame(records).set_index("date")


def _build_overlay(daily_index, biweekly_path, weekly_path):
    """Compute monthly composite with selective higher-res inputs."""
    from erweiterung.risk import geo_stress_composite as gsc

    orig = gsc._load_gdelt_monthly

    def patched(
        cache_path="data/cache/gdelt/monthly_aggregates.parquet",
        biweekly_path=biweekly_path,
        weekly_path=weekly_path,
    ):
        return orig(
            cache_path=cache_path, biweekly_path=biweekly_path, weekly_path=weekly_path
        )

    with patch.object(gsc, "_load_gdelt_monthly", side_effect=patched):
        monthly = compute_monthly_composite()
    daily = expand_composite_to_daily(monthly, daily_index, GeoStressPolicy())
    return (
        daily[["multiplier", "state", "composite_z_smoothed"]].rename(
            columns={"composite_z_smoothed": "composite_z"}
        ),
        monthly,
    )


def main():
    eq, xa = _load_panels()
    print(f"Panel: {len(eq)} days, eval window ~{len(eq) - 504} days")

    print("\nRunning A: Master baseline...")
    df_a = _walk_forward(eq, xa, LiveEngineConfig())

    configs = [
        ("M-only", None, None),
        ("M+BW", "data/cache/gdelt/biweekly_aggregates.parquet", None),
        (
            "M+BW+W",
            "data/cache/gdelt/biweekly_aggregates.parquet",
            "data/cache/gdelt/weekly_aggregates.parquet",
        ),
    ]
    results = {"A: baseline": (df_a, None)}

    for label, bw, wk in configs:
        print(f"\nRunning Composite ({label})...")
        overlay, monthly = _build_overlay(eq.index, bw, wk)
        state_dist = overlay["state"].value_counts().to_dict()
        df = _walk_forward(
            eq, xa, LiveEngineConfig(enable_geo_overlay=True), geo_overlay=overlay
        )
        results[f"B: {label}"] = (df, state_dist)
        print(f"  PAUSE days in overlay: {state_dist.get('PAUSE', 0)}")

    print("\n" + "=" * 100)
    print("METRICS (3.3y eval window)")
    print("=" * 100)
    header = f"{'Strategy':<25} {'AnnRet':>9} {'Sharpe':>8} {'Calmar':>8} {'MDD':>9} {'PAUSE_d':>9}"
    print(header)
    print("-" * 100)
    for label, (df, state_dist) in results.items():
        m = all_metrics(df["pnl"].dropna())
        pause_d = state_dist.get("PAUSE", 0) if state_dist else "-"
        print(
            f"  {label:<23} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+8.2%} "
            f"{pause_d:>9}"
        )

    print("\n" + "=" * 100)
    print("Calmar-Bootstrap vs Baseline (n=2000)")
    print("=" * 100)
    base = results["A: baseline"][0]
    for label, (df, _) in results.items():
        if label == "A: baseline":
            continue
        out = calmar_diff_bootstrap(
            df["pnl"].dropna(),
            base["pnl"].dropna(),
            n_bootstrap=2000,
            avg_block_size=20,
            seed=42,
        )
        if "error" not in out:
            p_gt = 1.0 - out["p_value_one_sided_greater"]
            ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
            print(
                f"  {label:<25} obs_diff={out['observed_diff']:+.3f}  95%CI {ci}  p(>0)={p_gt:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
