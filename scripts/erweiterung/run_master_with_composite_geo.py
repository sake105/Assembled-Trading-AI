#!/usr/bin/env python
"""End-to-End Master-Backtest mit Composite-Geo-Overlay.

Vergleicht drei LiveDecisionEngine-Varianten auf master_universe_panel:
- A: Baseline (no overlay)
- B: Master + Composite-Geo-Overlay
- C: Master + Composite-Geo-Overlay + News-Tilt (nur News-Window 2025+ informativ)

Walk-forward: Bootstrap auf 2-Year History, dann day-by-day update + decide.
Tagespnl = sa_weight * sa_leverage * eq_factor_return + xa_weight * xa_ew_leverage * xa_ew_return.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

# Cross-asset universe (ETFs broad market exposure)
XA_SYMBOLS = ["ACWI", "AGG", "ARKK"]


def _load_panels():
    panel = pd.read_parquet("data/sample/master_universe_panel.parquet")
    panel = panel.rename(columns={"timestamp": "date"})
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel = panel.sort_values(["symbol", "date"])
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    panel = panel.dropna(subset=["return"])

    xa_panel = panel[panel["symbol"].isin(XA_SYMBOLS)].copy()
    eq_panel = panel[~panel["symbol"].isin(XA_SYMBOLS)].copy()

    eq_wide = eq_panel.pivot(index="date", columns="symbol", values="return").fillna(0)
    xa_wide = xa_panel.pivot(index="date", columns="symbol", values="return").fillna(0)

    # Align indices and restrict to days where XA has data
    common = eq_wide.index.intersection(xa_wide.index)
    eq_wide = eq_wide.reindex(common)
    xa_wide = xa_wide.reindex(common)
    return eq_wide, xa_wide


def _run_engine_walk_forward(
    eq_wide: pd.DataFrame,
    xa_wide: pd.DataFrame,
    config: LiveEngineConfig,
    geo_overlay: pd.DataFrame | None = None,
    bootstrap_days: int = 504,
    news_tilt_scores: pd.Series | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest. Returns DataFrame with [date, pnl, sa_lev, geo_mult, ...]."""
    engine = LiveDecisionEngine(config)
    boot_eq = eq_wide.iloc[:bootstrap_days]
    boot_xa = xa_wide.iloc[:bootstrap_days]
    engine.bootstrap_from_history(boot_eq, boot_xa)

    if geo_overlay is not None:
        engine.attach_geo_overlay(geo_overlay)
    if news_tilt_scores is not None:
        engine.attach_news_tilt_scores(news_tilt_scores)

    records = []
    for i in range(bootstrap_days, len(eq_wide)):
        date = eq_wide.index[i]
        # Decide BEFORE update (so today's decision uses yesterday's state).
        # But state.last_date and current_geo_multiplier are stale.
        # Update first, then decide.
        engine.update_with_new_day(date, eq_wide.iloc[i], xa_wide.iloc[i])
        out = engine.decide_next()

        sa_lev = out["sa_leverage"]
        xa_lev = out["xa_ew_leverage"]
        # Equity factor return = mean of top picks today
        eq_top = out["eq_top_weights"]
        top_syms = eq_top[eq_top > 0].index
        eq_factor_ret = eq_wide.iloc[i].reindex(top_syms).fillna(0).mean()
        xa_ew_ret = xa_wide.iloc[i].mean()

        sa_pnl = config.sa_weight * sa_lev * eq_factor_ret
        xa_pnl = (1 - config.sa_weight) * xa_lev * xa_ew_ret
        total_pnl = sa_pnl + xa_pnl

        records.append(
            {
                "date": date,
                "pnl": total_pnl,
                "sa_lev": sa_lev,
                "xa_lev": xa_lev,
                "geo_mult": out.get("geo_multiplier", 1.0),
                "n_top": int((eq_top > 0).sum()),
            }
        )
    return pd.DataFrame(records).set_index("date")


def _build_geo_overlay(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = compute_monthly_composite()
    daily = expand_composite_to_daily(monthly, daily_index, GeoStressPolicy())
    return daily[["multiplier", "state", "composite_z_smoothed"]].rename(
        columns={"composite_z_smoothed": "composite_z"}
    )


def main():
    print("=" * 100)
    print("MASTER + COMPOSITE-GEO-OVERLAY: End-to-End-Backtest")
    print("=" * 100)

    eq_wide, xa_wide = _load_panels()
    print(
        f"Panel: {len(eq_wide)} days, {len(eq_wide.columns)} eq, "
        f"{len(xa_wide.columns)} xa, range {eq_wide.index[0].date()} to "
        f"{eq_wide.index[-1].date()}"
    )

    # Build daily geo overlay covering full backtest period
    geo_overlay = _build_geo_overlay(eq_wide.index)
    state_dist = geo_overlay["state"].value_counts()
    print(f"Geo-overlay state distribution: {dict(state_dist)}")

    cfg_base = LiveEngineConfig()
    cfg_geo = LiveEngineConfig(enable_geo_overlay=True)

    print("\nRunning A: Master (no overlay)...")
    df_base = _run_engine_walk_forward(eq_wide, xa_wide, cfg_base)
    print(
        f"  base: {len(df_base)} trading days, n_top mean={df_base['n_top'].mean():.1f}"
    )

    print("Running B: Master + Composite-Geo...")
    df_geo = _run_engine_walk_forward(eq_wide, xa_wide, cfg_geo, geo_overlay)
    pause_share = (df_geo["geo_mult"] < 0.7).mean()
    print(f"  geo: {len(df_geo)} days, multiplier<0.7 share={pause_share:.1%}")

    print("\n" + "=" * 100)
    print("METRICS")
    print("=" * 100)
    header = (
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} "
        f"{'Sortino':>9} {'Calmar':>8} {'MDD':>9}"
    )
    print(header)
    print("-" * 100)
    for label, df in [
        ("A: Master baseline", df_base),
        ("B: Master + Composite", df_geo),
    ]:
        m = all_metrics(df["pnl"].dropna())
        print(
            f"  {label:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+8.2%}"
        )

    # Calmar-Bootstrap
    out = calmar_diff_bootstrap(
        df_geo["pnl"].dropna(),
        df_base["pnl"].dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"\nB vs A Calmar-Bootstrap: obs_diff={out['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Sub-period analysis
    print("\n" + "=" * 100)
    print("SUB-PERIODS")
    print("=" * 100)
    periods = [
        ("Ukraine_2022", "2022-02-01", "2022-12-31"),
        ("Inflation_2022H2", "2022-07-01", "2022-12-31"),
        ("Banking_2023Q1", "2023-03-01", "2023-04-30"),
        ("Inauguration_2026", "2026-01-01", "2026-04-30"),
    ]
    print(
        f"{'Period':<22} {'A AnnRet':>10} {'B AnnRet':>10} "
        f"{'A MDD':>9} {'B MDD':>9} {'dMDD':>9}"
    )
    print("-" * 100)
    for label, start, end in periods:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        sub_a = df_base.loc[s:e, "pnl"].dropna()
        sub_b = df_geo.loc[s:e, "pnl"].dropna()
        if len(sub_a) < 5:
            continue
        ma = all_metrics(sub_a)
        mb = all_metrics(sub_b)
        d_mdd = mb["max_drawdown"] - ma["max_drawdown"]
        print(
            f"  {label:<20} "
            f"{ma.get('annualized_return', 0):>+9.2%} "
            f"{mb.get('annualized_return', 0):>+9.2%} "
            f"{ma.get('max_drawdown', 0):>+8.2%} "
            f"{mb.get('max_drawdown', 0):>+8.2%} "
            f"{d_mdd:>+8.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
