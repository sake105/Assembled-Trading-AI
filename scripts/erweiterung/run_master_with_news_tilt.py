#!/usr/bin/env python
"""End-to-End-Backtest mit Composite-Geo + News-Tilt.

Backtest-Window: News-Coverage (2025-12-22 to 2026-05-06, ~95 trading days).
Vergleicht drei LiveDecisionEngine-Varianten:
- A: Master baseline
- B: Master + Composite-Geo-Overlay
- C: Master + Composite-Geo + News-Tilt (per-day re-attached scores)

WICHTIG: News-Window ist klein (<100 days). Ergebnisse sind explorativ, nicht
statistisch signifikant aussagekräftig. Zweck: verifizieren dass die Integration
mechanisch funktioniert mit echten Daten und Edge-Direktion einschätzen.
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
from erweiterung.signals.news_tilt_builder import (  # noqa: E402
    build_daily_news_tilt,
    load_news_sentiment,
)

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
    common = eq_wide.index.intersection(xa_wide.index)
    return eq_wide.reindex(common), xa_wide.reindex(common)


def _build_geo_overlay(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = compute_monthly_composite()
    daily = expand_composite_to_daily(monthly, daily_index, GeoStressPolicy())
    return daily[["multiplier", "state", "composite_z_smoothed"]].rename(
        columns={"composite_z_smoothed": "composite_z"}
    )


def _walk_forward(
    eq_wide: pd.DataFrame,
    xa_wide: pd.DataFrame,
    config: LiveEngineConfig,
    bootstrap_days: int,
    eval_start: pd.Timestamp,
    geo_overlay: pd.DataFrame | None = None,
    news_z_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest. eval_start = first day to include in pnl."""
    engine = LiveDecisionEngine(config)
    engine.bootstrap_from_history(
        eq_wide.iloc[:bootstrap_days], xa_wide.iloc[:bootstrap_days]
    )

    if geo_overlay is not None:
        engine.attach_geo_overlay(geo_overlay)

    records = []
    for i in range(bootstrap_days, len(eq_wide)):
        date = eq_wide.index[i]
        # Per-day news-tilt refresh (z-scores can change daily)
        if news_z_panel is not None and date in news_z_panel.index:
            scores = news_z_panel.loc[date].dropna()
            if not scores.empty:
                engine.attach_news_tilt_scores(scores)

        engine.update_with_new_day(date, eq_wide.iloc[i], xa_wide.iloc[i])
        if date < eval_start:
            continue
        out = engine.decide_next()
        eq_top = out["eq_top_weights"]
        top_syms = eq_top[eq_top > 0].index
        eq_factor_ret = eq_wide.iloc[i].reindex(top_syms).fillna(0).mean()
        xa_ew_ret = xa_wide.iloc[i].mean()

        sa_pnl = config.sa_weight * out["sa_leverage"] * eq_factor_ret
        xa_pnl = (1 - config.sa_weight) * out["xa_ew_leverage"] * xa_ew_ret
        records.append(
            {
                "date": date,
                "pnl": sa_pnl + xa_pnl,
                "sa_lev": out["sa_leverage"],
                "geo_mult": out.get("geo_multiplier", 1.0),
                "n_top": int((eq_top > 0).sum()),
            }
        )
    return pd.DataFrame(records).set_index("date")


def main():
    print("=" * 100)
    print("MASTER + COMPOSITE + NEWS-TILT — End-to-End-Backtest")
    print("=" * 100)

    eq_wide, xa_wide = _load_panels()
    geo_overlay = _build_geo_overlay(eq_wide.index)

    news_df = load_news_sentiment()
    news_z = build_daily_news_tilt(
        news_df, eq_wide.index, rolling_days=14, decay_halflife_days=3
    )
    valid_z = news_z.dropna(how="all")
    print(
        f"News coverage: {len(valid_z)} days with at least 1 symbol "
        f"({valid_z.index.min().date()} to {valid_z.index.max().date()})"
    )
    print(f"Mean symbols/day with news: " f"{valid_z.notna().sum(axis=1).mean():.1f}")

    bootstrap_days = 504  # 2y bootstrap
    eval_start = pd.Timestamp("2025-12-22", tz="UTC")
    if eval_start <= eq_wide.index[bootstrap_days]:
        eval_start = eq_wide.index[bootstrap_days + 1]
    print(f"Eval window: {eval_start.date()} to {eq_wide.index[-1].date()}")

    print("\nRunning A: Master baseline...")
    df_a = _walk_forward(
        eq_wide, xa_wide, LiveEngineConfig(), bootstrap_days, eval_start
    )
    print(f"  A: {len(df_a)} eval days")

    print("Running B: Master + Composite-Geo...")
    df_b = _walk_forward(
        eq_wide,
        xa_wide,
        LiveEngineConfig(enable_geo_overlay=True),
        bootstrap_days,
        eval_start,
        geo_overlay=geo_overlay,
    )
    print(
        f"  B: {len(df_b)} eval days, geo<0.7 share={(df_b['geo_mult'] < 0.7).mean():.1%}"
    )

    print("Running C: Master + Composite + News-Tilt (strength=0.30)...")
    df_c = _walk_forward(
        eq_wide,
        xa_wide,
        LiveEngineConfig(
            enable_geo_overlay=True, enable_news_tilt=True, news_tilt_strength=0.30
        ),
        bootstrap_days,
        eval_start,
        geo_overlay=geo_overlay,
        news_z_panel=news_z,
    )
    print(f"  C: {len(df_c)} eval days")

    print("Running D: Master + Composite + News-Tilt (strength=0.60)...")
    df_d = _walk_forward(
        eq_wide,
        xa_wide,
        LiveEngineConfig(
            enable_geo_overlay=True, enable_news_tilt=True, news_tilt_strength=0.60
        ),
        bootstrap_days,
        eval_start,
        geo_overlay=geo_overlay,
        news_z_panel=news_z,
    )

    print("\n" + "=" * 100)
    print("METRICS")
    print("=" * 100)
    header = (
        f"{'Strategy':<40} {'AnnRet':>9} {'Sharpe':>8} "
        f"{'Sortino':>9} {'Calmar':>8} {'MDD':>9}"
    )
    print(header)
    print("-" * 100)
    for label, df in [
        ("A: Master baseline", df_a),
        ("B: Master + Composite", df_b),
        ("C: Master + Composite + NewsTilt 0.30", df_c),
        ("D: Master + Composite + NewsTilt 0.60", df_d),
    ]:
        m = all_metrics(df["pnl"].dropna())
        print(
            f"  {label:<38} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+8.2%}"
        )

    # Pair-wise Calmar-Bootstrap
    print("\n" + "=" * 100)
    print("PAIRWISE CALMAR-BOOTSTRAP (n=2000)")
    print("=" * 100)
    pairs = [
        ("B vs A", df_b, df_a),
        ("C vs A", df_c, df_a),
        ("C vs B", df_c, df_b),
        ("D vs B", df_d, df_b),
    ]
    for label, treat, base in pairs:
        out = calmar_diff_bootstrap(
            treat["pnl"].dropna(),
            base["pnl"].dropna(),
            n_bootstrap=2000,
            avg_block_size=10,
            seed=42,
        )
        if "error" in out:
            print(f"  {label}: {out['error']}")
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {label:<10} obs_diff={out['observed_diff']:+.3f}  95%CI {ci}  p(>0)={p_gt:.3f}"
        )

    print("\nNote: ~95 eval days, klein -> Statistik nur explorativ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
