#!/usr/bin/env python
"""19y Multi-Asset-Master-Backtest mit Composite-Overlay.

Echtes Master-70/30:
- SA (Equity): 22 stocks aus watchlist_2007_2026
- XA (Cross-Asset): 11 ETFs aus yfinance_long (AGG/DBC/EEM/EFA/GLD/HYG/IWM/QQQ/SLV/SPY/TLT)

Volles 19y Eval-Window (2007-01-04 bis 2026-04-01) inklusive GFC, COVID, Ukraine.
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

XA_ETFS = ["AGG", "DBC", "EEM", "EFA", "GLD", "HYG", "IWM", "QQQ", "SLV", "SPY", "TLT"]


def _load_eq_wide():
    df = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])
    df["return"] = df.groupby("symbol")["close"].pct_change()
    df = df.dropna(subset=["return"])
    return df.pivot(index="date", columns="symbol", values="return").fillna(0)


def _load_xa_wide():
    frames = []
    for sym in XA_ETFS:
        p = Path(f"data/cache/yfinance_long/{sym}.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        date_col = "date" if "date" in df.columns else "Date"
        df["date"] = pd.to_datetime(df[date_col], utc=True)
        df["symbol"] = sym
        df = df.sort_values("date")
        df["return"] = df["close"].pct_change()
        frames.append(df[["date", "symbol", "return"]].dropna())
    panel = pd.concat(frames, ignore_index=True)
    return panel.pivot(index="date", columns="symbol", values="return").fillna(0)


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
            {
                "date": date,
                "pnl": pnl,
                "sa_lev": out["sa_leverage"],
                "xa_lev": out["xa_ew_leverage"],
                "geo_mult": out.get("geo_multiplier", 1.0),
            }
        )
    return pd.DataFrame(records).set_index("date")


def _build_overlay(daily_index):
    monthly = compute_monthly_composite()
    daily = expand_composite_to_daily(monthly, daily_index, GeoStressPolicy())
    return daily[["multiplier", "state", "composite_z_smoothed"]].rename(
        columns={"composite_z_smoothed": "composite_z"}
    )


def main():
    print("=" * 100)
    print("19y MULTI-ASSET MASTER + COMPOSITE (22 stocks + 11 ETFs)")
    print("=" * 100)

    eq = _load_eq_wide()
    xa = _load_xa_wide()
    common = eq.index.intersection(xa.index)
    eq = eq.reindex(common)
    xa = xa.reindex(common)
    print(
        f"Panel: {len(eq)} days, {len(eq.columns)} SA stocks + {len(xa.columns)} XA ETFs"
    )
    print(f"Range: {eq.index[0].date()} to {eq.index[-1].date()}")

    cfg_base = LiveEngineConfig()  # default sa_weight=0.70
    cfg_geo = LiveEngineConfig(enable_geo_overlay=True)

    overlay = _build_overlay(eq.index)
    pause_d = (overlay["state"] == "PAUSE").sum()
    active_d = (overlay["state"] == "ACTIVE").sum()
    print(
        f"Composite state-dist: PAUSE={pause_d}d ACTIVE={active_d}d (post-GDELT 2013+)"
    )

    print("\nRunning A: Master baseline (70/30)...")
    df_a = _walk_forward(eq, xa, cfg_base)
    print(f"  A: {len(df_a)} eval days")

    print("Running B: Master + Composite-Geo...")
    df_b = _walk_forward(eq, xa, cfg_geo, geo_overlay=overlay)
    print(
        f"  B: {len(df_b)} eval days, geo<0.7 share={(df_b['geo_mult'] < 0.7).mean():.1%}"
    )

    print("\n" + "=" * 100)
    print("METRICS (full 19y eval window, multi-asset)")
    print("=" * 100)
    header = f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>9}"
    print(header)
    print("-" * 100)
    for label, df in [
        ("A: Master 70/30 baseline", df_a),
        ("B: Master + Composite", df_b),
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

    out = calmar_diff_bootstrap(
        df_b["pnl"].dropna(),
        df_a["pnl"].dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"\nB vs A Calmar-Bootstrap 19y: obs_diff={out['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Crisis sub-periods
    print("\n" + "=" * 100)
    print("CRISIS SUB-PERIODS")
    print("=" * 100)
    periods = [
        ("GFC_2008", "2008-09-01", "2009-06-30"),
        ("Flash_2010", "2010-05-01", "2010-06-30"),
        ("EU_Crisis_2011", "2011-07-01", "2011-12-31"),
        ("China_2015", "2015-08-01", "2015-10-31"),
        ("Q1_2016", "2016-01-01", "2016-03-31"),
        ("VolMageddon_2018", "2018-01-25", "2018-03-15"),
        ("Q4_2018", "2018-10-01", "2018-12-31"),
        ("COVID_2020", "2020-02-15", "2020-05-31"),
        ("Inflation_2022", "2022-01-01", "2022-12-31"),
        ("Ukraine_Q1_2022", "2022-02-15", "2022-06-30"),
        ("Banking_2023Q1", "2023-03-01", "2023-04-30"),
        ("Inauguration_2026", "2026-01-01", "2026-04-01"),
    ]
    print(
        f"{'Period':<22} {'A AnnRet':>10} {'B AnnRet':>10} "
        f"{'A MDD':>9} {'B MDD':>9} {'dMDD':>9} {'dReturn':>9}"
    )
    print("-" * 100)
    for label, start, end in periods:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        sub_a = df_a.loc[s:e, "pnl"].dropna()
        sub_b = df_b.loc[s:e, "pnl"].dropna()
        if len(sub_a) < 5:
            continue
        ma = all_metrics(sub_a)
        mb = all_metrics(sub_b)
        d_mdd = mb["max_drawdown"] - ma["max_drawdown"]
        d_ret = mb.get("annualized_return", 0) - ma.get("annualized_return", 0)
        print(
            f"  {label:<20} "
            f"{ma.get('annualized_return', 0):>+9.2%} "
            f"{mb.get('annualized_return', 0):>+9.2%} "
            f"{ma.get('max_drawdown', 0):>+8.2%} "
            f"{mb.get('max_drawdown', 0):>+8.2%} "
            f"{d_mdd:>+8.2%} "
            f"{d_ret:>+8.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
