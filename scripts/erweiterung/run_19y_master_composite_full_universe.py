#!/usr/bin/env python
"""19y-Master-Composite-Backtest auf vollem 200-Ticker-Universum.

Bessere Version von run_19y_master_composite.py:
- Universum: data/cache/yfinance_long/ (200 Ticker, period=max)
- Survivorship-Bias: trotzdem vorhanden — wir nehmen heutiges Universum.
- Filter: nur Symbole mit >=4000 trading days (Coverage ~16y+)
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


CACHE_DIR = Path("data/cache/yfinance_long")


def _load_universe_wide(min_days: int = 4000) -> pd.DataFrame:
    """Baue wide-format returns-panel aus yfinance_long cache.

    Filter: nur Symbole mit >= min_days trading days (sonst Survivorship-Spike).
    """
    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"{CACHE_DIR} not found — run fetch_full_universe_long_history.py first"
        )

    frames = []
    skipped = []
    for p in sorted(CACHE_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            if len(df) < min_days:
                skipped.append((p.stem, len(df)))
                continue
            df = df.reset_index()
            # Handle both 'date' and 'Date' from yfinance
            date_col = "date" if "date" in df.columns else "Date"
            df = df.rename(columns={date_col: "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["symbol"] = p.stem
            frames.append(df[["date", "symbol", "close"]])
        except Exception as e:
            skipped.append((p.stem, str(e)[:30]))

    print(
        f"Loaded: {len(frames)} symbols (skipped {len(skipped)} with <{min_days} days)"
    )
    if skipped[:5]:
        print(f"Skipped sample: {skipped[:5]}")

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    panel = panel.dropna(subset=["return"])
    eq = panel.pivot(index="date", columns="symbol", values="return").fillna(0)
    return eq


def _walk_forward_sa_only(eq, config, geo_overlay=None, bootstrap_days=504):
    engine = LiveDecisionEngine(config)
    xa_synthetic = pd.DataFrame(0.0, index=eq.index, columns=["_DUMMY_XA"])
    engine.bootstrap_from_history(
        eq.iloc[:bootstrap_days], xa_synthetic.iloc[:bootstrap_days]
    )
    if geo_overlay is not None:
        engine.attach_geo_overlay(geo_overlay)

    records = []
    for i in range(bootstrap_days, len(eq)):
        date = eq.index[i]
        engine.update_with_new_day(date, eq.iloc[i], xa_synthetic.iloc[i])
        out = engine.decide_next()
        eq_top = out["eq_top_weights"]
        top_syms = eq_top[eq_top > 0].index
        eq_factor_ret = eq.iloc[i].reindex(top_syms).fillna(0).mean()
        pnl = config.sa_weight * out["sa_leverage"] * eq_factor_ret
        records.append(
            {
                "date": date,
                "pnl": pnl,
                "sa_lev": out["sa_leverage"],
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
    print("19y MASTER + COMPOSITE auf VOLLEM 200-TICKER-UNIVERSUM")
    print("=" * 100)

    eq = _load_universe_wide(min_days=4000)
    print(
        f"Panel: {len(eq)} days, {len(eq.columns)} symbols, "
        f"{eq.index[0].date()} to {eq.index[-1].date()}"
    )
    print(f"Symbols sample: {sorted(eq.columns.tolist())[:15]}...")

    cfg_base = LiveEngineConfig(sa_weight=1.0)
    cfg_geo = LiveEngineConfig(sa_weight=1.0, enable_geo_overlay=True)

    overlay = _build_overlay(eq.index)
    pause_days = (overlay["state"] == "PAUSE").sum()
    active_days = (overlay["state"] == "ACTIVE").sum()
    print(f"Composite state-dist: PAUSE={pause_days}d ACTIVE={active_days}d")

    print("\nRunning A: SA-Master baseline...")
    df_a = _walk_forward_sa_only(eq, cfg_base)
    print(f"  A: {len(df_a)} eval days")

    print("Running B: SA-Master + Composite-Geo...")
    df_b = _walk_forward_sa_only(eq, cfg_geo, geo_overlay=overlay)
    print(
        f"  B: {len(df_b)} eval days, geo<0.7 share={(df_b['geo_mult'] < 0.7).mean():.1%}"
    )

    print("\n" + "=" * 100)
    print("METRICS (full 19y, n symbols)")
    print("=" * 100)
    header = f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>9}"
    print(header)
    print("-" * 100)
    for label, df in [
        ("A: SA-Master baseline", df_a),
        ("B: SA-Master + Composite", df_b),
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
            f"\nB vs A Calmar-Bootstrap: obs_diff={out['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Crisis sub-periods
    print("\n" + "=" * 100)
    print("CRISIS SUB-PERIODS")
    print("=" * 100)
    periods = [
        ("GFC_2008", "2008-09-01", "2009-06-30"),
        ("EU_Crisis_2011", "2011-07-01", "2011-12-31"),
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
