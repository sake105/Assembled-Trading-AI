#!/usr/bin/env python
"""Master V2 — Master_70_30 + EMA-Trend-Faktor + Profit-Lock-Overlay.

Pipeline
--------
1. 22 Mega-Caps 19y (data/sample/watchlist_2007_2026.parquet)
2. Equity-Faktor V2: 50% Mom-12/1 + 50% EMA-Trend (orthogonale Trends)
3. Vol-Targeting auf kombiniertem Equity-Faktor
4. Cross-Asset Hybrid (11 ETFs 19y)
5. Master Mix (70/30)
6. Profit-Lock Overlay
7. Calmar-Bootstrap vs Master V1 und 60/40
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
from erweiterung.qa.equity_curve_audit import audit_equity_curve  # noqa: E402
from erweiterung.strategies.ema_trend_cross_section import (  # noqa: E402
    EMATrendConfig,
    backtest_ema_trend,
)
from erweiterung.strategies.master_allocator import (  # noqa: E402
    MasterAllocator,
    MasterAllocatorConfig,
)
from erweiterung.strategies.profit_lock_overlay import (  # noqa: E402
    ProfitLockConfig,
    apply_profit_lock,
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
    print("=" * 100)
    print("MASTER V2 — Mom-12/1 + EMA-Trend + Profit-Lock + Cross-Asset")
    print("=" * 100)

    # Equity panel
    eq_panel = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})
    eq_panel["date"] = pd.to_datetime(eq_panel["date"], utc=True)
    eq_panel = eq_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq_panel["return"] = eq_panel.groupby("symbol")["close"].pct_change()
    print(f"Equity panel: {len(eq_panel)} rows, {eq_panel['symbol'].nunique()} symbols")

    # Faktor 1: Mom-12/1
    print("\nFaktor 1: Mom-12/1 ...")
    mom = momentum_12_1(eq_panel[["date", "symbol", "close"]])
    eq_with_mom = eq_panel.set_index(["date", "symbol"])
    eq_with_mom["mom_12_1"] = mom.reindex(eq_with_mom.index)
    eq_with_mom = eq_with_mom.reset_index()
    mom_factor = _cs_long_only(
        eq_with_mom.dropna(subset=["mom_12_1"]), "mom_12_1", quantile=0.3
    )
    mom_ret = mom_factor.groupby("date").agg(pnl=("pnl", "sum"))["pnl"]
    mom_ret.index = pd.to_datetime(mom_ret.index, utc=True)
    print(f"  Mom-12/1: {len(mom_ret)} days")

    # Faktor 2: EMA-Trend
    print("\nFaktor 2: EMA-Trend (EMA20/EMA60) ...")
    ema_ret = backtest_ema_trend(
        eq_panel[["date", "symbol", "close", "return"]].dropna(),
        EMATrendConfig(ema_fast=20, ema_slow=60, quantile_long=0.3, long_only=True),
    )
    ema_ret.index = pd.to_datetime(ema_ret.index, utc=True)
    print(f"  EMA-Trend: {len(ema_ret)} days")

    # Combined Equity Factor (50/50)
    aligned_eq = pd.concat({"mom": mom_ret, "ema": ema_ret}, axis=1).dropna()
    combined_eq = 0.5 * aligned_eq["mom"] + 0.5 * aligned_eq["ema"]
    print(f"Combined equity factor: {len(combined_eq)} days")

    # Cross-Asset
    print("\nCross-asset (11 ETFs, 19y) ...")
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

    # Master V1 baseline (only Mom-12/1)
    alloc_v1 = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out_v1 = alloc_v1.allocate(mom_ret, xa_rets)
    print(f"Master V1: {len(out_v1)} days")

    # Master V2 (combined Mom+EMA)
    alloc_v2 = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out_v2 = alloc_v2.allocate(combined_eq, xa_rets)
    print(f"Master V2 (combined): {len(out_v2)} days")

    # Apply Profit-Lock to V2
    pl_out = apply_profit_lock(
        out_v2["master_return"],
        ProfitLockConfig(
            lookback_days=20,
            trigger_return=0.06,
            multiplier_on_trigger=0.80,
            floor=0.50,
            cooldown_days=10,
        ),
    )
    master_v2_pl = pl_out["locked_return"]
    print(
        f"Master V2 + Profit-Lock: trigger fired on "
        f"{(pl_out['multiplier'] < 1.0).sum()} days ({(pl_out['multiplier'] < 1.0).mean():.1%})"
    )

    # 60/40 Classic
    classic = 0.60 * xa_rets["SPY"] + 0.40 * xa_rets["AGG"]

    # Performance
    print("\n" + "=" * 100)
    print("MASTER V2 vs V1 vs 60/40 (Long-History)")
    print("=" * 100)
    print(
        f"{'Strategy':<36} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    candidates = {
        "60_40_Classic": classic.loc[out_v2.index.min() : out_v2.index.max()],
        "Master_V1 (Mom only)": out_v1["master_return"],
        "Master_V2 (Mom+EMA)": out_v2["master_return"],
        "Master_V2 + ProfitLock": master_v2_pl,
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

    # Calmar-Bootstrap
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs 60/40 Classic")
    print("=" * 100)
    print(
        f"{'Challenger':<36} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    bench = candidates["60_40_Classic"]
    for name, ret in candidates.items():
        if "60_40" in name:
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(), bench.dropna(), n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<34} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci:>22} "
            f"{p_gt:>6.3f}"
        )

    # Master V2 vs Master V1
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP Master_V2 vs Master_V1")
    print("=" * 100)
    out = calmar_diff_bootstrap(
        out_v2["master_return"].dropna(),
        out_v1["master_return"].dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out:
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  V2 vs V1: obs_diff={out['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    out2 = calmar_diff_bootstrap(
        master_v2_pl.dropna(),
        out_v2["master_return"].dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    if "error" not in out2:
        p_gt = 1.0 - out2["p_value_one_sided_greater"]
        ci = f"[{out2['ci_low_2.5']:+.2f}, {out2['ci_high_97.5']:+.2f}]"
        print(
            f"  V2+ProfitLock vs V2: obs_diff={out2['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Audit
    print("\n" + "=" * 100)
    print("EQUITY-AUDIT Master_V2+ProfitLock")
    print("=" * 100)
    eq = (1 + master_v2_pl.fillna(0)).cumprod()
    eq.index = pd.to_datetime(eq.index, utc=True)
    audit = audit_equity_curve(eq, name="master_v2_pl")
    print(f"  Sharpe: {audit.overall_sharpe:.3f}, MDD: {audit.max_drawdown:.3f}")
    print(f"  Flags: {audit.flags}")

    # Save
    eq_df = pd.DataFrame(
        {
            "master_v1_return": out_v1["master_return"],
            "master_v2_return": out_v2["master_return"],
            "master_v2_pl_return": master_v2_pl,
            "master_v2_pl_equity": eq,
        }
    )
    eq_df.to_csv("output/erweiterung_master_v2_equity.csv")
    Path("output/erweiterung_master_v2_summary.json").write_text(
        json.dumps(
            {
                "n_days": int(len(out_v2)),
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
                "audit_flags": list(audit.flags),
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_v2_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
