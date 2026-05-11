#!/usr/bin/env python
"""Master-Allocator auf 19-Jahres-Historie — finale Validierung.

Pipeline: Long-History-Equity (22 Mega-Caps, 2007-2026) + Long-History-Cross-Asset
(11 ETFs, 2007-2026) → Master_70_30.
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
from erweiterung.strategies.master_allocator import (  # noqa: E402
    MasterAllocator,
    MasterAllocatorConfig,
)


def _cs_long_only(
    panel: pd.DataFrame, signal_col: str, quantile: float = 0.3
) -> pd.DataFrame:
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
    print("MASTER-ALLOCATOR LONG-HISTORY VALIDATION (2007-2026)")
    print("=" * 100)

    # Equity factor
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
    print(
        f"Equity factor: {len(eq_factor_ret)} days, "
        f"{eq_factor_ret.index.min().date()} -> {eq_factor_ret.index.max().date()}"
    )

    # Cross-asset long history
    xa_dir = Path("data/cache/yfinance_long")
    frames = []
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
        p = xa_dir / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = sym
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    xa_wide = panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    xa_rets = xa_wide.pct_change().dropna()
    print(
        f"Cross-asset: {len(xa_rets)} days, "
        f"{xa_rets.index.min().date()} -> {xa_rets.index.max().date()}"
    )

    # Master Allocator
    alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out = alloc.allocate(eq_factor_ret, xa_rets)
    print(f"Master series: {len(out)} days")

    # 60/40 Classic
    classic = 0.60 * xa_rets["SPY"] + 0.40 * xa_rets["AGG"]

    # Metrics
    print("\n" + "=" * 100)
    print("LONG-HISTORY MASTER vs 60/40 (2007-2026)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metrics_dump = {}
    candidates = {
        "60_40_Classic": classic.loc[out.index.min() : out.index.max()],
        "Pure_EquityFactor": eq_factor_ret.loc[out.index.min() : out.index.max()],
        "SA_VolTarget": out["sa_voltarget"],
        "XA_Hybrid": out["xa_hybrid"],
        "Master_70_30": out["master_return"],
    }
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Calmar-Bootstrap vs 60/40
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs 60/40 Classic (long history, 19y)")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    bench = candidates["60_40_Classic"]
    for name, ret in candidates.items():
        if name == "60_40_Classic":
            continue
        cb = calmar_diff_bootstrap(
            ret.dropna(), bench.dropna(), n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in cb:
            continue
        p_gt = 1.0 - cb["p_value_one_sided_greater"]
        ci = f"[{cb['ci_low_2.5']:+.2f}, {cb['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{cb['observed_diff']:>+8.3f} "
            f"{cb['mean_diff']:>+9.3f} "
            f"{ci:>22} "
            f"{p_gt:>6.3f}"
        )

    # Audit
    print("\n" + "=" * 100)
    print("EQUITY-AUDIT MASTER_70_30 (Long-History)")
    print("=" * 100)
    eq = (1 + out["master_return"].fillna(0)).cumprod()
    eq.index = pd.to_datetime(eq.index, utc=True)
    audit = audit_equity_curve(eq, name="master_long_history")
    print(f"  Sharpe: {audit.overall_sharpe:.3f}")
    print(f"  Lag-1 Autocorr: {audit.return_autocorr_lag1:.3f}")
    print(f"  Skew: {audit.skew:.3f}, Kurtosis: {audit.kurtosis:.3f}")
    print(f"  WD/Vol-Ratio: {audit.worst_day_vol_ratio:.2f}")
    print(f"  MDD: {audit.max_drawdown:.3f}")
    print(f"  Flags: {audit.flags}")

    # Save
    eq_df = pd.DataFrame(
        {
            "master_return": out["master_return"],
            "master_equity": eq,
            "60_40_return": bench,
        }
    )
    eq_df.to_csv("output/erweiterung_master_long_history_equity.csv")
    Path("output/erweiterung_master_long_history_summary.json").write_text(
        json.dumps(
            {
                "n_days": int(len(out)),
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
                "audit_sharpe": (
                    float(audit.overall_sharpe) if audit.overall_sharpe else None
                ),
                "audit_mdd": float(audit.max_drawdown) if audit.max_drawdown else None,
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_master_long_history_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
