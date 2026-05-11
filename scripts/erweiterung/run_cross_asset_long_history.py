#!/usr/bin/env python
"""Cross-Asset-Backtest auf 19-Jahres-ETF-Cache (2007-2026).

Nutzt data/cache/yfinance_long/ (gefetcht via fetch_long_history_etfs.py).
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
from erweiterung.portfolio.hierarchical_risk_parity import hrp_weights  # noqa: E402
from erweiterung.robustness.sub_period import STANDARD_EPOCHS_US_EQUITY  # noqa: E402
from erweiterung.strategies.volatility_targeting import (  # noqa: E402
    VolTargetConfig,
    apply_vol_targeting,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "HYG", "GLD", "SLV", "DBC"]


def main():
    cache_dir = Path("data/cache/yfinance_long")
    frames = []
    for sym in UNIVERSE:
        p = cache_dir / f"{sym}.parquet"
        if not p.exists():
            print(f"Missing: {sym}")
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        return 1
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["return"] = panel.groupby("symbol")["close"].pct_change()

    wide = panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    rets = wide.pct_change().dropna()
    print(
        f"Long-history panel: {len(rets)} days, {rets.shape[1]} assets, "
        f"{rets.index.min().date()} -> {rets.index.max().date()}"
    )

    corr = rets.corr()
    mean_corr = (corr.values.sum() - corr.shape[0]) / (corr.size - corr.shape[0])
    print(f"Mean off-diagonal correlation: {mean_corr:.3f}")

    strategies: dict[str, pd.Series] = {}

    # Equal-Weight 11
    strategies["EW_All_11"] = rets.mean(axis=1)

    # 60/40 Classic
    if "SPY" in rets.columns and "AGG" in rets.columns:
        strategies["60_40_Classic"] = 0.60 * rets["SPY"] + 0.40 * rets["AGG"]

    # Risk-Parity (Inverse-Vol)
    rv = rets.rolling(60, min_periods=10).std()
    inv = 1.0 / rv.replace(0, np.nan)
    weights_rp = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / rets.shape[1]).shift(1)
    strategies["Risk_Parity"] = (rets * weights_rp).sum(axis=1)

    # HRP static
    try:
        hw = hrp_weights(rets.dropna())
        strategies["HRP_Static"] = (rets * hw).sum(axis=1)
    except Exception as e:
        print(f"HRP failed: {e}")

    # VolTarget-EW
    ew = rets.mean(axis=1)
    vt = apply_vol_targeting(ew, VolTargetConfig(target_vol_annual=0.10))
    strategies["VolTarget_EW"] = vt["scaled_return"]

    # XAsset Mom Top-5
    mom = rets.rolling(252, min_periods=200).apply(
        lambda x: (1 + x[:-21]).prod() - 1 if len(x) > 21 else np.nan, raw=False
    )
    daily_idx = rets.index
    monthly_rebal = daily_idx[daily_idx.is_month_end | (daily_idx == daily_idx[-1])]
    cmom = pd.Series(0.0, index=daily_idx)
    cur_w = pd.Series(0.0, index=rets.columns)
    for d in daily_idx:
        if d in monthly_rebal:
            mt = mom.loc[d].dropna()
            if len(mt) >= 5:
                top5 = mt.nlargest(5).index
                cur_w = pd.Series(0.0, index=rets.columns)
                cur_w[top5] = 1.0 / 5.0
        if cur_w.sum() > 0:
            cmom.loc[d] = (rets.loc[d] * cur_w).sum()
    strategies["XAsset_Mom_Top5"] = cmom

    # Hybrid 50/50
    aligned = pd.concat(
        {"vt": strategies["VolTarget_EW"], "mom": cmom}, axis=1
    ).dropna()
    strategies["Hybrid_VT_Mom"] = 0.5 * aligned["vt"] + 0.5 * aligned["mom"]

    # Performance
    print("\n" + "=" * 100)
    print("LONG-HISTORY CROSS-ASSET (2007-2026, ~19y)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metrics_dump = {}
    for name, r in strategies.items():
        m = all_metrics(r.dropna())
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
    if "60_40_Classic" in strategies:
        print("\n" + "=" * 100)
        print("CALMAR-BOOTSTRAP vs 60/40 Classic (long history, 19y)")
        print("=" * 100)
        print(
            f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
        )
        print("-" * 100)
        baseline = strategies["60_40_Classic"]
        for name, r in strategies.items():
            if name == "60_40_Classic":
                continue
            out = calmar_diff_bootstrap(
                r.dropna(),
                baseline.dropna(),
                n_bootstrap=2000,
                avg_block_size=20,
                seed=42,
            )
            if "error" in out:
                continue
            p_gt = 1.0 - out["p_value_one_sided_greater"]
            ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
            print(
                f"  {name:<30} "
                f"{out['observed_diff']:>+8.3f} "
                f"{out['mean_diff']:>+9.3f} "
                f"{ci:>22} "
                f"{p_gt:>6.3f}"
            )

    # Sub-Period
    print("\n" + "=" * 100)
    print("SUB-PERIOD HIGHLIGHTS (key strategies)")
    print("=" * 100)
    print(f"{'Strategy':<26} {'Epoch':<26} {'AnnRet':>10} {'Sharpe':>8} {'MDD':>8}")
    print("-" * 100)
    for sname in [
        "60_40_Classic",
        "Risk_Parity",
        "VolTarget_EW",
        "XAsset_Mom_Top5",
        "Hybrid_VT_Mom",
    ]:
        if sname not in strategies:
            continue
        r = strategies[sname].dropna()
        for epoch in STANDARD_EPOCHS_US_EQUITY:
            mask = (r.index >= pd.Timestamp(epoch.start, tz="UTC")) & (
                r.index <= pd.Timestamp(epoch.end, tz="UTC")
            )
            sub = r[mask]
            if len(sub) < 30:
                continue
            m = all_metrics(sub)
            print(
                f"  {sname:<24} {epoch.name:<26} "
                f"{m.get('annualized_return', 0):>+9.2%} "
                f"{m.get('sharpe', 0):>+7.3f} "
                f"{m.get('max_drawdown', 0):>+7.2%}"
            )
        print()

    # Save
    eq_df = pd.DataFrame(
        {k: (1 + v.fillna(0)).cumprod() for k, v in strategies.items()}
    )
    eq_df.to_csv("output/erweiterung_cross_asset_long_history_equity.csv")
    Path("output/erweiterung_cross_asset_long_history_summary.json").write_text(
        json.dumps(
            {
                "corr_mean": float(mean_corr),
                "n_days": int(len(rets)),
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
            },
            indent=2,
            default=str,
        )
    )
    print("Saved -> output/erweiterung_cross_asset_long_history_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
