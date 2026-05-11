#!/usr/bin/env python
"""Cross-Asset-Backtest: echte Diversifikation über Asset-Klassen.

Universum
---------
- Equity: SPY (US-LC), QQQ (US-Tech), IWM (US-SC), EFA (Intl-Dev), EEM (EM)
- Bonds:  AGG (US-Agg), TLT (US-20y), HYG (US-HY)
- Commodities: GLD (Gold), SLV (Silver), DBC (Commodities)

Strategien
----------
- Pure Equal-Weight-Allocation
- 60/40 Stock/Bond Classic
- Risk-Parity (Inverse-Vol-Weighted)
- HRP-Diversified
- Vol-Targeted-Equal-Weight
- Trend-Following Mom-12/1 mit Risk-Overlay

Wichtig: Cross-Asset-Korrelationen sind deutlich niedriger als Single-Asset-
Faktoren → Diversifikations-Edge sollte hier echter sein.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.altdata.yfinance_cache_loader import load_universe_panel  # noqa: E402
from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.portfolio.hierarchical_risk_parity import hrp_weights  # noqa: E402
from erweiterung.strategies.volatility_targeting import (  # noqa: E402
    VolTargetConfig,
    apply_vol_targeting,
)

CROSS_ASSET_UNIVERSE = [
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
]
STOCK_BUCKET = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
BOND_BUCKET = ["AGG", "TLT", "HYG"]
COMMODITY_BUCKET = ["GLD", "SLV", "DBC"]


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _ann_metrics(ret: pd.Series) -> dict:
    if ret.empty or ret.std() == 0:
        return {"ann_return": 0, "sharpe": 0, "max_dd": 0, "calmar": 0}
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    vol = ret.std() * np.sqrt(252)
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "ann_return": float(ann),
        "ann_vol": float(vol),
        "sharpe": float(ann / vol) if vol > 0 else 0,
        "sortino": (
            float(ann / (ret[ret < 0].std() * np.sqrt(252)))
            if (ret < 0).any() and ret[ret < 0].std() > 0
            else 0
        ),
        "max_dd": float(dd),
        "calmar": float(ann / abs(dd)) if dd != 0 else 0,
    }


def main():
    cache_dir = "data/cache/yfinance"
    print(f"Loading cross-asset universe ({len(CROSS_ASSET_UNIVERSE)} ETFs) ...")
    panel = load_universe_panel(
        cache_dir,
        CROSS_ASSET_UNIVERSE,
        start=None,
        end=None,
        require_min_rows=200,
        skip_missing=False,
    )
    print(
        f"Loaded {len(panel)} rows, range {panel['date'].min()} -> {panel['date'].max()}"
    )

    # Wide format
    wide = panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    rets = wide.pct_change().dropna()
    print(f"Common returns: {len(rets)} days, {rets.shape[1]} assets")

    # Korrelations-Matrix für Diagnostik
    corr = rets.corr()
    print(
        f"\nMean off-diagonal correlation: {(corr.values.sum() - corr.shape[0]) / (corr.size - corr.shape[0]):.3f}"
    )

    strategies: dict[str, pd.Series] = {}

    # 1. Equal-Weight (all 11)
    strategies["EW_All_11"] = rets.mean(axis=1)

    # 2. 60/40 Classic (SPY + AGG)
    if "SPY" in rets.columns and "AGG" in rets.columns:
        strategies["60_40_Classic"] = 0.60 * rets["SPY"] + 0.40 * rets["AGG"]

    # 3. Stock/Bond/Commodity 50/30/20
    sb_ret = (
        rets[STOCK_BUCKET].mean(axis=1)
        if all(s in rets.columns for s in STOCK_BUCKET)
        else None
    )
    bd_ret = (
        rets[BOND_BUCKET].mean(axis=1)
        if all(s in rets.columns for s in BOND_BUCKET)
        else None
    )
    cm_ret = (
        rets[COMMODITY_BUCKET].mean(axis=1)
        if all(s in rets.columns for s in COMMODITY_BUCKET)
        else None
    )
    if sb_ret is not None and bd_ret is not None and cm_ret is not None:
        strategies["EW_Stocks_Bonds_Comm_50_30_20"] = (
            0.50 * sb_ret + 0.30 * bd_ret + 0.20 * cm_ret
        )

    # 4. Risk-Parity (Inverse-Vol)
    rv = rets.rolling(60, min_periods=10).std()
    inv = 1.0 / rv.replace(0, np.nan)
    weights_rp = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / rets.shape[1]).shift(1)
    strategies["Risk_Parity"] = (rets * weights_rp).sum(axis=1)

    # 5. HRP (static weights)
    try:
        hw = hrp_weights(rets.dropna())
        strategies["HRP_Static"] = (rets * hw).sum(axis=1)
    except Exception as e:
        print(f"HRP failed: {e}")

    # 6. Vol-Targeted Equal-Weight
    ew = rets.mean(axis=1)
    vt = apply_vol_targeting(ew, VolTargetConfig(target_vol_annual=0.10))
    strategies["VolTarget_EW"] = vt["scaled_return"]

    # 7. Cross-Asset Momentum 12-1 (selektiere top 5 ETFs based on 12-1 return, equal-weight)
    mom_12_1 = rets.rolling(252, min_periods=200).apply(
        lambda x: (1 + x[:-21]).prod() - 1 if len(x) > 21 else np.nan, raw=False
    )
    # Daily portfolio: rebalance monthly to top-5 momentum
    daily_idx = rets.index
    monthly_rebal = daily_idx[daily_idx.is_month_end | (daily_idx == daily_idx[-1])]
    cmom_returns = pd.Series(0.0, index=daily_idx)
    cur_weights = pd.Series(0.0, index=rets.columns)
    for d in daily_idx:
        if d in monthly_rebal:
            mom_today = mom_12_1.loc[d].dropna()
            if len(mom_today) >= 5:
                top5 = mom_today.nlargest(5).index
                cur_weights = pd.Series(0.0, index=rets.columns)
                cur_weights[top5] = 1.0 / 5.0
        if cur_weights.sum() > 0:
            cmom_returns.loc[d] = (rets.loc[d] * cur_weights).sum()
    strategies["XAsset_Mom_Top5"] = cmom_returns

    # 8. Combined: 50% VolTarget-EW + 50% XAsset-Mom-Top5 (Diversifier)
    aligned = pd.concat(
        {"vt": strategies["VolTarget_EW"], "mom": strategies["XAsset_Mom_Top5"]}, axis=1
    ).dropna()
    strategies["Hybrid_VT_Mom"] = 0.5 * aligned["vt"] + 0.5 * aligned["mom"]

    # ===== Metrics =====
    print("\n" + "=" * 100)
    print("CROSS-ASSET STRATEGIES PERFORMANCE")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    metrics_dump = {}
    for name, ret in strategies.items():
        m = _ann_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<30} "
            f"{m['ann_return']:>+8.2%} "
            f"{m['sharpe']:>+7.3f} "
            f"{m['sortino']:>+8.3f} "
            f"{m['calmar']:>+7.3f} "
            f"{m['max_dd']:>+7.2%}"
        )

    # Calmar-Bootstrap vs 60/40 Classic
    if "60_40_Classic" in strategies:
        print("\n" + "=" * 100)
        print("CALMAR-BOOTSTRAP vs 60/40 Classic (industry-standard benchmark)")
        print("=" * 100)
        print(
            f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
        )
        print("-" * 100)
        baseline = strategies["60_40_Classic"]
        for name, ret in strategies.items():
            if name == "60_40_Classic":
                continue
            out = calmar_diff_bootstrap(
                ret.dropna(),
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

    # Pro-Epoch Analysis
    print("\n" + "=" * 100)
    print("CROSS-ASSET SUB-PERIOD ANALYSIS (key strategies)")
    print("=" * 100)
    from erweiterung.robustness.sub_period import STANDARD_EPOCHS_US_EQUITY

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
        sr = strategies[sname].dropna()
        sr.index = (
            pd.to_datetime(sr.index, utc=True) if sr.index.tz is None else sr.index
        )
        for epoch in STANDARD_EPOCHS_US_EQUITY:
            mask = (sr.index >= pd.Timestamp(epoch.start, tz="UTC")) & (
                sr.index <= pd.Timestamp(epoch.end, tz="UTC")
            )
            sub = sr[mask]
            if len(sub) < 30:
                continue
            m = _ann_metrics(sub)
            print(
                f"  {sname:<24} {epoch.name:<26} "
                f"{m['ann_return']:>+9.2%} {m['sharpe']:>+7.3f} {m['max_dd']:>+7.2%}"
            )
        print()

    # Save
    eq_df = pd.DataFrame({k: _equity_curve(v) for k, v in strategies.items()})
    eq_df.to_csv("output/erweiterung_cross_asset_equity.csv")
    Path("output/erweiterung_cross_asset_summary.json").write_text(
        json.dumps({"metrics": metrics_dump}, indent=2, default=str)
    )
    print("Saved -> output/erweiterung_cross_asset_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
