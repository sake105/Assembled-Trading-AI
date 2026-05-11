#!/usr/bin/env python
"""Master-Pipeline — End-to-End Erweiterungs-Produktionspfad.

Pipeline
--------
1. Lade Equity-Universum (22 Mega-Caps Long-History oder beliebige Subset).
2. Berechne Mom-12/1-LongOnly als Equity-Faktor-Signal.
3. Lade Cross-Asset-Panel (11 ETFs).
4. Übergebe an MasterAllocator (70/30 Default).
5. Outputte Daily-Returns + Equity-Curve + Performance-Metriken.

Output
------
- output/erweiterung_master_pipeline_equity.csv
- output/erweiterung_master_pipeline_summary.json

Dieser Skript ist die **konsolidierte Produktionsversion** der Master-
Allocation-Forschung. Alle anderen Erweiterungs-Skripte sind Research-
Diagnostik; dieses Skript ist der finale konsumierbare Endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.altdata.yfinance_cache_loader import load_universe_panel  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.qa.equity_curve_audit import audit_equity_curve  # noqa: E402
from erweiterung.strategies.master_allocator import (  # noqa: E402
    MasterAllocator,
    MasterAllocatorConfig,
)

EQUITY_UNIVERSE_22 = [
    "AAPL",
    "ADBE",
    "AMZN",
    "AVGO",
    "COST",
    "CRM",
    "CVX",
    "GOOGL",
    "HD",
    "JNJ",
    "JPM",
    "MA",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "PEP",
    "PG",
    "TSLA",
    "UNH",
    "V",
    "XOM",
]
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


def _cs_long_only(
    panel: pd.DataFrame, signal_col: str, quantile: float = 0.3
) -> pd.DataFrame:
    """Long-Only Cross-Section: top quantile, equal-weight."""
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
    print("ERWEITERUNG MASTER PIPELINE (production-ready endpoint)")
    print("=" * 100)

    # === Equity-Faktor-Signal ===
    print("\nStep 1: Lade Equity-Long-History (22 Mega-Caps) ...")
    eq_panel = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})
    eq_panel["date"] = pd.to_datetime(eq_panel["date"], utc=True)
    eq_panel = eq_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq_panel["return"] = eq_panel.groupby("symbol")["close"].pct_change()
    print(
        f"  Equity panel: {len(eq_panel)} rows, {eq_panel['symbol'].nunique()} symbols, "
        f"{eq_panel['date'].min()} -> {eq_panel['date'].max()}"
    )

    print("\nStep 2: Compute Mom-12/1 long-only equity factor ...")
    mom = momentum_12_1(eq_panel[["date", "symbol", "close"]])
    eq_panel = eq_panel.set_index(["date", "symbol"])
    eq_panel["mom_12_1"] = mom.reindex(eq_panel.index)
    eq_panel = eq_panel.reset_index()
    eq_factor = _cs_long_only(
        eq_panel.dropna(subset=["mom_12_1"]), "mom_12_1", quantile=0.3
    )
    eq_factor_ret = eq_factor.groupby("date").agg(pnl=("pnl", "sum"))["pnl"]
    print(f"  Equity factor return series: {len(eq_factor_ret)} days")

    # === Cross-Asset-Panel ===
    print("\nStep 3: Lade Cross-Asset-Panel (11 ETFs) ...")
    xa_panel = load_universe_panel(
        "data/cache/yfinance",
        CROSS_ASSET_UNIVERSE,
        require_min_rows=200,
        skip_missing=False,
    )
    xa_wide = xa_panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    xa_rets = xa_wide.pct_change().dropna()
    print(
        f"  Cross-asset panel: {len(xa_rets)} days, {xa_rets.shape[1]} assets, "
        f"{xa_rets.index.min()} -> {xa_rets.index.max()}"
    )

    # === Master-Allocator ===
    print("\nStep 4: Run MasterAllocator (70/30 default) ...")
    cfg = MasterAllocatorConfig(sa_weight=0.70)
    alloc = MasterAllocator(cfg)
    out = alloc.allocate(eq_factor_ret, xa_rets)
    print(
        f"  Master return series: {len(out)} days, "
        f"{out.index.min()} -> {out.index.max()}"
    )

    # === Performance ===
    print("\nStep 5: Compute performance metrics ...")
    metrics = {}
    for col in [
        "sa_voltarget",
        "xa_voltarget_ew",
        "xa_mom_top_n",
        "xa_hybrid",
        "master_return",
    ]:
        m = all_metrics(out[col].dropna())
        metrics[col] = m

    print(
        f"\n{'Component':<22} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    for col, m in metrics.items():
        print(
            f"  {col:<20} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # === Audit ===
    print("\nStep 6: Equity-Curve Anomaly Audit ...")
    eq_curve = (1 + out["master_return"].fillna(0)).cumprod()
    eq_curve.index = pd.to_datetime(eq_curve.index, utc=True)
    audit = audit_equity_curve(eq_curve, name="master_pipeline")
    print(f"  Sharpe: {audit.overall_sharpe:.3f}")
    print(f"  Lag-1 Autocorr: {audit.return_autocorr_lag1:.3f}")
    print(f"  Skew: {audit.skew:.3f}, Kurtosis: {audit.kurtosis:.3f}")
    print(f"  WD/Vol-Ratio: {audit.worst_day_vol_ratio:.2f}")
    print(f"  Flags: {audit.flags}")

    # === Save ===
    out_path = Path("output/erweiterung_master_pipeline_equity.csv")
    pd.DataFrame(
        {
            "master_return": out["master_return"],
            "master_equity": (1 + out["master_return"].fillna(0)).cumprod(),
            "sa_voltarget_return": out["sa_voltarget"],
            "xa_hybrid_return": out["xa_hybrid"],
        }
    ).to_csv(out_path)
    print(f"\nSaved -> {out_path}")

    summary_path = Path("output/erweiterung_master_pipeline_summary.json")
    summary = {
        "pipeline_version": "1.0",
        "config": {
            "sa_weight": cfg.sa_weight,
            "sa_target_vol_annual": cfg.sa_target_vol_annual,
            "xa_target_vol_annual": cfg.xa_target_vol_annual,
            "xa_mom_top_n": cfg.xa_mom_top_n,
        },
        "data_inputs": {
            "equity_panel_rows": int(len(eq_panel)),
            "equity_symbols": int(eq_panel["symbol"].nunique()),
            "cross_asset_days": int(len(xa_rets)),
            "cross_asset_symbols": int(xa_rets.shape[1]),
            "master_days": int(len(out)),
            "date_start": str(out.index.min()),
            "date_end": str(out.index.max()),
        },
        "metrics": {
            col: {
                k: (
                    float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else v
                )
                for k, v in m.items()
                if not isinstance(v, (pd.Series, pd.DataFrame))
            }
            for col, m in metrics.items()
        },
        "audit": audit.to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
