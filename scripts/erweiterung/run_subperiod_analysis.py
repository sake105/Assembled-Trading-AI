#!/usr/bin/env python
"""Sub-Period-Analyse auf der Expanded-Universe-Equity-Curve.

Liest ``output/erweiterung_expanded_universe_equity.csv`` und zerlegt die
Backtest-Returns nach den Standard-Epochen aus
``erweiterung.robustness.sub_period``. Vergleicht zugleich mit der
Original-Equity-Curve aus ``output/equity_curve_baseline.csv`` falls vorhanden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.robustness.sub_period import (  # noqa: E402
    STANDARD_EPOCHS_US_EQUITY,
)


def _ann_metrics(ret: pd.Series) -> dict:
    if ret.empty or ret.std() == 0:
        return {"ann_return": None, "ann_vol": None, "sharpe": None, "max_dd": None}
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else None,
        "max_dd": float(dd.min()),
        "n_days": int(len(ret)),
    }


def main():
    eq_csv = Path("output/erweiterung_expanded_universe_equity.csv")
    if not eq_csv.exists():
        print(
            f"ERROR: {eq_csv} not found — run run_expanded_universe_backtest.py first."
        )
        return 1

    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    # Erw-Strategie auswählen (residual_momentum_LongOnly: bestes Sharpe pro Risiko)
    strategy_cols = [c for c in df.columns if not c.startswith("Unnamed")]
    target_strategies = [
        "residual_momentum_LongOnly",
        "combined_LongOnly_EqWeight",
        "momentum_12_1_LongOnly",
        "benchmark_equal_weight",
    ]
    target_strategies = [s for s in target_strategies if s in strategy_cols]

    summary: dict = {}
    for strat in target_strategies:
        eq = df[strat].dropna()
        ret = eq.pct_change().dropna()
        per_epoch = {}
        for epoch in STANDARD_EPOCHS_US_EQUITY:
            mask = (ret.index >= pd.Timestamp(epoch.start, tz="UTC")) & (
                ret.index <= pd.Timestamp(epoch.end, tz="UTC")
            )
            sub = ret[mask]
            if len(sub) < 5:
                continue
            per_epoch[epoch.name] = _ann_metrics(sub)
        summary[strat] = {
            "overall": _ann_metrics(ret),
            "by_epoch": per_epoch,
        }

    # Original-Vergleich
    orig_path = Path("output/equity_curve_baseline.csv")
    if orig_path.exists():
        orig = pd.read_csv(orig_path, parse_dates=["timestamp"])
        orig["date"] = pd.to_datetime(orig["timestamp"], utc=True)
        orig = orig.set_index("date").sort_index()
        orig_ret = orig["equity"].pct_change().dropna()
        per_epoch_orig = {}
        for epoch in STANDARD_EPOCHS_US_EQUITY:
            mask = (orig_ret.index >= pd.Timestamp(epoch.start, tz="UTC")) & (
                orig_ret.index <= pd.Timestamp(epoch.end, tz="UTC")
            )
            sub = orig_ret[mask]
            if len(sub) < 5:
                continue
            per_epoch_orig[epoch.name] = _ann_metrics(sub)
        summary["__original_baseline__"] = {
            "overall": _ann_metrics(orig_ret),
            "by_epoch": per_epoch_orig,
        }

    out_path = Path("output/erweiterung_subperiod_analysis.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved -> {out_path}")

    # Print table
    print("\n" + "=" * 100)
    print("SUB-PERIOD ANALYSIS")
    print("=" * 100)
    epochs_to_show = [
        "Post_GFC",
        "COVID_Crash",
        "Recovery_2020_2021",
        "Inflation_2022",
        "Modern_2023_plus",
    ]
    print(
        f"{'Strategy':<30} {'Epoch':<22} {'AnnRet':>10} {'Sharpe':>8} {'MDD':>8} {'Days':>5}"
    )
    print("-" * 100)
    for strat, data in summary.items():
        for epoch_name in epochs_to_show:
            ep = data["by_epoch"].get(epoch_name)
            if ep is None:
                continue
            print(
                f"  {strat:<28} {epoch_name:<22} "
                f"{(ep.get('ann_return') or 0):>+9.2%} "
                f"{(ep.get('sharpe') or 0):>+7.3f} "
                f"{(ep.get('max_dd') or 0):>+7.2%} "
                f"{ep.get('n_days', 0):>5d}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
