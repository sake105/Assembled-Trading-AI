#!/usr/bin/env python
"""Macro-Stress-Regime-Backtest auf der Erweiterungs-Equity.

Lädt VIX/yields/HY-Spreads aus ``output/macro.parquet`` + FRED-T10YIE,
berechnet einen Macro-Stress-Composite und schaltet zwischen Equal-Weight
(calm) und Momentum-12/1-LongOnly (stress).

Vergleicht direkt mit:
- Drawdown-Only-Switch (output/erweiterung_regime_conditional_equity.csv)
- Multi-Signal-Switch (output/erweiterung_multi_signal_regime_equity.csv)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.macro_stress_signals import (  # noqa: E402
    MacroStressConfig,
    load_macro_panel,
    macro_stress_composite,
)


def _convert(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return o


def _walk(o):
    if isinstance(o, dict):
        return {str(k): _walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(v) for v in o]
    return _convert(o)


def main():
    eq_csv = Path("output/erweiterung_expanded_universe_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found.")
        return 1

    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    bench_ret = df["benchmark_equal_weight"].pct_change()
    fac_ret = df["momentum_12_1_LongOnly"].pct_change()

    # Macro-Panel laden
    macro = load_macro_panel()
    if macro.empty:
        print("ERROR: no macro data found.")
        return 1
    print(
        f"Macro panel: {len(macro)} rows, {len(macro.columns)} cols, "
        f"{macro.index.min()} -> {macro.index.max()}"
    )

    # Auf Backtest-Range trimmen
    macro = macro.loc[df.index.min() : df.index.max()]

    # Test mehrere Threshold-Konfigurationen
    print("\n" + "=" * 100)
    print("MACRO-STRESS REGIME-CONDITIONAL BACKTEST")
    print("=" * 100)
    print(
        f"{'Threshold':<12} {'Stress%':>8} {'AnnRet':>9} {'Sharpe':>8} {'MDD':>8} {'in_stress_Sharpe':>18}"
    )
    print("-" * 100)

    canonical_alloc = None
    canonical_metrics = None
    for thr in (0.30, 0.45, 0.55, 0.65):
        cfg = MacroStressConfig(stress_threshold=thr)
        composite = macro_stress_composite(macro, cfg)
        # Reindex auf bench_ret-index, forward-fill
        regime = composite["regime"].reindex(bench_ret.index, method="ffill")
        regime_lag = regime.shift(1)
        alloc_ret = np.where(regime_lag == "stress", fac_ret, bench_ret)
        alloc_ret = pd.Series(alloc_ret, index=bench_ret.index).dropna()

        if alloc_ret.empty:
            continue
        metrics = all_metrics(alloc_ret)
        stress_share = (regime_lag == "stress").mean()

        in_stress = alloc_ret.loc[regime_lag.reindex(alloc_ret.index) == "stress"]
        if in_stress.empty or in_stress.std() == 0:
            in_stress_sharpe = 0.0
        else:
            in_stress_sharpe = (in_stress.mean() / in_stress.std()) * np.sqrt(252)

        print(
            f"  {thr:<10.2f} "
            f"{stress_share:>7.1%} "
            f"{metrics.get('annualized_return', 0):>+8.2%} "
            f"{metrics.get('sharpe', 0):>+7.3f} "
            f"{metrics.get('max_drawdown', 0):>+7.2%} "
            f"{in_stress_sharpe:>+17.3f}"
        )

        if thr == 0.55:  # canonical
            canonical_alloc = pd.DataFrame(
                {
                    "regime": regime_lag,
                    "composite": composite["composite_score"]
                    .reindex(bench_ret.index, method="ffill")
                    .shift(1),
                    "vix_spike": composite["vix_spike"]
                    .reindex(bench_ret.index, method="ffill")
                    .shift(1),
                    "yield_curve_stress": composite["yield_curve_stress"]
                    .reindex(bench_ret.index, method="ffill")
                    .shift(1),
                    "hy_spread_widening": composite["hy_spread_widening"]
                    .reindex(bench_ret.index, method="ffill")
                    .shift(1),
                    "real_yield_spike": composite["real_yield_spike"]
                    .reindex(bench_ret.index, method="ffill")
                    .shift(1),
                    "calm_return": bench_ret,
                    "stress_return": fac_ret,
                    "allocated_return": alloc_ret,
                }
            )
            canonical_metrics = metrics

    # Reference benchmarks
    print("\nReference (no switching):")
    bench_metrics = all_metrics(bench_ret.dropna())
    fac_metrics = all_metrics(fac_ret.dropna())
    print(
        f"  Pure Equal-Weight:  AnnRet={bench_metrics['annualized_return']:+.2%} "
        f"Sharpe={bench_metrics['sharpe']:+.3f} MDD={bench_metrics['max_drawdown']:+.2%}"
    )
    print(
        f"  Pure Mom-12/1-LO:   AnnRet={fac_metrics['annualized_return']:+.2%} "
        f"Sharpe={fac_metrics['sharpe']:+.3f} MDD={fac_metrics['max_drawdown']:+.2%}"
    )

    # Vergleich
    drawdown_only_csv = Path("output/erweiterung_regime_conditional_equity.csv")
    multi_signal_csv = Path("output/erweiterung_multi_signal_regime_equity.csv")
    print("\nVergleich:")
    for label, path in [
        ("Drawdown-Only-Switch", drawdown_only_csv),
        ("Multi-Signal-Switch", multi_signal_csv),
    ]:
        if not path.exists():
            continue
        do = pd.read_csv(path)
        if "date" in do.columns and "equity" in do.columns:
            do["date"] = pd.to_datetime(do["date"], utc=True)
            do = do.set_index("date").sort_index()
            do_ret = do["equity"].pct_change().dropna()
            do_m = all_metrics(do_ret)
            print(
                f"  {label:<28} AnnRet={do_m['annualized_return']:+.2%} "
                f"Sharpe={do_m['sharpe']:+.3f} MDD={do_m['max_drawdown']:+.2%}"
            )
    if canonical_metrics:
        print(
            f"  Macro-Regime (thr=0.55)      AnnRet={canonical_metrics['annualized_return']:+.2%} "
            f"Sharpe={canonical_metrics['sharpe']:+.3f} "
            f"MDD={canonical_metrics['max_drawdown']:+.2%}"
        )

    # Save
    if canonical_alloc is not None:
        canonical_alloc["equity"] = (
            1 + canonical_alloc["allocated_return"].fillna(0)
        ).cumprod()
        canonical_alloc.to_csv("output/erweiterung_macro_regime_equity.csv")

        summary = {
            "canonical_threshold": 0.55,
            "metrics": {k: _convert(v) for k, v in canonical_metrics.items()},
            "reference": {
                "equal_weight": {k: _convert(v) for k, v in bench_metrics.items()},
                "momentum_12_1_LongOnly": {
                    k: _convert(v) for k, v in fac_metrics.items()
                },
            },
        }
        Path("output/erweiterung_macro_regime_summary.json").write_text(
            json.dumps(_walk(summary), indent=2, default=str)
        )
        print("\nSaved -> output/erweiterung_macro_regime_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
