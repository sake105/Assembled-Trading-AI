#!/usr/bin/env python
"""Ensemble-Regime-Backtest: kombiniert Drawdown + Multi-Signal + Macro.

Lädt die drei Regime-Detector-Outputs und kombiniert sie via:
- weighted_mean (canonical)
- majority
- conservative
- any

Vergleich gegen die drei einzelnen Detectors + Pure Long-Only/Equal-Weight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.altdata.yfinance_cache_loader import (  # noqa: E402
    list_cached_symbols,
    load_universe_panel,
)
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.ensemble_regime import (  # noqa: E402
    EnsembleConfig,
    ensemble_regime,
)
from erweiterung.strategies.macro_stress_signals import (  # noqa: E402
    MacroStressConfig,
    load_macro_panel,
    macro_stress_composite,
)
from erweiterung.strategies.multi_signal_regime import (  # noqa: E402
    MultiSignalConfig,
    composite_stress_score,
)
from erweiterung.strategies.regime_conditional_allocator import (  # noqa: E402
    RegimeConfig,
    detect_regime,
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

    # ===== Drawdown-Detector =====
    dd_regime = detect_regime(bench_ret, RegimeConfig(drawdown_threshold=0.08))
    eq = (1 + bench_ret.fillna(0)).cumprod()
    rolling_max = eq.rolling(60, min_periods=1).max()
    dd_score = (1 - eq / rolling_max).abs().clip(0, 0.5) / 0.5

    # ===== Multi-Signal-Detector =====
    cached = list_cached_symbols("data/cache/yfinance")
    cs_panel = load_universe_panel(
        "data/cache/yfinance",
        cached,
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
        require_min_rows=200,
        skip_missing=True,
    )
    panel_pivot = (
        cs_panel.pivot_table(
            index="date", columns="symbol", values="return", aggfunc="first"
        )
        .sort_index()
        .reindex(bench_ret.index)
    )
    ms_out = composite_stress_score(
        bench_ret, panel_pivot, sentiment_panel=None, config=MultiSignalConfig()
    )
    ms_regime = ms_out["regime"]
    ms_score = ms_out["composite_score"]

    # ===== Macro-Detector =====
    macro = load_macro_panel()
    macro = macro.loc[df.index.min() : df.index.max()]
    macro_out = macro_stress_composite(macro, MacroStressConfig(stress_threshold=0.45))
    # Reindex auf bench-Index
    macro_regime = macro_out["regime"].reindex(bench_ret.index, method="ffill")
    macro_score = macro_out["composite_score"].reindex(bench_ret.index, method="ffill")

    # ===== Ensemble: 4 schemata testen =====
    print("\n" + "=" * 100)
    print("ENSEMBLE REGIME-CONDITIONAL BACKTEST")
    print("=" * 100)
    print(
        f"{'Scheme':<22} {'Threshold':>10} {'Stress%':>8} {'AnnRet':>9} {'Sharpe':>8} {'MDD':>8}"
    )
    print("-" * 100)

    canonical_alloc = None
    canonical_metrics = None

    schemes = [
        ("majority", None),
        ("conservative", None),
        ("any", None),
        ("weighted_mean", 0.40),
        ("weighted_mean", 0.50),
        ("weighted_mean", 0.60),
    ]
    for scheme, thr in schemes:
        if thr is None:
            cfg = EnsembleConfig(voting_scheme=scheme, smoothing_days=3)
            label_str = f"{scheme}"
        else:
            cfg = EnsembleConfig(voting_scheme=scheme, threshold=thr, smoothing_days=3)
            label_str = f"{scheme}@{thr:.2f}"

        out = ensemble_regime(
            drawdown_regime=dd_regime,
            multi_signal_regime_in=ms_regime,
            macro_regime_in=macro_regime,
            drawdown_score=dd_score,
            multi_signal_score=ms_score,
            macro_score=macro_score,
            config=cfg,
        )
        regime_lag = out["regime"].shift(1)
        alloc_ret = pd.Series(
            np.where(regime_lag == "stress", fac_ret, bench_ret),
            index=bench_ret.index,
        ).dropna()
        if alloc_ret.empty:
            continue
        m = all_metrics(alloc_ret)
        stress_share = (regime_lag == "stress").mean()
        thr_print = thr if thr is not None else float("nan")
        print(
            f"  {label_str:<20} "
            f"{(thr_print if thr is not None else 0):>10.2f} "
            f"{stress_share:>7.1%} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

        if scheme == "weighted_mean" and thr == 0.50:
            canonical_alloc = pd.DataFrame(
                {
                    "regime": regime_lag,
                    "ensemble_score": out["ensemble_score"].shift(1),
                    "drawdown_regime": dd_regime,
                    "ms_regime": ms_regime,
                    "macro_regime": macro_regime,
                    "calm_return": bench_ret,
                    "stress_return": fac_ret,
                    "allocated_return": alloc_ret,
                }
            )
            canonical_metrics = m

    # Reference
    print("\nReference:")
    for label, ret_in in [
        ("Pure Equal-Weight", bench_ret),
        ("Pure Mom-12/1 LO", fac_ret),
    ]:
        m = all_metrics(ret_in.dropna())
        print(
            f"  {label:<28} AnnRet={m['annualized_return']:+.2%} "
            f"Sharpe={m['sharpe']:+.3f} MDD={m['max_drawdown']:+.2%}"
        )

    # Vergleich gegen einzelne Detector-Outputs
    print("\nVergleich gegen einzelne Detectors:")
    for label, path in [
        ("Drawdown-Only", "output/erweiterung_regime_conditional_equity.csv"),
        ("Multi-Signal", "output/erweiterung_multi_signal_regime_equity.csv"),
        ("Macro-Only", "output/erweiterung_macro_regime_equity.csv"),
    ]:
        p = Path(path)
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "date" in d.columns and "equity" in d.columns:
            d["date"] = pd.to_datetime(d["date"], utc=True)
            d = d.set_index("date").sort_index()
            r = d["equity"].pct_change().dropna()
            m = all_metrics(r)
            print(
                f"  {label:<28} AnnRet={m['annualized_return']:+.2%} "
                f"Sharpe={m['sharpe']:+.3f} MDD={m['max_drawdown']:+.2%}"
            )
    if canonical_metrics:
        print(
            f"  Ensemble (weighted, thr=0.50) AnnRet={canonical_metrics['annualized_return']:+.2%} "
            f"Sharpe={canonical_metrics['sharpe']:+.3f} "
            f"MDD={canonical_metrics['max_drawdown']:+.2%}"
        )

    # Save canonical
    if canonical_alloc is not None:
        canonical_alloc["equity"] = (
            1 + canonical_alloc["allocated_return"].fillna(0)
        ).cumprod()
        canonical_alloc.to_csv("output/erweiterung_ensemble_regime_equity.csv")
        Path("output/erweiterung_ensemble_regime_summary.json").write_text(
            json.dumps(
                _walk(
                    {"metrics": {k: _convert(v) for k, v in canonical_metrics.items()}}
                ),
                indent=2,
                default=str,
            )
        )
        print("\nSaved -> output/erweiterung_ensemble_regime_*")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
