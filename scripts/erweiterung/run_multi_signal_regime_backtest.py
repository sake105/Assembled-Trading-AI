#!/usr/bin/env python
"""Multi-Signal Regime-Conditional Backtest auf der 195-Ticker-Equity.

Nutzt drawdown + realized-vol + cross-section-dispersion (+ optional news)
als kombiniertes Stress-Signal, schaltet zwischen Equal-Weight (calm) und
Momentum-12/1-LongOnly (stress).

Vergleicht direkt mit der drawdown-only-Variante aus
``run_regime_conditional_backtest.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.altdata.yfinance_cache_loader import (  # noqa: E402
    load_universe_panel,
    list_cached_symbols,
)
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.multi_signal_regime import (  # noqa: E402
    MultiSignalConfig,
    composite_stress_score,
)


def _convert(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
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


def load_news_panel() -> pd.DataFrame | None:
    """Wenn ``output/news_sentiment_daily.parquet`` existiert, lade ihn."""
    p = Path("output/news_sentiment_daily.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


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

    bench_ret = df["benchmark_equal_weight"].pct_change()
    fac_ret = df["momentum_12_1_LongOnly"].pct_change()

    # Cross-Section-Panel für Dispersion: aus dem Cache wiederherstellen
    cached = list_cached_symbols("data/cache/yfinance")
    cs_panel = load_universe_panel(
        "data/cache/yfinance",
        cached,
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
        require_min_rows=200,
        skip_missing=True,
    )
    panel_pivot = cs_panel.pivot_table(
        index="date", columns="symbol", values="return", aggfunc="first"
    ).sort_index()
    # Index auf bench_ret-Index trimmen
    panel_pivot = panel_pivot.reindex(bench_ret.index)

    # News-Panel (optional)
    news_panel = load_news_panel()
    if news_panel is not None:
        print(
            f"Found news panel: {len(news_panel)} rows, "
            f"{news_panel['timestamp'].min()} -> {news_panel['timestamp'].max()}"
        )

    # Composite-Score berechnen
    cfg = MultiSignalConfig()
    composite = composite_stress_score(
        bench_ret, panel_pivot, sentiment_panel=news_panel, config=cfg
    )

    # Regime t-1-shift (no look-ahead)
    regime_lag = composite["regime"].shift(1)

    # Allocate
    alloc = pd.DataFrame(
        {
            "regime": regime_lag,
            "composite": composite["composite_score"].shift(1),
            "drawdown": composite["drawdown"].shift(1),
            "realized_vol": composite["realized_vol"].shift(1),
            "dispersion": composite["dispersion"].shift(1),
            "news_anomaly": composite["news_anomaly"].shift(1),
            "calm_return": bench_ret,
            "stress_return": fac_ret,
        }
    )
    alloc["allocated_return"] = np.where(
        alloc["regime"] == "stress", alloc["stress_return"], alloc["calm_return"]
    )
    alloc = alloc.dropna(subset=["allocated_return"])

    # Performance
    bench_metrics = all_metrics(bench_ret.dropna())
    fac_metrics = all_metrics(fac_ret.dropna())
    multi_metrics = all_metrics(alloc["allocated_return"].dropna())

    stress_share = (alloc["regime"] == "stress").mean()

    in_stress = alloc.loc[alloc["regime"] == "stress", "allocated_return"]
    in_calm = alloc.loc[alloc["regime"] == "calm", "allocated_return"]

    print("\n" + "=" * 100)
    print("MULTI-SIGNAL REGIME-CONDITIONAL BACKTEST")
    print(f"  Stress-Share: {stress_share:.1%}")
    print(f"  Composite-Threshold: {cfg.stress_threshold}")
    print(
        f"  Signal-Weights: drawdown={cfg.weights['drawdown']:.2f}, "
        f"rv={cfg.weights['realized_vol']:.2f}, "
        f"dispersion={cfg.weights['dispersion']:.2f}, "
        f"news={cfg.weights['news_anomaly']:.2f}"
    )
    print("=" * 100)
    print(f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'MDD':>8}")
    print("-" * 100)
    for label, m in [
        ("Pure Equal-Weight (calm)", bench_metrics),
        ("Pure Mom-12/1 (stress)", fac_metrics),
        ("Multi-Signal-Switched", multi_metrics),
    ]:
        print(
            f"  {label:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Within-Regime
    print("\nWithin-Regime-Diagnostik:")
    for label, sub in [("In stress", in_stress), ("In calm", in_calm)]:
        if sub.empty or sub.std() == 0:
            print(f"  {label}: no data")
            continue
        ann_ret = (1 + sub).prod() ** (252 / len(sub)) - 1
        ann_vol = sub.std() * np.sqrt(252)
        eq = (1 + sub).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        print(
            f"  {label}: AnnRet={ann_ret:+.2%} "
            f"Sharpe={ann_ret/ann_vol if ann_vol > 0 else 0:+.3f} "
            f"MDD={dd:+.2%} days={len(sub)}"
        )

    # Vergleich gegen drawdown-only-Variante (falls vorhanden)
    drawdown_only_csv = Path("output/erweiterung_regime_conditional_equity.csv")
    if drawdown_only_csv.exists():
        do = pd.read_csv(drawdown_only_csv)
        if "date" in do.columns and "equity" in do.columns:
            do["date"] = pd.to_datetime(do["date"], utc=True)
            do = do.set_index("date").sort_index()
            do_ret = do["equity"].pct_change().dropna()
            do_metrics = all_metrics(do_ret)
            print("\nVergleich vs Drawdown-Only-Switch:")
            print(
                f"  Drawdown-Only         : AnnRet={do_metrics['annualized_return']:+.2%} "
                f"Sharpe={do_metrics['sharpe']:+.3f} "
                f"MDD={do_metrics['max_drawdown']:+.2%}"
            )
            print(
                f"  Multi-Signal          : AnnRet={multi_metrics['annualized_return']:+.2%} "
                f"Sharpe={multi_metrics['sharpe']:+.3f} "
                f"MDD={multi_metrics['max_drawdown']:+.2%}"
            )

    # Save
    alloc["equity"] = (1 + alloc["allocated_return"].fillna(0)).cumprod()
    alloc.to_csv("output/erweiterung_multi_signal_regime_equity.csv")

    summary = {
        "stress_share": float(stress_share),
        "config": {
            "stress_threshold": cfg.stress_threshold,
            "weights": cfg.weights,
        },
        "multi_signal": {k: _convert(v) for k, v in multi_metrics.items()},
        "pure_equal_weight": {k: _convert(v) for k, v in bench_metrics.items()},
        "pure_mom_12_1": {k: _convert(v) for k, v in fac_metrics.items()},
    }
    Path("output/erweiterung_multi_signal_regime_summary.json").write_text(
        json.dumps(_walk(summary), indent=2, default=str)
    )
    print("\nSaved -> output/erweiterung_multi_signal_regime_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
