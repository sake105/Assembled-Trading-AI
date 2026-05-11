#!/usr/bin/env python
"""News-Impact End-to-End Pipeline mit echten News-Daten.

Aktiviert die news_impact-Module (Skeleton vorher, jetzt mit Daten):
1. rolling_sentiment_baseline: rolling-mean per symbol
2. compute_surprise: realized - expected
3. standardized_surprise: z-score normalized
4. cross_section_surprise_rank: percentile per day
5. surprise_to_signal: long-top, short-bottom

Daten-Caveat: 423 News-Rows / 5 Monate / 91 Symbole = sparse. Demo der
Pipeline-Mechanik, kein Production-Backtest.

PR-Pfad: Module sind drop-in für Mainline-news-pipeline kompatibel.
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
from erweiterung.news_impact.news_surprise import (  # noqa: E402
    compute_surprise,
    cross_section_surprise_rank,
    rolling_sentiment_baseline,
    standardized_surprise,
    surprise_to_signal,
)


def main():
    print("=" * 100)
    print("NEWS-IMPACT PIPELINE (real news data, sparse 5 months)")
    print("=" * 100)

    p = Path("output/news_sentiment_fused.parquet")
    if not p.exists():
        print(f"ERROR: {p} not found.")
        return 1

    news = pd.read_parquet(p)
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    news["date"] = news["timestamp"].dt.normalize()
    news = news.rename(columns={"sentiment_score": "sentiment"})
    print(f"News rows: {len(news)}, symbols: {news['symbol'].nunique()}, "
          f"days: {news['date'].nunique()}")

    # Aggregate daily per symbol
    daily = news.groupby(["date", "symbol"]).agg(
        sentiment=("sentiment", "mean"),
        count=("sentiment", "count"),
    ).reset_index()
    print(f"Daily aggregated: {len(daily)} rows")

    # Step 1: Rolling baseline (in our sparse case: 30 days)
    print("\nStep 1: Rolling-Sentiment-Baseline (30d) ...")
    base = rolling_sentiment_baseline(
        daily, window=30, sentiment_col="sentiment",
        date_col="date", symbol_col="symbol",
    )
    print(f"  Baseline rows: {len(base)}, non-NaN: {base['baseline'].notna().sum()}")

    # Step 2: Surprise = realized - baseline.
    # rolling_sentiment_baseline returns the SAME df with added "baseline" col,
    # so we just pass it through.
    print("\nStep 2: News-Surprise ...")
    surprise = compute_surprise(base, sentiment_col="sentiment", baseline_col="baseline")
    print(f"  Surprise rows: {len(surprise)}")
    if "surprise" in surprise.columns:
        print(f"  Surprise stats: mean={surprise['surprise'].mean():.3f}, "
              f"std={surprise['surprise'].std():.3f}")

    # Step 3: Standardized (z-score) surprise — works on raw sentiment
    print("\nStep 3: Standardized Surprise (z-score, 20d) ...")
    std_surp = standardized_surprise(surprise, sentiment_col="sentiment", window=20)
    print(f"  Rows: {len(std_surp)}, columns: {list(std_surp.columns)}")

    # Step 4: Cross-section rank per day
    print("\nStep 4: Cross-Section Surprise Rank ...")
    ranked = cross_section_surprise_rank(std_surp)
    print(f"  Rows: {len(ranked)}")

    # Step 5: Surprise-to-Signal — returns Series of {-1, 0, +1}
    print("\nStep 5: Surprise-to-Signal (threshold=1.0) ...")
    sig_series = surprise_to_signal(ranked, threshold=1.0)
    # Attach to ranked DataFrame for downstream merge
    signal = ranked.copy()
    signal["position"] = sig_series.values if len(sig_series) == len(ranked) else 0
    if (signal["position"] != 0).any():
        n_active_days = (signal.groupby("date")["position"].apply(lambda s: (s != 0).any())).sum()
        print(f"  Days with active signals: {n_active_days}")

    # If we have positions, try simple backtest
    sym_in_news = sorted(signal["symbol"].unique())
    print(f"\nLoading prices for {len(sym_in_news)} news-symbols ...")
    try:
        panel = load_universe_panel(
            "data/cache/yfinance", sym_in_news,
            require_min_rows=20, skip_missing=True,
        )
    except Exception as e:
        print(f"Price load failed: {e}")
        return 1
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    prices = panel[["date", "symbol", "return"]].dropna()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.normalize()

    signal["date"] = pd.to_datetime(signal["date"], utc=True).dt.normalize()
    merged = signal.merge(prices, on=["date", "symbol"], how="inner")
    if merged.empty:
        print("No price overlap with news symbols")
        return 1

    # PnL daily (equal-weight long)
    daily_pnl = merged[merged["position"] > 0].groupby("date").apply(
        lambda g: g["return"].mean()
    )
    print(f"\nBacktest days: {len(daily_pnl)}")
    if len(daily_pnl) < 5:
        print("Too few days for meaningful metrics")
        return 1

    m = all_metrics(daily_pnl.dropna())
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 40)
    for k in ["annualized_return", "sharpe", "sortino", "max_drawdown"]:
        if k in m:
            print(f"  {k:<18} {m[k]:>+9.3f}")

    print("\n[OK] News-Impact-Pipeline End-to-End validiert mit echten Daten.")
    print("CAVEAT: 5-Monate-Sample, Sharpe nicht statistisch belastbar.")

    # Save
    Path("output/erweiterung_news_impact_signal.csv").write_text(signal.to_csv(index=False))
    Path("output/erweiterung_news_impact_summary.json").write_text(
        json.dumps(
            {
                "n_news_rows": int(len(news)),
                "n_symbols": int(news["symbol"].nunique()),
                "n_backtest_days": int(len(daily_pnl)),
                "metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                           for k, v in m.items() if not isinstance(v, (pd.Series, pd.DataFrame))},
            },
            indent=2, default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
