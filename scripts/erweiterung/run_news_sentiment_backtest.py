#!/usr/bin/env python
"""News-Sentiment-Strategie-Backtest auf den vorhandenen News-Daten.

Daten-Caveat
------------
Das Repo enthält nur 423 News-Rows über 2025-12 → 2026-05 (4.5 Monate).
Das ist **zu sparse** für statistisch belastbare Ergebnisse — dieser
Backtest ist eine Demo-Pipeline, kein finaler Strategie-Beweis.

Pipeline
--------
1. Lade output/news_sentiment_fused.parquet
2. Aggregiere zu Daily-Sentiment pro Symbol
3. Cross-Section-Signal: Long-Top-20%
4. Lade Preisdaten aus yfinance-Cache für betroffene Symbole
5. Backtest + Audit
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
from erweiterung.qa.equity_curve_audit import audit_equity_curve  # noqa: E402
from erweiterung.strategies.news_sentiment_strategy import (  # noqa: E402
    NewsSentimentConfig,
    aggregate_daily_sentiment,
    backtest_news_signal,
    cross_section_signal,
)


def main():
    news_path = Path("output/news_sentiment_fused.parquet")
    if not news_path.exists():
        print(f"ERROR: {news_path} not found.")
        return 1

    print(f"Loading news data: {news_path}")
    news = pd.read_parquet(news_path)
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    print(
        f"  Rows: {len(news)}, "
        f"Date range: {news['timestamp'].min()} -> {news['timestamp'].max()}"
    )
    print(f"  Unique symbols: {news['symbol'].nunique()}")

    # Aggregate
    print("\nAggregating daily sentiment ...")
    cfg = NewsSentimentConfig(
        aggregation_window_days=5,
        quantile_long=0.20,
        long_only=True,
        min_symbols_for_cross_section=5,
        smoothing_window=3,
    )
    agg = aggregate_daily_sentiment(news, cfg)
    print(
        f"  Aggregated: {len(agg)} rows ({agg['date'].nunique()} days, "
        f"{agg['symbol'].nunique()} symbols)"
    )

    # Cross-section signal
    sig = cross_section_signal(agg, cfg)
    print(
        f"  Cross-section signal: {len(sig)} rows, "
        f"{(sig['position'] == 1).sum()} long-positions across all days"
    )

    # Load price returns for affected symbols
    symbols = sorted(sig["symbol"].unique())
    print(f"\nLoading prices for {len(symbols)} symbols ...")
    try:
        panel = load_universe_panel(
            "data/cache/yfinance", symbols, require_min_rows=50, skip_missing=True
        )
    except Exception as e:
        print(f"Price load failed: {e}")
        return 1

    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    prices = panel[["date", "symbol", "return"]].dropna()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.normalize()
    print(f"  Price rows: {len(prices)}")
    print(
        f"  Missing symbols (no cache): {panel.attrs.get('skipped_symbols', [])[:10]}"
    )

    # Backtest
    print("\nRunning backtest ...")
    port = backtest_news_signal(agg, prices, cfg)
    if port.empty:
        print(
            "Portfolio empty — likely no overlap between news symbols and price cache"
        )
        return 1
    print(
        f"  Portfolio days: {len(port)}, "
        f"first: {port.index.min()}, last: {port.index.max()}"
    )

    # Metrics
    m = all_metrics(port.dropna())
    print("\n" + "=" * 80)
    print("NEWS-SENTIMENT BACKTEST RESULTS (Demo — Statistik nicht signifikant!)")
    print("=" * 80)
    print(f"  AnnRet: {m.get('annualized_return', 0):+.2%}")
    print(f"  Sharpe: {m.get('sharpe', 0):+.3f}")
    print(f"  Sortino: {m.get('sortino', 0):+.3f}")
    print(f"  Calmar: {m.get('calmar', 0):+.3f}")
    print(f"  MDD: {m.get('max_drawdown', 0):+.2%}")
    print(f"  N days: {len(port)}")

    # Audit (skip if too few days)
    if len(port) >= 60:
        eq = (1 + port.fillna(0)).cumprod()
        eq.index = pd.to_datetime(eq.index, utc=True)
        audit = audit_equity_curve(eq, name="news_sentiment")
        sh = audit.overall_sharpe if audit.overall_sharpe is not None else 0
        ac = audit.return_autocorr_lag1 if audit.return_autocorr_lag1 is not None else 0
        print(f"  Audit Sharpe: {sh:.3f}, Lag-1 Autocorr: {ac:.3f}")
        print(f"  Audit Flags: {audit.flags}")
    else:
        print(
            f"  AUDIT SKIPPED: only {len(port)} days -- TOO FEW for meaningful audit."
        )
        audit = type("S", (), {"flags": ["INSUFFICIENT_DATA"]})()

    # Compare to equal-weight of same symbol set
    ew_port = (
        prices.loc[prices["date"].between(port.index.min(), port.index.max())]
        .groupby("date")["return"]
        .mean()
    )
    m_ew = all_metrics(ew_port.dropna())
    print("\nBenchmark: Equal-Weight of same symbol-universe:")
    print(f"  AnnRet: {m_ew.get('annualized_return', 0):+.2%}")
    print(f"  Sharpe: {m_ew.get('sharpe', 0):+.3f}")
    print(f"  MDD: {m_ew.get('max_drawdown', 0):+.2%}")

    # Save
    pd.DataFrame(
        {
            "news_sentiment_return": port,
            "news_sentiment_equity": (1 + port.fillna(0)).cumprod(),
        }
    ).to_csv("output/erweiterung_news_sentiment_equity.csv")
    Path("output/erweiterung_news_sentiment_summary.json").write_text(
        json.dumps(
            {
                "caveat": "DEMO — News data only 2025-12 to 2026-05, statistik not significant",
                "n_news_rows": int(len(news)),
                "n_portfolio_days": int(len(port)),
                "metrics": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer))
                        else v
                    )
                    for k, v in m.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))
                },
                "audit_flags": list(audit.flags),
                "ew_benchmark_metrics": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer))
                        else v
                    )
                    for k, v in m_ew.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))
                },
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_news_sentiment_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
