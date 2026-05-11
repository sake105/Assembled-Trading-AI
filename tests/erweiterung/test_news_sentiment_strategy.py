"""Tests für news_sentiment_strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.news_sentiment_strategy import (
    NewsSentimentConfig,
    aggregate_daily_sentiment,
    backtest_news_signal,
    cross_section_signal,
)


def _synthetic_news(
    n_days: int = 50, n_symbols: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for d in range(n_days):
        for sym in [f"SYM{i}" for i in range(n_symbols)]:
            n_articles = int(rng.integers(0, 5))
            for _ in range(n_articles):
                rows.append(
                    {
                        "timestamp": base
                        + pd.Timedelta(days=d, hours=int(rng.integers(0, 24))),
                        "symbol": sym,
                        "sentiment_score": float(rng.normal(0, 0.5)),
                        "sentiment_volume": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_daily_sentiment_basic():
    df = _synthetic_news(n_days=20, n_symbols=5)
    agg = aggregate_daily_sentiment(df)
    assert "sentiment_mean" in agg.columns
    assert "sentiment_smoothed" in agg.columns
    assert (agg["sentiment_count"] > 0).all()


def test_aggregate_empty_returns_empty():
    out = aggregate_daily_sentiment(pd.DataFrame())
    assert out.empty


def test_cross_section_signal_positions():
    df = _synthetic_news(n_days=20, n_symbols=10)
    agg = aggregate_daily_sentiment(df)
    sig = cross_section_signal(agg)
    if not sig.empty:
        assert set(sig["position"].unique()).issubset({-1, 0, 1})


def test_cross_section_long_only():
    df = _synthetic_news(n_days=20, n_symbols=10)
    agg = aggregate_daily_sentiment(df)
    sig = cross_section_signal(agg, NewsSentimentConfig(long_only=True))
    if not sig.empty:
        assert (sig["position"] >= 0).all()


def test_cross_section_long_short_includes_shorts():
    df = _synthetic_news(n_days=30, n_symbols=15)
    agg = aggregate_daily_sentiment(df)
    sig = cross_section_signal(agg, NewsSentimentConfig(long_only=False))
    if not sig.empty:
        # Should have both +1 and -1 positions
        assert (sig["position"] == -1).any()
        assert (sig["position"] == 1).any()


def test_backtest_news_signal_runs():
    df = _synthetic_news(n_days=40, n_symbols=10)
    agg = aggregate_daily_sentiment(df)

    # Synthetic returns
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    rets = []
    for d in dates:
        for sym in [f"SYM{i}" for i in range(10)]:
            rets.append(
                {"date": d, "symbol": sym, "return": float(rng.normal(0.0005, 0.01))}
            )
    pr = pd.DataFrame(rets)

    port = backtest_news_signal(
        agg, pr, NewsSentimentConfig(min_symbols_for_cross_section=3)
    )
    assert isinstance(port, pd.Series)


def test_backtest_returns_empty_for_empty_signal():
    pr = pd.DataFrame({"date": [], "symbol": [], "return": []})
    port = backtest_news_signal(pd.DataFrame(), pr)
    assert port.empty


def test_min_cross_section_filter():
    """Days with < min_symbols are filtered out."""
    df = _synthetic_news(n_days=10, n_symbols=3)
    agg = aggregate_daily_sentiment(df)
    sig = cross_section_signal(
        agg, NewsSentimentConfig(min_symbols_for_cross_section=10)
    )
    # n_symbols=3 < 10, alle Tage werden gefiltert
    assert sig.empty
