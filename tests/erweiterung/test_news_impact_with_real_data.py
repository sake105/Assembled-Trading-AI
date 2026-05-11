"""Tests aller news_impact-Module mit ECHTEN News-Daten (korrekte APIs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


NEWS_DATA_EXISTS = Path("output/news_sentiment_fused.parquet").exists()
pytestmark = pytest.mark.skipif(
    not NEWS_DATA_EXISTS, reason="news_sentiment_fused.parquet missing"
)


@pytest.fixture(scope="module")
def real_news_df():
    df = pd.read_parquet("output/news_sentiment_fused.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df = df.rename(columns={"sentiment_score": "sentiment"})
    return df


@pytest.fixture(scope="module")
def real_returns_panel():
    """Long-format [date, symbol, return] for news_impact modules."""
    cache_dir = Path("data/cache/yfinance")
    if not cache_dir.exists():
        pytest.skip("yfinance cache missing")
    symbols = ["AAPL", "AMD", "META", "GOOGL", "NVDA", "MU", "ORCL", "PLTR", "IBM"]
    frames = []
    for s in symbols:
        p = cache_dir / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = s
        frames.append(df[["date", "symbol", "close"]])
    if not frames:
        pytest.skip("no price data")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.normalize()
    panel = panel.sort_values(["symbol", "date"])
    panel["return"] = panel.groupby("symbol")["close"].pct_change()
    return panel[["date", "symbol", "return"]].dropna()


def test_fit_news_decay_on_real_data(real_news_df, real_returns_panel):
    from erweiterung.news_impact.decay_model import fit_news_decay_model

    daily_news = real_news_df.groupby(["date", "symbol"], as_index=False).agg(
        sentiment=("sentiment", "mean")
    )
    fit = fit_news_decay_model(
        daily_news,
        real_returns_panel,
        sentiment_col="sentiment",
        date_col="date",
        symbol_col="symbol",
        horizons=(1, 2, 5),
    )
    assert hasattr(fit, "decay_lambda")
    assert hasattr(fit, "n_obs")
    assert fit.n_obs >= 0


def test_cumulative_news_impact_signal_with_fit(real_news_df, real_returns_panel):
    from erweiterung.news_impact.decay_model import (
        cumulative_news_impact_signal,
        fit_news_decay_model,
    )

    daily_news = real_news_df.groupby(["date", "symbol"], as_index=False).agg(
        sentiment=("sentiment", "mean")
    )
    fit = fit_news_decay_model(daily_news, real_returns_panel, horizons=(1, 3, 5))
    out = cumulative_news_impact_signal(daily_news, fit, horizon_max=5)
    assert isinstance(out, pd.DataFrame)


def test_cluster_news_tfidf_basic():
    from erweiterung.news_impact.event_clustering import cluster_news_tfidf

    headlines = [
        "Apple earnings beat consensus",
        "Apple Q3 earnings exceed expectations",
        "Federal Reserve raises rates",
        "Fed hikes interest rates",
        "Tesla unveils new model",
    ]
    clusters = cluster_news_tfidf(headlines, distance_threshold=0.5)
    assert len(clusters) == 5


def test_reactivity_panel_real_data(real_news_df, real_returns_panel):
    from erweiterung.news_impact.reactivity_index import reactivity_panel

    daily_news = real_news_df.groupby(["date", "symbol"], as_index=False).agg(
        sentiment=("sentiment", "mean")
    )
    out = reactivity_panel(daily_news, real_returns_panel, min_news=3)
    assert isinstance(out, pd.DataFrame)


def test_sentiment_spillover_matrix(real_news_df, real_returns_panel):
    from erweiterung.news_impact.cross_asset_spillover import sentiment_spillover_matrix

    daily_news = real_news_df.groupby(["date", "symbol"], as_index=False).agg(
        sentiment=("sentiment", "mean")
    )
    out = sentiment_spillover_matrix(daily_news, real_returns_panel, horizon_days=3)
    assert isinstance(out, pd.DataFrame)


def test_jensen_shannon_divergence():
    from erweiterung.news_impact.topic_drift import jensen_shannon_divergence

    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    jsd = jensen_shannon_divergence(p, q)
    assert 0.0 <= jsd <= 1.0


def test_jsd_zero_for_identical():
    from erweiterung.news_impact.topic_drift import jensen_shannon_divergence

    p = np.array([0.5, 0.3, 0.2])
    jsd = jensen_shannon_divergence(p, p)
    assert abs(jsd) < 1e-9


def test_classify_time_of_day_series(real_news_df):
    from erweiterung.news_impact.time_of_day_impact import classify_time_of_day

    ts_series = real_news_df["timestamp"].head(50)
    out = classify_time_of_day(ts_series)
    valid = {"pre_market", "intraday", "after_hours", "regular", "overnight"}
    assert set(out.dropna().unique()).issubset(valid)


def test_news_impact_end_to_end_pipeline(real_news_df, real_returns_panel):
    from erweiterung.news_impact.decay_model import (
        cumulative_news_impact_signal,
        fit_news_decay_model,
    )
    from erweiterung.news_impact.reactivity_index import reactivity_panel

    daily_news = real_news_df.groupby(["date", "symbol"], as_index=False).agg(
        sentiment=("sentiment", "mean")
    )
    fit = fit_news_decay_model(daily_news, real_returns_panel, horizons=(1, 2, 5))
    impact = cumulative_news_impact_signal(daily_news, fit, horizon_max=5)
    react = reactivity_panel(daily_news, real_returns_panel, min_news=3)
    assert isinstance(impact, pd.DataFrame)
    assert isinstance(react, pd.DataFrame)
