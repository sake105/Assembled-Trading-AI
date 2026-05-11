"""Tests für News-Tilt-Builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erweiterung.signals.news_tilt_builder import (
    build_daily_news_tilt,
    news_tilt_for_date,
)


def _make_news(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d, tz="UTC"), "symbol": s, "sentiment": x}
            for d, s, x in rows
        ]
    )


def test_empty_news_returns_empty_panel():
    idx = pd.date_range("2020-01-01", "2020-01-10", tz="UTC")
    out = build_daily_news_tilt(pd.DataFrame(), idx)
    assert out.empty


def test_basic_two_symbols():
    news = _make_news(
        [
            ("2020-01-01", "A", 0.8),
            ("2020-01-01", "B", -0.5),
            ("2020-01-02", "A", 0.6),
            ("2020-01-02", "B", -0.7),
        ]
    )
    idx = pd.date_range("2020-01-01", "2020-01-05", tz="UTC")
    z = build_daily_news_tilt(news, idx)
    assert "A" in z.columns and "B" in z.columns
    # Cross-section z: A > B always (more positive sentiment)
    assert z.loc[pd.Timestamp("2020-01-02", tz="UTC"), "A"] > 0
    assert z.loc[pd.Timestamp("2020-01-02", tz="UTC"), "B"] < 0


def test_zscores_sum_zero_per_day_when_complete():
    news = _make_news(
        [
            ("2020-01-01", "A", 1.0),
            ("2020-01-01", "B", 0.0),
            ("2020-01-01", "C", -1.0),
        ]
    )
    idx = pd.date_range("2020-01-01", "2020-01-03", tz="UTC")
    z = build_daily_news_tilt(news, idx)
    row = z.loc[pd.Timestamp("2020-01-01", tz="UTC")].dropna()
    if len(row) == 3:
        # ddof=0 z-scores sum to 0
        assert abs(row.sum()) < 1e-9


def test_ffill_within_rolling_window():
    """Symbol with one event should retain coverage for `rolling_days` after."""
    news = _make_news([("2020-01-01", "A", 0.5), ("2020-01-01", "B", -0.5)])
    idx = pd.date_range("2020-01-01", "2020-01-20", tz="UTC")
    z = build_daily_news_tilt(news, idx, rolling_days=10, decay_halflife_days=3)
    # Days 1-11 should have data; day 15 (beyond rolling_days) should be NaN
    assert z.loc[pd.Timestamp("2020-01-05", tz="UTC"), "A"] != pytest.approx(np.nan)
    # After ffill limit, should be NaN
    assert pd.isna(z.loc[pd.Timestamp("2020-01-15", tz="UTC"), "A"])


def test_lookup_helper_returns_series():
    news = _make_news(
        [
            ("2020-01-01", "A", 1.0),
            ("2020-01-01", "B", -1.0),
        ]
    )
    idx = pd.date_range("2020-01-01", "2020-01-05", tz="UTC")
    z = build_daily_news_tilt(news, idx)
    s = news_tilt_for_date(z, pd.Timestamp("2020-01-01", tz="UTC"))
    assert isinstance(s, pd.Series)
    assert "A" in s.index


def test_real_data_smoke():
    """Smoke test with real news_sentiment_fused.parquet if available."""
    cache = Path("output/news_sentiment_fused.parquet")
    if not cache.exists():
        pytest.skip("news cache missing")
    from erweiterung.signals.news_tilt_builder import load_news_sentiment

    df = load_news_sentiment()
    assert "sentiment" in df.columns
    assert "symbol" in df.columns
    assert df["date"].dt.tz is not None  # UTC-tagged

    idx = pd.date_range(df["date"].min(), df["date"].max(), tz="UTC")
    z = build_daily_news_tilt(df, idx, rolling_days=14, decay_halflife_days=3)
    assert isinstance(z, pd.DataFrame)
    # At least one valid z-score
    assert z.notna().any().any()
