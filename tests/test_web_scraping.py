"""Tests for M38d: Web Scraping Feature Extraction."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.data.altdata.web_scraping import (
    WebScrapingConfig,
    compute_job_posting_features,
    compute_app_rating_features,
    compute_website_traffic_features,
)


@pytest.mark.phase12
class TestJobPostingFeatures:
    def test_basic(self):
        data = pd.DataFrame({
            "symbol": ["AAPL"] * 20,
            "scrape_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "job_count": list(range(100, 120)),
        })
        result = compute_job_posting_features(data, as_of="2024-02-01")
        assert len(result) == 1
        assert "job_posting_count" in result.columns
        assert "job_posting_growth_30d" in result.columns
        assert result["job_posting_growth_30d"].iloc[0] > 0  # increasing

    def test_empty_input(self):
        result = compute_job_posting_features(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0

    def test_min_data_points(self):
        data = pd.DataFrame({
            "symbol": ["AAPL"] * 2,
            "scrape_date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "job_count": [100, 110],
        })
        cfg = WebScrapingConfig(min_data_points=5)
        result = compute_job_posting_features(data, as_of="2024-02-01", config=cfg)
        assert len(result) == 0

    def test_multiple_symbols(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "symbol": ["AAPL"] * 10 + ["GOOG"] * 10,
            "scrape_date": list(dates) * 2,
            "job_count": list(range(100, 110)) + list(range(200, 210)),
        })
        result = compute_job_posting_features(data, as_of="2024-02-01")
        assert len(result) == 2


@pytest.mark.phase12
class TestAppRatingFeatures:
    def test_basic(self):
        data = pd.DataFrame({
            "symbol": ["META"] * 15,
            "scrape_date": pd.date_range("2024-01-01", periods=15, freq="D"),
            "avg_rating": [4.2 + i * 0.01 for i in range(15)],
            "review_count": [50] * 15,
        })
        result = compute_app_rating_features(data, as_of="2024-02-01")
        assert len(result) == 1
        assert "app_rating" in result.columns
        assert "app_rating_trend" in result.columns
        assert "app_review_velocity" in result.columns
        assert result["app_rating_trend"].iloc[0] > 0  # improving

    def test_empty_input(self):
        result = compute_app_rating_features(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0


@pytest.mark.phase12
class TestWebTrafficFeatures:
    def test_basic(self):
        data = pd.DataFrame({
            "symbol": ["AMZN"] * 20,
            "scrape_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "estimated_visits": np.random.default_rng(42).poisson(1000000, 20),
        })
        result = compute_website_traffic_features(data, as_of="2024-02-01")
        assert len(result) == 1
        assert "web_traffic_index" in result.columns
        assert "web_traffic_trend" in result.columns

    def test_empty_input(self):
        result = compute_website_traffic_features(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0

    def test_pit_safety(self):
        """Data after as_of should be excluded."""
        data = pd.DataFrame({
            "symbol": ["AMZN"] * 10,
            "scrape_date": pd.date_range("2024-06-01", periods=10, freq="D"),
            "estimated_visits": [1000000] * 10,
        })
        result = compute_website_traffic_features(data, as_of="2024-01-01")
        assert len(result) == 0
