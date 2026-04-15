"""Tests for short interest features module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.short_interest_features import (
    compute_short_pct_float,
    compute_short_ratio,
    compute_short_squeeze_score,
    build_short_interest_features,
    get_short_interest_feature_names,
)


@pytest.mark.phase12
class TestShortPctFloat:
    def test_basic(self):
        assert compute_short_pct_float(1_000_000, 10_000_000) == pytest.approx(0.10)

    def test_zero_float(self):
        assert compute_short_pct_float(100, 0) == 0.0

    def test_high_short(self):
        result = compute_short_pct_float(5_000_000, 10_000_000)
        assert result == pytest.approx(0.50)


@pytest.mark.phase12
class TestShortRatio:
    def test_basic(self):
        result = compute_short_ratio(5_000_000, 1_000_000)
        assert result == pytest.approx(5.0)

    def test_zero_volume(self):
        assert compute_short_ratio(100, 0) == 0.0


@pytest.mark.phase12
class TestShortSqueezeScore:
    def test_low_risk(self):
        score = compute_short_squeeze_score(0.05, 2.0, 0.0, 0.0)
        assert 0 <= score <= 0.5

    def test_high_risk(self):
        score = compute_short_squeeze_score(0.30, 10.0, 0.10, 0.50)
        assert score > 0.7

    def test_range(self):
        score = compute_short_squeeze_score(0.15, 5.0, 0.05, 0.10)
        assert 0 <= score <= 1.0


@pytest.mark.phase12
class TestBuildShortInterestFeatures:
    def test_basic(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
            "settlement_date": list(dates) * 2,
            "short_interest": rng.integers(500_000, 5_000_000, 20),
            "shares_float": [10_000_000] * 20,
            "avg_volume": rng.integers(500_000, 2_000_000, 20),
        })
        result = build_short_interest_features(df)
        assert "si_pct_float" in result.columns
        assert "si_days_to_cover" in result.columns
        assert "si_squeeze_score" in result.columns
        assert len(result) == 20

    def test_empty(self):
        result = build_short_interest_features(pd.DataFrame())
        assert len(result) == 0

    def test_feature_names(self):
        names = get_short_interest_feature_names()
        assert len(names) == 4
        assert "si_pct_float" in names
