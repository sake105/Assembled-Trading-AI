"""Tests for seasonal calendar effect features."""

from __future__ import annotations

import pytest
import pandas as pd

pytest.importorskip("src.assembled_core.features.seasonal_features")
from src.assembled_core.features.seasonal_features import (
    build_seasonal_features,
    get_seasonal_feature_names,
)


@pytest.mark.phase12
class TestBuildSeasonalFeatures:
    def test_basic(self):
        dates = pd.bdate_range("2024-01-01", periods=252)
        result = build_seasonal_features(dates)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 252

    def test_all_columns_present(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        result = build_seasonal_features(dates)
        for name in get_seasonal_feature_names():
            assert name in result.columns, f"Missing: {name}"

    def test_january_effect(self):
        dates = pd.bdate_range("2024-01-02", periods=22)  # January only
        result = build_seasonal_features(dates)
        assert (result["seasonal_january"] == 1.0).all()

    def test_non_january(self):
        dates = pd.bdate_range("2024-06-01", periods=20)
        result = build_seasonal_features(dates)
        assert (result["seasonal_january"] == 0.0).all()

    def test_sell_in_may(self):
        # May dates should have -1
        dates = pd.bdate_range("2024-05-01", periods=22)
        result = build_seasonal_features(dates)
        assert (result["seasonal_sell_in_may"] == -1.0).all()

    def test_winter_positive(self):
        # November dates should have +1
        dates = pd.bdate_range("2024-11-01", periods=20)
        result = build_seasonal_features(dates)
        assert (result["seasonal_sell_in_may"] == 1.0).all()

    def test_turn_of_month(self):
        # First 3 days of month should be flagged
        dates = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-03-02"),
                pd.Timestamp("2024-03-15"),
            ]
        )
        result = build_seasonal_features(dates)
        assert result["seasonal_turn_of_month"].iloc[0] == 1.0
        assert result["seasonal_turn_of_month"].iloc[2] == 0.0

    def test_russell_reconstitution(self):
        dates = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-06-17"),
                pd.Timestamp("2024-06-28"),
                pd.Timestamp("2024-07-01"),
            ]
        )
        result = build_seasonal_features(dates)
        assert result["seasonal_russell_recon"].iloc[0] == 1.0
        assert result["seasonal_russell_recon"].iloc[2] == 0.0

    def test_from_series(self):
        series = pd.Series(pd.bdate_range("2024-01-01", periods=50))
        result = build_seasonal_features(series)
        assert len(result) == 50

    def test_feature_names(self):
        names = get_seasonal_feature_names()
        assert len(names) == 8


@pytest.mark.phase12
class TestDayOfWeekEffect:
    def test_monday_negative(self):
        # Find a Monday
        monday = pd.Timestamp("2024-01-01")  # This is a Monday
        result = build_seasonal_features(pd.DatetimeIndex([monday]))
        assert result["seasonal_day_of_week"].iloc[0] == -0.5

    def test_friday_positive(self):
        friday = pd.Timestamp("2024-01-05")  # This is a Friday
        result = build_seasonal_features(pd.DatetimeIndex([friday]))
        assert result["seasonal_day_of_week"].iloc[0] == 0.5
