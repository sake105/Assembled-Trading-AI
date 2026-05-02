"""Tests for M38c: Satellite / Geospatial Features."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

import pytest; pytest.importorskip("src.assembled_core.data.altdata.satellite_features")
from src.assembled_core.data.altdata.satellite_features import (
    SatelliteConfig,
    process_parking_lot_data,
    process_shipping_data,
    compute_nightlight_features,
)


@pytest.mark.phase12
class TestParkingLotData:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "symbol": ["WMT"] * 30,
            "observation_date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "occupancy_rate": rng.uniform(0.4, 0.9, 30),
        })
        result = process_parking_lot_data(data, as_of="2024-02-15")
        assert len(result) == 1
        assert "parking_lot_occupancy" in result.columns
        assert "parking_lot_trend_4w" in result.columns
        assert 0.0 <= result["parking_lot_occupancy"].iloc[0] <= 1.0

    def test_pit_lag(self):
        """Data within processing lag should be excluded."""
        data = pd.DataFrame({
            "symbol": ["WMT"] * 5,
            "observation_date": pd.date_range("2024-01-13", periods=5, freq="D"),
            "occupancy_rate": [0.7] * 5,
        })
        cfg = SatelliteConfig(processing_lag_days=5, min_observations=3)
        result = process_parking_lot_data(data, as_of="2024-01-15", config=cfg)
        assert len(result) == 0  # all data within lag window

    def test_empty_input(self):
        result = process_parking_lot_data(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0

    def test_multiple_symbols(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame({
            "symbol": ["WMT"] * 20 + ["TGT"] * 20,
            "observation_date": list(dates) * 2,
            "occupancy_rate": rng.uniform(0.3, 0.9, 40),
        })
        result = process_parking_lot_data(data, as_of="2024-02-01")
        assert len(result) == 2


@pytest.mark.phase12
class TestShippingData:
    def test_basic_v2(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "region": ["Shanghai"] * 20,
            "observation_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "vessel_count": rng.poisson(100, 20),
        })
        result = process_shipping_data(data, as_of="2024-02-01")
        assert len(result) == 1
        assert "shipping_volume_index" in result.columns
        assert "shipping_trend_4w" in result.columns

    def test_empty_input_v2(self):
        result = process_shipping_data(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0


@pytest.mark.phase12
class TestNightlightFeatures:
    def test_basic_v3(self):
        data = pd.DataFrame({
            "region": ["US_NE"] * 400 + ["US_NE"] * 30,
            "observation_date": list(pd.date_range("2022-01-01", periods=400, freq="D")) + list(pd.date_range("2024-01-01", periods=30, freq="D")),
            "light_intensity": [50.0] * 400 + [55.0] * 30,
        })
        result = compute_nightlight_features(data, as_of="2024-02-15")
        assert len(result) == 1
        assert "nightlight_intensity" in result.columns
        assert "nightlight_yoy_change" in result.columns
        assert result["nightlight_yoy_change"].iloc[0] > 0  # intensity increased

    def test_empty_input_v3(self):
        result = compute_nightlight_features(pd.DataFrame(), as_of="2024-01-01")
        assert len(result) == 0

    def test_no_yoy_data(self):
        data = pd.DataFrame({
            "region": ["EU_W"] * 10,
            "observation_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "light_intensity": [42.0] * 10,
        })
        result = compute_nightlight_features(data, as_of="2024-02-01")
        if len(result) > 0:
            assert result["nightlight_yoy_change"].iloc[0] == 0.0
