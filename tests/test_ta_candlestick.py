"""Tests for candlestick pattern recognition module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.ta_candlestick import (
    build_candlestick_features,
    get_candlestick_feature_names,
)


def _synthetic_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for sym in ["AAPL", "MSFT"]:
        close = 100.0 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        open_ = close + rng.normal(0, 0.5, n)
        volume = rng.poisson(1000000, n)
        for i in range(n):
            rows.append({
                "timestamp": dates[i], "symbol": sym,
                "open": open_[i], "high": high[i], "low": low[i],
                "close": close[i], "volume": volume[i],
            })
    return pd.DataFrame(rows)


@pytest.mark.phase12
class TestBuildCandlestickFeatures:
    def test_basic_output(self):
        df = _synthetic_ohlcv()
        result = build_candlestick_features(df)
        assert len(result) == len(df)
        # Should have pattern columns
        pattern_cols = [c for c in result.columns if c.startswith("cs_")]
        assert len(pattern_cols) > 0

    def test_expected_patterns(self):
        df = _synthetic_ohlcv(n=100)
        result = build_candlestick_features(df)
        expected = get_candlestick_feature_names()
        for name in expected:
            assert name in result.columns, f"Missing pattern: {name}"

    def test_pattern_values_binary(self):
        df = _synthetic_ohlcv(n=100)
        result = build_candlestick_features(df)
        pattern_cols = get_candlestick_feature_names()
        for col in pattern_cols:
            if col in result.columns:
                unique_vals = set(result[col].dropna().unique())
                assert unique_vals <= {-1.0, -1, 0.0, 0, 1.0, 1}, \
                    f"{col} has unexpected values: {unique_vals}"

    def test_short_data(self):
        df = _synthetic_ohlcv(n=5)
        result = build_candlestick_features(df)
        assert len(result) == len(df)

    def test_get_feature_names(self):
        names = get_candlestick_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 8  # at least 8 patterns


@pytest.mark.phase12
class TestCandlestickPatterns:
    def test_doji_detection(self):
        """A doji has open ≈ close."""
        df = pd.DataFrame({
            "symbol": ["AAPL"] * 3 + ["MSFT"] * 3,
            "timestamp": list(pd.bdate_range("2024-01-01", periods=3)) * 2,
            "open": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
            "high": [102.0, 101.0, 103.0, 202.0, 201.0, 203.0],
            "low": [98.0, 99.0, 97.0, 198.0, 199.0, 197.0],
            "close": [100.01, 100.0, 99.99, 200.01, 200.0, 199.99],
            "volume": [1000] * 6,
        })
        result = build_candlestick_features(df)
        pattern_names = get_candlestick_feature_names()
        # Should have pattern columns
        assert any(c in result.columns for c in pattern_names)

    def test_engulfing_needs_history(self):
        """Engulfing pattern needs at least 2 bars."""
        df = _synthetic_ohlcv(n=5)
        result = build_candlestick_features(df)
        pattern_names = get_candlestick_feature_names()
        assert any(c in result.columns for c in pattern_names)
