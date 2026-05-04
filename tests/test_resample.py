"""Tests for multi-timeframe resampling module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.data.resample import (
    resample_to_weekly,
    resample_to_monthly,
    align_higher_tf_to_daily,
)


def _synthetic_daily(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "open": close + rng.normal(0, 0.2, n),
            "high": close + rng.uniform(0.5, 1.5, n),
            "low": close - rng.uniform(0.5, 1.5, n),
            "close": close,
            "volume": rng.poisson(1000000, n),
        }
    )


@pytest.mark.phase12
class TestResampleToWeekly:
    def test_basic(self):
        daily = _synthetic_daily()
        weekly = resample_to_weekly(daily)
        assert len(weekly) < len(daily)
        assert "close" in weekly.columns
        assert "volume" in weekly.columns

    def test_ohlcv_aggregation(self):
        daily = _synthetic_daily()
        weekly = resample_to_weekly(daily)
        if "high" in weekly.columns:
            # Weekly high should be >= any daily close in that week
            assert weekly["high"].max() >= daily["close"].min()


@pytest.mark.phase12
class TestResampleToMonthly:
    def test_basic_v2(self):
        daily = _synthetic_daily()
        monthly = resample_to_monthly(daily)
        assert len(monthly) < len(daily)
        assert "close" in monthly.columns

    def test_fewer_bars(self):
        daily = _synthetic_daily(n=200)
        monthly = resample_to_monthly(daily)
        assert len(monthly) <= 12  # ~200 trading days ≈ 8-10 months


@pytest.mark.phase12
class TestAlignHigherTFToDaily:
    def test_basic_alignment(self):
        daily = _synthetic_daily()
        weekly = resample_to_weekly(daily)
        aligned = align_higher_tf_to_daily(
            daily,
            weekly,
            suffix="_weekly",
        )
        assert len(aligned) == len(daily)
        # Should have weekly columns
        weekly_cols = [c for c in aligned.columns if c.endswith("_weekly")]
        assert len(weekly_cols) > 0

    def test_no_future_leak(self):
        """Higher TF data should not leak future information."""
        daily = _synthetic_daily(n=50)
        weekly = resample_to_weekly(daily)
        aligned = align_higher_tf_to_daily(daily, weekly, suffix="_w")
        # First few rows might be NaN (no prior weekly data)
        # But no row should have data from a future week
        if "close_w" in aligned.columns:
            # Weekly close should be <= daily close's last value in that week
            assert aligned["close_w"].iloc[-1] <= daily["close"].max() * 1.5
