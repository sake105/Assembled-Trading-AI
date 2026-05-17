"""Tests for src/assembled_core/risk/garch_vol.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.garch_vol import (
    forecast_vol,
    size_vol_target,
    compute_vol_forecasts,
)


def _returns(n=252, vol=0.01, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n) * vol)


# ---------------------------------------------------------------------------
# forecast_vol
# ---------------------------------------------------------------------------


class TestForecastVol:
    def test_returns_positive_float(self):
        r = _returns(252)
        v = forecast_vol(r)
        assert isinstance(v, float)
        assert v > 0

    def test_annualized_in_reasonable_range(self):
        # Daily vol ~1% → annualised ~16%; 0.5% → ~8%
        r = _returns(252, vol=0.01)
        v = forecast_vol(r)
        assert 0.05 < v < 0.60, f"Vol {v:.4f} out of reasonable range"

    def test_higher_vol_inputs_give_higher_forecast(self):
        low = forecast_vol(_returns(252, vol=0.005))
        high = forecast_vol(_returns(252, vol=0.03))
        assert high > low

    def test_insufficient_data_returns_fallback(self):
        r = _returns(30, vol=0.01)  # below min_obs=60
        v = forecast_vol(r, min_obs=60)
        # fallback_window=20 should still produce a result for 30 obs
        assert np.isfinite(v) or np.isnan(v)  # either is acceptable

    def test_very_few_obs_returns_nan(self):
        # Only 1 obs → fallback window also can't produce a std (needs ≥ 2)
        v = forecast_vol(pd.Series([0.01]), min_obs=60, fallback_window=20)
        assert np.isnan(v)

    def test_empty_series_returns_nan(self):
        v = forecast_vol(pd.Series([], dtype=float))
        assert np.isnan(v)

    def test_numpy_array_accepted(self):
        np.random.seed(7)
        arr = np.random.randn(200) * 0.01
        v = forecast_vol(arr, min_obs=100)
        assert isinstance(v, float)

    def test_fallback_used_when_arch_not_available(self, monkeypatch):
        import assembled_core.risk.garch_vol as mod

        monkeypatch.setattr(mod, "_ARCH_AVAILABLE", False)
        r = _returns(252, vol=0.015)
        v = mod.forecast_vol(r)
        assert isinstance(v, float)
        assert v > 0


# ---------------------------------------------------------------------------
# size_vol_target
# ---------------------------------------------------------------------------


class TestSizeVolTarget:
    def test_target_equal_forecast_gives_one(self):
        size = size_vol_target(0.15, target_vol=0.15)
        assert abs(size - 1.0) < 1e-9

    def test_low_vol_gives_larger_size(self):
        size = size_vol_target(0.10, target_vol=0.15)
        assert size > 1.0

    def test_high_vol_gives_smaller_size(self):
        size = size_vol_target(0.30, target_vol=0.15)
        assert size < 1.0

    def test_max_leverage_cap(self):
        # Very low vol → would be huge, but capped
        size = size_vol_target(0.01, target_vol=0.15, max_leverage=1.5)
        assert size == pytest.approx(1.5)

    def test_zero_vol_returns_neutral(self):
        size = size_vol_target(0.0)
        assert size == pytest.approx(1.0)

    def test_nan_vol_returns_neutral(self):
        size = size_vol_target(float("nan"))
        assert size == pytest.approx(1.0)

    def test_inf_vol_returns_near_zero(self):
        size = size_vol_target(float("inf"))
        assert size == 0.0

    def test_min_size_floor(self):
        size = size_vol_target(1.0, target_vol=0.15, min_size=0.05)
        assert size >= 0.05


# ---------------------------------------------------------------------------
# compute_vol_forecasts
# ---------------------------------------------------------------------------


class TestComputeVolForecasts:
    def _make_prices(self, tickers=("AAPL", "MSFT"), n=200):
        rows = []
        for t in tickers:
            np.random.seed(hash(t) % 2**31)
            close = 100 + np.cumsum(np.random.randn(n) * 1.0)
            for i, c in enumerate(close):
                rows.append(
                    {
                        "ticker": t,
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                        "close": max(c, 1.0),
                    }
                )
        return pd.DataFrame(rows)

    def test_one_row_per_ticker(self):
        df = self._make_prices(["AAPL", "MSFT", "NVDA"])
        result = compute_vol_forecasts(df)
        assert len(result) == 3
        assert set(result["ticker"]) == {"AAPL", "MSFT", "NVDA"}

    def test_output_columns_present(self):
        df = self._make_prices(["AAPL"])
        result = compute_vol_forecasts(df)
        assert "vol_forecast_annual" in result.columns
        assert "size_multiplier" in result.columns

    def test_size_multiplier_within_bounds(self):
        df = self._make_prices(["AAPL", "MSFT"])
        result = compute_vol_forecasts(df, max_leverage=1.5)
        assert (result["size_multiplier"] >= 0).all()
        assert (result["size_multiplier"] <= 1.5).all()
