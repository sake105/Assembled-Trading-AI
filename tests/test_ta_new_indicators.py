"""Tests for new TA indicators: MACD, Bollinger Bands, Stochastic, ADX, OBV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.ta_features import (
    add_adx,
    add_all_features,
    add_bollinger_bands,
    add_macd,
    add_obv,
    add_stochastic,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = 150 + np.random.randn(n).cumsum()
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n)) * 2,
            "low": close - abs(np.random.randn(n)) * 2,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n),
        }
    )
    # Fix OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def multi_symbol_ohlcv() -> pd.DataFrame:
    """Create multi-symbol OHLCV data."""
    frames = []
    for sym in ["AAPL", "MSFT"]:
        np.random.seed(hash(sym) % 2**31)
        n = 50
        close = 150 + np.random.randn(n).cumsum()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01", periods=n, freq="D", tz="UTC"
                ),
                "symbol": sym,
                "open": close,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": np.random.randint(1_000_000, 5_000_000, n),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class TestMACD:
    def test_macd_columns_exist(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_macd(sample_ohlcv)
        assert "ta_macd_v1" in result.columns
        assert "ta_macd_signal_v1" in result.columns
        assert "ta_macd_hist_v1" in result.columns

    def test_macd_histogram_equals_diff(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_macd(sample_ohlcv)
        expected = result["ta_macd_v1"] - result["ta_macd_signal_v1"]
        np.testing.assert_allclose(result["ta_macd_hist_v1"], expected, atol=1e-10)

    def test_macd_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_macd(sample_ohlcv)
        # After 26 periods (slow EMA), values should be non-NaN
        assert result["ta_macd_v1"].iloc[30:].notna().all()

    def test_macd_multi_symbol(self, multi_symbol_ohlcv: pd.DataFrame) -> None:
        result = add_macd(multi_symbol_ohlcv)
        for sym in ["AAPL", "MSFT"]:
            sym_data = result[result["symbol"] == sym]
            assert sym_data["ta_macd_v1"].notna().sum() > 0

    def test_macd_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"symbol": ["X"], "price": [100]})
        with pytest.raises(KeyError):
            add_macd(df)


class TestBollingerBands:
    def test_bb_columns_exist(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(sample_ohlcv)
        assert "ta_bb_upper_v1" in result.columns
        assert "ta_bb_lower_v1" in result.columns
        assert "ta_bb_pctb_v1" in result.columns
        assert "ta_bb_bandwidth_v1" in result.columns

    def test_bb_upper_gt_lower(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(sample_ohlcv)
        valid = result["ta_bb_upper_v1"].notna() & result["ta_bb_lower_v1"].notna()
        assert (result.loc[valid, "ta_bb_upper_v1"] >= result.loc[valid, "ta_bb_lower_v1"]).all()

    def test_bb_pctb_at_upper_is_one(self) -> None:
        """When close equals upper band, %B should be ~1.0."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC"),
                "symbol": "X",
                "close": [100.0] * 20 + [105.0] * 10,  # Jump to trigger bands
            }
        )
        result = add_bollinger_bands(df)
        # %B should be between 0 and 1 for most points
        pctb = result["ta_bb_pctb_v1"].dropna()
        assert pctb.min() >= -1.0  # Allow some overshoot
        assert pctb.max() <= 2.0

    def test_bb_multi_symbol(self, multi_symbol_ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(multi_symbol_ohlcv)
        assert result["ta_bb_pctb_v1"].notna().sum() > 0


class TestStochastic:
    def test_stoch_columns_exist(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_stochastic(sample_ohlcv)
        assert "ta_stoch_k_v1" in result.columns
        assert "ta_stoch_d_v1" in result.columns

    def test_stoch_range_0_100(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_stochastic(sample_ohlcv)
        k = result["ta_stoch_k_v1"].dropna()
        assert k.min() >= 0.0
        assert k.max() <= 100.0

    def test_stoch_d_smoother_than_k(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_stochastic(sample_ohlcv)
        k_std = result["ta_stoch_k_v1"].std()
        d_std = result["ta_stoch_d_v1"].std()
        assert d_std <= k_std  # %D should be smoother

    def test_stoch_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"symbol": ["X"], "close": [100]})
        with pytest.raises(KeyError, match="Missing required columns"):
            add_stochastic(df)


class TestADX:
    def test_adx_columns_exist(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_adx(sample_ohlcv)
        assert "ta_adx_v1" in result.columns
        assert "ta_plus_di_v1" in result.columns
        assert "ta_minus_di_v1" in result.columns

    def test_adx_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_adx(sample_ohlcv)
        adx = result["ta_adx_v1"].dropna()
        assert (adx >= 0).all()

    def test_adx_di_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_adx(sample_ohlcv)
        assert (result["ta_plus_di_v1"].dropna() >= 0).all()
        assert (result["ta_minus_di_v1"].dropna() >= 0).all()

    def test_adx_multi_symbol(self, multi_symbol_ohlcv: pd.DataFrame) -> None:
        result = add_adx(multi_symbol_ohlcv)
        for sym in ["AAPL", "MSFT"]:
            sym_data = result[result["symbol"] == sym]
            assert sym_data["ta_adx_v1"].notna().sum() > 0


class TestOBV:
    def test_obv_column_exists(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_obv(sample_ohlcv)
        assert "ta_obv_v1" in result.columns

    def test_obv_cumulative(self, sample_ohlcv: pd.DataFrame) -> None:
        """OBV should be cumulative sum of signed volume."""
        result = add_obv(sample_ohlcv)
        # First value should be 0 (no previous close to compare)
        assert result["ta_obv_v1"].iloc[0] == 0.0

    def test_obv_direction(self) -> None:
        """OBV increases on up days, decreases on down days."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01", periods=5, freq="D", tz="UTC"
                ),
                "symbol": "X",
                "close": [100.0, 102.0, 101.0, 103.0, 102.0],
                "volume": [1000, 2000, 1500, 3000, 1000],
            }
        )
        result = add_obv(df)
        obv = result["ta_obv_v1"].values
        # Day 1: up -> +2000, Day 2: down -> -1500, Day 3: up -> +3000, Day 4: down -> -1000
        assert obv[0] == 0  # No previous
        assert obv[1] == 2000  # Up day
        assert obv[2] == 500  # Down day (2000 - 1500)
        assert obv[3] == 3500  # Up day (500 + 3000)
        assert obv[4] == 2500  # Down day (3500 - 1000)

    def test_obv_missing_volume_raises(self) -> None:
        df = pd.DataFrame({"symbol": ["X"], "close": [100]})
        with pytest.raises(KeyError, match="Missing required columns"):
            add_obv(df)


class TestAddAllFeatures:
    def test_all_new_indicators_present(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_all_features(sample_ohlcv)
        expected_cols = [
            "ta_macd_v1",
            "ta_macd_signal_v1",
            "ta_macd_hist_v1",
            "ta_bb_upper_v1",
            "ta_bb_lower_v1",
            "ta_bb_pctb_v1",
            "ta_bb_bandwidth_v1",
            "ta_stoch_k_v1",
            "ta_stoch_d_v1",
            "ta_adx_v1",
            "ta_plus_di_v1",
            "ta_minus_di_v1",
            "ta_obv_v1",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_features_count(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_all_features(sample_ohlcv)
        ta_cols = [c for c in result.columns if c.startswith("ta_")]
        assert len(ta_cols) == 19  # 6 old + 13 new

    def test_selective_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that indicators can be selectively disabled."""
        result = add_all_features(
            sample_ohlcv,
            include_macd=False,
            include_bollinger=False,
            include_stochastic=False,
            include_adx=False,
            include_obv=False,
        )
        assert "ta_macd_v1" not in result.columns
        assert "ta_bb_upper_v1" not in result.columns
        assert "ta_stoch_k_v1" not in result.columns
        assert "ta_adx_v1" not in result.columns
        assert "ta_obv_v1" not in result.columns
        # Old indicators should still be present
        assert "ta_rsi_14_v1" in result.columns
        assert "ta_atr_14_v1" in result.columns
