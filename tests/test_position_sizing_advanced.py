"""Tests for advanced position sizing: Kelly, Risk Parity, Vol-Scaled."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import (
    compute_kelly_weights,
    compute_risk_parity_weights,
    compute_vol_scaled_weights,
)


@pytest.fixture
def long_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "direction": ["LONG", "LONG", "LONG", "LONG"],
            "score": [0.8, 0.6, 0.7, 0.5],
        }
    )


@pytest.fixture
def mixed_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "direction": ["LONG", "FLAT", "LONG"],
            "score": [0.8, 0.6, 0.7],
        }
    )


@pytest.fixture
def volatilities() -> dict[str, float]:
    return {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.30, "AMZN": 0.35}


class TestKellyWeights:
    def test_basic_output(self, long_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(long_signals)
        assert "symbol" in result.columns
        assert "target_weight" in result.columns
        assert "kelly_raw" in result.columns
        assert len(result) == 4

    def test_weights_non_negative(self, long_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(long_signals)
        assert (result["target_weight"] >= 0).all()

    def test_weights_sum_le_one(self, long_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(long_signals)
        assert result["target_weight"].sum() <= 1.0 + 1e-10

    def test_half_kelly_smaller(self, long_signals: pd.DataFrame) -> None:
        full = compute_kelly_weights(long_signals, fraction=1.0)
        half = compute_kelly_weights(long_signals, fraction=0.5)
        assert half["target_weight"].sum() <= full["target_weight"].sum() + 1e-10

    def test_with_explicit_win_rates(self, long_signals: pd.DataFrame) -> None:
        win_rates = {"AAPL": 0.60, "MSFT": 0.55, "GOOGL": 0.58, "AMZN": 0.52}
        result = compute_kelly_weights(long_signals, win_rates=win_rates)
        assert len(result) == 4

    def test_max_weight_cap(self, long_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(long_signals, max_weight=0.10)
        assert (result["target_weight"] <= 0.10 + 1e-10).all()

    def test_top_n(self, long_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(long_signals, top_n=2)
        assert len(result) == 2

    def test_empty_signals(self) -> None:
        empty = pd.DataFrame(columns=["symbol", "direction", "score"])
        result = compute_kelly_weights(empty)
        assert len(result) == 0

    def test_flat_signals_excluded(self, mixed_signals: pd.DataFrame) -> None:
        result = compute_kelly_weights(mixed_signals)
        assert "MSFT" not in result["symbol"].values


class TestRiskParity:
    def test_basic_output_v2(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_risk_parity_weights(long_signals, volatilities)
        assert len(result) == 4
        assert "target_weight" in result.columns
        assert "volatility" in result.columns

    def test_lower_vol_higher_weight(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_risk_parity_weights(long_signals, volatilities)
        msft_w = result[result["symbol"] == "MSFT"]["target_weight"].iloc[0]
        amzn_w = result[result["symbol"] == "AMZN"]["target_weight"].iloc[0]
        # MSFT (vol=0.20) should have higher weight than AMZN (vol=0.35)
        assert msft_w > amzn_w

    def test_weights_sum_le_one_v2(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_risk_parity_weights(long_signals, volatilities)
        assert result["target_weight"].sum() <= 1.0 + 1e-10

    def test_max_weight_cap_v2(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_risk_parity_weights(
            long_signals, volatilities, max_weight=0.20
        )
        assert (result["target_weight"] <= 0.20 + 1e-10).all()

    def test_empty_signals_v2(self, volatilities: dict[str, float]) -> None:
        empty = pd.DataFrame(columns=["symbol", "direction"])
        result = compute_risk_parity_weights(empty, volatilities)
        assert len(result) == 0


class TestVolScaled:
    def test_basic_output_v3(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_vol_scaled_weights(long_signals, volatilities)
        assert len(result) == 4
        assert "target_weight" in result.columns

    def test_lower_vol_higher_weight_v2(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_vol_scaled_weights(long_signals, volatilities, target_vol=0.15)
        msft_w = result[result["symbol"] == "MSFT"]["target_weight"].iloc[0]
        amzn_w = result[result["symbol"] == "AMZN"]["target_weight"].iloc[0]
        assert msft_w > amzn_w

    def test_higher_target_vol_higher_weights(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        low = compute_vol_scaled_weights(
            long_signals, volatilities, target_vol=0.10
        )
        high = compute_vol_scaled_weights(
            long_signals, volatilities, target_vol=0.25
        )
        assert high["target_weight"].sum() >= low["target_weight"].sum() - 1e-10

    def test_weights_sum_le_one_v3(
        self, long_signals: pd.DataFrame, volatilities: dict[str, float]
    ) -> None:
        result = compute_vol_scaled_weights(long_signals, volatilities)
        assert result["target_weight"].sum() <= 1.0 + 1e-10

    def test_empty_signals_v3(self, volatilities: dict[str, float]) -> None:
        empty = pd.DataFrame(columns=["symbol", "direction"])
        result = compute_vol_scaled_weights(empty, volatilities)
        assert len(result) == 0
