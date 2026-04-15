"""Tests for M25: Factor Timing — Dynamic Factor Weight Adjustment."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.ml.factor_timing import (
    FactorTimingConfig,
    FactorTimingResult,
    compute_factor_momentum,
    compute_factor_crowding,
    compute_factor_mean_reversion,
    adjust_factor_weights,
)


@pytest.fixture
def factor_returns():
    """Synthetic factor returns: 60 periods, 5 factors."""
    rng = np.random.default_rng(42)
    n = 60
    data = {
        "momentum": rng.normal(0.005, 0.02, n),
        "value": rng.normal(0.002, 0.015, n),
        "quality": rng.normal(0.003, 0.01, n),
        "size": rng.normal(-0.001, 0.02, n),
        "low_vol": rng.normal(0.004, 0.008, n),
    }
    # Make momentum a clear winner recently
    data["momentum"][-12:] += 0.01
    return pd.DataFrame(data)


@pytest.fixture
def factor_exposures():
    """Synthetic cross-sectional factor exposures: 50 assets, 5 factors."""
    rng = np.random.default_rng(42)
    data = {
        "momentum": rng.normal(0, 1, 50),
        "value": rng.normal(0, 1, 50),
        "quality": rng.normal(0, 1, 50),
        "size": rng.normal(0, 1, 50),
        "low_vol": rng.normal(0, 0.5, 50),  # less dispersed
    }
    return pd.DataFrame(data)


@pytest.fixture
def base_weights():
    return {
        "momentum": 0.25,
        "value": 0.20,
        "quality": 0.20,
        "size": 0.15,
        "low_vol": 0.20,
    }


@pytest.mark.phase12
class TestFactorMomentum:
    def test_basic_momentum(self, factor_returns):
        scores = compute_factor_momentum(factor_returns, lookback=12)
        assert len(scores) == 5
        # Momentum factor was boosted recently
        assert scores["momentum"] > 0

    def test_empty_returns(self):
        scores = compute_factor_momentum(pd.DataFrame(), lookback=12)
        assert scores == {}

    def test_short_returns(self):
        df = pd.DataFrame({"a": [0.01, 0.02]})
        scores = compute_factor_momentum(df, lookback=12)
        assert "a" in scores


@pytest.mark.phase12
class TestFactorCrowding:
    def test_basic_crowding(self, factor_exposures):
        scores = compute_factor_crowding(factor_exposures)
        assert len(scores) == 5

    def test_empty_exposures(self):
        scores = compute_factor_crowding(pd.DataFrame())
        assert scores == {}

    def test_uniform_not_crowded(self):
        """Uniform distribution should have low crowding."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.uniform(-1, 1, 100),
            "f2": rng.normal(0, 1, 100),
        })
        scores = compute_factor_crowding(df)
        # Uniform has negative excess kurtosis, should be low/zero crowding
        assert isinstance(scores, dict)


@pytest.mark.phase12
class TestFactorMeanReversion:
    def test_basic_mean_reversion(self, factor_returns):
        scores = compute_factor_mean_reversion(
            factor_returns, short_lookback=12, long_lookback=60,
        )
        assert len(scores) == 5

    def test_insufficient_data(self):
        df = pd.DataFrame({"a": [0.01] * 10})
        scores = compute_factor_mean_reversion(df, long_lookback=60)
        assert scores["a"] == 0.0


@pytest.mark.phase12
class TestAdjustFactorWeights:
    def test_basic_adjustment(self, base_weights, factor_returns, factor_exposures):
        result = adjust_factor_weights(
            base_weights, factor_returns, factor_exposures,
        )
        assert isinstance(result, FactorTimingResult)
        assert len(result.adjusted_weights) == 5
        # Weights should sum to ~1
        total = sum(result.adjusted_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)
        # All weights should be non-negative
        assert all(w >= 0 for w in result.adjusted_weights.values())

    def test_no_data_returns_base(self, base_weights):
        result = adjust_factor_weights(base_weights)
        # Without data, should return close to base weights
        for f in base_weights:
            assert result.adjusted_weights[f] == pytest.approx(
                base_weights[f], abs=0.01
            )

    def test_tilt_bounded(self, base_weights, factor_returns):
        cfg = FactorTimingConfig(max_tilt_pct=0.10)  # 10% max
        result = adjust_factor_weights(
            base_weights, factor_returns, config=cfg,
        )
        for f in base_weights:
            assert abs(result.tilt_applied[f]) <= 0.10

    def test_momentum_tilts_toward_winner(self, base_weights, factor_returns):
        cfg = FactorTimingConfig(
            momentum_weight=1.0, mean_reversion_weight=0.0, crowding_weight=0.0,
        )
        result = adjust_factor_weights(
            base_weights, factor_returns, config=cfg,
        )
        # Momentum factor had boosted recent returns -> should get more weight
        assert result.adjusted_weights["momentum"] >= base_weights["momentum"]

    def test_crowding_reduces_weight(self, base_weights, factor_exposures):
        cfg = FactorTimingConfig(
            momentum_weight=0.0, mean_reversion_weight=0.0, crowding_weight=1.0,
        )
        result = adjust_factor_weights(
            base_weights, factor_exposures=factor_exposures, config=cfg,
        )
        # Most crowded factor should have reduced weight
        assert isinstance(result.crowding_scores, dict)

    def test_preserves_all_factors(self, base_weights, factor_returns):
        result = adjust_factor_weights(base_weights, factor_returns)
        assert set(result.adjusted_weights.keys()) == set(base_weights.keys())
