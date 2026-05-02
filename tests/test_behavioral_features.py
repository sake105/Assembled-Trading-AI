"""Tests for M41 Behavioral Finance Factors."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.features.behavioral_features")
from src.assembled_core.features.behavioral_features import (
    capital_gains_overhang,
    anchoring_52w_high,
    round_number_proximity,
    abnormal_volume,
    max_effect,
    abnormal_turnover,
    compute_behavioral_composite,
)


def _make_price_volume(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    volumes = pd.Series(rng.lognormal(14, 0.5, n), index=dates)
    return prices, volumes, pd.Series(returns, index=dates)


@pytest.mark.phase12
class TestCapitalGainsOverhang:
    def test_basic(self):
        prices, volumes, _ = _make_price_volume()
        cgo = capital_gains_overhang(prices, volumes)
        assert len(cgo) == len(prices)
        # CGO should be bounded
        valid = cgo.dropna()
        assert valid.abs().max() < 10

    def test_trending_up(self):
        prices = pd.Series(np.arange(100, 200, dtype=float))
        volumes = pd.Series(np.ones(100) * 1e6)
        cgo = capital_gains_overhang(prices, volumes, lookback=50)
        # Trending up → positive CGO (paper gains)
        assert cgo.iloc[-1] > 0

    def test_zero_volume(self):
        prices = pd.Series([100.0] * 50)
        volumes = pd.Series([0.0] * 50)
        cgo = capital_gains_overhang(prices, volumes)
        # Should handle gracefully
        assert not cgo.isna().all()


@pytest.mark.phase12
class TestAnchoring52WHigh:
    def test_basic_v2(self):
        prices, _, _ = _make_price_volume()
        prox = anchoring_52w_high(prices)
        assert len(prox) == len(prices)
        valid = prox.dropna()
        assert (valid <= 1.01).all()  # can't exceed 52W high much
        assert (valid >= 0).all()

    def test_at_high(self):
        # Price at 52W high → proximity = 1.0
        prices = pd.Series(np.arange(1, 101, dtype=float))
        prox = anchoring_52w_high(prices, lookback=50)
        assert prox.iloc[-1] == pytest.approx(1.0)


@pytest.mark.phase12
class TestRoundNumberProximity:
    def test_near_round(self):
        prices = pd.Series([98.0, 49.0, 199.0])
        prox = round_number_proximity(prices)
        # 98 is near 100, 49 near 50, 199 near 200
        assert prox.iloc[0] > 0
        assert prox.iloc[1] > 0

    def test_at_round(self):
        prices = pd.Series([100.0, 50.0])
        prox = round_number_proximity(prices)
        # At round number → not below it → lower proximity
        assert len(prox) == 2


@pytest.mark.phase12
class TestAbnormalVolume:
    def test_basic_v3(self):
        _, volumes, _ = _make_price_volume()
        abn = abnormal_volume(volumes)
        assert len(abn) == len(volumes)
        valid = abn.dropna()
        assert (valid > 0).all()

    def test_spike(self):
        volumes = pd.Series([1e6] * 100)
        volumes.iloc[-1] = 5e6  # 5x spike
        abn = abnormal_volume(volumes, lookback=30)
        assert abn.iloc[-1] > 3.0  # should be ~5x


@pytest.mark.phase12
class TestMaxEffect:
    def test_basic_v4(self):
        _, _, returns = _make_price_volume()
        mx = max_effect(returns)
        assert len(mx) == len(returns)
        valid = mx.dropna()
        assert (valid >= 0).all() or (valid <= 0.5).all()

    def test_known_max(self):
        returns = pd.Series([0.01, -0.02, 0.05, 0.01, 0.02])
        mx = max_effect(returns, lookback=5)
        assert mx.iloc[-1] == pytest.approx(0.05)


@pytest.mark.phase12
class TestAbnormalTurnover:
    def test_basic_v5(self):
        _, volumes, _ = _make_price_volume()
        abn = abnormal_turnover(volumes, shares_outstanding=1e8)
        assert len(abn) == len(volumes)

    def test_high_turnover(self):
        volumes = pd.Series([1e6] * 100)
        volumes.iloc[-1] = 1e7  # 10x spike
        abn = abnormal_turnover(volumes, shares_outstanding=1e8, lookback=30)
        assert abn.iloc[-1] > 5.0


@pytest.mark.phase12
class TestBehavioralComposite:
    def test_basic_v6(self):
        prices, volumes, returns = _make_price_volume(n=300)
        composite = compute_behavioral_composite(prices, volumes, returns)
        assert len(composite) == len(prices)

    def test_custom_weights(self):
        prices, volumes, returns = _make_price_volume()
        w = {"cgo": 2.0, "anchor_52w": 1.0, "abn_vol": 0.5,
             "max_effect": 1.0, "abn_turnover": 0.5}
        composite = compute_behavioral_composite(prices, volumes, returns, weights=w)
        assert len(composite) == len(prices)

    def test_no_crash(self):
        """Should handle short series gracefully."""
        prices = pd.Series([100.0, 101.0, 99.0])
        volumes = pd.Series([1e6, 1.5e6, 0.8e6])
        returns = pd.Series([0.0, 0.01, -0.02])
        composite = compute_behavioral_composite(prices, volumes, returns)
        assert len(composite) == 3
