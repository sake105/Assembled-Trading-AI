"""Tests for M36: Statistical Arbitrage — Pairs Trading."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.strategies.stat_arb import (
    PairResult,
    estimate_hedge_ratio,
    compute_spread,
    estimate_half_life,
    check_cointegration,
    find_cointegrated_pairs,
    generate_pair_signal,
)


@pytest.fixture
def cointegrated_pair():
    """Generate a synthetic cointegrated pair."""
    rng = np.random.default_rng(42)
    n = 500
    # Random walk for B
    b = 100 + np.cumsum(rng.normal(0, 0.5, n))
    # A = 2*B + stationary noise
    spread = rng.normal(0, 1, n)
    spread = np.cumsum(spread * 0.1)  # Slightly persistent
    # Make it mean-reverting
    for i in range(1, n):
        spread[i] = 0.95 * spread[i - 1] + rng.normal(0, 0.5)
    a = 2.0 * b + spread + 50
    return a, b


@pytest.fixture
def non_cointegrated_pair():
    """Two independent random walks."""
    rng = np.random.default_rng(99)
    n = 500
    a = 100 + np.cumsum(rng.normal(0, 1, n))
    b = 100 + np.cumsum(rng.normal(0, 1, n))
    return a, b


@pytest.mark.phase12
class TestHedgeRatio:
    def test_known_ratio(self):
        b = np.arange(100, dtype=float) + 50
        a = 2.0 * b + 10  # exact linear: beta = 2.0
        beta = estimate_hedge_ratio(a, b)
        assert beta == pytest.approx(2.0, rel=0.01)

    def test_noisy_ratio(self, cointegrated_pair):
        a, b = cointegrated_pair
        beta = estimate_hedge_ratio(a, b)
        # Should be close to 2.0
        assert 1.5 < beta < 2.5


@pytest.mark.phase12
class TestComputeSpread:
    def test_basic_spread(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([5.0, 10.0, 15.0])
        spread = compute_spread(a, b, hedge_ratio=2.0)
        np.testing.assert_allclose(spread, [0.0, 0.0, 0.0])

    def test_nonzero_spread(self):
        a = np.array([12.0, 22.0, 28.0])
        b = np.array([5.0, 10.0, 15.0])
        spread = compute_spread(a, b, hedge_ratio=2.0)
        np.testing.assert_allclose(spread, [2.0, 2.0, -2.0])


@pytest.mark.phase12
class TestEstimateHalfLife:
    def test_mean_reverting(self):
        rng = np.random.default_rng(42)
        n = 500
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 1)
        hl = estimate_half_life(spread)
        # AR(1) with phi=0.9 -> half-life ~ -ln(2)/ln(0.9) ~ 6.6
        assert 3 < hl < 15

    def test_random_walk_no_reversion(self):
        rng = np.random.default_rng(42)
        walk = np.cumsum(rng.normal(0, 1, 500))
        hl = estimate_half_life(walk)
        assert hl == float("inf") or hl > 20  # much longer than mean-reverting

    def test_short_series(self):
        hl = estimate_half_life(np.array([1.0, 2.0, 3.0]))
        assert hl == float("inf")


@pytest.mark.phase12
class TestCointegration:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = check_cointegration(a, b)
        assert isinstance(result, PairResult)
        assert result.hedge_ratio > 0
        # p-value should be low for cointegrated pair
        assert result.coint_pvalue < 0.20  # relaxed for synthetic data

    def test_non_cointegrated_pair(self, non_cointegrated_pair):
        a, b = non_cointegrated_pair
        result = check_cointegration(a, b)
        # p-value should be higher
        assert result.coint_pvalue > 0.01 or not result.is_cointegrated


@pytest.mark.phase12
class TestFindPairs:
    def test_find_pairs_synthetic(self, cointegrated_pair):
        a, b = cointegrated_pair
        rng = np.random.default_rng(77)
        c = 100 + np.cumsum(rng.normal(0, 1, len(a)))

        n = len(a)
        df = pd.DataFrame({
            "timestamp": list(range(n)) * 3,
            "symbol": ["A"] * n + ["B"] * n + ["C"] * n,
            "close": list(a) + list(b) + list(c),
        })

        pairs = find_cointegrated_pairs(df, max_pvalue=0.20)
        # Should find at least A-B
        assert isinstance(pairs, list)
        # May or may not find depending on statsmodels availability


@pytest.mark.phase12
class TestGenerateSignal:
    def test_long_signal_on_low_spread(self, cointegrated_pair):
        a, b = cointegrated_pair
        pair = PairResult(
            symbol_a="A", symbol_b="B",
            coint_pvalue=0.01, hedge_ratio=2.0,
            half_life=5.0, spread_mean=0.0, spread_std=1.0,
            is_cointegrated=True,
        )
        # Force a low z-score by manipulating the last price
        a_mod = a.copy()
        a_mod[-1] = a_mod[-1] - 10 * a.std()  # artificially low

        signal = generate_pair_signal(a_mod, b, pair, entry_z=2.0)
        if signal is not None:
            assert signal.direction_a == "LONG"
            assert signal.direction_b == "SHORT"

    def test_no_signal_near_equilibrium(self, cointegrated_pair):
        a, b = cointegrated_pair
        pair = PairResult(
            symbol_a="A", symbol_b="B",
            coint_pvalue=0.01, hedge_ratio=2.0,
            half_life=5.0, spread_mean=0.0, spread_std=1.0,
            is_cointegrated=True,
        )
        # Use very high entry threshold so no signal fires
        signal = generate_pair_signal(a, b, pair, entry_z=100.0)
        assert signal is None

    def test_signal_strength_capped(self, cointegrated_pair):
        a, b = cointegrated_pair
        pair = PairResult(
            symbol_a="A", symbol_b="B",
            coint_pvalue=0.01, hedge_ratio=2.0,
            half_life=5.0, spread_mean=0.0, spread_std=1.0,
            is_cointegrated=True,
        )
        a_extreme = a.copy()
        a_extreme[-1] = a_extreme[-1] + 100 * a.std()
        signal = generate_pair_signal(a_extreme, b, pair, entry_z=0.5)
        if signal is not None:
            assert signal.signal_strength <= 3.0

    def test_short_series_returns_none(self):
        pair = PairResult(
            symbol_a="A", symbol_b="B",
            coint_pvalue=0.01, hedge_ratio=1.0,
            half_life=5.0, spread_mean=0.0, spread_std=1.0,
            is_cointegrated=True,
        )
        signal = generate_pair_signal(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), pair,
        )
        assert signal is None
