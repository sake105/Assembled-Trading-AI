"""Tests for erweiterung.meta."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.meta import regime_router, strategy_orchestrator


def test_equal_weight_combination(synthetic_returns):
    out = strategy_orchestrator.equal_weight_combination(synthetic_returns)
    assert len(out) == len(synthetic_returns)
    np.testing.assert_array_almost_equal(
        out.values, synthetic_returns.mean(axis=1).values
    )


def test_inverse_vol_combination(synthetic_returns):
    out = strategy_orchestrator.inverse_vol_combination(synthetic_returns, lookback=60)
    assert len(out) == len(synthetic_returns)
    assert out.iloc[60:].notna().any()


def test_hedge_algorithm_basic(synthetic_returns):
    combined, weights = strategy_orchestrator.hedge_algorithm(
        synthetic_returns, eta=0.05
    )
    assert len(combined) == len(synthetic_returns)
    # Weights should sum to ~1 each row
    row_sums = weights.sum(axis=1)
    assert (abs(row_sums - 1.0) < 1e-6).all()


def test_regime_aware_combination(synthetic_returns):
    regime = pd.Series(
        np.random.default_rng(0).integers(0, 3, len(synthetic_returns)),
        index=synthetic_returns.index,
    )
    mapping = {
        0: ["AAA", "BBB"],
        1: ["CCC"],
        2: ["DDD", "EEE"],
    }
    out = strategy_orchestrator.regime_aware_combination(
        synthetic_returns, regime, mapping
    )
    assert len(out) == len(synthetic_returns)


def test_vol_regime(synthetic_returns):
    out = regime_router.vol_regime(synthetic_returns.iloc[:, 0], window=21)
    valid = out.dropna()
    assert valid.isin([0, 1, 2]).all()


def test_trend_regime(synthetic_returns):
    prices = (1 + synthetic_returns.iloc[:, 0]).cumprod() * 100
    out = regime_router.trend_regime(prices, slow=200, fast=50)
    valid = out.dropna()
    assert valid.isin([0, 1, 2]).all()


def test_composite_regime(synthetic_returns):
    prices = (1 + synthetic_returns.iloc[:, 0]).cumprod() * 100
    out = regime_router.composite_regime(
        synthetic_returns.iloc[:, 0], prices, vol_window=21
    )
    valid = out.dropna()
    assert valid.isin([0, 1, 2, 3]).all()
