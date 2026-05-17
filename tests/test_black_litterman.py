"""Tests for Black-Litterman portfolio optimizer."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.portfolio.black_litterman import (
    BlackLittermanOptimizer,
    robust_bl_shrinkage,
)


def _synthetic_market(n_assets: int = 5, seed: int = 42) -> tuple:
    """Create synthetic market data for BL tests."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM_{i}" for i in range(n_assets)]

    # Market cap weights
    caps = rng.uniform(100, 1000, n_assets)
    market_weights = caps / caps.sum()

    # Covariance matrix (must be PSD)
    L = rng.normal(0, 0.01, (n_assets, n_assets))
    cov = L @ L.T + np.eye(n_assets) * 0.001
    sigma = pd.DataFrame(cov, index=symbols, columns=symbols)

    return symbols, market_weights, sigma


@pytest.mark.fast
class TestBlackLittermanOptimizer:
    def test_implied_returns(self):
        symbols, mw, sigma = _synthetic_market()
        bl = BlackLittermanOptimizer(risk_aversion=2.5)
        pi = bl.compute_implied_returns(
            pd.Series(mw, index=symbols),
            sigma,
        )
        assert len(pi) == len(symbols)
        assert all(np.isfinite(pi))

    def test_posterior_returns_no_views(self):
        symbols, mw, sigma = _synthetic_market()
        bl = BlackLittermanOptimizer(risk_aversion=2.5)
        pi = bl.compute_implied_returns(pd.Series(mw, index=symbols), sigma)
        # Without views, posterior should equal prior
        posterior = bl.compute_posterior_returns(
            pi,
            sigma,
            views={},
        )
        np.testing.assert_allclose(posterior, pi, atol=1e-6)

    def test_optimize_weights_sum_to_one(self):
        symbols, mw, sigma = _synthetic_market()
        bl = BlackLittermanOptimizer(risk_aversion=2.5)
        weights = bl.optimize(
            market_weights=pd.Series(mw, index=symbols),
            sigma=sigma,
            views={},  # no views
        )
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 0.01

    def test_optimize_with_views(self):
        symbols, mw, sigma = _synthetic_market()
        bl = BlackLittermanOptimizer(risk_aversion=2.5)
        views = {symbols[0]: 0.10}  # bullish on first symbol
        weights = bl.optimize(
            market_weights=pd.Series(mw, index=symbols),
            sigma=sigma,
            views=views,
        )
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 0.05

    def test_long_only_constraint(self):
        symbols, mw, sigma = _synthetic_market()
        bl = BlackLittermanOptimizer(risk_aversion=2.5)
        weights = bl.optimize(
            market_weights=pd.Series(mw, index=symbols),
            sigma=sigma,
            views={},
        )
        assert all(w >= -0.001 for w in weights.values)


@pytest.mark.fast
class TestRobustBLShrinkage:
    def test_basic(self):
        symbols, mw, sigma = _synthetic_market()
        mu_bl = np.array([0.05, 0.06, 0.04, 0.07, 0.03])
        shrunk = robust_bl_shrinkage(mu_bl, sigma.values)
        assert len(shrunk) == len(mu_bl)
        assert all(np.isfinite(shrunk))

    def test_shrinkage_toward_zero(self):
        symbols, mw, sigma = _synthetic_market()
        mu_bl = np.array([0.05, 0.06, 0.04, 0.07, 0.03])
        shrunk = robust_bl_shrinkage(mu_bl, sigma.values, kappa_scale=10.0)
        # Heavy shrinkage should reduce magnitude
        assert np.linalg.norm(shrunk) <= np.linalg.norm(mu_bl) + 0.01
