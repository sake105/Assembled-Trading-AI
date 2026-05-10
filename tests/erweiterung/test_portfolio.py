"""Tests for erweiterung.portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.portfolio import (
    black_litterman,
    cvar_optimizer,
    hierarchical_risk_parity as hrp,
    kelly_sizing,
    risk_parity,
)


def test_hrp_weights_sum_to_one(synthetic_returns):
    w = hrp.hrp_weights(synthetic_returns)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
    assert len(w) == synthetic_returns.shape[1]


def test_hrp_weights_diversified(synthetic_returns):
    w = hrp.hrp_weights(synthetic_returns)
    # No single weight should dominate (>0.6)
    assert w.max() < 0.6


def test_correlation_distance_symmetric():
    corr = pd.DataFrame(
        [[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    d = hrp.correlation_distance(corr)
    assert (d.values == d.values.T).all()
    np.testing.assert_array_almost_equal(np.diag(d.values), 0)


def test_quasi_diag_returns_permutation():
    # Proper linkage tree for n=4: merge 0&1, then 2&3, then merge those.
    linkage = [(0, 1, 0.5), (2, 3, 0.4), (4, 5, 0.6)]
    order = hrp.quasi_diag_order(linkage, n=4)
    assert sorted(order) == [0, 1, 2, 3]


def test_market_implied_returns(synthetic_returns):
    cov = synthetic_returns.cov() * 252
    w = pd.Series(0.2, index=cov.index)
    pi = black_litterman.market_implied_returns(cov, w, risk_aversion=2.5)
    assert len(pi) == cov.shape[0]
    # Implied returns sollten nicht zu wild sein
    assert pi.abs().max() < 1.0


def test_black_litterman_posterior(synthetic_returns):
    cov = synthetic_returns.cov() * 252
    w_mkt = pd.Series([0.4, 0.3, 0.2, 0.05, 0.05], index=cov.index)
    pi = black_litterman.market_implied_returns(cov, w_mkt)
    # View: AAA outperforms BBB by 5%
    P = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.05])
    views = black_litterman.BLViews(P=P, Q=Q, confidence=[0.5])
    mu_bl, sigma_bl = black_litterman.black_litterman_posterior(
        cov, pi, views, tau=0.05
    )
    assert len(mu_bl) == cov.shape[0]
    assert mu_bl["AAA"] > mu_bl["BBB"]


def test_mvo_long_only_caps(synthetic_returns):
    cov = synthetic_returns.cov() * 252
    mu = pd.Series([0.10, 0.08, 0.06, -0.02, 0.04], index=cov.index)
    w = black_litterman.mean_variance_optimal_weights(
        mu, cov, long_only=True, max_weight=0.3
    )
    assert (w >= 0).all()
    assert (w <= 0.3 + 1e-9).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_cvar_optimal_weights(synthetic_returns):
    weights, metrics = cvar_optimizer.cvar_optimal_weights(
        synthetic_returns, confidence=0.95, long_only=True, max_weight=0.4
    )
    assert abs(weights.sum() - 1.0) < 1e-3
    assert (weights >= -1e-6).all()
    assert "cvar" in metrics


def test_risk_parity_weights(synthetic_returns):
    cov = synthetic_returns.cov() * 252
    w = risk_parity.risk_parity_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w > 0).all()
    # ERC: contributions should be approximately equal
    sigma = float(np.sqrt(w.values @ cov.values @ w.values))
    rc = w.values * (cov.values @ w.values) / sigma
    assert (rc.max() / rc.min()) < 5.0  # close to equal but allow tolerance


def test_kelly_single():
    f = kelly_sizing.fractional_kelly_single(
        expected_return=0.001, variance=0.0004, fraction=0.25
    )
    assert f > 0
    assert f <= 1.0


def test_kelly_negative_returns():
    f = kelly_sizing.fractional_kelly_single(
        expected_return=-0.002, variance=0.0004, fraction=0.25
    )
    assert f < 0


def test_confidence_discounted_kelly():
    f0 = kelly_sizing.confidence_discounted_kelly(0.001, 0.0004, confidence=0.0)
    f1 = kelly_sizing.confidence_discounted_kelly(0.001, 0.0004, confidence=1.0)
    assert f0 == 0
    assert f1 > 0


def test_multi_asset_kelly_caps(synthetic_returns):
    mu = pd.Series([0.10, 0.05, 0.02, -0.01, 0.03], index=synthetic_returns.columns)
    cov = synthetic_returns.cov() * 252
    w = kelly_sizing.multi_asset_kelly(mu, cov, fraction=0.25, max_per_asset=0.20)
    assert (w.abs() <= 0.20 + 1e-9).all()
