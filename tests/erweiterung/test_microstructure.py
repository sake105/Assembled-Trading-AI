"""Tests for erweiterung.microstructure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.microstructure import liquidity_proxies, vpin


def test_amihud_basic():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "symbol": ["A"] * n,
            "return": rng.normal(0, 0.02, n),
            "dollar_volume": rng.uniform(1e6, 1e8, n),
        }
    )
    out = liquidity_proxies.amihud_illiquidity(df, window=21)
    assert "amihud" in out.columns
    valid = out["amihud"].dropna()
    assert (valid >= 0).all()


def test_roll_spread_basic():
    rng = np.random.default_rng(0)
    # construct prices with bid-ask bouncing
    p = pd.Series(np.cumsum(rng.normal(0, 0.5, 200))).abs() + 100
    p = p + rng.choice([-0.05, 0.05], size=200)
    s = liquidity_proxies.roll_spread_estimator(p)
    assert np.isfinite(s) or pd.isna(s)


def test_corwin_schultz():
    rng = np.random.default_rng(0)
    high = pd.Series(101 + rng.normal(0, 0.5, 100))
    low = pd.Series(99 + rng.normal(0, 0.5, 100))
    s = liquidity_proxies.corwin_schultz_spread(high, low)
    assert (s.dropna() >= 0).all()


def test_kyle_lambda():
    rng = np.random.default_rng(0)
    n = 200
    sv = pd.Series(rng.normal(0, 1, n))
    rets = 0.001 * sv + pd.Series(rng.normal(0, 0.005, n))
    lam = liquidity_proxies.kyle_lambda(rets, sv, window=60)
    valid = lam.dropna()
    assert valid.shape[0] > 0


def test_bulk_volume_classify():
    p = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 0.1, 200)))
    v = pd.Series(np.random.default_rng(0).integers(1000, 10000, 200))
    vb, vs = vpin.bulk_volume_classify(p, v)
    assert (vb >= 0).all()
    assert (vs >= 0).all()


def test_compute_vpin():
    rng = np.random.default_rng(0)
    n = 1000
    p = pd.Series(np.cumsum(rng.normal(0, 0.05, n)) + 100)
    v = pd.Series(rng.integers(500, 5000, n))
    out = vpin.compute_vpin(p, v, n_buckets=20)
    valid = out.dropna()
    assert valid.shape[0] > 0
    assert (valid >= 0).all()
