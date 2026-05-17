"""Tests for portfolio covariance estimation module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.portfolio.covariance")
from src.assembled_core.portfolio.covariance import (
    estimate_covariance,
    returns_from_prices,
    _ewm_covariance,
    _ensure_psd,
)


def _synthetic_returns(n: int = 200, k: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Correlated returns via Cholesky
    L = np.array(
        [
            [1.0, 0, 0, 0],
            [0.5, 0.866, 0, 0],
            [0.3, 0.2, 0.933, 0],
            [0.1, 0.1, 0.1, 0.980],
        ]
    )[:k, :k]
    raw = rng.normal(0, 0.01, (n, k))
    correlated = raw @ L.T
    dates = pd.bdate_range("2020-01-01", periods=n)
    symbols = [f"SYM_{i}" for i in range(k)]
    return pd.DataFrame(correlated, index=dates, columns=symbols)


@pytest.mark.fast
class TestEstimateCovariance:
    def test_sample_shape(self):
        ret = _synthetic_returns()
        cov = estimate_covariance(ret, method="sample")
        assert cov.shape == (4, 4)
        assert list(cov.index) == list(cov.columns)

    def test_sample_symmetric(self):
        ret = _synthetic_returns()
        cov = estimate_covariance(ret, method="sample")
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_sample_positive_diagonal(self):
        ret = _synthetic_returns()
        cov = estimate_covariance(ret, method="sample")
        assert all(cov.values[i, i] > 0 for i in range(4))

    def test_ledoit_wolf(self):
        pytest.importorskip("sklearn")
        ret = _synthetic_returns()
        cov = estimate_covariance(ret, method="ledoit_wolf")
        assert cov.shape == (4, 4)
        # Shrinkage should produce valid covariance
        eigvals = np.linalg.eigvalsh(cov.values)
        assert all(eigvals > 0)

    def test_ewm_method(self):
        ret = _synthetic_returns()
        cov = estimate_covariance(ret, method="ewm", ewm_halflife=30)
        assert cov.shape == (4, 4)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_dcc_garch_method(self):
        ret = _synthetic_returns(n=120)
        cov = estimate_covariance(ret, method="dcc_garch")
        assert cov.shape == (4, 4)
        # Should be symmetric
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-8)

    def test_annualize_flag(self):
        ret = _synthetic_returns()
        cov_ann = estimate_covariance(ret, method="sample", annualize=True)
        cov_raw = estimate_covariance(ret, method="sample", annualize=False)
        # Annualized should be ~252x raw
        ratio = cov_ann.values[0, 0] / max(cov_raw.values[0, 0], 1e-20)
        assert 250 < ratio < 254

    def test_empty_returns(self):
        ret = pd.DataFrame()
        cov = estimate_covariance(ret)
        assert cov.empty

    def test_single_column(self):
        ret = _synthetic_returns()
        cov = estimate_covariance(ret[["SYM_0"]])
        assert cov.empty  # requires >= 2 columns


@pytest.mark.fast
class TestEWMCovariance:
    def test_basic(self):
        ret = _synthetic_returns(n=100, k=3)
        cov = _ewm_covariance(ret, halflife=30)
        assert cov.shape == (3, 3)
        # Symmetric
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_short_data(self):
        ret = _synthetic_returns(n=5, k=2)
        cov = _ewm_covariance(ret, halflife=10, min_periods=10)
        # Below min_periods → zeros
        assert cov[0, 1] == 0.0


@pytest.mark.fast
class TestEnsurePSD:
    def test_already_psd(self):
        cov = pd.DataFrame(
            np.eye(3) * 0.01,
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        result = _ensure_psd(cov)
        np.testing.assert_allclose(result.values, cov.values, atol=1e-6)

    def test_negative_eigenvalue_fixed(self):
        # Create a matrix with a negative eigenvalue
        arr = np.array([[1.0, 1.5], [1.5, 1.0]])  # eigenvalues: 2.5, -0.5
        cov = pd.DataFrame(arr, index=["A", "B"], columns=["A", "B"])
        result = _ensure_psd(cov)
        eigvals = np.linalg.eigvalsh(result.values)
        assert all(eigvals > 0)


@pytest.mark.fast
class TestReturnsFromPrices:
    def test_basic_log_returns(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {
                "timestamp": list(dates) * 2,
                "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
                "close": [100, 101, 102, 103, 104, 200, 202, 204, 206, 208],
            }
        )
        ret = returns_from_prices(prices, log_returns=True)
        assert "AAPL" in ret.columns
        assert "MSFT" in ret.columns
        assert len(ret) == 4  # drops first NaN row

    def test_simple_returns(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {
                "timestamp": list(dates) * 2,
                "symbol": ["A"] * 3 + ["B"] * 3,
                "close": [100.0, 110.0, 121.0, 200.0, 210.0, 220.5],
            }
        )
        ret = returns_from_prices(prices, log_returns=False)
        assert ret["A"].iloc[0] == pytest.approx(0.1, abs=0.001)
