"""Tests for HMM regime detection module."""

from __future__ import annotations

import pytest

pytest.importorskip("src.assembled_core.ml.regime_hmm")
import pytest
import numpy as np
import pandas as pd

from src.assembled_core.ml.regime_hmm import RegimeHMM, MultiFeatureRegimeHMM


def _synthetic_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Two regimes: bull (positive mean, low vol) and bear (negative mean, high vol)
    regime = np.concatenate([np.zeros(250), np.ones(250)])
    returns = np.where(
        regime == 0,
        rng.normal(0.001, 0.01, n),
        rng.normal(-0.002, 0.025, n),
    )
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(returns, index=dates)


@pytest.mark.fast
class TestRegimeHMM:
    def test_fit(self):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            pytest.skip("hmmlearn not installed")
        returns = _synthetic_returns()
        hmm = RegimeHMM(n_regimes=2)
        hmm.fit(returns)
        assert hmm._is_fitted

    def test_predict_regime(self):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            pytest.skip("hmmlearn not installed")
        returns = _synthetic_returns()
        hmm = RegimeHMM(n_regimes=2)
        hmm.fit(returns)
        regimes = hmm.predict_regime(returns)
        assert len(regimes) == len(returns)
        assert set(regimes.unique()) <= {"bull", "bear", "sideways"}

    def test_predict_proba(self):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            pytest.skip("hmmlearn not installed")
        returns = _synthetic_returns()
        hmm = RegimeHMM(n_regimes=2)
        hmm.fit(returns)
        proba = hmm.predict_regime_proba(returns)
        assert proba.shape[0] == len(returns)
        # Probabilities should sum to ~1
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_not_fitted_raises(self):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            pytest.skip("hmmlearn not installed")
        hmm = RegimeHMM()
        with pytest.raises(RuntimeError):
            hmm.predict_regime(_synthetic_returns())


@pytest.mark.fast
class TestMultiFeatureRegimeHMM:
    def test_fallback_without_hmmlearn(self):
        features = pd.DataFrame(
            {
                "ret": np.random.default_rng(42).normal(0, 0.01, 100),
                "vol": np.random.default_rng(42).uniform(0.01, 0.03, 100),
            }
        )
        mf = MultiFeatureRegimeHMM(n_regimes=2)
        proba = mf.predict_proba(features)
        assert proba.shape[0] == len(features)

    def test_crisis_alert(self):
        features = pd.DataFrame(
            {
                "ret": np.random.default_rng(42).normal(-0.02, 0.03, 50),
                "vol": np.random.default_rng(42).uniform(0.02, 0.05, 50),
            }
        )
        mf = MultiFeatureRegimeHMM(n_regimes=2)
        alert = mf.crisis_alert(features, threshold=0.3)
        assert "alert" in alert
        assert isinstance(alert["alert"], bool)
