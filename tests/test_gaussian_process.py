"""Tests for M19b: Gaussian Process Regression."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("src.assembled_core.ml.gaussian_process")
from src.assembled_core.ml.gaussian_process import (
    GPRResult,
    FactorGPR,
    build_gpr_position_sizing_signal,
)


@pytest.fixture
def train_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(0, 0.2, 100)
    return X, y


@pytest.fixture
def test_data():
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, (20, 5))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(0, 0.2, 20)
    return X, y


@pytest.mark.phase12
class TestFactorGPR:
    def test_fit_and_predict(self, train_data, test_data):
        X_train, y_train = train_data
        X_test, _ = test_data

        gpr = FactorGPR(max_train_samples=100)
        gpr.fit(X_train, y_train)
        result = gpr.predict(X_test)

        assert isinstance(result, GPRResult)
        assert len(result.mean) == len(X_test)
        assert len(result.std) == len(X_test)
        assert all(result.std > 0)
        assert all(result.lower_95 < result.upper_95)

    def test_is_fitted_flag(self, train_data):
        X, y = train_data
        gpr = FactorGPR()
        assert gpr.is_fitted is False
        gpr.fit(X, y)
        assert gpr.is_fitted is True

    def test_predict_before_fit_raises(self):
        gpr = FactorGPR()
        with pytest.raises(RuntimeError, match="fit"):
            gpr.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_subsampling(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (1000, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 1000)

        gpr = FactorGPR(max_train_samples=50)
        gpr.fit(X, y)
        result = gpr.predict(X[:5])
        assert len(result.mean) == 5

    def test_confidence_inversely_proportional_to_std(self, train_data, test_data):
        X_train, y_train = train_data
        X_test, _ = test_data

        gpr = FactorGPR(max_train_samples=100)
        gpr.fit(X_train, y_train)
        result = gpr.predict(X_test)

        # Confidence should be inversely related to std
        for i in range(len(result.mean)):
            assert result.confidence[i] == pytest.approx(1.0 / result.std[i], rel=1e-5)

    def test_credible_intervals_95(self, train_data, test_data):
        X_train, y_train = train_data
        X_test, _ = test_data

        gpr = FactorGPR(max_train_samples=100)
        gpr.fit(X_train, y_train)
        result = gpr.predict(X_test)

        # 95% CI should be mean +/- 1.96*std
        np.testing.assert_allclose(
            result.lower_95,
            result.mean - 1.96 * result.std,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result.upper_95,
            result.mean + 1.96 * result.std,
            rtol=1e-5,
        )

    def test_pandas_input(self, train_data, test_data):
        import pandas as pd

        X_train, y_train = train_data
        X_test, _ = test_data

        gpr = FactorGPR(max_train_samples=100)
        gpr.fit(
            pd.DataFrame(X_train),
            pd.Series(y_train),
        )
        result = gpr.predict(pd.DataFrame(X_test))
        assert len(result.mean) == len(X_test)


@pytest.mark.phase12
class TestPositionSizingSignal:
    def test_basic_sizing(self):
        result = GPRResult(
            mean=np.array([0.5, 0.3, -0.2]),
            std=np.array([0.1, 0.5, 0.2]),
            lower_95=np.zeros(3),
            upper_95=np.ones(3),
            confidence=np.array([10.0, 2.0, 5.0]),
        )
        sizing = build_gpr_position_sizing_signal(result)
        assert len(sizing) == 3
        # Higher confidence should scale up the signal
        # Position 0 has highest confidence -> should have largest magnitude

    def test_confidence_scaling_zero(self):
        result = GPRResult(
            mean=np.array([1.0, 1.0]),
            std=np.array([0.1, 0.5]),
            lower_95=np.zeros(2),
            upper_95=np.ones(2),
            confidence=np.array([10.0, 2.0]),
        )
        # With scaling=0, confidence is ignored
        sizing = build_gpr_position_sizing_signal(result, confidence_scaling=0.0)
        np.testing.assert_allclose(sizing, result.mean, rtol=1e-5)
