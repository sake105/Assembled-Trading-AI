"""Tests for Bayesian Neural Network (MC Dropout)."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("sklearn")

from src.assembled_core.ml.bayesian_nn import (  # noqa: E402
    MCDropoutMLP,
    BNNPrediction,
)


def _synthetic_regression(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.1, n)
    return X.astype(np.float32), y.astype(np.float32)


@pytest.mark.phase12
class TestMCDropoutMLP:
    def test_fit_predict(self):
        X, y = _synthetic_regression()
        model = MCDropoutMLP(
            hidden_sizes=[16, 8],
            n_mc_samples=10,
            n_epochs=20,
        )
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert isinstance(pred, BNNPrediction)
        assert len(pred.mean) == 10
        assert len(pred.std) == 10

    def test_uncertainty_positive(self):
        X, y = _synthetic_regression()
        model = MCDropoutMLP(hidden_sizes=[16], n_mc_samples=10, n_epochs=20)
        model.fit(X, y)
        pred = model.predict(X[:20])
        assert (pred.std >= 0).all()

    def test_confidence_property(self):
        X, y = _synthetic_regression()
        model = MCDropoutMLP(hidden_sizes=[16], n_mc_samples=10, n_epochs=20)
        model.fit(X, y)
        pred = model.predict(X[:10])
        conf = pred.confidence
        assert len(conf) == 10
        assert (conf > 0).all()

    def test_sharpe_sizing(self):
        X, y = _synthetic_regression()
        model = MCDropoutMLP(hidden_sizes=[16], n_mc_samples=10, n_epochs=20)
        model.fit(X, y)
        pred = model.predict(X[:10])
        sizing = pred.sharpe_sizing
        assert len(sizing) == 10

    def test_not_fitted_raises(self):
        model = MCDropoutMLP()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))

    def test_samples_shape(self):
        X, y = _synthetic_regression()
        model = MCDropoutMLP(hidden_sizes=[16], n_mc_samples=15, n_epochs=20)
        model.fit(X, y)
        pred = model.predict(X[:5])
        if pred.samples is not None:
            assert pred.samples.shape[1] == 5
