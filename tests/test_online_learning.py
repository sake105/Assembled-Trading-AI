"""Tests for online learning module (Plan 2.7)."""

from __future__ import annotations

import pytest

pytest.importorskip("src.assembled_core.ml.online_learning")
import pytest
import numpy as np

from src.assembled_core.ml.online_learning import (
    EWRLSModel,
    RetrainingTrigger,
    compute_model_age_confidence,
)


@pytest.mark.phase12
class TestEWRLSModel:
    def test_init(self):
        model = EWRLSModel(n_features=3)
        assert model.beta.shape == (3,)
        assert model.P.shape == (3, 3)
        assert model.n_updates == 0

    def test_single_update(self):
        model = EWRLSModel(n_features=2, forgetting_factor=0.99)
        x = np.array([1.0, 0.5])
        error = model.update(x, y=1.0)
        assert isinstance(error, float)
        assert model.n_updates == 1

    def test_predict(self):
        model = EWRLSModel(n_features=2)
        model.beta = np.array([0.5, 0.3])
        pred = model.predict(np.array([2.0, 1.0]))
        assert pred == pytest.approx(1.3, abs=0.01)

    def test_batch_update(self):
        rng = np.random.default_rng(42)
        true_beta = np.array([0.5, -0.3, 0.2])
        X = rng.normal(0, 1, (100, 3))
        y = X @ true_beta + rng.normal(0, 0.01, 100)

        model = EWRLSModel(n_features=3, forgetting_factor=0.99)
        errors = model.batch_update(X, y)
        assert len(errors) == 100
        # After 100 updates, beta should be close to true_beta
        np.testing.assert_allclose(model.beta, true_beta, atol=0.1)

    def test_forgetting_adapts(self):
        """Model with low forgetting factor adapts faster to regime change."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 2))
        # Regime 1: beta = [1, 0]
        y1 = X[:100] @ np.array([1.0, 0.0]) + rng.normal(0, 0.01, 100)
        # Regime 2: beta = [0, 1]
        y2 = X[100:] @ np.array([0.0, 1.0]) + rng.normal(0, 0.01, 100)
        y = np.concatenate([y1, y2])

        fast = EWRLSModel(n_features=2, forgetting_factor=0.95)
        slow = EWRLSModel(n_features=2, forgetting_factor=0.995)

        fast.batch_update(X, y)
        slow.batch_update(X, y)

        # Fast model should be closer to regime-2 beta
        fast_dist = np.linalg.norm(fast.beta - np.array([0.0, 1.0]))
        slow_dist = np.linalg.norm(slow.beta - np.array([0.0, 1.0]))
        assert fast_dist < slow_dist


@pytest.mark.phase12
class TestRetrainingTrigger:
    def test_no_trigger_good_ic(self):
        trigger = RetrainingTrigger(ic_threshold=0.0, consecutive_bad_days=5)
        result = trigger.check(ic_value=0.05, prediction_error=0.01)
        assert result is False

    def test_trigger_bad_ic(self):
        trigger = RetrainingTrigger(ic_threshold=0.0, consecutive_bad_days=3)
        trigger.check(-0.01, 0.01)
        trigger.check(-0.02, 0.01)
        result = trigger.check(-0.03, 0.01)
        assert result is True

    def test_reset(self):
        trigger = RetrainingTrigger(consecutive_bad_days=3)
        trigger._bad_day_count = 5
        trigger._error_history = [0.1, 0.2]
        trigger.reset()
        assert trigger._bad_day_count == 0
        assert len(trigger._error_history) == 0

    def test_error_spike_trigger(self):
        rng = np.random.default_rng(42)
        trigger = RetrainingTrigger(error_zscore_threshold=2.0)
        # Build error history with some variance (std > 0)
        for _ in range(50):
            trigger.check(0.05, rng.normal(0.01, 0.002))
        # Very large error spike
        result = trigger.check(0.05, 10.0)
        assert result is True


@pytest.mark.phase12
class TestModelAgeConfidence:
    def test_fresh_model(self):
        conf = compute_model_age_confidence(0)
        assert conf == 1.0

    def test_halflife(self):
        conf = compute_model_age_confidence(30, half_life_days=30)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_old_model(self):
        conf = compute_model_age_confidence(180, half_life_days=30)
        assert conf < 0.05

    def test_monotonic_decay(self):
        values = [compute_model_age_confidence(d) for d in range(0, 100, 10)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
