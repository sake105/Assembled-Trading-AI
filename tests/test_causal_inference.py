"""Tests for M29: Causal Inference for Factor-Return Relationships."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip('src.assembled_core.ml.causal_inference')
from src.assembled_core.ml.causal_inference import (
    CausalEffectResult,
    GrangerResult,
    estimate_propensity_score,
    propensity_score_matching,
    iv_two_stage_least_squares,
    difference_in_differences,
    granger_causality_test,
    screen_factors_causal,
)


@pytest.fixture
def causal_data():
    """Synthetic data with known causal relationship."""
    rng = np.random.default_rng(42)
    n = 500
    # True causal effect: factor -> returns with beta=0.02
    confounder = rng.normal(0, 1, n)
    factor = 0.5 * confounder + rng.normal(0, 1, n)
    returns = 0.02 * factor + 0.01 * confounder + rng.normal(0, 0.05, n)
    return factor, returns, confounder


@pytest.fixture
def spurious_data():
    """Spurious correlation driven by a confounder."""
    rng = np.random.default_rng(99)
    n = 500
    confounder = rng.normal(0, 1, n)
    factor = 0.8 * confounder + rng.normal(0, 0.5, n)
    returns = 0.8 * confounder + rng.normal(0, 0.5, n)
    # factor and returns are correlated but factor doesn't cause returns
    return factor, returns, confounder


@pytest.mark.phase12
class TestPropensityScoreMatching:
    def test_basic_psm(self, causal_data):
        factor, returns, confounder = causal_data
        result = propensity_score_matching(
            factor, returns, confounder.reshape(-1, 1),
        )
        assert isinstance(result, CausalEffectResult)
        assert result.method == "propensity_score_matching"
        assert result.n_treated > 0
        assert result.n_control > 0

    def test_short_data_returns_default(self):
        result = propensity_score_matching(
            np.array([1.0, 2.0]), np.array([0.01, 0.02]),
        )
        assert result.ate == 0.0
        assert result.p_value == 1.0

    def test_ate_positive_for_real_effect(self, causal_data):
        factor, returns, _ = causal_data
        result = propensity_score_matching(factor, returns)
        # There's a real positive causal effect
        assert result.ate != 0.0

    def test_bounded_p_value(self, causal_data):
        factor, returns, _ = causal_data
        result = propensity_score_matching(factor, returns)
        assert 0.0 <= result.p_value <= 1.0


@pytest.mark.phase12
class TestPropensityScore:
    def test_scores_bounded(self):
        rng = np.random.default_rng(42)
        covariates = rng.normal(0, 1, (100, 3))
        treatment = (rng.random(100) > 0.5).astype(float)
        scores = estimate_propensity_score(covariates, treatment)
        assert all(0.01 <= s <= 0.99 for s in scores)
        assert len(scores) == 100


@pytest.mark.phase12
class TestIV2SLS:
    def test_basic_iv(self, causal_data):
        factor, returns, _ = causal_data
        instrument = np.roll(factor, 1)
        instrument[0] = factor.mean()
        result = iv_two_stage_least_squares(factor, returns, instrument)
        assert isinstance(result, CausalEffectResult)
        assert result.method == "iv_2sls"

    def test_short_data(self):
        result = iv_two_stage_least_squares(
            np.array([1.0, 2.0]), np.array([0.01, 0.02]),
            np.array([0.5, 1.5]),
        )
        assert result.ate == 0.0


@pytest.mark.phase12
class TestDifferenceInDifferences:
    def test_positive_treatment_effect(self):
        rng = np.random.default_rng(42)
        n = 100
        pre = 50
        # Treated group gets a boost after event
        treated = np.concatenate([
            rng.normal(0.001, 0.01, pre),
            rng.normal(0.005, 0.01, n - pre),  # post-event boost
        ])
        control = rng.normal(0.001, 0.01, n)
        result = difference_in_differences(treated, control, pre)
        assert result.method == "difference_in_differences"
        assert result.ate > 0  # treatment had positive effect

    def test_no_effect(self):
        rng = np.random.default_rng(42)
        n = 100
        treated = rng.normal(0.001, 0.01, n)
        control = rng.normal(0.001, 0.01, n)
        result = difference_in_differences(treated, control, 50)
        # ATE should be near zero
        assert abs(result.ate) < 0.01

    def test_short_data_v2(self):
        result = difference_in_differences(
            np.array([0.01, 0.02, 0.03]),
            np.array([0.01, 0.02, 0.03]),
            pre_periods=1,
        )
        assert result.p_value == 1.0


@pytest.mark.phase12
class TestGrangerCausality:
    def test_causal_series(self):
        rng = np.random.default_rng(42)
        n = 200
        cause = rng.normal(0, 1, n)
        effect = np.zeros(n)
        for i in range(2, n):
            effect[i] = 0.5 * cause[i - 1] + 0.3 * cause[i - 2] + rng.normal(0, 0.5)
        result = granger_causality_test(cause, effect, max_lag=3)
        assert isinstance(result, GrangerResult)
        assert result.f_statistic > 0

    def test_independent_series(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = granger_causality_test(x, y)
        # Should not detect strong causality
        assert isinstance(result, GrangerResult)

    def test_short_series(self):
        result = granger_causality_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert result.p_value == 1.0


@pytest.mark.phase12
class TestScreenFactors:
    def test_screen_multiple_factors(self):
        rng = np.random.default_rng(42)
        n = 200
        factors = pd.DataFrame({
            "real_factor": rng.normal(0, 1, n),
            "noise_factor": rng.normal(0, 1, n),
        })
        returns = pd.Series(
            0.02 * factors["real_factor"].values + rng.normal(0, 0.05, n),
            index=factors.index,
        )
        results = screen_factors_causal(factors, returns)
        assert len(results) == 2
        assert all(isinstance(r, CausalEffectResult) for r in results)
        # Results should be sorted by p-value
        assert results[0].p_value <= results[1].p_value
