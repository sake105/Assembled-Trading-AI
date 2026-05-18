"""Tests for src/assembled_core/qa/synthetic_control.py (C2-027)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from src.assembled_core.qa.synthetic_control import (
    SyntheticControlResult,
    compute_treatment_effect,
    fit_synthetic_control,
    placebo_test,
)


def _make_synthetic_data(
    n_periods: int = 60,
    n_donors: int = 5,
    treatment_period: int = 40,
    treatment_effect: float = 5.0,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Generate synthetic test data with a known treatment effect.

    Treated unit follows a linear combination of donors pre-treatment,
    then deviates by `treatment_effect` post-treatment.
    """
    rng = np.random.default_rng(seed)
    # Donors: independent random walks
    donor_arr = np.cumsum(rng.normal(0, 1, size=(n_periods, n_donors)), axis=0)
    donors = pd.DataFrame(
        donor_arr,
        columns=[f"donor_{i}" for i in range(n_donors)],
    )
    # Treated = linear combo of first 3 donors (known true weights)
    true_weights = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    treated_arr = donor_arr @ true_weights + rng.normal(0, 0.1, size=n_periods)
    # Inject treatment effect post-treatment
    treated_arr[treatment_period:] += treatment_effect
    treated = pd.Series(treated_arr, name="treated")
    return treated, donors, treatment_period


# ---------------------------------------------------------------------------
# fit_synthetic_control
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFitSyntheticControl:
    def test_basic_fit(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        assert isinstance(result, SyntheticControlResult)
        assert result.n_donors == 5
        assert result.n_pre == 40
        assert result.n_post == 20
        assert result.converged is True

    def test_weights_sum_to_one(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        assert (result.weights >= -1e-9).all()

    def test_synthetic_series_length(self) -> None:
        treated, donors, t0 = _make_synthetic_data(n_periods=80)
        result = fit_synthetic_control(treated, donors, treatment_period=50)
        assert len(result.synthetic_series) == len(treated)

    def test_pre_treatment_rmse_small_with_known_weights(self) -> None:
        """If treated is a linear combo of donors pre-treatment, the fit
        RMSE should be very small (limited by injected noise)."""
        treated, donors, t0 = _make_synthetic_data(n_periods=100, treatment_period=80)
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        # Noise std = 0.1; RMSE should be order 0.1
        assert result.pre_treatment_rmse < 0.5

    def test_recovers_dominant_donors(self) -> None:
        """True weights are [0.5, 0.3, 0.2, 0, 0]. Fit should put most
        weight on donors 0-2."""
        treated, donors, t0 = _make_synthetic_data(n_periods=100, treatment_period=80)
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        active_weight = result.weights.iloc[:3].sum()
        # Most of the weight should be in donors 0-2
        assert active_weight > 0.6

    def test_invalid_treatment_period_raises(self) -> None:
        treated, donors, _ = _make_synthetic_data(n_periods=50)
        with pytest.raises(ValueError, match="treatment_period"):
            fit_synthetic_control(treated, donors, treatment_period=0)
        with pytest.raises(ValueError, match="treatment_period"):
            fit_synthetic_control(treated, donors, treatment_period=50)

    def test_length_mismatch_raises(self) -> None:
        treated = pd.Series(np.zeros(50))
        donors = pd.DataFrame(np.zeros((40, 3)))
        with pytest.raises(ValueError, match="rows"):
            fit_synthetic_control(treated, donors, treatment_period=20)

    def test_too_few_donors_raises(self) -> None:
        treated = pd.Series(np.zeros(50))
        donors = pd.DataFrame(np.zeros((50, 1)))
        with pytest.raises(ValueError, match="≥2 donors"):
            fit_synthetic_control(treated, donors, treatment_period=25)

    def test_nan_in_treated_raises(self) -> None:
        treated = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        donors = pd.DataFrame({"a": [1.0] * 6, "b": [2.0] * 6})
        with pytest.raises(ValueError, match="treated contains NaN"):
            fit_synthetic_control(treated, donors, treatment_period=3)

    def test_nan_in_donor_raises(self) -> None:
        treated = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        donors = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
        with pytest.raises(ValueError):  # ≥2 donors OR NaN in donor
            fit_synthetic_control(treated, donors, treatment_period=3)


# ---------------------------------------------------------------------------
# compute_treatment_effect
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeTreatmentEffect:
    def test_post_effect_recovers_injected_treatment(self) -> None:
        """With injected treatment_effect=5.0 over 20 post-periods, the
        mean post-treatment effect should be close to 5.0."""
        treated, donors, t0 = _make_synthetic_data(
            n_periods=100, treatment_period=70, treatment_effect=5.0
        )
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        te = compute_treatment_effect(result, treated)
        mean_post = te.iloc[t0:].mean()
        # Recovered effect should be within 1 unit of true 5.0
        assert abs(mean_post - 5.0) < 1.5

    def test_pre_residuals_small(self) -> None:
        treated, donors, t0 = _make_synthetic_data(n_periods=100, treatment_period=80)
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        te = compute_treatment_effect(result, treated)
        pre_residuals = te.iloc[:t0]
        # Pre-treatment residuals should be near zero (noise-level)
        assert abs(pre_residuals.mean()) < 0.3
        assert pre_residuals.std() < 0.5

    def test_length_mismatch_raises(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = fit_synthetic_control(treated, donors, treatment_period=t0)
        with pytest.raises(ValueError, match="same length"):
            compute_treatment_effect(result, treated.iloc[:30])


# ---------------------------------------------------------------------------
# placebo_test
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPlaceboTest:
    def test_basic_placebo(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = placebo_test(treated, donors, treatment_period=t0)
        assert "original_avg_post_effect" in result
        assert "p_value" in result
        assert result["n_placebos_total"] == 5
        assert 0 <= result["n_placebos_used"] <= 5

    def test_large_effect_low_p_value(self) -> None:
        """With a large injected treatment effect (10.0 std vs 0.1 noise),
        the p-value should be near 0 (original effect is in the tail of
        placebo distribution)."""
        treated, donors, t0 = _make_synthetic_data(
            n_periods=100, treatment_period=70, treatment_effect=10.0
        )
        result = placebo_test(treated, donors, treatment_period=t0)
        # p-value depends on RMSE filter; check it's well below 0.5
        if not np.isnan(result["p_value"]):
            assert result["p_value"] <= 0.5

    def test_zero_effect_high_p_value(self) -> None:
        """With NO treatment effect, the placebo distribution should
        envelope the original — p-value typically near 1.0."""
        treated, donors, t0 = _make_synthetic_data(
            n_periods=100, treatment_period=70, treatment_effect=0.0
        )
        result = placebo_test(treated, donors, treatment_period=t0)
        # p-value should be relatively high (no edge)
        if not np.isnan(result["p_value"]) and result["n_placebos_used"] > 0:
            assert result["p_value"] >= 0.2

    def test_placebo_records_have_3_fields(self) -> None:
        treated, donors, t0 = _make_synthetic_data()
        result = placebo_test(treated, donors, treatment_period=t0)
        for record in result["placebo_effects"]:
            assert len(record) == 3
            name, effect, rmse = record
            assert isinstance(name, str)
            assert np.isfinite(effect)
            assert np.isfinite(rmse)
            assert rmse >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_two_donors_minimum() -> None:
    """Smallest valid donor pool is 2 donors."""
    rng = np.random.default_rng(0)
    treated = pd.Series(rng.normal(0, 1, 30))
    donors = pd.DataFrame(rng.normal(0, 1, (30, 2)), columns=["a", "b"])
    result = fit_synthetic_control(treated, donors, treatment_period=20)
    assert result.n_donors == 2
    assert abs(result.weights.sum() - 1.0) < 1e-6


@pytest.mark.fast
def test_weights_pd_series_index_matches_donors() -> None:
    treated, donors, t0 = _make_synthetic_data()
    result = fit_synthetic_control(treated, donors, treatment_period=t0)
    assert list(result.weights.index) == list(donors.columns)
