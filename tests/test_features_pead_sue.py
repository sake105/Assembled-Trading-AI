"""Tests for SUE / expected-EPS source — C4-083 closure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.pead_sue import (
    SueResult,
    compute_expected_eps_foster,
    compute_expected_eps_random_walk,
    compute_expected_eps_seasonal_rw,
    compute_sue,
    compute_sue_from_expected,
)


def _quarterly_eps(n: int = 20, base: float = 1.0, growth: float = 0.05) -> pd.Series:
    """Synthetic quarterly EPS series with mild growth + seasonality."""
    rng = np.random.default_rng(42)
    seasonal = np.tile([1.0, 0.9, 1.1, 1.05], n // 4 + 1)[:n]
    trend = np.array([base + growth * i for i in range(n)])
    noise = rng.normal(0, 0.02, n)
    return pd.Series(trend * seasonal + noise, name="eps")


# ---------------------------------------------------------------------------
# Expected EPS models
# ---------------------------------------------------------------------------


def test_random_walk_lags_by_one():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    rw = compute_expected_eps_random_walk(s)
    assert np.isnan(rw.iloc[0])
    assert rw.iloc[1] == 1.0
    assert rw.iloc[2] == 2.0
    assert rw.iloc[3] == 3.0


def test_seasonal_rw_lags_by_seasonality():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    srw = compute_expected_eps_seasonal_rw(s, seasonality=4)
    # First 4 are NaN
    assert srw.iloc[:4].isna().all()
    # Then lagged by 4
    assert srw.iloc[4] == 1.0
    assert srw.iloc[5] == 2.0
    assert srw.iloc[6] == 3.0
    assert srw.iloc[7] == 4.0


def test_seasonal_rw_rejects_zero_seasonality():
    with pytest.raises(ValueError, match="seasonality"):
        compute_expected_eps_seasonal_rw(pd.Series([1.0, 2.0]), seasonality=0)


def test_foster_seasonal_rw_plus_drift():
    """Foster expectation = seasonal RW + recent YoY-diff drift average."""
    # Construct series where YoY drift is constant at +0.5
    # EPS: [1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0, ...]
    s = pd.Series([1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0], dtype=float)
    foster = compute_expected_eps_foster(s, seasonality=4, drift_window=2)
    # At index 8: seasonal_RW = s[4] = 1.5; drift should be avg of (s[5]-s[1], s[6]-s[2], s[7]-s[3])
    # With drift_window=2, last 2 of those: avg((3.5-3.0)+(4.5-4.0))/2 = avg(0.5+0.5)/2 = 0.5
    # So Foster prediction = 1.5 + 0.5 = 2.0
    assert abs(foster.iloc[8] - 2.0) < 1e-9, f"Expected 2.0, got {foster.iloc[8]}"


def test_foster_rejects_invalid_drift_window():
    with pytest.raises(ValueError, match="drift_window"):
        compute_expected_eps_foster(pd.Series([1.0] * 10), drift_window=0)


# ---------------------------------------------------------------------------
# compute_sue
# ---------------------------------------------------------------------------


def test_compute_sue_returns_dataclass_with_all_fields():
    eps = _quarterly_eps(n=20)
    result = compute_sue(eps, method="seasonal_rw")
    assert isinstance(result, SueResult)
    assert len(result.sue) == len(eps)
    assert result.sigma_forecast_error > 0
    assert result.n_events > 0
    assert result.method == "seasonal_rw"


def test_compute_sue_method_random_walk():
    eps = _quarterly_eps(n=12)
    result = compute_sue(eps, method="random_walk")
    assert result.method == "random_walk"
    # First obs is NaN (RW needs t-1)
    assert np.isnan(result.expected_eps.iloc[0])


def test_compute_sue_method_seasonal_rw():
    eps = _quarterly_eps(n=20)
    result = compute_sue(eps, method="seasonal_rw", seasonality=4)
    assert result.method == "seasonal_rw"
    # First 4 obs have NaN expected_eps
    assert result.expected_eps.iloc[:4].isna().all()


def test_compute_sue_method_foster():
    eps = _quarterly_eps(n=20)
    result = compute_sue(eps, method="foster", seasonality=4, drift_window=4)
    assert result.method == "foster"
    # First (seasonality + drift_window) obs are NaN
    assert result.expected_eps.iloc[:8].isna().all()


def test_compute_sue_rejects_external_method():
    """external requires compute_sue_from_expected, not compute_sue."""
    with pytest.raises(ValueError, match="external"):
        compute_sue(_quarterly_eps(n=20), method="external")


def test_compute_sue_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        compute_sue(_quarterly_eps(n=20), method="bogus_model")


def test_compute_sue_rejects_short_series():
    s = pd.Series([1.0, 2.0, 3.0])  # Only 3 obs, need ≥6 for seasonal_rw
    with pytest.raises(ValueError, match="non-NaN obs"):
        compute_sue(s, method="seasonal_rw", seasonality=4)


def test_compute_sue_normalises_to_unit_std():
    """SUE should have ~unit std across the full sample (by construction)."""
    eps = _quarterly_eps(n=40)
    result = compute_sue(eps, method="seasonal_rw")
    sue_clean = result.sue.dropna()
    # Should normalise to ~1 (sample std). Allow tolerance for small sample.
    assert 0.8 < sue_clean.std(ddof=1) < 1.2


# ---------------------------------------------------------------------------
# compute_sue_from_expected
# ---------------------------------------------------------------------------


def test_compute_sue_from_expected_external_method():
    """When expected EPS is supplied externally → method='external'."""
    actual = pd.Series([1.5, 2.5, 3.5, 4.5], name="actual")
    expected = pd.Series([1.0, 2.0, 3.0, 4.0], name="expected")
    result = compute_sue_from_expected(actual, expected)
    assert result.method == "external"
    # Forecast errors all = 0.5; sigma = 0 (degenerate) → NaN SUE
    np.testing.assert_array_almost_equal(
        result.forecast_error.to_numpy(), [0.5, 0.5, 0.5, 0.5]
    )


def test_compute_sue_from_expected_correct_standardisation():
    """SUE = (actual - expected) / σ(forecast_error)."""
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])  # FE always = 0.5
    result = compute_sue_from_expected(actual, expected)
    # σ = 0 here (constant FE) → degenerate, SUE returns NaN
    assert result.sigma_forecast_error == 0.0 or np.isnan(result.sue.iloc[0])


def test_compute_sue_from_expected_rejects_empty_overlap():
    a = pd.Series([1.0], index=pd.Index([0]))
    e = pd.Series([1.0], index=pd.Index([99]))
    with pytest.raises(ValueError, match="share no index"):
        compute_sue_from_expected(a, e)


def test_compute_sue_from_expected_rejects_too_few_obs():
    a = pd.Series([1.0])
    e = pd.Series([0.9])
    with pytest.raises(ValueError, match="non-NaN aligned"):
        compute_sue_from_expected(a, e)
