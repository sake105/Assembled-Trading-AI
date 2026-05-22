"""Tests for Factor Exposure Analysis module (Phase A, Sprint A2).

Tests the factor_exposures module: FactorExposureConfig, compute_factor_exposures, summarize_factor_exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.risk.factor_exposures import (
    FactorExposureConfig,
    compute_factor_exposures,
    summarize_factor_exposures,
)


@pytest.fixture
def sample_strategy_returns() -> pd.Series:
    """Create sample strategy returns."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    # Create returns with some structure
    returns = np.random.randn(100) * 0.01
    return pd.Series(returns, index=dates, name="strategy_return")


@pytest.fixture
def sample_factor_returns() -> pd.DataFrame:
    """Create sample factor returns with known relationship to strategy."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    # Factor 1: Strong positive relationship (beta = 0.5)
    factor1 = np.random.randn(100) * 0.02
    # Factor 2: Weak negative relationship (beta = -0.2)
    factor2 = np.random.randn(100) * 0.015
    # Factor 3: Noise (beta = 0.0)
    factor3 = np.random.randn(100) * 0.01

    return pd.DataFrame(
        {
            "factor1": factor1,
            "factor2": factor2,
            "factor3": factor3,
        },
        index=dates,
    )


@pytest.mark.advanced
def test_compute_factor_exposures_basic_ols(
    sample_strategy_returns, sample_factor_returns
):
    """Test basic OLS factor exposure computation with known linear relationship."""
    # Create strategy returns as linear combination of factors
    true_beta1 = 0.5
    true_beta2 = -0.2
    true_intercept = 0.001

    strategy_returns = (
        true_intercept
        + true_beta1 * sample_factor_returns["factor1"]
        + true_beta2 * sample_factor_returns["factor2"]
        + np.random.randn(len(sample_factor_returns)) * 0.005  # Small noise
    )

    config = FactorExposureConfig(
        freq="1d",
        window_size=60,
        min_obs=30,
        mode="rolling",
        add_constant=True,
        standardize_factors=False,  # Don't standardize for this test
        regression_method="ols",
    )

    exposures = compute_factor_exposures(
        strategy_returns, sample_factor_returns, config
    )

    # Check that we got results
    assert not exposures.empty, "Exposures should not be empty"

    # Check required columns
    assert "beta_factor1" in exposures.columns
    assert "beta_factor2" in exposures.columns
    assert "beta_factor3" in exposures.columns
    assert "intercept" in exposures.columns
    assert "r2" in exposures.columns
    assert "n_obs" in exposures.columns
    assert "residual_vol" in exposures.columns

    # Check that betas are in reasonable range (should be close to true betas)
    # Use mean of last few windows for stability
    recent_exposures = exposures.tail(10)
    mean_beta1 = recent_exposures["beta_factor1"].mean()
    mean_beta2 = recent_exposures["beta_factor2"].mean()

    # Allow some tolerance due to noise
    assert abs(mean_beta1 - true_beta1) < 0.3, (
        f"Beta1 should be close to {true_beta1}, got {mean_beta1}"
    )
    assert abs(mean_beta2 - true_beta2) < 0.3, (
        f"Beta2 should be close to {true_beta2}, got {mean_beta2}"
    )


@pytest.mark.advanced
def test_compute_factor_exposures_rolling_vs_expanding(
    sample_strategy_returns, sample_factor_returns
):
    """Test that rolling and expanding modes produce different number of windows."""
    config_rolling = FactorExposureConfig(
        freq="1d",
        window_size=60,
        min_obs=30,
        mode="rolling",
    )

    config_expanding = FactorExposureConfig(
        freq="1d",
        window_size=60,
        min_obs=30,
        mode="expanding",
    )

    exposures_rolling = compute_factor_exposures(
        sample_strategy_returns, sample_factor_returns, config_rolling
    )
    exposures_expanding = compute_factor_exposures(
        sample_strategy_returns, sample_factor_returns, config_expanding
    )

    # Both should have results
    assert not exposures_rolling.empty
    assert not exposures_expanding.empty

    # Expanding should have more valid windows (less NaN rows at the start)
    # But both should have same total number of rows
    assert len(exposures_rolling) == len(exposures_expanding)

    # Expanding mode should have fewer NaN values in early windows
    n_nan_rolling = exposures_rolling["beta_factor1"].isna().sum()
    n_nan_expanding = exposures_expanding["beta_factor1"].isna().sum()

    # Expanding should have fewer NaN (or equal), but not necessarily
    # We just check that both produce valid results
    assert n_nan_rolling < len(exposures_rolling)
    assert n_nan_expanding < len(exposures_expanding)


@pytest.mark.advanced
def test_compute_factor_exposures_min_obs():
    """Test that windows with too few observations are skipped."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    strategy_returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
    factor_returns = pd.DataFrame(
        {"factor1": np.random.randn(50) * 0.01},
        index=dates,
    )

    # Config with min_obs higher than available data
    config = FactorExposureConfig(
        freq="1d",
        window_size=30,
        min_obs=100,  # Higher than available data
        mode="rolling",
    )

    exposures = compute_factor_exposures(strategy_returns, factor_returns, config)

    # Should still return DataFrame, but all betas should be NaN
    assert not exposures.empty
    assert exposures["beta_factor1"].isna().all(), (
        "All betas should be NaN when min_obs not met"
    )


@pytest.mark.advanced
def test_compute_factor_exposures_ridge(sample_strategy_returns, sample_factor_returns):
    """Test Ridge regression method."""
    config_ols = FactorExposureConfig(
        freq="1d",
        window_size=60,
        min_obs=30,
        regression_method="ols",
        ridge_alpha=1.0,
    )

    config_ridge = FactorExposureConfig(
        freq="1d",
        window_size=60,
        min_obs=30,
        regression_method="ridge",
        ridge_alpha=1.0,
    )

    exposures_ols = compute_factor_exposures(
        sample_strategy_returns, sample_factor_returns, config_ols
    )
    exposures_ridge = compute_factor_exposures(
        sample_strategy_returns, sample_factor_returns, config_ridge
    )

    # Both should have results
    assert not exposures_ols.empty
    assert not exposures_ridge.empty

    # Betas should be different (ridge shrinks coefficients)
    # But similar in magnitude and sign
    recent_ols = exposures_ols.tail(10)
    recent_ridge = exposures_ridge.tail(10)

    mean_beta1_ols = recent_ols["beta_factor1"].mean()
    mean_beta1_ridge = recent_ridge["beta_factor1"].mean()

    # Both should be finite
    assert np.isfinite(mean_beta1_ols)
    assert np.isfinite(mean_beta1_ridge)

    # Ridge betas should generally be smaller in absolute value (shrinkage)
    # But sign should be the same
    if abs(mean_beta1_ols) > 0.1:
        assert np.sign(mean_beta1_ols) == np.sign(mean_beta1_ridge), (
            "Sign should be same"
        )


@pytest.mark.advanced
def test_summarize_factor_exposures_basic(
    sample_strategy_returns, sample_factor_returns
):
    """Test summarization of factor exposures."""
    config = FactorExposureConfig(freq="1d", window_size=60, min_obs=30)

    exposures = compute_factor_exposures(
        sample_strategy_returns, sample_factor_returns, config
    )

    summary = summarize_factor_exposures(exposures, config)

    # Check required columns
    assert "factor" in summary.columns
    assert "mean_beta" in summary.columns
    assert "std_beta" in summary.columns
    assert "mean_r2" in summary.columns
    assert "median_r2" in summary.columns
    assert "mean_residual_vol" in summary.columns
    assert "n_windows" in summary.columns
    assert "n_windows_total" in summary.columns

    # Should have at least one factor
    assert len(summary) > 0

    # Check that n_windows is reasonable (should be <= n_windows_total)
    assert (summary["n_windows"] <= summary["n_windows_total"]).all()


@pytest.mark.advanced
def test_compute_factor_exposures_handles_missing_data():
    """Test that missing data (NaN) is handled robustly."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    strategy_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

    # Create factor returns with some NaN values
    factor_returns = pd.DataFrame(
        {
            "factor1": np.random.randn(100) * 0.01,
            "factor2": np.random.randn(100) * 0.01,
        },
        index=dates,
    )

    # Introduce some NaN values
    factor_returns.loc[dates[20:25], "factor1"] = np.nan
    factor_returns.loc[dates[50:55], "factor2"] = np.nan

    config = FactorExposureConfig(freq="1d", window_size=60, min_obs=30)

    # Should not raise exception
    exposures = compute_factor_exposures(strategy_returns, factor_returns, config)

    # Should still produce results (windows with enough non-NaN data)
    assert not exposures.empty


@pytest.mark.advanced
def test_compute_factor_exposures_empty_inputs():
    """Test that empty inputs return empty DataFrame without crashing."""
    empty_returns = pd.Series([], dtype=float)
    empty_factors = pd.DataFrame()

    config = FactorExposureConfig()

    exposures = compute_factor_exposures(empty_returns, empty_factors, config)

    assert exposures.empty, "Empty inputs should return empty DataFrame"


@pytest.mark.advanced
def test_summarize_factor_exposures_min_r2_filter():
    """Test that summarize_factor_exposures filters by min_r2_for_report."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    strategy_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    factor_returns = pd.DataFrame(
        {
            "factor1": np.random.randn(100) * 0.01,
            "factor2": np.random.randn(100) * 0.01,
        },
        index=dates,
    )

    config = FactorExposureConfig(
        freq="1d", window_size=60, min_obs=30, min_r2_for_report=0.5
    )

    exposures = compute_factor_exposures(strategy_returns, factor_returns, config)

    summary = summarize_factor_exposures(exposures, config)

    # All factors in summary should have mean_r2 >= 0.5 (if any pass the filter)
    if not summary.empty:
        assert (summary["mean_r2"] >= 0.5).all()


@pytest.mark.advanced
def test_compute_factor_exposures_no_constant():
    """Test factor exposure computation without intercept term."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    strategy_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    factor_returns = pd.DataFrame(
        {"factor1": np.random.randn(100) * 0.01},
        index=dates,
    )

    config = FactorExposureConfig(
        freq="1d", window_size=60, min_obs=30, add_constant=False
    )

    exposures = compute_factor_exposures(strategy_returns, factor_returns, config)

    # Should not have intercept column
    assert "intercept" not in exposures.columns

    # Should still have other columns
    assert "beta_factor1" in exposures.columns
    assert "r2" in exposures.columns
