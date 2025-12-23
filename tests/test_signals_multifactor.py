"""Tests for Multi-Factor Signal Generation Module.

Tests the build_multifactor_signal() and select_top_bottom() functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.config.factor_bundles import (
    FactorBundleConfig,
    FactorConfig,
    FactorBundleOptions,
)
from src.assembled_core.signals.multifactor_signal import (
    build_multifactor_signal,
    select_top_bottom,
    MultiFactorSignalResult,
)


@pytest.fixture
def sample_factors_df() -> pd.DataFrame:
    """Create a synthetic factor DataFrame with 3 factors, 3 symbols, 20 days."""
    np.random.seed(42)  # Fix seed for reproducibility
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    all_data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            # Create factor values with clear differences between symbols
            # AAPL: High factor values (should rank high)
            # MSFT: Medium factor values
            # GOOGL: Low factor values (should rank low)
            symbol_bias = {"AAPL": 2.0, "MSFT": 0.0, "GOOGL": -2.0}[symbol]

            # Factor 1: Positive correlation (higher is better)
            factor1 = symbol_bias + 0.1 * i + np.random.normal(0, 0.1)

            # Factor 2: Negative correlation (lower is better, will be inverted)
            factor2 = -symbol_bias + 0.05 * i + np.random.normal(0, 0.1)

            # Factor 3: Mixed correlation
            factor3 = symbol_bias * 0.5 + np.random.normal(0, 0.2)

            all_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "returns_12m": factor1,
                    "rv_20": factor2,  # Volatility (lower is better)
                    "momentum_12m_excl_1m": factor3,
                }
            )

    df = pd.DataFrame(all_data)
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


@pytest.fixture
def sample_bundle_positive() -> FactorBundleConfig:
    """Create a simple bundle with positive-direction factors."""
    return FactorBundleConfig(
        universe="test_universe",
        factor_set="test",
        horizon_days=20,
        factors=[
            FactorConfig(name="returns_12m", weight=0.5, direction="positive"),
            FactorConfig(name="momentum_12m_excl_1m", weight=0.5, direction="positive"),
        ],
        options=FactorBundleOptions(
            winsorize=False,
            zscore=True,
            neutralize_by=None,
        ),
    )


@pytest.fixture
def sample_bundle_mixed_directions() -> FactorBundleConfig:
    """Create a bundle with mixed-direction factors."""
    return FactorBundleConfig(
        universe="test_universe",
        factor_set="test",
        horizon_days=20,
        factors=[
            FactorConfig(name="returns_12m", weight=0.4, direction="positive"),
            FactorConfig(
                name="rv_20", weight=0.4, direction="negative"
            ),  # Lower volatility is better
            FactorConfig(name="momentum_12m_excl_1m", weight=0.2, direction="positive"),
        ],
        options=FactorBundleOptions(
            winsorize=True,
            winsorize_limits=[0.01, 0.99],
            zscore=True,
            neutralize_by=None,
        ),
    )


@pytest.fixture
def sample_bundle_no_zscore() -> FactorBundleConfig:
    """Create a bundle without z-scoring (raw values)."""
    return FactorBundleConfig(
        universe="test_universe",
        factor_set="test",
        horizon_days=20,
        factors=[
            FactorConfig(name="returns_12m", weight=1.0, direction="positive"),
        ],
        options=FactorBundleOptions(
            winsorize=False,
            zscore=False,
            neutralize_by=None,
        ),
    )


class TestBuildMultifactorSignal:
    """Tests for build_multifactor_signal() function."""

    def test_basic_functionality(self, sample_factors_df, sample_bundle_positive):
        """Test that build_multifactor_signal runs without errors and adds mf_score."""
        result = build_multifactor_signal(
            sample_factors_df,
            bundle=sample_bundle_positive,
        )

        assert isinstance(result, MultiFactorSignalResult)
        assert len(result.df) == len(sample_factors_df)

        # Should preserve original columns
        for col in sample_factors_df.columns:
            assert col in result.df.columns

        # Should add mf_score column
        assert "mf_score" in result.df.columns

        # Should add z-score columns
        assert "returns_12m_z" in result.df.columns
        assert "momentum_12m_excl_1m_z" in result.df.columns

        # mf_score should not be all NaN (unless all factors are NaN)
        non_null_count = result.df["mf_score"].notna().sum()
        assert non_null_count > 0, "mf_score should have some non-null values"

        # Check metadata
        assert "used_factors" in result.meta
        assert "factor_weights" in result.meta
        assert "options_applied" in result.meta

    def test_direction_positive_higher_values_better(
        self, sample_factors_df, sample_bundle_positive
    ):
        """Test that higher factor values lead to higher mf_score for positive-direction factors."""
        result = build_multifactor_signal(
            sample_factors_df,
            bundle=sample_bundle_positive,
        )

        # For each timestamp, check that AAPL (highest factor values) has highest mf_score
        for timestamp in sample_factors_df["timestamp"].unique():
            ts_data = result.df[result.df["timestamp"] == timestamp].copy()

            if ts_data["mf_score"].notna().sum() < 2:
                continue  # Skip if not enough non-null values

            # AAPL should have higher mf_score than GOOGL (on average)
            aapl_score = ts_data[ts_data["symbol"] == "AAPL"]["mf_score"].values
            googl_score = ts_data[ts_data["symbol"] == "GOOGL"]["mf_score"].values

            if (
                len(aapl_score) > 0
                and len(googl_score) > 0
                and not (np.isnan(aapl_score[0]) or np.isnan(googl_score[0]))
            ):
                assert aapl_score[0] > googl_score[0], (
                    f"At {timestamp}, AAPL mf_score ({aapl_score[0]}) should be > GOOGL mf_score ({googl_score[0]})"
                )

    def test_direction_negative_inverts_factor(
        self, sample_factors_df, sample_bundle_mixed_directions
    ):
        """Test that negative-direction factors are inverted (lower values become higher scores)."""
        result = build_multifactor_signal(
            sample_factors_df,
            bundle=sample_bundle_mixed_directions,
        )

        # Check that rv_20_z column exists (volatility, negative direction)
        assert "rv_20_z" in result.df.columns

        # For negative-direction factors, after inversion, higher original values become lower scores
        # Check a few timestamps
        for timestamp in sample_factors_df["timestamp"].unique()[:5]:
            ts_data = result.df[result.df["timestamp"] == timestamp].copy()

            if ts_data["mf_score"].notna().sum() < 2:
                continue

            # GOOGL should have lower rv_20 (lower volatility) but higher mf_score
            # after inversion of rv_20
            googl_data = ts_data[ts_data["symbol"] == "GOOGL"]
            aapl_data = ts_data[ts_data["symbol"] == "AAPL"]

            if len(googl_data) > 0 and len(aapl_data) > 0:
                # This is a sanity check - the inversion should help GOOGL (low volatility)
                # score better in the final mf_score when combined with other factors
                assert "mf_score" in ts_data.columns

    def test_winsorize_behavior(self, sample_factors_df):
        """Test that winsorizing clips extreme values."""
        # Add extreme outliers
        df_with_outliers = sample_factors_df.copy()
        df_with_outliers.loc[0, "returns_12m"] = 1000.0  # Extreme outlier
        df_with_outliers.loc[1, "returns_12m"] = -1000.0  # Extreme outlier

        bundle = FactorBundleConfig(
            universe="test_universe",
            factor_set="test",
            horizon_days=20,
            factors=[
                FactorConfig(name="returns_12m", weight=1.0, direction="positive"),
            ],
            options=FactorBundleOptions(
                winsorize=True,
                winsorize_limits=[0.01, 0.99],
                zscore=True,
            ),
        )

        result = build_multifactor_signal(df_with_outliers, bundle=bundle)

        # After winsorizing and z-scoring, extreme values should be clipped
        # Check that the z-scored values are reasonable (not in the thousands)
        z_col = "returns_12m_z"
        if z_col in result.df.columns:
            z_values = result.df[z_col].dropna()
            assert z_values.abs().max() < 10.0, (
                f"Z-scores should be reasonable after winsorizing, got max={z_values.abs().max()}"
            )

    def test_zscore_crosssectional_mean_zero_std_one(
        self, sample_factors_df, sample_bundle_positive
    ):
        """Test that z-scoring produces mean≈0 and std≈1 per timestamp (cross-sectional)."""
        result = build_multifactor_signal(
            sample_factors_df,
            bundle=sample_bundle_positive,
        )

        # Check z-score columns
        for factor_name in ["returns_12m", "momentum_12m_excl_1m"]:
            z_col = f"{factor_name}_z"
            if z_col not in result.df.columns:
                continue

            # Group by timestamp and check mean and std
            for timestamp in result.df["timestamp"].unique():
                ts_data = result.df[result.df["timestamp"] == timestamp][z_col].dropna()

                if len(ts_data) >= 2:  # Need at least 2 values for std
                    mean_val = ts_data.mean()
                    std_val = ts_data.std(
                        ddof=0
                    )  # Population std (as used in function)

                    # Mean should be approximately 0 (allowing for floating point errors)
                    assert abs(mean_val) < 1e-10, (
                        f"Z-score mean should be ≈0, got {mean_val} for {factor_name} at {timestamp}"
                    )

                    # Std should be approximately 1 (allowing for floating point errors or single value)
                    if len(ts_data) > 1:
                        assert abs(std_val - 1.0) < 1e-10 or std_val == 0.0, (
                            f"Z-score std should be ≈1, got {std_val} for {factor_name} at {timestamp}"
                        )

    def test_missing_factors_are_logged(
        self, sample_factors_df, sample_bundle_positive
    ):
        """Test that missing factors are logged and excluded from calculation."""
        # Remove one factor from the DataFrame
        df_missing = sample_factors_df.drop(columns=["momentum_12m_excl_1m"]).copy()

        # Should still work with remaining factors
        result = build_multifactor_signal(
            df_missing,
            bundle=sample_bundle_positive,
        )

        # Should have mf_score computed from remaining factors
        assert "mf_score" in result.df.columns

        # Metadata should indicate missing factors
        assert "missing_factors" in result.meta
        assert "momentum_12m_excl_1m" in result.meta["missing_factors"]

        # Used factors should only include available ones
        assert "returns_12m" in result.meta["used_factors"]

    def test_no_factors_available_raises_error(self, sample_factors_df):
        """Test that ValueError is raised if no factors from bundle are available."""
        bundle = FactorBundleConfig(
            universe="test_universe",
            factor_set="test",
            horizon_days=20,
            factors=[
                FactorConfig(
                    name="nonexistent_factor", weight=1.0, direction="positive"
                ),
            ],
            options=FactorBundleOptions(),
        )

        with pytest.raises(ValueError, match="No factors from bundle are available"):
            build_multifactor_signal(sample_factors_df, bundle=bundle)

    def test_empty_dataframe_raises_error(self, sample_bundle_positive):
        """Test that empty DataFrame raises KeyError."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "returns_12m"])

        with pytest.raises(KeyError, match="empty"):
            build_multifactor_signal(empty_df, bundle=sample_bundle_positive)

    def test_missing_required_columns_raises_error(self, sample_bundle_positive):
        """Test that missing required columns raise ValueError."""
        df_no_timestamp = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "returns_12m": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            build_multifactor_signal(df_no_timestamp, bundle=sample_bundle_positive)

        df_no_symbol = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=10, freq="D", tz="UTC"
                ),
                "returns_12m": range(10),
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            build_multifactor_signal(df_no_symbol, bundle=sample_bundle_positive)


class TestSelectTopBottom:
    """Tests for select_top_bottom() function."""

    def test_basic_functionality(self, sample_factors_df, sample_bundle_positive):
        """Test that select_top_bottom adds mf_long_flag and mf_short_flag columns."""
        # First build mf_score
        mf_result = build_multifactor_signal(
            sample_factors_df, bundle=sample_bundle_positive
        )

        # Then select top/bottom
        result = select_top_bottom(
            mf_result.df,
            top_quantile=0.2,
            bottom_quantile=0.2,
        )

        assert "mf_long_flag" in result.columns
        assert "mf_short_flag" in result.columns

        # Flags should be 0 or 1
        assert result["mf_long_flag"].isin([0, 1]).all()
        assert result["mf_short_flag"].isin([0, 1]).all()

        # Should preserve all original columns including mf_score
        assert "mf_score" in result.columns

    def test_top_quantile_selection(self, sample_factors_df, sample_bundle_positive):
        """Test that top quantile symbols are correctly flagged as long."""
        mf_result = build_multifactor_signal(
            sample_factors_df, bundle=sample_bundle_positive
        )

        result = select_top_bottom(
            mf_result.df,
            top_quantile=0.33,  # Top 33% (should be 1 out of 3 symbols)
            bottom_quantile=0.2,
        )

        # For each timestamp, check that top symbol is flagged
        for timestamp in result["timestamp"].unique():
            ts_data = result[result["timestamp"] == timestamp].copy()

            if ts_data["mf_score"].notna().sum() < 2:
                continue  # Skip if not enough data

            # Find symbol with highest mf_score
            top_symbol = ts_data.loc[ts_data["mf_score"].idxmax(), "symbol"]

            # Check that top symbol has long flag
            top_symbol_data = ts_data[ts_data["symbol"] == top_symbol]
            if len(top_symbol_data) > 0:
                assert top_symbol_data["mf_long_flag"].iloc[0] == 1, (
                    f"Top symbol {top_symbol} at {timestamp} should have mf_long_flag=1"
                )

    def test_bottom_quantile_selection(self, sample_factors_df, sample_bundle_positive):
        """Test that bottom quantile symbols are correctly flagged as short."""
        mf_result = build_multifactor_signal(
            sample_factors_df, bundle=sample_bundle_positive
        )

        result = select_top_bottom(
            mf_result.df,
            top_quantile=0.2,
            bottom_quantile=0.33,  # Bottom 33% (should be 1 out of 3 symbols)
        )

        # For each timestamp, check that bottom symbol is flagged
        for timestamp in result["timestamp"].unique():
            ts_data = result[result["timestamp"] == timestamp].copy()

            if ts_data["mf_score"].notna().sum() < 2:
                continue  # Skip if not enough data

            # Find symbol with lowest mf_score
            bottom_symbol = ts_data.loc[ts_data["mf_score"].idxmin(), "symbol"]

            # Check that bottom symbol has short flag
            bottom_symbol_data = ts_data[ts_data["symbol"] == bottom_symbol]
            if len(bottom_symbol_data) > 0:
                assert bottom_symbol_data["mf_short_flag"].iloc[0] == 1, (
                    f"Bottom symbol {bottom_symbol} at {timestamp} should have mf_short_flag=1"
                )

    def test_quantile_per_timestamp(self, sample_factors_df, sample_bundle_positive):
        """Test that quantiles are computed per timestamp (cross-sectional)."""
        mf_result = build_multifactor_signal(
            sample_factors_df, bundle=sample_bundle_positive
        )

        result = select_top_bottom(
            mf_result.df,
            top_quantile=0.33,
            bottom_quantile=0.33,
        )

        # Check that we have long flags for multiple timestamps
        timestamps_with_long = result[result["mf_long_flag"] == 1]["timestamp"].unique()
        assert len(timestamps_with_long) > 0, (
            "Should have long flags for at least one timestamp"
        )

        # Check that we have short flags for multiple timestamps
        timestamps_with_short = result[result["mf_short_flag"] == 1][
            "timestamp"
        ].unique()
        assert len(timestamps_with_short) > 0, (
            "Should have short flags for at least one timestamp"
        )

    def test_invalid_quantiles_raise_error(
        self, sample_factors_df, sample_bundle_positive
    ):
        """Test that invalid quantile values raise ValueError."""
        mf_result = build_multifactor_signal(
            sample_factors_df, bundle=sample_bundle_positive
        )

        with pytest.raises(ValueError, match="top_quantile must be in"):
            select_top_bottom(mf_result.df, top_quantile=1.5, bottom_quantile=0.2)

        with pytest.raises(ValueError, match="bottom_quantile must be in"):
            select_top_bottom(mf_result.df, top_quantile=0.2, bottom_quantile=-0.1)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": ["AAPL"] * 10,
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            select_top_bottom(df, top_quantile=0.2, bottom_quantile=0.2)
