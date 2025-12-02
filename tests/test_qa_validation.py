"""Tests for qa.validation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase9

from src.assembled_core.qa.validation import (
    ModelValidationResult,
    run_full_model_validation,
    validate_data_quality,
    validate_overfitting,
    validate_performance,
)


@pytest.fixture
def good_metrics() -> dict[str, float | None]:
    """Metrics that should pass validation."""
    return {
        "sharpe_ratio": 1.5,
        "max_drawdown_pct": -15.0,  # -15% drawdown
        "total_trades": 100,
        "cagr": 0.20,
        "start_capital": 10000.0,
    }


@pytest.fixture
def bad_metrics() -> dict[str, float | None]:
    """Metrics that should fail validation."""
    return {
        "sharpe_ratio": 0.3,  # Too low
        "max_drawdown_pct": -30.0,  # Too high drawdown
        "total_trades": 10,  # Too few trades
        "cagr": -0.05,
        "start_capital": 10000.0,
    }


@pytest.fixture
def partial_metrics() -> dict[str, float | None]:
    """Metrics with some missing keys (should generate warnings)."""
    return {
        "sharpe_ratio": 1.5,
        # max_drawdown_pct missing
        "total_trades": 100,
    }


@pytest.fixture
def clean_feature_df() -> pd.DataFrame:
    """Feature DataFrame with no missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "ta_sma_20": np.random.randn(100),
        "insider_net_buy_20d": np.random.randn(100),
    })


@pytest.fixture
def dirty_feature_df() -> pd.DataFrame:
    """Feature DataFrame with missing values above threshold."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    })
    # Make feature2 have > 5% missing values
    df.loc[0:10, "feature2"] = np.nan  # 11/100 = 11% missing
    return df


class TestValidatePerformance:
    """Tests for validate_performance function."""

    def test_validate_performance_ok(self, good_metrics):
        """Test that good metrics pass validation."""
        result = validate_performance(good_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        assert result.is_ok is True
        assert len(result.errors) == 0
        assert result.model_name == "performance_validation"
        assert "sharpe_checked" in result.metadata
        assert result.metadata["sharpe_ratio"] == 1.5

    def test_validate_performance_bad_sharpe(self, bad_metrics):
        """Test that low Sharpe ratio causes validation failure."""
        result = validate_performance(bad_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("Sharpe ratio" in err for err in result.errors)
        assert any("0.3" in err for err in result.errors)

    def test_validate_performance_bad_drawdown(self, bad_metrics):
        """Test that high max drawdown causes validation failure."""
        result = validate_performance(bad_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("drawdown" in err.lower() for err in result.errors)
        assert any("-30.0" in err or "30%" in err for err in result.errors)

    def test_validate_performance_bad_trade_count(self, bad_metrics):
        """Test that low trade count causes validation failure."""
        result = validate_performance(bad_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("trades" in err.lower() for err in result.errors)
        assert any("10" in err for err in result.errors)

    def test_validate_performance_missing_sharpe(self, partial_metrics):
        """Test that missing Sharpe ratio generates warning but no hard failure."""
        # Note: If sharpe is None, we still check other metrics
        # But if sharpe exists and is good, and drawdown is missing, we get a warning
        result = validate_performance(partial_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        # Sharpe exists and is good, but drawdown is missing
        assert len(result.warnings) > 0
        assert any("drawdown" in warn.lower() for warn in result.warnings)

    def test_validate_performance_missing_all(self):
        """Test with minimal metrics (should have warnings but no errors if thresholds are lenient)."""
        minimal_metrics = {"sharpe_ratio": None, "max_drawdown_pct": None, "total_trades": None}
        result = validate_performance(minimal_metrics, min_sharpe=1.0, max_drawdown=0.20, min_trades=30)
        
        # No errors (because missing values are warnings, not errors when thresholds aren't violated)
        assert result.is_ok is True  # No hard errors
        assert len(result.warnings) > 0
        assert all("not available" in warn.lower() or "missing" in warn.lower() for warn in result.warnings)


class TestValidateOverfitting:
    """Tests for validate_overfitting function."""

    def test_validate_overfitting_ok(self):
        """Test that good deflated Sharpe passes validation."""
        result = validate_overfitting(deflated_sharpe=0.8, threshold=0.5)
        
        assert result.is_ok is True
        assert len(result.errors) == 0
        assert result.model_name == "overfitting_validation"

    def test_validate_overfitting_fail(self):
        """Test that low deflated Sharpe causes validation failure."""
        result = validate_overfitting(deflated_sharpe=0.3, threshold=0.5)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("Sharpe" in err or "overfitting" in err.lower() for err in result.errors)
        assert any("0.3" in err for err in result.errors)

    def test_validate_overfitting_none(self):
        """Test that None deflated Sharpe generates warning but no hard failure."""
        result = validate_overfitting(deflated_sharpe=None, threshold=0.5)
        
        assert result.is_ok is True  # Warning only, not a hard failure
        assert len(result.warnings) > 0
        assert any("not available" in warn.lower() or "cannot validate" in warn.lower() for warn in result.warnings)
        assert len(result.errors) == 0

    def test_validate_overfitting_at_threshold(self):
        """Test behavior exactly at threshold."""
        result = validate_overfitting(deflated_sharpe=0.5, threshold=0.5)
        
        # At threshold should pass (>= threshold)
        assert result.is_ok is True
        assert len(result.errors) == 0


class TestValidateDataQuality:
    """Tests for validate_data_quality function."""

    def test_validate_data_quality_ok(self, clean_feature_df):
        """Test that clean DataFrame passes validation."""
        result = validate_data_quality(clean_feature_df, max_missing_fraction=0.05)
        
        assert result.is_ok is True
        assert len(result.errors) == 0
        assert result.model_name == "data_quality_validation"
        assert result.metadata["total_rows"] == 100
        assert result.metadata["total_columns"] == 5

    def test_validate_data_quality_fail(self, dirty_feature_df):
        """Test that DataFrame with too many missing values fails validation."""
        result = validate_data_quality(dirty_feature_df, max_missing_fraction=0.05)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("feature2" in err for err in result.errors)
        assert any("missing" in err.lower() for err in result.errors)

    def test_validate_data_quality_empty(self):
        """Test that empty DataFrame fails validation."""
        empty_df = pd.DataFrame()
        result = validate_data_quality(empty_df, max_missing_fraction=0.05)
        
        assert result.is_ok is False
        assert len(result.errors) > 0
        assert any("empty" in err.lower() for err in result.errors)

    def test_validate_data_quality_warning(self):
        """Test that small amount of missing values generates warning but passes."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        # Add 2% missing values (below threshold, should pass but warn)
        df.loc[0:1, "feature2"] = np.nan  # 2/100 = 2% missing
        
        result = validate_data_quality(df, max_missing_fraction=0.05)
        
        assert result.is_ok is True  # Below threshold, passes
        assert len(result.warnings) > 0  # But generates warning
        assert any("feature2" in warn for warn in result.warnings)


class TestRunFullModelValidation:
    """Tests for run_full_model_validation function."""

    def test_run_full_model_validation_all_ok(self, good_metrics, clean_feature_df):
        """Test full validation with all checks passing."""
        result = run_full_model_validation(
            model_name="trend_baseline",
            metrics=good_metrics,
            feature_df=clean_feature_df,
            deflated_sharpe=0.8
        )
        
        assert result.model_name == "trend_baseline"
        assert result.is_ok is True
        assert len(result.errors) == 0
        assert "performance" in result.metadata["validation_details"]
        assert "overfitting" in result.metadata["validation_details"]
        assert "data_quality" in result.metadata["validation_details"]

    def test_run_full_model_validation_multiple_failures(self, bad_metrics, dirty_feature_df):
        """Test full validation with multiple validation failures."""
        result = run_full_model_validation(
            model_name="test_model",
            metrics=bad_metrics,
            feature_df=dirty_feature_df,
            deflated_sharpe=0.3  # Also fails overfitting check
        )
        
        assert result.model_name == "test_model"
        assert result.is_ok is False
        assert len(result.errors) > 1  # Multiple errors from different checks
        assert result.metadata["total_errors"] > 1

    def test_run_full_model_validation_no_feature_df(self, good_metrics):
        """Test full validation without feature DataFrame."""
        result = run_full_model_validation(
            model_name="trend_baseline",
            metrics=good_metrics,
            feature_df=None,
            deflated_sharpe=0.8
        )
        
        assert result.model_name == "trend_baseline"
        assert result.is_ok is True
        assert len(result.warnings) > 0
        assert any("not provided" in warn.lower() or "skipping" in warn.lower() for warn in result.warnings)
        assert result.metadata["validation_details"]["data_quality"]["is_ok"] is None

    def test_run_full_model_validation_custom_config(self, good_metrics, clean_feature_df):
        """Test full validation with custom configuration."""
        config = {
            "min_sharpe": 2.0,  # Higher threshold
            "max_drawdown": 0.10,  # Stricter drawdown limit
            "min_trades": 50,
            "overfitting_threshold": 0.7,
            "max_missing_fraction": 0.01,
        }
        
        # good_metrics has sharpe=1.5, which is below 2.0, so should fail
        result = run_full_model_validation(
            model_name="trend_baseline",
            metrics=good_metrics,
            feature_df=clean_feature_df,
            deflated_sharpe=0.8,
            config=config
        )
        
        assert result.model_name == "trend_baseline"
        assert result.is_ok is False  # Fails because sharpe 1.5 < 2.0
        assert len(result.errors) > 0
        assert any("Sharpe" in err for err in result.errors)

    def test_run_full_model_validation_aggregates_warnings(self, partial_metrics):
        """Test that full validation aggregates warnings from all checks."""
        result = run_full_model_validation(
            model_name="test_model",
            metrics=partial_metrics,  # Missing some metrics
            feature_df=None,  # Missing feature_df
            deflated_sharpe=None  # Missing deflated_sharpe
        )
        
        assert result.model_name == "test_model"
        # Multiple warnings from different checks
        assert len(result.warnings) > 1
        assert result.metadata["total_warnings"] > 1

