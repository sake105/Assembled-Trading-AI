"""Tests for ML Factor Models Module (Phase E1).

Tests the factor_models functions:
- prepare_ml_dataset()
- run_time_series_cv()
- evaluate_ml_predictions()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    evaluate_ml_predictions,
    prepare_ml_dataset,
    run_time_series_cv,
)


@pytest.fixture
def sample_factor_panel_df() -> pd.DataFrame:
    """Create sample factor panel with constructed linear relationship to forward returns."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    all_data = []
    for date in dates:
        for symbol in symbols:
            # Create factors with some structure
            factor_mom = np.random.randn() * 0.5  # Momentum factor
            factor_value = np.random.randn() * 0.3  # Value factor
            factor_quality = np.random.randn() * 0.4  # Quality factor
            factor_vol = np.random.uniform(0.1, 0.3)  # Volatility factor (positive)
            
            # Construct forward return with simple linear relationship + noise
            # True relationship: fwd_return = 0.3 * mom + 0.2 * value + 0.1 * quality - 0.5 * vol + noise
            true_fwd_return = (
                0.3 * factor_mom +
                0.2 * factor_value +
                0.1 * factor_quality -
                0.5 * factor_vol +
                np.random.randn() * 0.1  # Noise
            )
            
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "factor_mom": factor_mom,
                "factor_value": factor_value,
                "factor_quality": factor_quality,
                "factor_vol": factor_vol,
                "fwd_return_20d": true_fwd_return,
            })
    
    return pd.DataFrame(all_data).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_prepare_ml_dataset_basic(sample_factor_panel_df):
    """Test that prepare_ml_dataset correctly extracts features and labels."""
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        feature_cols=None,  # Auto-detect
    )
    
    X, y = prepare_ml_dataset(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
    )
    
    # Check structure
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0
    
    # Check that features are detected (should find factor_* columns)
    assert len(X.columns) > 0
    assert all(col.startswith("factor_") for col in X.columns)
    
    # Check that label is extracted correctly
    assert len(y) > 0
    assert not y.isna().all()  # Should have some valid labels
    
    # Check that X and y are aligned (same index)
    assert X.index.equals(y.index)


def test_prepare_ml_dataset_with_explicit_features(sample_factor_panel_df):
    """Test prepare_ml_dataset with explicitly specified features."""
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        feature_cols=["factor_mom", "factor_value"],  # Explicit features
    )
    
    X, y = prepare_ml_dataset(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
    )
    
    # Check that only specified features are used
    assert set(X.columns) == {"factor_mom", "factor_value"}
    assert len(X) == len(y)


def test_prepare_ml_dataset_time_filtering(sample_factor_panel_df):
    """Test that time filtering works correctly."""
    test_start = pd.Timestamp("2020-06-01", tz="UTC")
    test_end = pd.Timestamp("2020-08-31", tz="UTC")
    
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        test_start=test_start,
        test_end=test_end,
    )
    
    X, y = prepare_ml_dataset(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
    )
    
    # Check that data is filtered (should be less than full dataset)
    assert len(X) < len(sample_factor_panel_df)
    
    # Verify timestamps are in range (if we can check via index alignment)
    # Note: We can't directly check timestamps from X/y, but we know filtering happened
    assert len(X) > 0


def test_prepare_ml_dataset_empty_df_raises():
    """Test that empty DataFrame raises appropriate error."""
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "factor_mom", "fwd_return_20d"])
    
    experiment = MLExperimentConfig(label_col="fwd_return_20d")
    
    with pytest.raises(ValueError, match="empty"):
        prepare_ml_dataset(empty_df, experiment)


def test_prepare_ml_dataset_missing_label_raises(sample_factor_panel_df):
    """Test that missing label column raises error."""
    experiment = MLExperimentConfig(label_col="nonexistent_label")
    
    with pytest.raises(ValueError, match="not found"):
        prepare_ml_dataset(sample_factor_panel_df, experiment)


def test_run_time_series_cv_linear_model(sample_factor_panel_df):
    """Test time-series CV with linear model."""
    pytest.importorskip("sklearn")
    
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        n_splits=3,
        min_train_samples=50,
    )
    
    model_cfg = MLModelConfig(
        name="linear_test",
        model_type="linear",
        params={},
    )
    
    result = run_time_series_cv(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
        model_cfg=model_cfg,
    )
    
    # Check structure
    assert "predictions_df" in result
    assert "metrics_df" in result
    assert "global_metrics" in result
    assert "per_split_metrics" in result
    
    # Check predictions
    predictions_df = result["predictions_df"]
    assert len(predictions_df) > 0
    assert "y_true" in predictions_df.columns
    assert "y_pred" in predictions_df.columns
    assert "split_index" in predictions_df.columns
    
    # Check metrics
    global_metrics = result["global_metrics"]
    assert "mse" in global_metrics
    assert "mae" in global_metrics
    assert "r2" in global_metrics
    assert "n_splits" in global_metrics
    
    # Check per-split metrics
    per_split = result["per_split_metrics"]
    assert len(per_split) <= experiment.n_splits  # Can be less if some splits failed
    
    # Check metrics DataFrame
    metrics_df = result["metrics_df"]
    if not metrics_df.empty:
        assert "mse" in metrics_df.columns
        assert "mae" in metrics_df.columns
        assert "r2" in metrics_df.columns


def test_run_time_series_cv_ridge_model(sample_factor_panel_df):
    """Test time-series CV with Ridge model."""
    pytest.importorskip("sklearn")
    
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        n_splits=3,
        min_train_samples=50,
    )
    
    model_cfg = MLModelConfig(
        name="ridge_test",
        model_type="ridge",
        params={"alpha": 0.1},
    )
    
    result = run_time_series_cv(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
        model_cfg=model_cfg,
    )
    
    assert len(result["predictions_df"]) > 0
    assert result["global_metrics"]["r2"] is not None


def test_run_time_series_cv_random_forest(sample_factor_panel_df):
    """Test time-series CV with Random Forest model."""
    pytest.importorskip("sklearn")
    
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        n_splits=3,
        min_train_samples=50,
        standardize=False,  # Random Forest doesn't need standardization
    )
    
    model_cfg = MLModelConfig(
        name="rf_test",
        model_type="random_forest",
        params={"n_estimators": 10, "max_depth": 5},  # Small model for speed
    )
    
    result = run_time_series_cv(
        factor_panel_df=sample_factor_panel_df,
        experiment=experiment,
        model_cfg=model_cfg,
    )
    
    assert len(result["predictions_df"]) > 0
    assert result["global_metrics"]["r2"] is not None


def test_run_time_series_cv_insufficient_data(sample_factor_panel_df):
    """Test that insufficient data raises appropriate error."""
    pytest.importorskip("sklearn")
    
    # Create very small dataset
    small_df = sample_factor_panel_df.head(20)
    
    experiment = MLExperimentConfig(
        label_col="fwd_return_20d",
        n_splits=10,  # Too many splits for small data
        min_train_samples=5,
    )
    
    model_cfg = MLModelConfig(name="linear_test", model_type="linear")
    
    with pytest.raises(ValueError, match="Insufficient data|No valid splits"):
        run_time_series_cv(small_df, experiment, model_cfg)


def test_evaluate_ml_predictions_high_correlation():
    """Test evaluate_ml_predictions with high correlation (good predictions)."""
    pytest.importorskip("sklearn")
    
    # Create predictions with high correlation
    np.random.seed(42)
    n_samples = 100
    
    # True returns
    y_true = np.random.randn(n_samples) * 0.02
    
    # Predictions: highly correlated with true (but not perfect)
    y_pred = y_true + np.random.randn(n_samples) * 0.005  # Small noise
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D", tz="UTC")
    symbols = ["AAPL"] * n_samples
    
    predictions_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbols,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    
    metrics = evaluate_ml_predictions(predictions_df, horizon_days=20)
    
    # Check that metrics are computed
    assert metrics["mse"] is not None
    assert metrics["mae"] is not None
    assert metrics["r2"] is not None
    assert metrics["n_samples"] == n_samples
    
    # With high correlation, R² should be high (> 0.8)
    assert metrics["r2"] > 0.8
    
    # IC should be positive and high
    if metrics["ic_mean"] is not None:
        assert metrics["ic_mean"] > 0.5
    
    # Directional accuracy should be high (> 60%)
    assert metrics["directional_accuracy"] > 0.6


def test_evaluate_ml_predictions_low_correlation():
    """Test evaluate_ml_predictions with low correlation (poor predictions)."""
    pytest.importorskip("sklearn")
    
    # Create predictions with low correlation (random noise)
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randn(n_samples) * 0.02
    y_pred = np.random.randn(n_samples) * 0.02  # Independent of y_true
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D", tz="UTC")
    symbols = ["AAPL"] * n_samples
    
    predictions_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbols,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    
    metrics = evaluate_ml_predictions(predictions_df, horizon_days=20)
    
    # With low correlation, R² should be low (possibly negative)
    assert metrics["r2"] < 0.5
    
    # IC should be close to zero
    if metrics["ic_mean"] is not None:
        assert abs(metrics["ic_mean"]) < 0.2
    
    # Directional accuracy should be around 50% (random)
    assert 0.4 < metrics["directional_accuracy"] < 0.6


def test_evaluate_ml_predictions_portfolio_metrics():
    """Test that portfolio metrics are computed when timestamp and symbol are available."""
    pytest.importorskip("sklearn")
    
    # Create predictions with clear signal for portfolio construction
    np.random.seed(42)
    n_timestamps = 20
    n_symbols = 10
    
    dates = pd.date_range("2020-01-01", periods=n_timestamps, freq="D", tz="UTC")
    symbols = [f"STOCK{i}" for i in range(n_symbols)]
    
    all_data = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            # Create predictions with signal: higher predictions → higher returns
            y_pred = i * 0.01  # Clear ranking
            y_true = y_pred + np.random.randn() * 0.005  # Small noise
            
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "y_true": y_true,
                "y_pred": y_pred,
            })
    
    predictions_df = pd.DataFrame(all_data)
    
    metrics = evaluate_ml_predictions(predictions_df, horizon_days=20)
    
    # Check portfolio metrics
    assert metrics["ls_return_mean"] is not None
    assert metrics["ls_sharpe"] is not None
    
    # With clear signal, Sharpe should be positive
    assert metrics["ls_sharpe"] > 0


def test_evaluate_ml_predictions_no_timestamp():
    """Test that evaluate_ml_predictions works without timestamp (no IC/portfolio metrics)."""
    # Predictions without timestamp
    np.random.seed(42)
    y_true = np.random.randn(50) * 0.02
    y_pred = y_true + np.random.randn(50) * 0.005
    
    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    
    metrics = evaluate_ml_predictions(predictions_df, horizon_days=20)
    
    # Classical metrics should still work
    assert metrics["mse"] is not None
    assert metrics["r2"] is not None
    
    # IC and portfolio metrics should be None (need timestamp/symbol for cross-sectional analysis)
    assert metrics["ic_mean"] is None
    assert metrics["ls_sharpe"] is None


def test_evaluate_ml_predictions_empty():
    """Test that empty predictions DataFrame returns None values."""
    empty_df = pd.DataFrame(columns=["y_true", "y_pred"])
    
    metrics = evaluate_ml_predictions(empty_df, horizon_days=20)
    
    assert metrics["n_samples"] == 0
    assert metrics["mse"] is None
    assert metrics["r2"] is None


def test_ml_model_config_validation():
    """Test that MLModelConfig validates model_type."""
    # Valid model types
    for model_type in ["linear", "ridge", "lasso", "random_forest"]:
        cfg = MLModelConfig(name="test", model_type=model_type)  # type: ignore
        assert cfg.model_type == model_type
    
    # Invalid model type
    with pytest.raises(ValueError, match="Unsupported model_type"):
        MLModelConfig(name="test", model_type="invalid_type")  # type: ignore


def test_ml_experiment_config_validation():
    """Test that MLExperimentConfig accepts valid parameters."""
    # Valid config
    cfg = MLExperimentConfig(label_col="fwd_return_20d", n_splits=5)
    assert cfg.n_splits == 5
    assert cfg.min_train_samples >= 100  # Default should be reasonable


def test_run_time_series_cv_no_sklearn_raises(sample_factor_panel_df):
    """Test that missing sklearn raises ImportError with helpful message."""
    # This test will only run if sklearn is actually missing
    # In most cases, sklearn will be available, so this test might be skipped
    
    experiment = MLExperimentConfig(label_col="fwd_return_20d", n_splits=2)
    model_cfg = MLModelConfig(name="test", model_type="linear")
    
    # Try to import sklearn to see if it's available
    try:
        import sklearn  # noqa: F401
        sklearn_available = True
    except ImportError:
        sklearn_available = False
    
    if not sklearn_available:
        with pytest.raises(ImportError, match="scikit-learn"):
            run_time_series_cv(sample_factor_panel_df, experiment, model_cfg)
    else:
        # If sklearn is available, test runs normally
        pytest.skip("sklearn is available, cannot test ImportError")

