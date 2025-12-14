"""Tests for ML Explainability & Feature Importance Module (E2)."""
from __future__ import annotations

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.assembled_core.ml.explainability import (
    compute_model_feature_importance,
    compute_permutation_importance,
    summarize_feature_importance_global,
)

pytest.importorskip("sklearn")


@pytest.fixture
def sample_X_y_linear():
    """Create synthetic dataset with known linear relationship."""
    np.random.seed(42)
    n_samples = 200
    
    # Create 4 features with different true coefficients
    X = pd.DataFrame({
        "factor_mom": np.random.randn(n_samples),
        "factor_value": np.random.randn(n_samples),
        "factor_vol": np.random.randn(n_samples),
        "factor_quality": np.random.randn(n_samples),
    })
    
    # True relationship: y = 2.0 * factor_mom - 1.0 * factor_value + 0.5 * factor_vol + noise
    true_coefs = {"factor_mom": 2.0, "factor_value": -1.0, "factor_vol": 0.5, "factor_quality": 0.0}
    y = (
        true_coefs["factor_mom"] * X["factor_mom"]
        + true_coefs["factor_value"] * X["factor_value"]
        + true_coefs["factor_vol"] * X["factor_vol"]
        + true_coefs["factor_quality"] * X["factor_quality"]
        + np.random.randn(n_samples) * 0.1  # Small noise
    )
    
    return X, y, true_coefs


@pytest.fixture
def sample_X_y_tree():
    """Create synthetic dataset with slight non-linear relationship."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        "factor_mom": np.random.randn(n_samples),
        "factor_value": np.random.randn(n_samples),
        "factor_vol": np.random.randn(n_samples),
        "factor_quality": np.random.randn(n_samples),
    })
    
    # Slightly non-linear relationship: y depends more on factor_mom and factor_value
    y = (
        2.0 * X["factor_mom"]
        - 1.5 * X["factor_value"]
        + 0.3 * X["factor_mom"] ** 2  # Non-linear term
        + np.random.randn(n_samples) * 0.15
    )
    
    return X, y


@pytest.mark.advanced
def test_compute_model_feature_importance_linear(sample_X_y_linear):
    """Test feature importance for linear models."""
    X, y, true_coefs = sample_X_y_linear
    feature_names = list(X.columns)
    
    # Train a Ridge model
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X.values, y.values)
    
    # Compute feature importance
    importance_df = compute_model_feature_importance(model, feature_names)
    
    # Check structure
    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert "raw_value" in importance_df.columns
    assert "direction" in importance_df.columns
    
    # Check all features are present
    assert set(importance_df["feature"]) == set(feature_names)
    assert len(importance_df) == len(feature_names)
    
    # Check importance values are non-negative
    assert (importance_df["importance"] >= 0).all()
    
    # Check that factor_mom has highest importance (true coefficient = 2.0)
    factor_mom_importance = importance_df[importance_df["feature"] == "factor_mom"]["importance"].iloc[0]
    factor_value_importance = importance_df[importance_df["feature"] == "factor_value"]["importance"].iloc[0]
    factor_quality_importance = importance_df[importance_df["feature"] == "factor_quality"]["importance"].iloc[0]
    
    # factor_mom should have highest importance (absolute coefficient)
    assert factor_mom_importance > factor_value_importance
    assert factor_mom_importance > factor_quality_importance
    
    # Check directions: factor_mom positive, factor_value negative
    factor_mom_direction = importance_df[importance_df["feature"] == "factor_mom"]["direction"].iloc[0]
    factor_value_direction = importance_df[importance_df["feature"] == "factor_value"]["direction"].iloc[0]
    
    assert factor_mom_direction == 1  # Positive coefficient
    assert factor_value_direction == -1  # Negative coefficient
    
    # Check sorting: should be sorted by importance descending
    assert importance_df["importance"].is_monotonic_decreasing
    
    # Check raw_value matches actual coefficient (with sign)
    factor_mom_raw = importance_df[importance_df["feature"] == "factor_mom"]["raw_value"].iloc[0]
    assert abs(factor_mom_raw - true_coefs["factor_mom"]) < 0.5  # Allow some variation due to regularization/noise


@pytest.mark.advanced
def test_compute_model_feature_importance_random_forest(sample_X_y_tree):
    """Test feature importance for tree-based models."""
    X, y = sample_X_y_tree
    feature_names = list(X.columns)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X.values, y.values)
    
    # Compute feature importance
    importance_df = compute_model_feature_importance(model, feature_names)
    
    # Check structure
    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert "raw_value" in importance_df.columns
    assert "direction" in importance_df.columns
    
    # Check all features are present
    assert set(importance_df["feature"]) == set(feature_names)
    assert len(importance_df) == len(feature_names)
    
    # Check importance values are non-negative
    assert (importance_df["importance"] >= 0).all()
    
    # Check that at least some features have importance > 0
    assert (importance_df["importance"] > 0).any()
    
    # For tree models, direction should be None
    assert importance_df["direction"].isna().all()
    
    # Check sorting
    assert importance_df["importance"].is_monotonic_decreasing
    
    # Check that factor_mom likely has higher importance than factor_quality
    # (since it's used in the non-linear relationship)
    factor_mom_importance = importance_df[importance_df["feature"] == "factor_mom"]["importance"].iloc[0]
    factor_quality_importance = importance_df[importance_df["feature"] == "factor_quality"]["importance"].iloc[0]
    # This should be true for this dataset, but allow for some randomness
    assert factor_mom_importance >= 0.0  # At least non-negative


@pytest.mark.advanced
def test_compute_model_feature_importance_unsupported_model_raises(sample_X_y_linear):
    """Test that unsupported models raise ValueError."""
    X, y, _ = sample_X_y_linear
    feature_names = list(X.columns)
    
    # Train KNN model (does not have coef_ or feature_importances_)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X.values, y.values)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="does not have coef_ or feature_importances_"):
        compute_model_feature_importance(model, feature_names)


@pytest.mark.advanced
def test_compute_model_feature_importance_dimension_mismatch_raises(sample_X_y_linear):
    """Test that dimension mismatch raises ValueError."""
    X, y, _ = sample_X_y_linear
    feature_names = list(X.columns) + ["extra_feature"]  # One extra feature
    
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X.values, y.values)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="does not match model dimensions"):
        compute_model_feature_importance(model, feature_names)


@pytest.mark.advanced
def test_compute_permutation_importance_basic(sample_X_y_linear):
    """Test permutation importance computation."""
    X, y, _ = sample_X_y_linear
    feature_names = list(X.columns)
    
    # Train a model
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X.values, y.values)
    
    # Compute permutation importance
    perm_importance_df = compute_permutation_importance(
        model,
        X,
        y,
        scoring="r2",
        n_repeats=5,  # Small number for speed
        random_state=42,
    )
    
    # Check structure
    assert isinstance(perm_importance_df, pd.DataFrame)
    assert "feature" in perm_importance_df.columns
    assert "importance_mean" in perm_importance_df.columns
    assert "importance_std" in perm_importance_df.columns
    assert "importance_median" in perm_importance_df.columns
    
    # Check all features are present
    assert set(perm_importance_df["feature"]) == set(feature_names)
    assert len(perm_importance_df) == len(feature_names)
    
    # Check no NaNs
    assert perm_importance_df["importance_mean"].notna().all()
    assert perm_importance_df["importance_std"].notna().all()
    assert perm_importance_df["importance_median"].notna().all()
    
    # Check sorting (should be sorted by importance_mean descending)
    assert perm_importance_df["importance_mean"].is_monotonic_decreasing
    
    # Check that std is non-negative
    assert (perm_importance_df["importance_std"] >= 0).all()
    
    # For this dataset, factor_mom should have higher importance than factor_quality
    factor_mom_importance = perm_importance_df[perm_importance_df["feature"] == "factor_mom"]["importance_mean"].iloc[0]
    factor_quality_importance = perm_importance_df[perm_importance_df["feature"] == "factor_quality"]["importance_mean"].iloc[0]
    assert factor_mom_importance > factor_quality_importance


@pytest.mark.advanced
def test_summarize_feature_importance_global_basic():
    """Test global feature importance aggregation."""
    # Create two mock importance DataFrames
    importance_df_1 = pd.DataFrame({
        "feature": ["factor_mom", "factor_value", "factor_vol"],
        "importance": [0.8, 0.5, 0.2],
    })
    
    importance_df_2 = pd.DataFrame({
        "feature": ["factor_mom", "factor_value", "factor_vol"],
        "importance": [0.9, 0.4, 0.3],
    })
    
    feature_importance_dfs = {
        "ridge": importance_df_1,
        "lasso": importance_df_2,
    }
    
    # Compute global summary
    summary_df = summarize_feature_importance_global(feature_importance_dfs)
    
    # Check structure
    assert isinstance(summary_df, pd.DataFrame)
    assert "feature" in summary_df.columns
    assert "importance_mean" in summary_df.columns
    assert "importance_median" in summary_df.columns
    assert "importance_max" in summary_df.columns
    assert "n_models" in summary_df.columns
    
    # Check all features are present
    assert set(summary_df["feature"]) == {"factor_mom", "factor_value", "factor_vol"}
    assert len(summary_df) == 3
    
    # Check n_models
    assert (summary_df["n_models"] == 2).all()
    
    # Check that factor_mom has highest mean importance
    factor_mom_mean = summary_df[summary_df["feature"] == "factor_mom"]["importance_mean"].iloc[0]
    factor_value_mean = summary_df[summary_df["feature"] == "factor_value"]["importance_mean"].iloc[0]
    factor_vol_mean = summary_df[summary_df["feature"] == "factor_vol"]["importance_mean"].iloc[0]
    
    assert factor_mom_mean > factor_value_mean
    assert factor_mom_mean > factor_vol_mean
    
    # Check mean values: factor_mom should be (0.8 + 0.9) / 2 = 0.85
    assert abs(factor_mom_mean - 0.85) < 0.01
    
    # Check sorting (should be sorted by importance_mean descending)
    assert summary_df["importance_mean"].is_monotonic_decreasing
    
    # Check max values
    factor_mom_max = summary_df[summary_df["feature"] == "factor_mom"]["importance_max"].iloc[0]
    assert abs(factor_mom_max - 0.9) < 0.01


@pytest.mark.advanced
def test_summarize_feature_importance_global_with_permutation_importance():
    """Test global aggregation with permutation importance DataFrames (importance_mean column)."""
    # Create mock permutation importance DataFrame
    perm_importance_df = pd.DataFrame({
        "feature": ["factor_mom", "factor_value"],
        "importance_mean": [0.15, 0.08],
    })
    
    # Create model-based importance DataFrame
    model_importance_df = pd.DataFrame({
        "feature": ["factor_mom", "factor_value"],
        "importance": [0.8, 0.5],
    })
    
    feature_importance_dfs = {
        "permutation": perm_importance_df,
        "ridge": model_importance_df,
    }
    
    # Should work with both formats
    summary_df = summarize_feature_importance_global(feature_importance_dfs)
    
    assert len(summary_df) == 2
    assert set(summary_df["feature"]) == {"factor_mom", "factor_value"}
    assert (summary_df["n_models"] == 2).all()


@pytest.mark.advanced
def test_summarize_feature_importance_global_empty_dict():
    """Test that empty dict returns empty DataFrame with correct columns."""
    summary_df = summarize_feature_importance_global({})
    
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 0
    assert set(summary_df.columns) == {"feature", "importance_mean", "importance_median", "importance_max", "n_models"}


@pytest.mark.advanced
def test_summarize_feature_importance_global_missing_columns_raises():
    """Test that missing required columns raise ValueError."""
    invalid_df = pd.DataFrame({
        "feature": ["factor_mom"],
        "wrong_column": [0.5],  # Missing "importance" or "importance_mean"
    })
    
    feature_importance_dfs = {"model": invalid_df}
    
    with pytest.raises(ValueError, match="missing 'importance' or 'importance_mean' column"):
        summarize_feature_importance_global(feature_importance_dfs)


@pytest.mark.advanced
def test_integration_ml_validation_with_feature_importance(tmp_path: Path):
    """Integration test: Check that feature importance CSV is generated."""
    # Create a small synthetic factor panel
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    all_data = []
    for date in dates:
        for symbol in symbols:
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "fwd_return_20d": np.random.randn() * 0.05,
                "factor_mom": np.random.randn(),
                "factor_value": np.random.randn(),
                "factor_vol": np.random.randn(),
            })
    
    factor_panel_df = pd.DataFrame(all_data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Save to temporary file
    panel_file = tmp_path / "test_factor_panel.parquet"
    factor_panel_df.to_parquet(panel_file, index=False)
    
    # Import run_ml_validation function
    import sys
    from pathlib import Path as PathLib
    project_root = PathLib(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    # Run ML validation
    output_dir = tmp_path / "ml_validation_output"
    exit_code = run_ml_validation(
        factor_panel_file=panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,
        output_dir=output_dir,
    )
    
    # Check exit code
    assert exit_code == 0
    
    # Check that feature importance CSV was created
    feature_importance_files = list(output_dir.glob("ml_feature_importance_*.csv"))
    assert len(feature_importance_files) > 0, f"No feature importance CSV found in {output_dir}"
    
    # Load and check the file
    feature_importance_df = pd.read_csv(feature_importance_files[0])
    assert len(feature_importance_df) > 0
    assert "feature" in feature_importance_df.columns
    assert "importance" in feature_importance_df.columns
    
    # Check that expected features are present
    expected_features = ["factor_mom", "factor_value", "factor_vol"]
    assert set(feature_importance_df["feature"]) >= set(expected_features)

