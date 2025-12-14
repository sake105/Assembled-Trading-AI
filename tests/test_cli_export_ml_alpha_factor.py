"""Tests for ML Alpha Factor Export Script (E3).

This module tests the export_ml_alpha_factor functionality for generating
ML alpha factors from factor panels and merging them back into panels.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.ml.export_ml_alpha_factor import export_ml_alpha_factor

pytest.importorskip("sklearn")

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_factor_panel_file(tmp_path: Path) -> Path:
    """
    Creates a synthetic factor panel DataFrame and saves it as Parquet.
    
    - 5 symbols, 80 days
    - 3 factor columns with a clear linear relationship to fwd_return_20d + noise
    - fwd_return_20d as label
    
    Returns:
        Path to the saved Parquet file
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=80, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    all_data = []
    for date in dates:
        for symbol in symbols:
            # Generate factors
            factor_mom = np.random.normal(0, 0.5)
            factor_value = np.random.normal(0, 0.5)
            factor_quality = np.random.normal(0, 0.5)

            # Create a clear linear relationship for fwd_return_20d
            fwd_return_20d = (
                0.3 * factor_mom
                - 0.2 * factor_value
                + 0.1 * factor_quality
                + np.random.normal(0, 0.05)  # Noise
            )

            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "fwd_return_20d": fwd_return_20d,
                "factor_mom": factor_mom,
                "factor_value": factor_value,
                "factor_quality": factor_quality,
                "returns_12m": np.random.normal(0, 0.1),  # Additional factor-like column
                "trend_strength_50": np.random.normal(0, 1),
            })

    df = pd.DataFrame(all_data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    panel_file = tmp_path / "test_factor_panel.parquet"
    df.to_parquet(panel_file, index=False)
    return panel_file


@pytest.mark.advanced
def test_export_ml_alpha_factor_basic(sample_factor_panel_file, tmp_path: Path):
    """Test basic ML alpha factor export functionality."""
    output_dir = tmp_path / "ml_alpha_output"
    
    # Run export
    output_path = export_ml_alpha_factor(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,  # Small number for speed
        output_dir=output_dir,
    )
    
    # Check output file exists
    assert output_path.exists(), f"Output file should exist: {output_path}"
    
    # Load output panel
    ml_alpha_panel_df = pd.read_parquet(output_path)
    
    # Check structure
    assert isinstance(ml_alpha_panel_df, pd.DataFrame)
    assert len(ml_alpha_panel_df) > 0
    
    # Check required columns are present
    assert "timestamp" in ml_alpha_panel_df.columns
    assert "symbol" in ml_alpha_panel_df.columns
    assert "fwd_return_20d" in ml_alpha_panel_df.columns
    
    # Check ML alpha column exists
    ml_alpha_cols = [col for col in ml_alpha_panel_df.columns if col.startswith("ml_alpha_")]
    assert len(ml_alpha_cols) > 0, "ML alpha column should be present"
    
    ml_alpha_col = ml_alpha_cols[0]
    assert ml_alpha_col == "ml_alpha_ridge_20d", f"Expected ml_alpha_ridge_20d, got {ml_alpha_col}"
    
    # Check that some rows have predictions (test samples from CV)
    n_with_predictions = ml_alpha_panel_df[ml_alpha_col].notna().sum()
    assert n_with_predictions > 0, "At least some rows should have ML alpha predictions"
    
    # Check correlation between ML alpha and forward returns (only for non-NaN rows)
    ml_alpha_valid = ml_alpha_panel_df[[ml_alpha_col, "fwd_return_20d"]].dropna()
    if len(ml_alpha_valid) > 10:  # Need enough samples for correlation
        correlation = ml_alpha_valid[ml_alpha_col].corr(ml_alpha_valid["fwd_return_20d"])
        # Correlation should be positive (model should learn the relationship)
        assert correlation > 0.3, f"ML alpha should correlate with forward returns, got {correlation:.4f}"
    
    # Check that original factor columns are preserved
    assert "factor_mom" in ml_alpha_panel_df.columns
    assert "factor_value" in ml_alpha_panel_df.columns
    assert "factor_quality" in ml_alpha_panel_df.columns


@pytest.mark.advanced
def test_export_ml_alpha_factor_custom_column_name(sample_factor_panel_file, tmp_path: Path):
    """Test ML alpha export with custom column name."""
    output_dir = tmp_path / "ml_alpha_output"
    custom_column_name = "my_custom_ml_alpha"
    
    # Run export with custom column name
    output_path = export_ml_alpha_factor(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,
        output_dir=output_dir,
        column_name=custom_column_name,
    )
    
    # Load output panel
    ml_alpha_panel_df = pd.read_parquet(output_path)
    
    # Check custom column name is used
    assert custom_column_name in ml_alpha_panel_df.columns, "Custom column name should be present"
    assert ml_alpha_panel_df[custom_column_name].notna().sum() > 0, "Custom column should have predictions"


@pytest.mark.advanced
def test_export_ml_alpha_factor_random_forest(sample_factor_panel_file, tmp_path: Path):
    """Test ML alpha export with Random Forest model."""
    output_dir = tmp_path / "ml_alpha_output"
    
    # Run export with Random Forest
    output_path = export_ml_alpha_factor(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="random_forest",
        model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        n_splits=3,
        output_dir=output_dir,
    )
    
    # Check output file exists
    assert output_path.exists()
    
    # Load output panel
    ml_alpha_panel_df = pd.read_parquet(output_path)
    
    # Check ML alpha column exists (should be ml_alpha_random_forest_20d)
    ml_alpha_cols = [col for col in ml_alpha_panel_df.columns if col.startswith("ml_alpha_")]
    assert len(ml_alpha_cols) > 0
    assert "ml_alpha_random_forest_20d" in ml_alpha_cols


@pytest.mark.advanced
def test_export_ml_alpha_factor_cli_subcommand(sample_factor_panel_file, tmp_path: Path):
    """Test ML alpha export via CLI subprocess."""
    output_dir = tmp_path / "ml_alpha_output"
    
    # Run via subprocess
    script_path = ROOT / "research" / "ml" / "export_ml_alpha_factor.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--factor-panel-file", str(sample_factor_panel_file),
        "--label-col", "fwd_return_20d",
        "--model-type", "ridge",
        "--model-param", "alpha=0.1",
        "--n-splits", "3",
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    
    # Check exit code
    assert result.returncode == 0, f"CLI command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    
    # Check output file was created
    output_files = list(output_dir.glob("ml_alpha_panel_*.parquet"))
    assert len(output_files) > 0, "Output Parquet file should be created"
    
    # Check predictions file was created
    predictions_files = list(output_dir.glob("ml_alpha_predictions_*.parquet"))
    assert len(predictions_files) > 0, "Predictions Parquet file should be created"
    
    # Check metadata file was created
    metadata_files = list(output_dir.glob("ml_alpha_metadata_*.json"))
    assert len(metadata_files) > 0, "Metadata JSON file should be created"


@pytest.mark.advanced
def test_export_ml_alpha_factor_preserves_original_columns(sample_factor_panel_file, tmp_path: Path):
    """Test that original factor columns are preserved in output."""
    output_dir = tmp_path / "ml_alpha_output"
    
    # Load original panel to get column list
    original_df = pd.read_parquet(sample_factor_panel_file)
    original_columns = set(original_df.columns)
    
    # Run export
    output_path = export_ml_alpha_factor(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,
        output_dir=output_dir,
    )
    
    # Load output panel
    ml_alpha_panel_df = pd.read_parquet(output_path)
    
    # Check all original columns are present (plus ML alpha column)
    for col in original_columns:
        assert col in ml_alpha_panel_df.columns, f"Original column {col} should be preserved"
    
    # Check ML alpha column was added
    ml_alpha_cols = [col for col in ml_alpha_panel_df.columns if col.startswith("ml_alpha_")]
    assert len(ml_alpha_cols) > 0, "ML alpha column should be added"


@pytest.mark.advanced
def test_export_ml_alpha_factor_only_test_samples_have_predictions(sample_factor_panel_file, tmp_path: Path):
    """Test that only test samples (OOS) have predictions, training samples remain NaN."""
    output_dir = tmp_path / "ml_alpha_output"
    
    # Run export
    output_path = export_ml_alpha_factor(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,
        output_dir=output_dir,
    )
    
    # Load output panel
    ml_alpha_panel_df = pd.read_parquet(output_path)
    
    # Find ML alpha column
    ml_alpha_cols = [col for col in ml_alpha_panel_df.columns if col.startswith("ml_alpha_")]
    assert len(ml_alpha_cols) > 0
    ml_alpha_col = ml_alpha_cols[0]
    
    # Check that some rows have NaN (training samples)
    n_with_nan = ml_alpha_panel_df[ml_alpha_col].isna().sum()
    assert n_with_nan > 0, "Some rows should have NaN (training samples don't have predictions)"
    
    # Check that some rows have predictions (test samples)
    n_with_predictions = ml_alpha_panel_df[ml_alpha_col].notna().sum()
    assert n_with_predictions > 0, "Some rows should have predictions (test samples)"
    
    # Total should match
    assert n_with_nan + n_with_predictions == len(ml_alpha_panel_df), "All rows should be either NaN or have predictions"

