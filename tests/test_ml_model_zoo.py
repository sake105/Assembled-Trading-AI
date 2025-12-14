"""Tests for Model Zoo Factor Validation.

This module tests the model zoo functionality for comparing multiple ML models
on factor panels.
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

from research.ml.model_zoo_factor_validation import (
    build_default_model_zoo,
    run_model_zoo_for_panel,
    write_model_zoo_summary,
)


@pytest.fixture
def sample_factor_panel_tmpfile(tmp_path: Path) -> Path:
    """
    Creates a synthetic factor panel DataFrame and saves it as Parquet.
    
    - 4 symbols, 80 days
    - 4 factor columns with a linear relationship to fwd_return_20d + noise
    - fwd_return_20d as label
    
    Returns:
        Path to the saved Parquet file
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=80, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    all_data = []
    for symbol in symbols:
        # Generate factors
        factor_mom = np.random.normal(0, 0.5, len(dates))
        factor_value = np.random.normal(0, 0.5, len(dates))
        factor_quality = np.random.normal(0, 0.5, len(dates))
        factor_vol = np.random.normal(0, 0.5, len(dates))

        # Create a clear linear relationship for fwd_return_20d
        fwd_return_20d = (
            0.15 * factor_mom
            - 0.10 * factor_value
            + 0.08 * factor_quality
            - 0.05 * factor_vol
            + np.random.normal(0, 0.02, len(dates))  # Noise
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "fwd_return_20d": fwd_return_20d,
            "factor_mom": factor_mom,
            "factor_value": factor_value,
            "factor_quality": factor_quality,
            "factor_vol": factor_vol,
            "returns_1m": np.random.normal(0, 0.02, len(dates)),  # Example of other factor-like columns
            "trend_strength_20": np.random.normal(0, 1, len(dates)),
        })
        all_data.append(df)

    panel_df = pd.concat(all_data, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    # Save to Parquet
    output_file = tmp_path / "sample_factor_panel.parquet"
    panel_df.to_parquet(output_file, index=False)
    
    return output_file


@pytest.mark.advanced
def test_build_default_model_zoo():
    """Test that build_default_model_zoo returns a list of MLModelConfig instances."""
    pytest.importorskip("sklearn")
    
    models = build_default_model_zoo()
    
    assert isinstance(models, list)
    assert len(models) >= 3, "Should have at least 3 models (linear, ridge, random_forest)"
    
    # Check that all models have required attributes
    for model in models:
        assert hasattr(model, "name")
        assert hasattr(model, "model_type")
        assert hasattr(model, "params")
        assert model.model_type in ["linear", "ridge", "lasso", "random_forest"]
    
    # Check that we have at least one of each type (rough check)
    model_types = [m.model_type for m in models]
    assert "linear" in model_types
    assert "ridge" in model_types
    assert "random_forest" in model_types


@pytest.mark.advanced
def test_run_model_zoo_for_panel_basic(sample_factor_panel_tmpfile: Path, tmp_path: Path):
    """Test that run_model_zoo_for_panel creates summary CSV with valid metrics."""
    pytest.importorskip("sklearn")
    
    output_dir = tmp_path / "model_zoo_output"
    
    # Run model zoo
    summary_df = run_model_zoo_for_panel(
        factor_panel_path=sample_factor_panel_tmpfile,
        label_col="fwd_return_20d",
        output_dir=output_dir,
        experiment_cfg_kwargs={
            "n_splits": 3,  # Use fewer splits for faster tests
            "min_train_samples": 20,  # Lower threshold for small dataset
        },
    )
    
    # Check that summary DataFrame was returned
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) > 0, "Summary should contain at least one model"
    
    # Write summary to CSV (run_model_zoo_for_panel doesn't write CSV itself)
    write_model_zoo_summary(
        summary_df=summary_df,
        output_dir=output_dir,
        write_markdown=False,
    )
    
    # Check that summary CSV was created
    csv_path = output_dir / "ml_model_zoo_summary.csv"
    assert csv_path.exists(), f"Summary CSV should exist at {csv_path}"
    
    # Reload and verify CSV
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == len(summary_df)
    
    # Check that we have multiple models
    assert len(summary_df) >= 3, "Should have at least 3 models in zoo"
    
    # Check that successful models have valid metrics (not all NaN)
    if "test_r2_mean" in summary_df.columns:
        successful_models = summary_df[summary_df["test_r2_mean"].notna()]
    else:
        successful_models = summary_df
    
    if len(successful_models) > 0:
        # Check that test_r2_mean is numeric and reasonable
        assert successful_models["test_r2_mean"].dtype in [np.float64, np.float32], \
            "test_r2_mean should be numeric"
        
        # Check that ic_mean exists for successful models (may be None/NaN if insufficient data)
        # We just check that the column exists, not that it's populated
        assert "ic_mean" in successful_models.columns
        assert "ic_ir" in successful_models.columns
        
        # Check that model_name and model_type are present
        assert "model_name" in successful_models.columns
        assert "model_type" in successful_models.columns


@pytest.mark.advanced
def test_model_zoo_prefers_regularized_model(sample_factor_panel_tmpfile: Path, tmp_path: Path):
    """
    Test that regularized models (Ridge) perform reasonably well.
    
    This is a rough structural check - we don't enforce strict statistical
    requirements, just verify that Ridge models achieve a reasonable R².
    """
    pytest.importorskip("sklearn")
    
    output_dir = tmp_path / "model_zoo_output_regularized"
    
    # Run model zoo
    summary_df = run_model_zoo_for_panel(
        factor_panel_path=sample_factor_panel_tmpfile,
        label_col="fwd_return_20d",
        output_dir=output_dir,
        experiment_cfg_kwargs={
            "n_splits": 3,
            "min_train_samples": 20,
        },
    )
    
    # Filter successful Ridge models
    if "test_r2_mean" in summary_df.columns:
        ridge_models = summary_df[
            (summary_df["model_type"] == "ridge") &
            (summary_df["test_r2_mean"].notna())
        ]
    else:
        ridge_models = summary_df[summary_df["model_type"] == "ridge"]
    
    if len(ridge_models) > 0:
        # Check that at least one Ridge model has reasonable performance
        max_ridge_r2 = ridge_models["test_r2_mean"].max()
        
        # With our synthetic data (clear linear relationship), Ridge should achieve decent R²
        # We use a lower threshold (0.3) to account for small dataset and noise
        assert max_ridge_r2 > 0.3, \
            f"At least one Ridge model should achieve R² > 0.3, got {max_ridge_r2:.4f}"
    
    # Also check that at least one model overall achieved reasonable performance
    if "test_r2_mean" in summary_df.columns:
        successful_models = summary_df[summary_df["test_r2_mean"].notna()]
    else:
        successful_models = summary_df
    
    if len(successful_models) > 0:
        max_r2 = successful_models["test_r2_mean"].max()
        assert max_r2 > 0.2, \
            f"At least one model should achieve R² > 0.2, got {max_r2:.4f}"


@pytest.mark.advanced
def test_ml_model_zoo_cli_subcommand(sample_factor_panel_tmpfile: Path, tmp_path: Path, monkeypatch):
    """Test ml_model_zoo CLI subcommand via subprocess."""
    pytest.importorskip("sklearn")
    
    output_dir = tmp_path / "cli_output"
    output_dir.mkdir()
    
    # Run CLI command
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "ml_model_zoo",
        "--factor-panel-file", str(sample_factor_panel_tmpfile),
        "--label-col", "fwd_return_20d",
        "--n-splits", "3",
        "--output-dir", str(output_dir),
        "--no-markdown",  # Skip Markdown for faster test
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes timeout
    )
    
    # Check exit code
    assert result.returncode == 0, \
        f"CLI command failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    
    # Check that summary CSV was created
    csv_path = output_dir / "ml_model_zoo_summary.csv"
    assert csv_path.exists(), f"Summary CSV should exist at {csv_path}"
    
    # Verify CSV is readable and contains data
    summary_df = pd.read_csv(csv_path)
    assert len(summary_df) > 0, "Summary CSV should contain at least one row"
    assert "model_name" in summary_df.columns
    assert "model_type" in summary_df.columns


@pytest.mark.advanced
def test_write_model_zoo_summary(tmp_path: Path):
    """Test that write_model_zoo_summary creates CSV and optional Markdown."""
    output_dir = tmp_path / "summary_output"
    
    # Create a sample summary DataFrame
    summary_df = pd.DataFrame({
        "model_name": ["linear", "ridge_0_1", "rf_depth_3"],
        "model_type": ["linear", "ridge", "random_forest"],
        "test_r2_mean": [0.5, 0.6, 0.55],
        "ic_mean": [0.1, 0.15, 0.12],
        "ic_ir": [0.5, 0.8, 0.6],
        "ls_sharpe": [None, 1.2, None],
        "n_samples": [100, 100, 100],
    })
    
    # Test CSV only
    csv_path, md_path = write_model_zoo_summary(
        summary_df=summary_df,
        output_dir=output_dir,
        write_markdown=False,
    )
    
    assert csv_path.exists()
    assert md_path is None
    
    # Verify CSV content
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == len(summary_df)
    assert list(loaded_df.columns) == list(summary_df.columns)
    
    # Test CSV + Markdown
    csv_path2, md_path2 = write_model_zoo_summary(
        summary_df=summary_df,
        output_dir=output_dir,
        write_markdown=True,
    )
    
    assert csv_path2.exists()
    assert md_path2 is not None
    assert md_path2.exists()
    
    # Verify Markdown content
    md_content = md_path2.read_text(encoding="utf-8")
    assert "# Model Zoo Comparison Summary" in md_content
    assert "linear" in md_content  # Should contain model names
    assert "ridge_0_1" in md_content

