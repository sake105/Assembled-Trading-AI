"""Tests for ML Factor Validation CLI workflow."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_factor_panel_file(tmp_path: Path) -> Path:
    """Create a sample factor panel file for testing."""
    # Create factor panel with 3 symbols × 60 days
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    np.random.seed(42)
    all_data = []
    
    for date in dates:
        for symbol in symbols:
            # Simple factors
            factor_mom = np.random.randn() * 0.5
            factor_value = np.random.randn() * 0.3
            
            # Forward return with simple relationship + noise
            fwd_return = 0.3 * factor_mom + 0.2 * factor_value + np.random.randn() * 0.1
            
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "factor_mom": factor_mom,
                "factor_value": factor_value,
                "fwd_return_20d": fwd_return,
            })
    
    df = pd.DataFrame(all_data)
    
    # Save as parquet
    panel_file = tmp_path / "factor_panel.parquet"
    df.to_parquet(panel_file, index=False)
    
    return panel_file


def test_ml_validate_factors_basic(sample_factor_panel_file: Path, tmp_path: Path):
    """Test basic ML validation via direct function call."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="linear",
        model_params=None,
        n_splits=3,
        test_start=None,
        test_end=None,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check that output files exist
    metrics_csv = output_dir / "ml_metrics_linear_fwd_return_20d.csv"
    report_md = output_dir / "ml_validation_report_linear_fwd_return_20d.md"
    
    assert metrics_csv.exists(), f"Metrics CSV not found: {metrics_csv}"
    assert report_md.exists(), f"Report MD not found: {report_md}"
    
    # Check CSV structure
    metrics_df = pd.read_csv(metrics_csv)
    assert len(metrics_df) > 0
    assert "mse" in metrics_df.columns
    assert "r2" in metrics_df.columns
    
    # Check Markdown report has content
    report_content = report_md.read_text(encoding="utf-8")
    assert len(report_content) > 0
    assert "ML Factor Validation Report" in report_content
    assert "Classical ML Metrics" in report_content


def test_ml_validate_factors_with_ridge(sample_factor_panel_file: Path, tmp_path: Path):
    """Test ML validation with Ridge model and custom parameters."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output_ridge"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.1},
        n_splits=3,
        test_start=None,
        test_end=None,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check outputs
    metrics_csv = output_dir / "ml_metrics_ridge_fwd_return_20d.csv"
    report_md = output_dir / "ml_validation_report_ridge_fwd_return_20d.md"
    portfolio_csv = output_dir / "ml_portfolio_metrics_ridge_fwd_return_20d.csv"
    
    assert metrics_csv.exists()
    assert report_md.exists()
    assert portfolio_csv.exists()
    
    # Check portfolio metrics
    portfolio_df = pd.read_csv(portfolio_csv)
    assert len(portfolio_df) == 1  # One row
    assert "model_name" in portfolio_df.columns
    assert "ls_sharpe" in portfolio_df.columns


def test_ml_validate_factors_via_cli_subcommand(sample_factor_panel_file: Path, tmp_path: Path):
    """Test ML validation via CLI subcommand."""
    pytest.importorskip("sklearn")
    
    output_dir = tmp_path / "ml_output_cli"
    
    # Run via CLI subprocess
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "ml_validate_factors",
        "--factor-panel-file", str(sample_factor_panel_file),
        "--label-col", "fwd_return_20d",
        "--model-type", "linear",
        "--n-splits", "3",
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    # Should succeed
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    
    # Check outputs
    metrics_csv = output_dir / "ml_metrics_linear_fwd_return_20d.csv"
    report_md = output_dir / "ml_validation_report_linear_fwd_return_20d.md"
    
    assert metrics_csv.exists()
    assert report_md.exists()


def test_ml_validate_factors_missing_file(tmp_path: Path):
    """Test that missing factor panel file raises clear error."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    non_existent_file = tmp_path / "nonexistent.parquet"
    
    exit_code = run_ml_validation(
        factor_panel_file=non_existent_file,
        label_col="fwd_return_20d",
        model_type="linear",
        model_params=None,
        n_splits=3,
        output_dir=tmp_path / "output",
    )
    
    assert exit_code == 1  # Should return error code


def test_ml_validate_factors_missing_label_col(tmp_path: Path):
    """Test that missing label column raises clear error."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    # Create factor panel without label column
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "factor_mom": np.random.randn(10),
    })
    
    panel_file = tmp_path / "no_label.parquet"
    df.to_parquet(panel_file, index=False)
    
    exit_code = run_ml_validation(
        factor_panel_file=panel_file,
        label_col="fwd_return_20d",  # Not in DataFrame
        model_type="linear",
        model_params=None,
        n_splits=3,
        output_dir=tmp_path / "output",
    )
    
    assert exit_code == 1  # Should return error code


def test_ml_validate_factors_random_forest(sample_factor_panel_file: Path, tmp_path: Path):
    """Test ML validation with Random Forest model."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output_rf"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="random_forest",
        model_params={"n_estimators": 10, "max_depth": 5},  # Small for speed
        n_splits=3,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check outputs
    metrics_csv = output_dir / "ml_metrics_random_forest_fwd_return_20d.csv"
    assert metrics_csv.exists()


def test_ml_validate_factors_with_time_filter(sample_factor_panel_file: Path, tmp_path: Path):
    """Test ML validation with time range filtering."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output_filtered"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="linear",
        model_params=None,
        n_splits=2,
        test_start="2020-03-01",
        test_end="2020-04-30",
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check that outputs are created (might be smaller due to filtering)
    metrics_csv = output_dir / "ml_metrics_linear_fwd_return_20d.csv"
    assert metrics_csv.exists()


def test_ml_validate_factors_csv_input(tmp_path: Path):
    """Test that CSV input format is supported."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    # Create CSV factor panel
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 30,
        "factor_mom": np.random.randn(30),
        "fwd_return_20d": np.random.randn(30) * 0.02,
    })
    
    csv_file = tmp_path / "factor_panel.csv"
    df.to_csv(csv_file, index=False)
    
    output_dir = tmp_path / "ml_output_csv"
    
    exit_code = run_ml_validation(
        factor_panel_file=csv_file,
        label_col="fwd_return_20d",
        model_type="linear",
        model_params=None,
        n_splits=2,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check outputs
    metrics_csv = output_dir / "ml_metrics_linear_fwd_return_20d.csv"
    assert metrics_csv.exists()


def test_ml_validate_factors_no_sklearn_raises(sample_factor_panel_file: Path, tmp_path: Path, monkeypatch):
    """Test that missing sklearn raises clear error."""
    # Mock sklearn import to raise ImportError
    import sys
    from unittest.mock import MagicMock
    
    # Check if sklearn is actually available
    try:
        import sklearn  # noqa: F401
        sklearn_available = True
    except ImportError:
        sklearn_available = False
    
    if sklearn_available:
        # If sklearn is available, we can't test the ImportError case
        # But we can verify the check exists in the code
        pytest.skip("sklearn is available, cannot test ImportError")
    
    # If sklearn is not available, the function should raise ImportError
    from scripts.run_ml_factor_validation import run_ml_validation
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="linear",
        output_dir=tmp_path / "output",
    )
    
    # Should return error code if sklearn check fails
    # (The actual check happens in run_time_series_cv, which raises ImportError)
    # So this test mainly verifies that the error is caught properly
    assert exit_code == 1


def test_ml_validate_factors_with_model_params(sample_factor_panel_file: Path, tmp_path: Path):
    """Test ML validation with custom model parameters."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output_params"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="ridge",
        model_params={"alpha": 0.5, "max_iter": 500},
        n_splits=3,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check that outputs exist
    metrics_csv = output_dir / "ml_metrics_ridge_fwd_return_20d.csv"
    assert metrics_csv.exists()


def test_ml_validate_factors_predictions_sample(sample_factor_panel_file: Path, tmp_path: Path):
    """Test that predictions sample file is created."""
    pytest.importorskip("sklearn")
    
    from scripts.run_ml_factor_validation import run_ml_validation
    
    output_dir = tmp_path / "ml_output_sample"
    
    exit_code = run_ml_validation(
        factor_panel_file=sample_factor_panel_file,
        label_col="fwd_return_20d",
        model_type="linear",
        n_splits=3,
        output_dir=output_dir,
    )
    
    assert exit_code == 0
    
    # Check predictions sample file
    predictions_file = output_dir / "ml_predictions_sample_linear_fwd_return_20d.parquet"
    assert predictions_file.exists()
    
    # Check structure
    predictions_df = pd.read_parquet(predictions_file)
    assert "y_true" in predictions_df.columns
    assert "y_pred" in predictions_df.columns
    assert len(predictions_df) > 0

