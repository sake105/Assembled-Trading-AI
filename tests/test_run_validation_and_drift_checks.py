# tests/test_run_validation_and_drift_checks.py
"""Tests for run_validation_and_drift_checks script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_validation_and_drift_checks import (
    extract_features_from_dataset,
    extract_labels_from_dataset,
    find_latest_ml_dataset,
    load_ml_dataset,
    run_drift_checks,
    run_validation_checks,
    write_summary_report,
)

pytestmark = pytest.mark.phase9


@pytest.fixture
def sample_ml_dataset():
    """Create a sample ML dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features
    feature_data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1D", tz="UTC"),
        "symbol": ["AAPL"] * (n_samples // 2) + ["MSFT"] * (n_samples // 2),
        "label": np.random.choice([0, 1], size=n_samples),
        "ta_ema_20": np.random.normal(100, 5, n_samples),
        "ta_ema_50": np.random.normal(100, 5, n_samples),
        "ta_rsi": np.random.uniform(30, 70, n_samples),
        "insider_net_buy": np.random.normal(0, 1000, n_samples),
        "shipping_congestion_score": np.random.uniform(0, 100, n_samples),
    }
    
    return pd.DataFrame(feature_data)


@pytest.fixture
def reference_ml_dataset():
    """Create a reference ML dataset for drift detection."""
    np.random.seed(123)  # Different seed to create different distribution
    n_samples = 100
    
    feature_data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="1D", tz="UTC"),
        "symbol": ["AAPL"] * (n_samples // 2) + ["MSFT"] * (n_samples // 2),
        "label": np.random.choice([0, 1], size=n_samples),
        "ta_ema_20": np.random.normal(105, 5, n_samples),  # Shifted mean
        "ta_ema_50": np.random.normal(105, 5, n_samples),
        "ta_rsi": np.random.uniform(30, 70, n_samples),
        "insider_net_buy": np.random.normal(500, 1000, n_samples),  # Shifted mean
        "shipping_congestion_score": np.random.uniform(0, 100, n_samples),
    }
    
    return pd.DataFrame(feature_data)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_features_from_dataset(self, sample_ml_dataset: pd.DataFrame):
        """Test feature extraction from ML dataset."""
        features = extract_features_from_dataset(sample_ml_dataset)
        
        assert isinstance(features, pd.DataFrame)
        assert "label" not in features.columns
        assert "timestamp" not in features.columns
        assert "symbol" not in features.columns
        
        # Should have feature columns
        assert "ta_ema_20" in features.columns
        assert "insider_net_buy" in features.columns
        assert len(features.columns) > 0

    def test_extract_labels_from_dataset(self, sample_ml_dataset: pd.DataFrame):
        """Test label extraction from ML dataset."""
        labels = extract_labels_from_dataset(sample_ml_dataset)
        
        assert labels is not None
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_ml_dataset)
        assert labels.name == "label"

    def test_extract_labels_from_dataset_no_label_column(self):
        """Test label extraction when label column doesn't exist."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        labels = extract_labels_from_dataset(df)
        assert labels is None

    def test_find_latest_ml_dataset(self, tmp_path: Path):
        """Test finding the latest ML dataset."""
        ml_dir = tmp_path / "ml_datasets"
        ml_dir.mkdir(parents=True, exist_ok=True)
        
        # Create two parquet files
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        df2 = pd.DataFrame({"col2": [4, 5, 6]})
        
        file1 = ml_dir / "dataset1.parquet"
        file2 = ml_dir / "dataset2.parquet"
        
        df1.to_parquet(file1)
        import time
        time.sleep(0.1)  # Small delay to ensure different mtime
        df2.to_parquet(file2)
        
        latest = find_latest_ml_dataset(ml_dir)
        assert latest is not None
        assert latest == file2

    def test_find_latest_ml_dataset_empty_directory(self, tmp_path: Path):
        """Test finding latest dataset when directory is empty."""
        ml_dir = tmp_path / "ml_datasets"
        ml_dir.mkdir(parents=True, exist_ok=True)
        
        latest = find_latest_ml_dataset(ml_dir)
        assert latest is None

    def test_load_ml_dataset(self, sample_ml_dataset: pd.DataFrame, tmp_path: Path):
        """Test loading ML dataset from parquet file."""
        dataset_file = tmp_path / "test_dataset.parquet"
        sample_ml_dataset.to_parquet(dataset_file)
        
        loaded = load_ml_dataset(dataset_file)
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_ml_dataset)
        assert list(loaded.columns) == list(sample_ml_dataset.columns)

    def test_load_ml_dataset_file_not_found(self, tmp_path: Path):
        """Test loading non-existent dataset."""
        dataset_file = tmp_path / "nonexistent.parquet"
        
        with pytest.raises(FileNotFoundError):
            load_ml_dataset(dataset_file)


class TestValidationChecks:
    """Tests for validation checks."""

    def test_run_validation_checks_with_metrics(self, sample_ml_dataset: pd.DataFrame):
        """Test validation checks with performance metrics."""
        metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": -15.0,
            "total_trades": 100
        }
        
        results = run_validation_checks(
            model_name="test_model",
            current_dataset=sample_ml_dataset,
            performance_metrics=metrics
        )
        
        assert "model_name" in results
        assert "is_ok" in results
        assert "errors" in results
        assert "warnings" in results
        assert "metadata" in results
        
        assert results["model_name"] == "test_model"
        assert isinstance(results["is_ok"], bool)

    def test_run_validation_checks_without_metrics(self, sample_ml_dataset: pd.DataFrame):
        """Test validation checks without performance metrics."""
        results = run_validation_checks(
            model_name="test_model",
            current_dataset=sample_ml_dataset,
            performance_metrics=None
        )
        
        assert "warnings" in results
        assert len(results["warnings"]) > 0  # Should warn about missing metrics


class TestDriftChecks:
    """Tests for drift detection checks."""

    def test_run_drift_checks_with_reference(self, sample_ml_dataset: pd.DataFrame, reference_ml_dataset: pd.DataFrame):
        """Test drift checks with reference dataset."""
        results = run_drift_checks(
            current_dataset=sample_ml_dataset,
            reference_dataset=reference_ml_dataset,
            psi_threshold=0.2,
            severe_threshold=0.3
        )
        
        assert "feature_drift" in results
        assert "label_drift" in results
        assert "performance_drift" in results
        
        feature_drift = results["feature_drift"]
        if feature_drift:
            assert "total_features_checked" in feature_drift
            assert "features_with_drift" in feature_drift
            assert "overall_severity" in feature_drift

    def test_run_drift_checks_without_reference(self, sample_ml_dataset: pd.DataFrame):
        """Test drift checks without reference dataset."""
        results = run_drift_checks(
            current_dataset=sample_ml_dataset,
            reference_dataset=None,
            psi_threshold=0.2
        )
        
        assert "feature_drift" in results
        assert results["feature_drift"] is None or "overall_severity" in results["feature_drift"]


class TestSummaryReport:
    """Tests for summary report generation."""

    def test_write_summary_report(self, tmp_path: Path, sample_ml_dataset: pd.DataFrame):
        """Test writing summary report."""
        from datetime import datetime
        
        output_path = tmp_path / "summary.md"
        
        validation_results = {
            "model_name": "test_model",
            "is_ok": True,
            "errors": [],
            "warnings": ["Warning 1"],
            "metadata": {}
        }
        
        drift_results = {
            "feature_drift": {
                "total_features_checked": 5,
                "features_with_drift": [],
                "overall_severity": "NONE"
            },
            "label_drift": None,
            "performance_drift": None
        }
        
        write_summary_report(
            output_path=output_path,
            model_name="test_model",
            current_dataset_path=Path("test_current.parquet"),
            reference_dataset_path=None,
            validation_results=validation_results,
            drift_results=drift_results,
            timestamp=datetime.utcnow()
        )
        
        assert output_path.exists()
        
        content = output_path.read_text(encoding="utf-8")
        
        # Check for key sections
        assert "Validation and Drift Checks Summary" in content
        assert "Validation Result" in content
        assert "Drift Status" in content
        assert "test_model" in content

    def test_write_summary_report_contains_keywords(self, tmp_path: Path):
        """Test that summary report contains required keywords."""
        from datetime import datetime
        
        output_path = tmp_path / "summary.md"
        
        validation_results = {
            "model_name": "test_model",
            "is_ok": False,
            "errors": ["Error 1", "Error 2"],
            "warnings": ["Warning 1"],
            "metadata": {}
        }
        
        drift_results = {
            "feature_drift": {
                "total_features_checked": 10,
                "features_with_drift": [
                    {"feature": "ta_ema_20", "psi": 0.25, "drift_flag": "MODERATE"}
                ],
                "overall_severity": "MODERATE"
            },
            "label_drift": {
                "psi": 0.15,
                "drift_detected": False,
                "drift_severity": "NONE"
            },
            "performance_drift": None
        }
        
        write_summary_report(
            output_path=output_path,
            model_name="test_model",
            current_dataset_path=Path("current.parquet"),
            reference_dataset_path=Path("reference.parquet"),
            validation_results=validation_results,
            drift_results=drift_results,
            timestamp=datetime.utcnow()
        )
        
        content = output_path.read_text(encoding="utf-8")
        
        # Check for required keywords
        assert "Validation Result" in content
        assert "Drift Status" in content
        assert "Feature Drift" in content
        assert "Label Drift" in content


class TestScriptIntegration:
    """Integration tests for the script."""

    def test_script_runs_with_sample_data(self, tmp_path: Path, sample_ml_dataset: pd.DataFrame, reference_ml_dataset: pd.DataFrame):
        """Test that script runs successfully with sample data."""
        script_path = ROOT / "scripts" / "run_validation_and_drift_checks.py"
        
        # Create dataset files
        ml_datasets_dir = tmp_path / "ml_datasets"
        ml_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        current_dataset_file = ml_datasets_dir / "current.parquet"
        reference_dataset_file = ml_datasets_dir / "reference.parquet"
        
        sample_ml_dataset.to_parquet(current_dataset_file)
        reference_ml_dataset.to_parquet(reference_dataset_file)
        
        output_dir = tmp_path / "monitoring"
        output_path = output_dir / "summary.md"
        
        # Run script
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--current-dataset", str(current_dataset_file),
                "--reference-dataset", str(reference_dataset_file),
                "--output", str(output_path),
                "--model-name", "test_model"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check exit code
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check that output file was created
        assert output_path.exists(), "Summary report file was not created"
        
        # Check file content
        content = output_path.read_text(encoding="utf-8")
        assert "Validation Result" in content
        assert "Drift Status" in content

    def test_script_without_reference_dataset(self, tmp_path: Path, sample_ml_dataset: pd.DataFrame):
        """Test script runs without reference dataset."""
        script_path = ROOT / "scripts" / "run_validation_and_drift_checks.py"
        
        # Create dataset file
        ml_datasets_dir = tmp_path / "ml_datasets"
        ml_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        current_dataset_file = ml_datasets_dir / "current.parquet"
        sample_ml_dataset.to_parquet(current_dataset_file)
        
        output_dir = tmp_path / "monitoring"
        output_path = output_dir / "summary.md"
        
        # Run script without reference dataset
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--current-dataset", str(current_dataset_file),
                "--output", str(output_path)
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed (reference is optional)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_path.exists()

    def test_script_file_not_found_error(self, tmp_path: Path):
        """Test script handles file not found errors gracefully."""
        script_path = ROOT / "scripts" / "run_validation_and_drift_checks.py"
        
        output_path = tmp_path / "summary.md"
        
        # Run script with non-existent dataset
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--current-dataset", str(tmp_path / "nonexistent.parquet"),
                "--output", str(output_path)
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail with exit code 1
        assert result.returncode == 1
        
        # Error message could be in stdout or stderr (depending on logging setup)
        error_output = (result.stdout + result.stderr).lower()
        assert "not found" in error_output or "file not found" in error_output


class TestReportContent:
    """Tests for report content validation."""

    def test_report_contains_validation_section(self, tmp_path: Path):
        """Test that report contains validation section with keywords."""
        from datetime import datetime
        
        output_path = tmp_path / "summary.md"
        
        validation_results = {
            "model_name": "test_model",
            "is_ok": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        drift_results = {
            "feature_drift": None,
            "label_drift": None,
            "performance_drift": None
        }
        
        write_summary_report(
            output_path=output_path,
            model_name="test_model",
            current_dataset_path=None,
            reference_dataset_path=None,
            validation_results=validation_results,
            drift_results=drift_results,
            timestamp=datetime.utcnow()
        )
        
        content = output_path.read_text(encoding="utf-8")
        
        # Must contain validation result section
        assert "## Validation Result" in content
        assert "Status:" in content

    def test_report_contains_drift_section(self, tmp_path: Path):
        """Test that report contains drift status section."""
        from datetime import datetime
        
        output_path = tmp_path / "summary.md"
        
        validation_results = {
            "model_name": "test_model",
            "is_ok": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        drift_results = {
            "feature_drift": {
                "total_features_checked": 5,
                "features_with_drift": [],
                "overall_severity": "NONE"
            },
            "label_drift": None,
            "performance_drift": None
        }
        
        write_summary_report(
            output_path=output_path,
            model_name="test_model",
            current_dataset_path=None,
            reference_dataset_path=None,
            validation_results=validation_results,
            drift_results=drift_results,
            timestamp=datetime.utcnow()
        )
        
        content = output_path.read_text(encoding="utf-8")
        
        # Must contain drift status section
        assert "## Drift Status" in content
        assert "Feature Drift" in content

