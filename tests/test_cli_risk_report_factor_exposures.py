"""Tests for Risk Report Factor Exposures Integration.

Tests the integration of factor exposure analysis into the risk report generation workflow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_backtest_dir(tmp_path: Path) -> Path:
    """Create a sample backtest directory with equity curve and performance report."""
    backtest_dir = tmp_path / "backtest_sample"
    backtest_dir.mkdir()
    
    # Create equity curve CSV
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    equity_values = 10000 * (1 + np.random.randn(100).cumsum() * 0.01)
    equity_df = pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values,
    })
    equity_df.to_csv(backtest_dir / "equity_curve.csv", index=False)
    
    # Create minimal performance report (optional, but some workflows expect it)
    performance_report = backtest_dir / "performance_report.md"
    performance_report.write_text("# Performance Report\n\nFinal PF: 1.15\n", encoding="utf-8")
    
    return backtest_dir


@pytest.fixture
def sample_factor_returns_file(tmp_path: Path) -> Path:
    """Create a sample factor returns file with 2-3 factors."""
    factor_file = tmp_path / "factor_returns.parquet"
    
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    factor_returns_df = pd.DataFrame({
        "timestamp": dates,
        "factor1": np.random.randn(100) * 0.01,
        "factor2": np.random.randn(100) * 0.015,
        "factor3": np.random.randn(100) * 0.008,
    })
    factor_returns_df = factor_returns_df.set_index("timestamp")
    factor_returns_df.to_parquet(factor_file)
    
    return factor_file


@pytest.mark.advanced
def test_risk_report_with_factor_exposures_creates_files(
    sample_backtest_dir: Path,
    sample_factor_returns_file: Path,
    tmp_path: Path,
):
    """Test that CLI with --enable-factor-exposures creates expected CSV files."""
    import subprocess
    import sys
    
    output_dir = tmp_path / "risk_report_output"
    output_dir.mkdir()
    
    # Run risk report generation
    script_path = Path(__file__).parent.parent / "scripts" / "generate_risk_report.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--backtest-dir",
        str(sample_backtest_dir),
        "--output-dir",
        str(output_dir),
        "--enable-factor-exposures",
        "--factor-returns-file",
        str(sample_factor_returns_file),
        "--factor-exposures-window",
        "60",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    # Check exit code (should be 0 for success)
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # Check that factor exposure files were created
    detail_file = output_dir / "factor_exposures_detail.csv"
    summary_file = output_dir / "factor_exposures_summary.csv"
    
    assert detail_file.exists(), f"factor_exposures_detail.csv should exist in {output_dir}"
    assert summary_file.exists(), f"factor_exposures_summary.csv should exist in {output_dir}"
    
    # Check that files are not empty
    detail_df = pd.read_csv(detail_file, index_col=0, parse_dates=True)
    summary_df = pd.read_csv(summary_file)
    
    assert not detail_df.empty, "factor_exposures_detail.csv should not be empty"
    assert not summary_df.empty, "factor_exposures_summary.csv should not be empty"
    
    # Check that summary has expected columns
    expected_cols = ["factor", "mean_beta", "std_beta", "mean_r2", "mean_residual_vol", "n_windows", "n_windows_total"]
    for col in expected_cols:
        assert col in summary_df.columns, f"Summary should have column '{col}'"


@pytest.mark.advanced
def test_risk_report_with_factor_exposures_updates_markdown(
    sample_backtest_dir: Path,
    sample_factor_returns_file: Path,
    tmp_path: Path,
):
    """Test that risk_report.md contains Factor Exposures section."""
    import subprocess
    import sys
    
    output_dir = tmp_path / "risk_report_output"
    output_dir.mkdir()
    
    # Run risk report generation
    script_path = Path(__file__).parent.parent / "scripts" / "generate_risk_report.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--backtest-dir",
        str(sample_backtest_dir),
        "--output-dir",
        str(output_dir),
        "--enable-factor-exposures",
        "--factor-returns-file",
        str(sample_factor_returns_file),
        "--factor-exposures-window",
        "60",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # Check that markdown report exists and contains Factor Exposures section
    report_md = output_dir / "risk_report.md"
    assert report_md.exists(), f"risk_report.md should exist in {output_dir}"
    
    report_content = report_md.read_text(encoding="utf-8")
    
    assert "## Factor Exposures" in report_content, "Report should contain '## Factor Exposures' section"
    assert "Factor Exposure Summary" in report_content, "Report should contain 'Factor Exposure Summary' subsection"


@pytest.mark.advanced
def test_risk_report_without_factor_exposures_no_files(
    sample_backtest_dir: Path,
    tmp_path: Path,
):
    """Test that without --enable-factor-exposures flag, factor exposure files are not created."""
    import subprocess
    import sys
    
    output_dir = tmp_path / "risk_report_output"
    output_dir.mkdir()
    
    # Run risk report generation WITHOUT factor exposures
    script_path = Path(__file__).parent.parent / "scripts" / "generate_risk_report.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--backtest-dir",
        str(sample_backtest_dir),
        "--output-dir",
        str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # Check that factor exposure files were NOT created
    detail_file = output_dir / "factor_exposures_detail.csv"
    summary_file = output_dir / "factor_exposures_summary.csv"
    
    assert not detail_file.exists(), "factor_exposures_detail.csv should NOT exist without --enable-factor-exposures"
    assert not summary_file.exists(), "factor_exposures_summary.csv should NOT exist without --enable-factor-exposures"
    
    # Check that basic report still exists
    report_md = output_dir / "risk_report.md"
    assert report_md.exists(), "risk_report.md should still exist"
    
    # Check that Factor Exposures section is NOT in report
    report_content = report_md.read_text(encoding="utf-8")
    assert "## Factor Exposures" not in report_content, "Report should NOT contain Factor Exposures section without flag"

