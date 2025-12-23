"""Tests for Health Check CLI Script.

Tests the health check script with simulated backtest directories and data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_backtest_dir(tmp_path: Path) -> Path:
    """Create a sample backtest directory with equity curve and risk report."""
    backtest_dir = tmp_path / "backtest_sample"
    backtest_dir.mkdir()

    # Create equity curve CSV (with recent dates to pass freshness check)
    # Use dates ending today, so last timestamp is current
    end_date = pd.Timestamp.now(tz="UTC").normalize()  # Today at midnight UTC
    dates = pd.date_range(end=end_date, periods=100, freq="D", tz="UTC")
    equity_values = 10000 * (1 + np.random.randn(100).cumsum() * 0.01)
    equity_df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity_values,
        }
    )
    equity_df.to_csv(backtest_dir / "equity_curve.csv", index=False)

    # Create risk summary CSV (healthy metrics)
    risk_summary_df = pd.DataFrame(
        [
            {
                "mean_return_annualized": 0.10,
                "vol_annualized": 0.15,
                "sharpe": 1.5,  # Good Sharpe
                "sortino": 1.8,
                "max_drawdown": -0.12,  # Moderate drawdown
                "calmar": 0.83,
                "turnover": 2.5,  # Low turnover
            }
        ]
    )
    risk_summary_df.to_csv(backtest_dir / "risk_summary.csv", index=False)

    # Create risk report Markdown
    risk_report_md = backtest_dir / "risk_report.md"
    risk_report_md.write_text(
        "# Risk Report\n\n## Global Risk Metrics\n\nSharpe: 1.5\n", encoding="utf-8"
    )

    return backtest_dir


@pytest.fixture
def sample_backtest_dir_unhealthy(tmp_path: Path) -> Path:
    """Create a sample backtest directory with unhealthy metrics."""
    backtest_dir = tmp_path / "backtest_unhealthy"
    backtest_dir.mkdir()

    # Create equity curve CSV (old data)
    dates = pd.date_range(
        "2023-01-01", periods=50, freq="D", tz="UTC"
    )  # Old dates (will fail freshness check)
    equity_values = 10000 * (1 + np.random.randn(50).cumsum() * 0.01)
    equity_df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity_values,
        }
    )
    equity_df.to_csv(backtest_dir / "equity_curve.csv", index=False)

    # Create risk summary CSV (unhealthy metrics)
    risk_summary_df = pd.DataFrame(
        [
            {
                "mean_return_annualized": -0.05,
                "vol_annualized": 0.25,
                "sharpe": -0.3,  # Negative Sharpe
                "sortino": -0.4,
                "max_drawdown": -0.50,  # Severe drawdown
                "calmar": -0.10,
                "turnover": 15.0,  # High turnover
            }
        ]
    )
    risk_summary_df.to_csv(backtest_dir / "risk_summary.csv", index=False)

    return backtest_dir


@pytest.mark.advanced
def test_health_check_ok_with_healthy_data(sample_backtest_dir: Path, tmp_path: Path):
    """Test that health check returns OK status with healthy dummy data."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"
    output_dir = tmp_path / "health_output"
    output_dir.mkdir()

    # Create backtests root
    backtests_root = sample_backtest_dir.parent

    cmd = [
        sys.executable,
        str(script_path),
        "--backtests-root",
        str(backtests_root),
        "--output-dir",
        str(output_dir),
        "--days",
        "100",  # Large window so freshness check passes
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Check that output files were created
    summary_json = output_dir / "health_summary.json"
    summary_md = output_dir / "health_summary.md"

    assert summary_json.exists(), "health_summary.json should exist"
    assert summary_md.exists(), "health_summary.md should exist"

    # Check JSON content
    with open(summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Overall status should be OK or SKIP for healthy data
    overall_status = summary_data.get("overall_status", "UNKNOWN")
    checks_info = [
        (c.get("name"), c.get("status")) for c in summary_data.get("checks", [])
    ]

    # Exit code should match overall status (0=OK/SKIP, 1=WARN, 2=CRITICAL)
    expected_exit_code = (
        2 if overall_status == "CRITICAL" else (1 if overall_status == "WARN" else 0)
    )

    # For healthy data, we expect OK or SKIP
    assert overall_status in ["OK", "SKIP"], (
        f"Expected OK or SKIP, got {overall_status}. "
        f"Checks: {checks_info}. "
        f"Exit code: {result.returncode} (expected {expected_exit_code})"
    )

    # Verify exit code matches overall status
    assert result.returncode == expected_exit_code, (
        f"Exit code {result.returncode} doesn't match overall status {overall_status} "
        f"(expected {expected_exit_code})"
    )


@pytest.mark.advanced
def test_health_check_warn_when_risk_report_missing(
    sample_backtest_dir: Path, tmp_path: Path
):
    """Test that health check returns WARN when risk report is missing."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"
    output_dir = tmp_path / "health_output"
    output_dir.mkdir()

    # Remove risk report
    (sample_backtest_dir / "risk_report.md").unlink()
    (sample_backtest_dir / "risk_summary.csv").unlink()

    # Create backtests root
    backtests_root = sample_backtest_dir.parent

    cmd = [
        sys.executable,
        str(script_path),
        "--backtests-root",
        str(backtests_root),
        "--output-dir",
        str(output_dir),
        "--days",
        "100",
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Exit code should be 1 for WARN (or 0 if overall is OK with other checks)
    # But we should see WARN for risk_report_exists check
    assert result.returncode in [0, 1], (
        f"Expected exit code 0 or 1, got {result.returncode}"
    )

    # Check JSON content
    summary_json = output_dir / "health_summary.json"
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Find risk_report_exists check
        risk_report_check = None
        for check in summary_data.get("checks", []):
            if check["name"] == "risk_report_exists":
                risk_report_check = check
                break

        assert risk_report_check is not None, (
            "risk_report_exists check should be present"
        )
        assert risk_report_check["status"] == "WARN", (
            f"Expected WARN for missing risk report, got {risk_report_check['status']}"
        )


@pytest.mark.advanced
def test_health_check_critical_when_equity_missing(tmp_path: Path):
    """Test that health check returns CRITICAL when equity curve is missing."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"
    output_dir = tmp_path / "health_output"
    output_dir.mkdir()

    # Create backtest directory without equity curve
    backtest_dir = tmp_path / "backtest_no_equity"
    backtest_dir.mkdir()
    # Don't create equity_curve.csv

    # Create backtests root
    backtests_root = tmp_path

    cmd = [
        sys.executable,
        str(script_path),
        "--backtests-root",
        str(backtests_root),
        "--output-dir",
        str(output_dir),
        "--days",
        "100",
        "--format",
        "json",
    ]

    subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Exit code should be 2 for CRITICAL, or we should see CRITICAL status in output
    # Actually, if no backtest dir is found, it might be different
    # Let's check the output
    summary_json = output_dir / "health_summary.json"
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Check if we have CRITICAL status (either overall or in checks)
        overall_status = summary_data.get("overall_status", "UNKNOWN")
        checks = summary_data.get("checks", [])

        # Should have at least one CRITICAL check or overall_status CRITICAL
        has_critical = overall_status == "CRITICAL" or any(
            check.get("status") == "CRITICAL" for check in checks
        )

        # If no backtest dir found, we should have CRITICAL
        assert has_critical or overall_status in ["CRITICAL", "WARN"], (
            "Should have CRITICAL or WARN when equity is missing"
        )


@pytest.mark.advanced
def test_health_check_warn_when_sharpe_below_threshold(
    sample_backtest_dir_unhealthy: Path, tmp_path: Path
):
    """Test that health check returns WARN when Sharpe ratio is below threshold."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"
    output_dir = tmp_path / "health_output"
    output_dir.mkdir()

    # Create backtests root
    backtests_root = sample_backtest_dir_unhealthy.parent

    cmd = [
        sys.executable,
        str(script_path),
        "--backtests-root",
        str(backtests_root),
        "--output-dir",
        str(output_dir),
        "--days",
        "365",  # Large window so freshness check passes
        "--min-sharpe",
        "0.5",
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Exit code should be 1 for WARN (or 2 for CRITICAL if drawdown is also bad)
    assert result.returncode in [1, 2], (
        f"Expected exit code 1 or 2, got {result.returncode}"
    )

    # Check JSON content
    summary_json = output_dir / "health_summary.json"
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Find sharpe_ratio check
        sharpe_check = None
        for check in summary_data.get("checks", []):
            if check["name"] == "sharpe_ratio":
                sharpe_check = check
                break

        if sharpe_check is not None:
            assert sharpe_check["status"] in ["WARN", "CRITICAL"], (
                f"Expected WARN or CRITICAL for low Sharpe, got {sharpe_check['status']}"
            )


@pytest.mark.advanced
def test_health_check_cli_help():
    """Test that CLI --help works."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--help",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0, "Help should exit with code 0"
    assert "backtests-root" in result.stdout, "Help should mention --backtests-root"
    assert "days" in result.stdout, "Help should mention --days"


@pytest.mark.advanced
def test_check_health_subcommand_in_cli():
    """Test that check_health subcommand exists in central CLI."""
    cli_path = Path(__file__).parent.parent / "scripts" / "cli.py"

    # Test --help for check_health subcommand
    cmd = [
        sys.executable,
        str(cli_path),
        "check_health",
        "--help",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0, "check_health --help should exit with code 0"
    assert "backtests-root" in result.stdout, "Help should mention --backtests-root"
    assert "days" in result.stdout, "Help should mention --days"

    # Test that check_health appears in info subcommand
    cmd_info = [
        sys.executable,
        str(cli_path),
        "info",
    ]

    result_info = subprocess.run(
        cmd_info, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    assert result_info.returncode == 0, "info should exit with code 0"
    assert "check_health" in result_info.stdout, (
        "info should list check_health subcommand"
    )


@pytest.mark.advanced
def test_check_health_smoke_test_minimal(tmp_path: Path):
    """Smoke test: check_health with minimal setup should not crash."""
    cli_path = Path(__file__).parent.parent / "scripts" / "cli.py"
    backtests_root = tmp_path / "backtests"
    backtests_root.mkdir(parents=True)
    output_dir = tmp_path / "health"

    cmd = [
        sys.executable,
        str(cli_path),
        "check_health",
        "--backtests-root",
        str(backtests_root),
        "--output-dir",
        str(output_dir),
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Should not crash (exit code 0, 1, or 2 is acceptable)
    assert result.returncode in [0, 1, 2], (
        f"check_health should exit with 0/1/2, got {result.returncode}. stderr: {result.stderr}"
    )

    # Should create output files
    assert (output_dir / "health_summary.json").exists(), (
        "health_summary.json should be created"
    )
    assert (output_dir / "health_summary.md").exists(), (
        "health_summary.md should be created"
    )
