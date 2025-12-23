"""Tests for Risk Report CLI workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_backtest_dir(tmp_path: Path) -> Path:
    """Create a sample backtest output directory with equity_curve and positions."""
    backtest_dir = tmp_path / "backtest_output"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Create equity curve
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    equity_values = 10000.0 * (1 + np.cumsum(np.random.normal(0.001, 0.01, len(dates))))

    equity_df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity_values,
        }
    )
    equity_df.to_csv(backtest_dir / "equity_curve.csv", index=False)

    # Create positions
    symbols = ["AAPL", "MSFT", "GOOGL"]
    positions_data = []
    for date in dates:
        for symbol in symbols:
            positions_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "weight": np.random.uniform(0.2, 0.4),
                }
            )

    positions_df = pd.DataFrame(positions_data)
    # Normalize weights per timestamp to sum to ~1.0
    for date in dates:
        mask = positions_df["timestamp"] == date
        weights = positions_df.loc[mask, "weight"]
        positions_df.loc[mask, "weight"] = weights / weights.sum() * 1.0

    positions_df.to_csv(backtest_dir / "positions.csv", index=False)

    return backtest_dir


@pytest.fixture
def sample_regime_file(tmp_path: Path) -> Path:
    """Create a sample regime state file."""
    regime_dir = tmp_path / "regime"
    regime_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")

    # Create regimes: bull (0-25), bear (26-45), neutral (46-59)
    regimes = ["bull"] * 26 + ["bear"] * 20 + ["neutral"] * 14

    regime_df = pd.DataFrame(
        {
            "timestamp": dates,
            "regime_label": regimes,
            "regime_trend_score": np.random.randn(len(dates)),
        }
    )

    regime_file = regime_dir / "regime_state.csv"
    regime_df.to_csv(regime_file, index=False)

    return regime_file


@pytest.fixture
def sample_factor_panel_file(tmp_path: Path) -> Path:
    """Create a sample factor panel file."""
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    factor_data = []
    for date in dates:
        for symbol in symbols:
            factor_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "returns_12m": np.random.randn(),
                    "rv_20": np.random.uniform(0.1, 0.3),
                    "earnings_eps_surprise_last": np.random.randn() * 0.1,
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_file = factor_dir / "factor_panel.csv"
    factor_df.to_csv(factor_file, index=False)

    return factor_file


def test_risk_report_basic(sample_backtest_dir: Path, tmp_path: Path):
    """Test basic risk report generation."""
    from scripts.generate_risk_report import generate_risk_report

    output_dir = tmp_path / "risk_output"

    exit_code = generate_risk_report(
        backtest_dir=sample_backtest_dir,
        regime_file=None,
        factor_panel_file=None,
        output_dir=output_dir,
    )

    assert exit_code == 0

    # Check that required outputs exist
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"

    assert summary_csv.exists(), f"risk_summary.csv not found in {output_dir}"
    assert report_md.exists(), f"risk_report.md not found in {output_dir}"

    # Check CSV structure
    summary_df = pd.read_csv(summary_csv)
    assert len(summary_df) == 1  # Single row with all metrics
    assert "mean_return_annualized" in summary_df.columns
    assert "vol_annualized" in summary_df.columns
    assert "sharpe" in summary_df.columns
    assert "n_periods" in summary_df.columns

    # Check Markdown report exists and has content
    report_content = report_md.read_text(encoding="utf-8")
    assert len(report_content) > 0
    assert "Risk Report" in report_content
    assert "Global Risk Metrics" in report_content

    # Check that exposure timeseries is created if positions available
    exposure_csv = output_dir / "exposure_timeseries.csv"
    assert exposure_csv.exists(), (
        "exposure_timeseries.csv should be created when positions are available"
    )

    exposure_df = pd.read_csv(exposure_csv)
    assert "gross_exposure" in exposure_df.columns
    assert "net_exposure" in exposure_df.columns
    assert "hhi_concentration" in exposure_df.columns


def test_risk_report_with_regime_and_factors(
    sample_backtest_dir: Path,
    sample_regime_file: Path,
    sample_factor_panel_file: Path,
    tmp_path: Path,
):
    """Test risk report with regime and factor data."""
    from scripts.generate_risk_report import generate_risk_report

    output_dir = tmp_path / "risk_output_full"

    exit_code = generate_risk_report(
        backtest_dir=sample_backtest_dir,
        regime_file=sample_regime_file,
        factor_panel_file=sample_factor_panel_file,
        output_dir=output_dir,
    )

    assert exit_code == 0

    # Check that all outputs exist
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"
    regime_csv = output_dir / "risk_by_regime.csv"
    factor_csv = output_dir / "risk_by_factor_group.csv"

    assert summary_csv.exists()
    assert report_md.exists()
    assert regime_csv.exists(), (
        "risk_by_regime.csv should be created when regime file is provided"
    )
    assert factor_csv.exists(), (
        "risk_by_factor_group.csv should be created when factor panel is provided"
    )

    # Check regime CSV structure
    regime_df = pd.read_csv(regime_csv)
    assert "regime" in regime_df.columns
    assert "n_periods" in regime_df.columns
    assert "sharpe" in regime_df.columns
    assert len(regime_df) > 0  # Should have at least one regime

    # Check factor CSV structure
    factor_df = pd.read_csv(factor_csv)
    assert "factor_group" in factor_df.columns
    assert "correlation_with_returns" in factor_df.columns
    assert len(factor_df) > 0  # Should have at least one factor group

    # Check that Markdown report includes regime and factor sections
    report_content = report_md.read_text(encoding="utf-8")
    assert "Risk by Regime" in report_content
    assert "Performance Attribution by Factor Group" in report_content


def test_risk_report_missing_positions(sample_backtest_dir: Path, tmp_path: Path):
    """Test that risk report works even if positions file is missing."""
    from scripts.generate_risk_report import generate_risk_report

    # Remove positions file
    positions_file = sample_backtest_dir / "positions.csv"
    if positions_file.exists():
        positions_file.unlink()

    output_dir = tmp_path / "risk_output_no_positions"

    exit_code = generate_risk_report(
        backtest_dir=sample_backtest_dir,
        regime_file=None,
        factor_panel_file=None,
        output_dir=output_dir,
    )

    assert exit_code == 0

    # Should still create summary and report
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"

    assert summary_csv.exists()
    assert report_md.exists()

    # Exposure timeseries should not exist (no positions)
    exposure_csv = output_dir / "exposure_timeseries.csv"
    assert not exposure_csv.exists(), (
        "exposure_timeseries.csv should not be created without positions"
    )


def test_risk_report_missing_regime_file(sample_backtest_dir: Path, tmp_path: Path):
    """Test that risk report handles missing regime file gracefully."""
    from scripts.generate_risk_report import generate_risk_report

    # Use a non-existent regime file
    non_existent_regime = tmp_path / "non_existent" / "regime.csv"

    output_dir = tmp_path / "risk_output_no_regime"

    exit_code = generate_risk_report(
        backtest_dir=sample_backtest_dir,
        regime_file=non_existent_regime,  # Doesn't exist
        factor_panel_file=None,
        output_dir=output_dir,
    )

    assert exit_code == 0  # Should not crash

    # Should still create basic outputs
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"

    assert summary_csv.exists()
    assert report_md.exists()

    # Regime CSV should not exist
    regime_csv = output_dir / "risk_by_regime.csv"
    assert not regime_csv.exists(), (
        "risk_by_regime.csv should not be created without valid regime file"
    )


def test_risk_report_missing_backtest_dir(tmp_path: Path):
    """Test that risk report handles missing backtest directory."""
    from scripts.generate_risk_report import generate_risk_report

    non_existent_dir = tmp_path / "non_existent_backtest"

    exit_code = generate_risk_report(
        backtest_dir=non_existent_dir,
        regime_file=None,
        factor_panel_file=None,
        output_dir=tmp_path / "output",
    )

    assert exit_code == 1  # Should return error code


def test_risk_report_missing_equity_curve(tmp_path: Path):
    """Test that risk report handles missing equity curve file."""
    from scripts.generate_risk_report import generate_risk_report

    # Create backtest dir but without equity curve
    backtest_dir = tmp_path / "backtest_no_equity"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Create positions but no equity curve
    positions_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "weight": [0.5] * 10,
        }
    )
    positions_df.to_csv(backtest_dir / "positions.csv", index=False)

    exit_code = generate_risk_report(
        backtest_dir=backtest_dir,
        regime_file=None,
        factor_panel_file=None,
        output_dir=tmp_path / "output",
    )

    assert exit_code == 1  # Should return error code (equity curve is required)


def test_risk_report_cli_via_subcommand(
    sample_backtest_dir: Path, tmp_path: Path, monkeypatch
):
    """Test risk report via CLI subcommand."""
    import subprocess

    output_dir = tmp_path / "risk_output_cli"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run via CLI
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_risk_report.py"),
        "--backtest-dir",
        str(sample_backtest_dir),
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Should succeed
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Check outputs
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"

    assert summary_csv.exists()
    assert report_md.exists()


def test_risk_report_parquet_files(tmp_path: Path):
    """Test that risk report works with Parquet files."""
    from scripts.generate_risk_report import generate_risk_report

    backtest_dir = tmp_path / "backtest_parquet"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Create equity curve as Parquet
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    equity_df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": 10000.0
            * (1 + np.cumsum(np.random.normal(0.001, 0.01, len(dates)))),
        }
    )
    equity_df.to_parquet(backtest_dir / "equity_curve.parquet", index=False)

    # Create positions as Parquet
    positions_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "weight": [1.0] * len(dates),
        }
    )
    positions_df.to_parquet(backtest_dir / "positions.parquet", index=False)

    output_dir = tmp_path / "risk_output_parquet"

    exit_code = generate_risk_report(
        backtest_dir=backtest_dir,
        regime_file=None,
        factor_panel_file=None,
        output_dir=output_dir,
    )

    assert exit_code == 0

    # Check outputs
    summary_csv = output_dir / "risk_summary.csv"
    report_md = output_dir / "risk_report.md"

    assert summary_csv.exists()
    assert report_md.exists()
