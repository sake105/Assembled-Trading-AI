"""Tests for TCA Report CLI Script.

This module tests the CLI script scripts/generate_tca_report.py and its integration with scripts/cli.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_backtest_dir(tmp_path: Path) -> Path:
    """Create a sample backtest directory with trades and equity curve."""
    backtest_dir = tmp_path / "backtest_output"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Create trades DataFrame
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    trades_df = pd.DataFrame(
        {
            "timestamp": dates[:5],
            "symbol": ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"],
            "side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
            "qty": [10, 20, 10, 15, 20],
            "price": [100.0, 250.0, 105.0, 150.0, 255.0],
        }
    )
    trades_df.to_csv(backtest_dir / "trades.csv", index=False)

    # Create equity curve DataFrame
    equity_df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": 10000.0 * (1 + pd.Series(range(len(dates))) * 0.001),
        }
    )
    equity_df.to_csv(backtest_dir / "equity_curve.csv", index=False)

    return backtest_dir


@pytest.fixture
def sample_backtest_dir_no_equity(tmp_path: Path) -> Path:
    """Create a sample backtest directory with trades but no equity curve."""
    backtest_dir = tmp_path / "backtest_output_no_equity"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # Create trades DataFrame only
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    trades_df = pd.DataFrame(
        {
            "timestamp": dates[:5],
            "symbol": ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"],
            "side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
            "qty": [10, 20, 10, 15, 20],
            "price": [100.0, 250.0, 105.0, 150.0, 255.0],
        }
    )
    trades_df.to_csv(backtest_dir / "trades.csv", index=False)

    return backtest_dir


@pytest.mark.advanced
def test_tca_report_basic(sample_backtest_dir: Path, tmp_path: Path):
    """Test basic TCA report generation via direct function call."""
    from scripts.generate_tca_report import generate_tca_report

    output_dir = tmp_path / "tca_output"

    exit_code = generate_tca_report(
        backtest_dir=sample_backtest_dir,
        output_dir=output_dir,
        method="simple",
        commission_bps=0.5,
        spread_bps=5.0,
        slippage_bps=3.0,
    )

    assert exit_code == 0

    # Check that output files exist
    tca_trades_file = output_dir / "tca_trades.csv"
    tca_summary_file = output_dir / "tca_summary.csv"
    tca_report_file = output_dir / "tca_report.md"
    tca_risk_summary_file = output_dir / "tca_risk_summary.csv"

    assert tca_trades_file.exists(), f"TCA trades file not found: {tca_trades_file}"
    assert tca_summary_file.exists(), f"TCA summary file not found: {tca_summary_file}"
    assert tca_report_file.exists(), f"TCA report file not found: {tca_report_file}"

    # Check that risk summary exists (because equity curve was provided)
    assert tca_risk_summary_file.exists(), (
        f"TCA risk summary file not found: {tca_risk_summary_file}"
    )

    # Check that files are not empty
    assert tca_trades_file.stat().st_size > 0
    assert tca_summary_file.stat().st_size > 0
    assert tca_report_file.stat().st_size > 0

    # Check TCA trades structure
    tca_trades_df = pd.read_csv(tca_trades_file)
    assert "cost_total" in tca_trades_df.columns
    assert len(tca_trades_df) == 5  # 5 trades

    # Check TCA summary structure
    tca_summary_df = pd.read_csv(tca_summary_file)
    assert "total_cost" in tca_summary_df.columns
    assert "n_trades" in tca_summary_df.columns


@pytest.mark.advanced
def test_tca_report_without_equity_curve(
    sample_backtest_dir_no_equity: Path, tmp_path: Path
):
    """Test TCA report generation without equity curve (no risk summary)."""
    from scripts.generate_tca_report import generate_tca_report

    output_dir = tmp_path / "tca_output_no_equity"

    exit_code = generate_tca_report(
        backtest_dir=sample_backtest_dir_no_equity,
        output_dir=output_dir,
        method="simple",
    )

    assert exit_code == 0

    # Check that basic output files exist
    tca_trades_file = output_dir / "tca_trades.csv"
    tca_summary_file = output_dir / "tca_summary.csv"
    tca_report_file = output_dir / "tca_report.md"

    assert tca_trades_file.exists()
    assert tca_summary_file.exists()
    assert tca_report_file.exists()

    # Risk summary should not exist (no equity curve)
    tca_risk_summary_file = output_dir / "tca_risk_summary.csv"
    assert not tca_risk_summary_file.exists(), (
        "Risk summary should not exist without equity curve"
    )


@pytest.mark.advanced
def test_tca_report_cli_subcommand(sample_backtest_dir: Path, tmp_path: Path):
    """Test TCA report generation via CLI subcommand."""
    output_dir = tmp_path / "tca_output_cli"

    # Run CLI command
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "cli.py"),
            "tca_report",
            "--backtest-dir",
            str(sample_backtest_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Check exit code
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"

    # Check that output files exist
    tca_trades_file = output_dir / "tca_trades.csv"
    tca_summary_file = output_dir / "tca_summary.csv"
    tca_report_file = output_dir / "tca_report.md"

    assert tca_trades_file.exists(), f"TCA trades file not found: {tca_trades_file}"
    assert tca_summary_file.exists(), f"TCA summary file not found: {tca_summary_file}"
    assert tca_report_file.exists(), f"TCA report file not found: {tca_report_file}"


@pytest.mark.advanced
def test_tca_report_with_custom_cost_params(sample_backtest_dir: Path, tmp_path: Path):
    """Test TCA report with custom cost parameters."""
    from scripts.generate_tca_report import generate_tca_report

    output_dir = tmp_path / "tca_output_custom"

    exit_code = generate_tca_report(
        backtest_dir=sample_backtest_dir,
        output_dir=output_dir,
        method="simple",
        commission_bps=1.0,  # Custom commission
        spread_bps=10.0,  # Custom spread
        slippage_bps=5.0,  # Custom slippage
    )

    assert exit_code == 0

    # Check that costs are higher with larger parameters
    tca_trades_file = output_dir / "tca_trades.csv"
    tca_trades_df = pd.read_csv(tca_trades_file)

    total_cost_custom = tca_trades_df["cost_total"].sum()

    # Compare with default parameters
    output_dir_default = tmp_path / "tca_output_default"
    generate_tca_report(
        backtest_dir=sample_backtest_dir,
        output_dir=output_dir_default,
        method="simple",
        commission_bps=0.5,
        spread_bps=5.0,
        slippage_bps=3.0,
    )
    tca_trades_df_default = pd.read_csv(output_dir_default / "tca_trades.csv")
    total_cost_default = tca_trades_df_default["cost_total"].sum()

    # Custom costs should be higher
    assert total_cost_custom > total_cost_default, (
        "Higher cost parameters should result in higher total costs"
    )


@pytest.mark.advanced
def test_tca_report_markdown_content(sample_backtest_dir: Path, tmp_path: Path):
    """Test that TCA report Markdown contains expected content."""
    from scripts.generate_tca_report import generate_tca_report

    output_dir = tmp_path / "tca_output_md"

    generate_tca_report(
        backtest_dir=sample_backtest_dir,
        output_dir=output_dir,
    )

    report_file = output_dir / "tca_report.md"
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")

    # Check for key sections
    assert "# Transaction Cost Analysis (TCA) Report" in content
    assert "Summary Statistics" in content
    assert "Daily TCA Summary" in content
    assert "Output Files" in content

    # Check that numbers are present
    assert "Total Trades:" in content
    assert "Total Costs:" in content
