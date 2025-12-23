"""Tests for analyze_factors CLI workflow."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_price_data(tmp_path: Path) -> Path:
    """Create sample price data files for testing (one file per symbol).

    LocalParquetPriceDataSource expects: <local_data_root>/<freq>/<SYMBOL>.parquet
    """
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    # Create directory structure: <tmp_path>/1d/
    data_dir = tmp_path / "1d"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create one parquet file per symbol
    for symbol in symbols:
        base_price = (
            100.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 150.0)
        )
        data = []
        for i, date in enumerate(dates):
            price = base_price + i * 0.1 + (hash(symbol) % 10) * 0.01
            data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000000.0,
                }
            )

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Save to <tmp_path>/1d/<SYMBOL>.parquet
        symbol_file = data_dir / f"{symbol}.parquet"
        df.to_parquet(symbol_file, index=False)

    return tmp_path  # Return the root directory (local_data_root)


@pytest.fixture
def sample_universe_file(tmp_path: Path) -> Path:
    """Create sample universe file for testing."""
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n")
    return universe_file


def test_analyze_factors_cli_basic(
    tmp_path: Path, sample_price_data: Path, sample_universe_file: Path, monkeypatch
):
    """Test basic analyze_factors CLI functionality."""
    # Set up environment for local data source
    # sample_price_data is now the root directory (tmp_path)
    data_dir = sample_price_data

    monkeypatch.setenv("ASSEMBLED_LOCAL_DATA_ROOT", str(data_dir))
    monkeypatch.setenv("ASSEMBLED_DATA_SOURCE", "local")

    # Create output directory
    output_dir = tmp_path / "output" / "factor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyze_factors via subprocess
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "analyze_factors",
        "--freq",
        "1d",
        "--symbols-file",
        str(sample_universe_file),
        "--data-source",
        "local",
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2020-02-20",
        "--factor-set",
        "core",
        "--horizon-days",
        "5",
        "--quantiles",
        "5",
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Check exit code
    assert result.returncode == 0, (
        f"Command failed with exit code {result.returncode}. "
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    # Check that output files were created
    base_name = "factor_analysis_core_5d_1d"
    expected_files = [
        output_dir / f"{base_name}_report.md",
        output_dir / f"{base_name}_ic_summary.csv",
    ]

    # Portfolio summary may not exist if there's insufficient data
    portfolio_file = output_dir / f"{base_name}_portfolio_summary.csv"

    for expected_file in expected_files:
        assert expected_file.exists(), f"Expected file {expected_file} was not created"
        assert expected_file.stat().st_size > 0, f"File {expected_file} is empty"

    # Portfolio summary is optional (may be empty if insufficient data)
    if portfolio_file.exists():
        df = pd.read_csv(portfolio_file)
        assert not df.empty, (
            f"Portfolio summary CSV {portfolio_file} should not be empty if it exists"
        )


def test_analyze_factors_direct_call(
    tmp_path: Path, sample_price_data: Path, sample_universe_file: Path, monkeypatch
):
    """Test analyze_factors by calling the function directly."""
    from scripts.run_factor_analysis import run_factor_analysis_from_args
    from argparse import Namespace

    # Set up environment
    # sample_price_data is now the root directory (tmp_path)
    data_dir = sample_price_data
    monkeypatch.setenv("ASSEMBLED_LOCAL_DATA_ROOT", str(data_dir))
    monkeypatch.setenv("ASSEMBLED_DATA_SOURCE", "local")

    # Create output directory
    output_dir = tmp_path / "output" / "factor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create args namespace
    args = Namespace(
        freq="1d",
        symbols=None,
        symbols_file=sample_universe_file,
        universe=None,
        data_source="local",
        start_date="2020-01-01",
        end_date="2020-02-20",
        factor_set="core",
        horizon_days=5,
        quantiles=5,
        output_dir=str(output_dir),
    )

    # Run function
    exit_code = run_factor_analysis_from_args(args)

    assert exit_code == 0, "Function should return exit code 0"

    # Check that output files were created
    base_name = "factor_analysis_core_5d_1d"
    expected_files = [
        output_dir / f"{base_name}_report.md",
        output_dir / f"{base_name}_ic_summary.csv",
    ]

    # Portfolio summary may not exist if there's insufficient data
    portfolio_file = output_dir / f"{base_name}_portfolio_summary.csv"

    for expected_file in expected_files:
        assert expected_file.exists(), f"Expected file {expected_file} was not created"

    # Portfolio summary is optional (may be empty if insufficient data)
    if portfolio_file.exists():
        df = pd.read_csv(portfolio_file)
        assert not df.empty, (
            f"Portfolio summary CSV {portfolio_file} should not be empty if it exists"
        )


def test_analyze_factors_with_vol_liquidity(
    tmp_path: Path, sample_price_data: Path, sample_universe_file: Path, monkeypatch
):
    """Test analyze_factors with vol_liquidity factor set."""
    from scripts.run_factor_analysis import run_factor_analysis_from_args
    from argparse import Namespace

    # Set up environment
    # sample_price_data is now the root directory (tmp_path)
    data_dir = sample_price_data
    monkeypatch.setenv("ASSEMBLED_LOCAL_DATA_ROOT", str(data_dir))
    monkeypatch.setenv("ASSEMBLED_DATA_SOURCE", "local")

    # Create output directory
    output_dir = tmp_path / "output" / "factor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create args namespace
    args = Namespace(
        freq="1d",
        symbols=None,
        symbols_file=sample_universe_file,
        universe=None,
        data_source="local",
        start_date="2020-01-01",
        end_date="2020-02-20",
        factor_set="core+vol_liquidity",
        horizon_days=10,
        quantiles=5,
        output_dir=str(output_dir),
    )

    # Run function
    exit_code = run_factor_analysis_from_args(args)

    assert exit_code == 0, "Function should return exit code 0"

    # Check that output files were created
    base_name = "factor_analysis_core+vol_liquidity_10d_1d"
    expected_files = [
        output_dir / f"{base_name}_report.md",
        output_dir / f"{base_name}_ic_summary.csv",
    ]

    # Portfolio summary may not exist if there's insufficient data
    portfolio_file = output_dir / f"{base_name}_portfolio_summary.csv"

    for expected_file in expected_files:
        assert expected_file.exists(), f"Expected file {expected_file} was not created"

        # Check CSV content if it exists
        if expected_file.suffix == ".csv" and expected_file.exists():
            df = pd.read_csv(expected_file)
            assert not df.empty, f"CSV file {expected_file} should not be empty"

    # Portfolio summary is optional (may be empty if insufficient data)
    if portfolio_file.exists():
        df = pd.read_csv(portfolio_file)
        assert not df.empty, (
            f"Portfolio summary CSV {portfolio_file} should not be empty if it exists"
        )
