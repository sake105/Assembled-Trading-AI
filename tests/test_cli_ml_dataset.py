"""Tests for CLI ML dataset builder subcommand."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.phase7
def test_cli_build_ml_dataset_basic(tmp_path: Path):
    """Test that build_ml_dataset CLI command works with sample data."""
    # Check if sample data exists
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"
    if not sample_file.exists():
        pytest.skip(f"Sample data file not found: {sample_file}")

    # Output path in temp directory
    output_path = tmp_path / "test_ml_dataset.parquet"

    # Run CLI command
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "build_ml_dataset",
        "--strategy",
        "trend_baseline",
        "--freq",
        "1d",
        "--price-file",
        str(sample_file),
        "--out",
        str(output_path),
        "--label-horizon-days",
        "10",
        "--success-threshold",
        "0.02",
    ]

    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )

    # Check exit code
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

    # Check that output file exists
    assert output_path.exists(), f"Output file not created: {output_path}"

    # Load and verify dataset
    df = pd.read_parquet(output_path)

    assert not df.empty, "ML dataset is empty"
    assert "label" in df.columns, "Missing 'label' column"
    assert df["label"].dtype in [int, "int64", "int32"], (
        "Label column should be integer"
    )
    assert set(df["label"].unique()).issubset({0, 1}), (
        "Label should only contain 0 or 1"
    )

    # Check for required metadata columns
    assert "symbol" in df.columns or "open_time" in df.columns, (
        "Missing metadata columns"
    )

    # Check for at least some feature columns
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "label",
            "symbol",
            "open_time",
            "open_price",
            "close_time",
            "pnl_pct",
            "horizon_days",
        ]
    ]
    assert len(feature_cols) > 0, "No feature columns found"


@pytest.mark.phase7
def test_cli_build_ml_dataset_help():
    """Test that build_ml_dataset --help works."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "build_ml_dataset",
        "--help",
    ]

    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=10
    )

    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert (
        "build_ml_dataset" in result.stdout or "Build ML-ready dataset" in result.stdout
    )
    assert "--strategy" in result.stdout
    assert "--freq" in result.stdout
    assert "--label-horizon-days" in result.stdout
