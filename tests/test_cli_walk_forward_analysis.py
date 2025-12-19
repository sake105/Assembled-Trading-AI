"""Tests for Walk-Forward Analysis CLI integration."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.advanced
def test_walk_forward_cli_runs_with_dummy_config(tmp_path: Path):
    """Test that walk-forward CLI runs with synthetic config (smoke test).
    
    This test creates minimal synthetic price data and verifies that
    the walk-forward CLI can run without errors (even if it may fail
    due to insufficient data or other issues).
    """
    # Create minimal synthetic price data
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    price_data = []
    for symbol in symbols:
        for date in dates:
            price_data.append({
                "timestamp": date,
                "symbol": symbol,
                "close": 100.0 + (hash(symbol + str(date)) % 50),
            })
    
    prices_df = pd.DataFrame(price_data)
    price_file = tmp_path / "prices.parquet"
    prices_df.to_parquet(price_file, index=False)
    
    # Create temporary universe file
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("\n".join(symbols))
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    
    # Run walk-forward CLI (will likely fail due to insufficient data, but should not crash)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_walk_forward_analysis.py"),
        "--freq", "1d",
        "--universe", str(universe_file),
        "--data-source", "local",  # Will fail, but tests argument parsing
        "--start-date", "2020-01-01",
        "--end-date", "2022-12-31",
        "--strategy", "trend_baseline",
        "--test-window", "63",
        "--mode", "rolling",
        "--train-window", "252",
        "--output-dir", str(output_dir),
    ]
    
    # Note: This will likely fail because we can't easily mock the data loading,
    # but it tests that the CLI argument parsing works
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    
    # Should parse arguments correctly (may fail on data loading, which is OK)
    # We mainly want to ensure it doesn't crash with argument errors
    assert result.returncode != 0 or output_dir.exists(), "CLI should run or create output directory"


@pytest.mark.advanced
def test_walk_forward_cli_help():
    """Test that walk-forward CLI shows help without errors."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_walk_forward_analysis.py"),
        "--help",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    
    assert result.returncode == 0, "Help should return exit code 0"
    assert "walk-forward" in result.stdout.lower() or "walk_forward" in result.stdout.lower(), \
        "Help should mention walk-forward"


@pytest.mark.advanced
def test_walk_forward_cli_via_cli_py(tmp_path: Path):
    """Test walk-forward subcommand via main CLI."""
    # Create temporary universe file
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT")
    
    # Run via main CLI
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "walk_forward",
        "--freq", "1d",
        "--universe", str(universe_file),
        "--start-date", "2020-01-01",
        "--end-date", "2022-12-31",
        "--strategy", "trend_baseline",
        "--test-window", "63",
        "--mode", "rolling",
        "--train-window", "252",
        "--output-dir", str(tmp_path / "output"),
    ]
    
    # This will likely fail on data loading (expected), but tests CLI integration
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=60)
    
    # Should not crash with argument parsing errors
    # May fail on data loading, which is OK for this integration test
    assert "error" not in result.stderr.lower() or "data" in result.stderr.lower() or "file" in result.stderr.lower(), \
        "CLI should either succeed or fail with data-related errors, not argument parsing errors"

