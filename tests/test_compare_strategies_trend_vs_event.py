"""Tests for compare_strategies_trend_vs_event.py CLI script."""
from __future__ import annotations

import shutil
import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.phase6


@pytest.fixture
def synthetic_prices(tmp_path: Path) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            # Simple upward trend with some noise
            base_price = 100.0 + (i * 0.1) + (i % 10) * 0.5
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "open": base_price * 0.99,
                "high": base_price * 1.02,
                "low": base_price * 0.98,
                "close": base_price,
                "volume": 1000000.0
            })
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_price_file(tmp_path: Path, synthetic_prices: pd.DataFrame) -> Path:
    """Create a sample price file in parquet format.
    
    First tries to copy data/sample/eod_sample.parquet if it exists,
    otherwise creates synthetic data.
    """
    price_file = tmp_path / "eod_sample.parquet"
    
    # Try to copy existing sample file first
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"
    if sample_file.exists():
        shutil.copy(sample_file, price_file)
    else:
        # Fallback to synthetic data
        synthetic_prices.to_parquet(price_file)
    
    return price_file


def test_compare_script_runs_and_creates_summary(sample_price_file: Path, tmp_path: Path):
    """Test that compare_strategies script runs and creates summary files."""
    script_path = ROOT / "scripts" / "compare_strategies_trend_vs_event.py"
    
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--freq", "1d",
            "--price-file", str(sample_price_file),
            "--out", str(tmp_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120  # May take longer for two backtests
    )
    
    # Should succeed
    assert result.returncode == 0, (
        f"Script failed with exit code {result.returncode}. "
        f"Output: {result.stdout}\nError: {result.stderr}"
    )
    
    # Check that comparison_summary.md exists
    summary_md = tmp_path / "comparison_summary.md"
    assert summary_md.exists(), f"comparison_summary.md should exist in {tmp_path}"
    
    # Check file content
    content = summary_md.read_text(encoding="utf-8")
    assert "trend baseline" in content.lower(), "Summary should mention trend baseline"
    assert "event insider shipping" in content.lower(), "Summary should mention event insider shipping"
    
    # Check that comparison_summary.csv exists
    summary_csv = tmp_path / "comparison_summary.csv"
    assert summary_csv.exists(), f"comparison_summary.csv should exist in {tmp_path}"
    
    # Check CSV content
    df = pd.read_csv(summary_csv)
    assert "strategy" in df.columns, "CSV should have 'strategy' column"
    assert len(df) == 2, "CSV should have 2 rows (one for each strategy)"
    assert "trend_baseline" in df["strategy"].values, "CSV should contain trend_baseline"
    assert "event_insider_shipping" in df["strategy"].values, "CSV should contain event_insider_shipping"


def test_compare_script_with_no_costs(sample_price_file: Path, tmp_path: Path):
    """Test that compare_strategies script works with --no-costs flag."""
    script_path = ROOT / "scripts" / "compare_strategies_trend_vs_event.py"
    
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--freq", "1d",
            "--price-file", str(sample_price_file),
            "--out", str(tmp_path),
            "--no-costs",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120
    )
    
    # Should succeed
    assert result.returncode == 0, (
        f"Script failed with exit code {result.returncode}. "
        f"Output: {result.stdout}\nError: {result.stderr}"
    )
    
    # Check that files were created
    summary_md = tmp_path / "comparison_summary.md"
    assert summary_md.exists(), "comparison_summary.md should exist"
    
    summary_csv = tmp_path / "comparison_summary.csv"
    assert summary_csv.exists(), "comparison_summary.csv should exist"

