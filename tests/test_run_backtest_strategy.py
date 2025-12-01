# tests/test_run_backtest_strategy.py
"""Tests for run_backtest_strategy.py CLI script."""
from __future__ import annotations

import sys
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pandas import Timedelta

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR

pytestmark = pytest.mark.phase4


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
    """Create a sample price file in parquet format."""
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    synthetic_prices.to_parquet(price_file)
    
    return price_file


@pytest.fixture
def sample_universe_file(tmp_path: Path) -> Path:
    """Create a sample universe file."""
    universe_file = tmp_path / "watchlist.txt"
    universe_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
    return universe_file


def test_run_backtest_strategy_smoke(tmp_path: Path, sample_price_file: Path, monkeypatch):
    """Test basic backtest run that completes successfully."""
    import src.assembled_core.config as config_module
    
    # Override OUTPUT_DIR
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        # Create reports directory
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        
        # Run script
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(sample_price_file),
                "--start-capital", "10000",
                "--out", str(tmp_path),
                "--generate-report"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should succeed
        assert result.returncode == 0, f"Script failed with exit code {result.returncode}. Output: {result.stdout}\nError: {result.stderr}"
        
        # Check that report was generated
        report_files = list((tmp_path / "reports").glob("qa_report_trend_baseline_1d_*.md"))
        assert len(report_files) > 0, "QA report should be generated"
        
        # Check report content
        report_content = report_files[0].read_text(encoding="utf-8")
        assert "Performance Metrics" in report_content or "QA Report" in report_content
        assert "trend_baseline" in report_content or "1d" in report_content
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_backtest_strategy_missing_prices(tmp_path: Path, monkeypatch):
    """Test that script exits with code 1 when price file is missing."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(tmp_path / "nonexistent.parquet")
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail with exit code 1
        assert result.returncode == 1, f"Script should fail but got exit code {result.returncode}"
        assert "not found" in result.stderr.lower() or "FileNotFoundError" in result.stderr or "File not found" in result.stdout
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_backtest_strategy_invalid_freq(tmp_path: Path, monkeypatch):
    """Test that script exits with code 1 for invalid frequency."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "invalid"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail (argparse should reject invalid choice)
        assert result.returncode != 0, "Script should fail for invalid frequency"
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_backtest_strategy_with_universe(tmp_path: Path, sample_price_file: Path, sample_universe_file: Path, monkeypatch):
    """Test backtest with universe file."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    # Also need to update the base dir for watchlist lookup
    import src.assembled_core.config as config_module
    original_base_dir = getattr(config_module, "_BASE_DIR_CACHE", None)
    
    try:
        # Create aggregates directory with daily.parquet
        (tmp_path / "aggregates").mkdir(parents=True, exist_ok=True)
        
        # Copy price file to expected location (only if different)
        target_price_file = tmp_path / "aggregates" / "daily.parquet"
        import shutil
        if sample_price_file.resolve() != target_price_file.resolve():
            shutil.copy(sample_price_file, target_price_file)
        
        # Create watchlist in tmp_path (will be used as base)
        (tmp_path / "watchlist.txt").write_text("AAPL\nMSFT\n", encoding="utf-8")
        
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--universe", str(tmp_path / "watchlist.txt"),
                "--out", str(tmp_path),
                "--start-capital", "10000"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should succeed
        assert result.returncode == 0, f"Script failed. Output: {result.stdout}\nError: {result.stderr}"
        assert "Backtest completed" in result.stdout or "Final PF" in result.stdout
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_backtest_strategy_no_costs(tmp_path: Path, sample_price_file: Path, monkeypatch):
    """Test backtest without transaction costs."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(sample_price_file),
                "--no-costs",
                "--start-capital", "10000"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"Script failed. Output: {result.stdout}\nError: {result.stderr}"
        assert "With Costs: False" in result.stdout
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_backtest_strategy_custom_costs(tmp_path: Path, sample_price_file: Path, monkeypatch):
    """Test backtest with custom cost parameters."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(sample_price_file),
                "--commission-bps", "1.0",
                "--spread-w", "0.3",
                "--impact-w", "0.6",
                "--start-capital", "10000"
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"Script failed. Output: {result.stdout}\nError: {result.stderr}"
        assert "commission_bps=1.0" in result.stdout or "Cost Model" in result.stdout
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)

