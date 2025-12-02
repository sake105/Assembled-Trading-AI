# tests/test_run_backtest_strategy.py
"""Tests for run_backtest_strategy.py CLI script."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    _ = getattr(config_module, "_BASE_DIR_CACHE", None)  # original_base_dir
    
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


@pytest.mark.phase6
def test_run_backtest_event_insider_shipping(tmp_path: Path, sample_price_file: Path, monkeypatch):
    """Test backtest with event_insider_shipping strategy."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        # Create reports directory
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        
        # Run script with event_insider_shipping strategy
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(sample_price_file),
                "--strategy", "event_insider_shipping",
                "--start-capital", "10000",
                "--out", str(tmp_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should succeed
        assert result.returncode == 0, (
            f"Script failed with exit code {result.returncode}. "
            f"Output: {result.stdout}\nError: {result.stderr}"
        )
        
        # Check that event strategy was used
        assert "event_insider_shipping" in result.stdout.lower() or "Event Strategy" in result.stdout
        assert "Backtest completed" in result.stdout or "Final PF" in result.stdout
        
        # Verify that event features were loaded/used (should appear in logs)
        assert "insider" in result.stdout.lower() or "shipping" in result.stdout.lower() or "event" in result.stdout.lower()
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


@pytest.mark.phase6
def test_event_strategy_generates_trades_with_sample_events(tmp_path: Path, sample_price_file: Path, monkeypatch):
    """Test that event strategy generates trades when sample events are available."""
    import src.assembled_core.config as config_module
    
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        # Create event data directory and generate sample events
        event_dir = ROOT / "data" / "sample" / "events"
        event_dir.mkdir(parents=True, exist_ok=True)
        
        # Read price data to get symbols and dates
        prices = pd.read_parquet(sample_price_file)
        if prices["timestamp"].dtype != "datetime64[ns, UTC]":
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        
        symbols = prices["symbol"].unique()
        dates = prices["timestamp"].sort_values().unique()
        first_symbol = symbols[0]
        event_dates = dates[len(dates) // 2 : len(dates) // 2 + 5]
        
        # Generate insider events
        from src.assembled_core.data.insider_ingest import normalize_insider
        insider_raw = pd.DataFrame({
            "timestamp": event_dates,
            "symbol": [first_symbol] * len(event_dates),
            "trades_count": [5, 3, 4, 6, 2],
            "net_shares": [2000, -3000, 1500, -2500, 3000],  # Mix of buy/sell
            "role": ["CEO", "CFO", "Director", "CEO", "Director"],
        })
        insider_events = normalize_insider(insider_raw)
        insider_path = event_dir / "insider_sample.parquet"
        insider_events.to_parquet(insider_path, index=False)
        
        # Generate shipping events
        from src.assembled_core.data.shipping_routes_ingest import normalize_shipping
        shipping_raw = pd.DataFrame({
            "timestamp": event_dates,
            "route_id": [f"R{i:03d}" for i in range(len(event_dates))],
            "port_from": ["LAX"] * len(event_dates),
            "port_to": ["SHG"] * len(event_dates),
            "symbol": [first_symbol] * len(event_dates),
            "ships": [10, 20, 5, 30, 15],
            "congestion_score": [20, 80, 40, 90, 50],  # Low = bullish, High = bearish
        })
        shipping_events = normalize_shipping(shipping_raw)
        shipping_path = event_dir / "shipping_sample.parquet"
        shipping_events.to_parquet(shipping_path, index=False)
        
        # Create reports directory
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        
        # Run backtest with event strategy
        script_path = ROOT / "scripts" / "run_backtest_strategy.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--freq", "1d",
                "--price-file", str(sample_price_file),
                "--strategy", "event_insider_shipping",
                "--start-capital", "10000",
                "--out", str(tmp_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should succeed
        assert result.returncode == 0, (
            f"Script failed with exit code {result.returncode}. "
            f"Output: {result.stdout}\nError: {result.stderr}"
        )
        
        # Check that sample events were used
        assert "Using sample insider events" in result.stdout or "Using sample shipping events" in result.stdout
        
        # Check that trades were generated (Total Trades > 0)
        assert "Total Trades:" in result.stdout
        # Extract trade count from output
        import re
        trades_match = re.search(r"Total Trades:\s*(\d+)", result.stdout)
        if trades_match:
            trade_count = int(trades_match.group(1))
            assert trade_count > 0, f"Expected at least 1 trade, but got {trade_count}"
        else:
            # Fallback: check for "Backtest completed" which indicates successful run
            assert "Backtest completed" in result.stdout
    
    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)

