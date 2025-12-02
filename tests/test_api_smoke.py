# tests/test_api_smoke.py
"""Smoke tests for FastAPI endpoints."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import pandas Timedelta for test data
from pandas import Timedelta

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_orders_file(tmp_path: Path):
    """Create a sample orders file for testing."""
    # Override OUTPUT_DIR for this test
    orders_file = tmp_path / "orders_1d.csv"
    orders_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create sample orders
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [1.0, 1.0],
        "price": [100.0, 200.0]
    }
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(orders_file, index=False)
    
    return tmp_path


def test_orders_endpoint_with_sample_file(client: TestClient, sample_orders_file: Path, monkeypatch):
    """Test GET /api/v1/orders/{freq} with sample file."""
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", sample_orders_file)
    
    try:
        response = client.get("/api/v1/orders/1d")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frequency" in data
        assert "orders" in data
        assert "count" in data
        assert "total_notional" in data
        
        assert data["frequency"] == "1d"
        assert data["count"] == 2
        assert len(data["orders"]) == 2
        assert data["total_notional"] == 300.0  # 100 + 200
        
        # Check order structure
        order = data["orders"][0]
        assert "timestamp" in order
        assert "symbol" in order
        assert "side" in order
        assert "qty" in order
        assert "price" in order
        assert "notional" in order
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_orders_endpoint_file_not_found(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/orders/{freq} when file doesn't exist."""
    # Temporarily override OUTPUT_DIR to empty directory
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/orders/1d")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_performance_backtest_curve_endpoint_with_sample_file(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/performance/{freq}/backtest-curve with sample file."""
    # Create sample equity curve
    curve_file = tmp_path / "equity_curve_1d.csv"
    curve_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    }
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(curve_file, index=False)
    
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/performance/1d/backtest-curve")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frequency" in data
        assert "points" in data
        assert "count" in data
        assert "start_equity" in data
        assert "end_equity" in data
        
        assert data["frequency"] == "1d"
        assert data["count"] == 10
        assert len(data["points"]) == 10
        assert data["start_equity"] == 10000.0
        assert data["end_equity"] == 10090.0
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_performance_metrics_endpoint_with_sample_file(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/performance/{freq}/metrics with sample file."""
    # Create sample equity curve
    curve_file = tmp_path / "equity_curve_1d.csv"
    curve_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    }
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(curve_file, index=False)
    
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/performance/1d/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "freq" in data
        assert "final_pf" in data
        assert "sharpe" in data
        assert "rows" in data
        assert "first" in data
        assert "last" in data
        
        assert data["freq"] == "1d"
        assert data["rows"] == 10
        assert data["final_pf"] > 0
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_signals_1d_empty_or_missing(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/signals/1d when price file doesn't exist."""
    # Temporarily override OUTPUT_DIR to empty directory
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/signals/1d")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower() or "empty" in data["detail"].lower()
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_signals_1d_ok(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/signals/1d with valid price file."""
    # Create sample price file
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create valid DataFrame with enough rows for EMA calculation (need at least slow period)
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(100)], utc=True),
        "symbol": ["AAPL"] * 50 + ["MSFT"] * 50,
        "close": [100.0 + i * 0.1 for i in range(50)] + [200.0 + i * 0.1 for i in range(50)]
    }
    df = pd.DataFrame(data)
    df.to_parquet(price_file)
    
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/signals/1d")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "frequency" in data
        assert "signals" in data
        assert "count" in data
        assert "first_timestamp" in data
        assert "last_timestamp" in data
        
        assert data["frequency"] == "1d"
        assert data["count"] > 0, "Should have at least one signal"
        assert len(data["signals"]) > 0, "Should have at least one signal in list"
        
        # Check signal structure
        signal = data["signals"][0]
        assert "timestamp" in signal
        assert "symbol" in signal
        assert "signal_type" in signal
        assert "price" in signal
        assert signal["signal_type"] in ["BUY", "SELL", "NEUTRAL"]
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_portfolio_current_missing_file(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/portfolio/{freq}/current when file doesn't exist."""
    # Temporarily override OUTPUT_DIR to empty directory
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/portfolio/1d/current")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_portfolio_current_ok(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/portfolio/{freq}/current with valid file."""
    # Create sample portfolio equity file
    equity_file = tmp_path / "portfolio_equity_1d.csv"
    equity_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create valid DataFrame
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    }
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(equity_file, index=False)
    
    # Create portfolio report (freq-specific)
    report_file = tmp_path / "portfolio_report_1d.md"
    report_file.write_text(
        "# Portfolio Report (1d)\n\n"
        "- Final PF: 1.0007\n"
        "- Sharpe: 0.1566\n"
        "- Trades: 2\n",
        encoding="utf-8"
    )
    
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/portfolio/1d/current")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "equity" in data
        assert "cash" in data
        assert "positions" in data
        assert "pf" in data or "performance_factor" in data
        assert "sharpe" in data
        assert "total_trades" in data
        assert "start_capital" in data
        
        assert data["equity"] == 10090.0  # Last equity value
        assert data["total_trades"] == 2
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_qa_status_no_files(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/qa/status when no files exist (should return error status)."""
    # Temporarily override OUTPUT_DIR to empty directory
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/qa/status?freq=1d")
        
        # Should return 200 even if status is error
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "timestamp" in data
        assert "checks" in data
        assert "summary" in data
        
        assert data["overall_status"] == "error"
        assert len(data["checks"]) == 3  # prices, orders, portfolio
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_get_qa_status_ok(client: TestClient, tmp_path: Path, monkeypatch):
    """Test GET /api/v1/qa/status with valid files (should return ok status)."""
    from datetime import datetime
    
    # Create minimal valid files
    # Prices
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    prices_df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(10)], utc=True),
        "symbol": ["AAPL"] * 10,
        "close": [100.0 + i * 0.1 for i in range(10)]
    })
    prices_df.to_parquet(price_file)
    
    # Orders
    orders_file = tmp_path / "orders_1d.csv"
    orders_df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [1.0, 1.0],
        "price": [100.0, 200.0]
    })
    orders_df["timestamp"] = orders_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    orders_df.to_csv(orders_file, index=False)
    
    # Portfolio equity
    portfolio_file = tmp_path / "portfolio_equity_1d.csv"
    portfolio_df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    })
    portfolio_df["timestamp"] = portfolio_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    portfolio_df.to_csv(portfolio_file, index=False)
    
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module
    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)
    
    try:
        response = client.get("/api/v1/qa/status?freq=1d")
        
        # Should return 200
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "timestamp" in data
        assert "checks" in data
        assert "summary" in data
        
        assert data["overall_status"] == "ok"
        assert len(data["checks"]) == 3  # prices, orders, portfolio
        
        # All checks should be ok
        for check in data["checks"]:
            assert check["status"] == "ok", f"Check {check['check_name']} should be ok but got {check['status']}"
        
        # Summary should reflect ok status
        assert data["summary"]["ok"] == 3
        assert data["summary"]["error"] == 0
        assert data["summary"]["warning"] == 0
    
    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)

