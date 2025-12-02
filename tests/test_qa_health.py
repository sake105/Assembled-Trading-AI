# tests/test_qa_health.py
"""Tests for QA health check functions."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.health import (
    aggregate_qa_status,
    check_orders,
    check_portfolio,
    check_prices,
)


def test_check_prices_file_not_found(tmp_path: Path):
    """Test check_prices when file does not exist."""
    result = check_prices("1d", output_dir=tmp_path)
    
    assert result.name == "prices"
    assert result.status == "error"
    assert "not found" in result.message.lower()
    assert result.details is not None
    assert "file" in result.details


def test_check_prices_empty_file(tmp_path: Path):
    """Test check_prices when file is empty."""
    # Create empty parquet file
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty DataFrame with correct schema
    df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    df.to_parquet(price_file)
    
    result = check_prices("1d", output_dir=tmp_path)
    
    assert result.name == "prices"
    assert result.status == "warning"
    assert "empty" in result.message.lower()


def test_check_prices_missing_columns(tmp_path: Path):
    """Test check_prices when required columns are missing."""
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with wrong columns
    df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    df.to_parquet(price_file)
    
    result = check_prices("1d", output_dir=tmp_path)
    
    assert result.name == "prices"
    assert result.status == "error"
    assert "missing" in result.message.lower() or "required columns" in result.message.lower()


def test_check_prices_ok(tmp_path: Path):
    """Test check_prices with valid file."""
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create valid DataFrame
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 3, utc=True),
        "symbol": ["AAPL", "MSFT", "AAPL"],
        "close": [100.0, 200.0, 101.0]
    }
    df = pd.DataFrame(data)
    df.to_parquet(price_file)
    
    result = check_prices("1d", output_dir=tmp_path)
    
    assert result.name == "prices"
    assert result.status == "ok"
    assert result.details is not None
    assert result.details["rows"] == 3
    assert result.details["symbols"] == 2


def test_check_orders_file_not_found(tmp_path: Path):
    """Test check_orders when file does not exist."""
    result = check_orders("1d", output_dir=tmp_path)
    
    assert result.name == "orders"
    assert result.status == "error"
    assert "not found" in result.message.lower()


def test_check_orders_empty_file(tmp_path: Path):
    """Test check_orders when file is empty."""
    orders_file = tmp_path / "orders_1d.csv"
    orders_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty CSV with correct schema
    df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    df.to_csv(orders_file, index=False)
    
    result = check_orders("1d", output_dir=tmp_path)
    
    assert result.name == "orders"
    assert result.status == "warning"
    assert "empty" in result.message.lower()


def test_check_orders_ok(tmp_path: Path):
    """Test check_orders with valid file."""
    orders_file = tmp_path / "orders_1d.csv"
    orders_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create valid DataFrame
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [1.0, 1.0],
        "price": [100.0, 200.0]
    }
    df = pd.DataFrame(data)
    df.to_csv(orders_file, index=False)
    
    result = check_orders("1d", output_dir=tmp_path)
    
    assert result.name == "orders"
    assert result.status == "ok"
    assert result.details is not None
    assert result.details["rows"] == 2


def test_check_portfolio_file_not_found(tmp_path: Path):
    """Test check_portfolio when file does not exist."""
    result = check_portfolio("1d", output_dir=tmp_path)
    
    assert result.name == "portfolio"
    assert result.status == "error"
    assert "not found" in result.message.lower()


def test_check_portfolio_too_few_rows(tmp_path: Path):
    """Test check_portfolio when file has too few rows."""
    portfolio_file = tmp_path / "portfolio_equity_1d.csv"
    portfolio_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with only 2 rows (minimum is 5)
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True),
        "equity": [10000.0, 10050.0]
    }
    df = pd.DataFrame(data)
    # Ensure timestamp is written as string in CSV
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(portfolio_file, index=False)
    
    result = check_portfolio("1d", output_dir=tmp_path)
    
    assert result.name == "portfolio"
    assert result.status == "warning"
    assert "too few rows" in result.message.lower() or "few" in result.message.lower()


def test_check_portfolio_with_nans(tmp_path: Path):
    """Test check_portfolio when equity column has NaNs."""
    portfolio_file = tmp_path / "portfolio_equity_1d.csv"
    portfolio_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with NaNs
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 10, utc=True),
        "equity": [10000.0] * 5 + [None] * 5  # 5 NaNs
    }
    df = pd.DataFrame(data)
    df.to_csv(portfolio_file, index=False)
    
    result = check_portfolio("1d", output_dir=tmp_path)
    
    assert result.name == "portfolio"
    assert result.status == "error"
    assert "nan" in result.message.lower()


def test_check_portfolio_ok(tmp_path: Path):
    """Test check_portfolio with valid file."""
    portfolio_file = tmp_path / "portfolio_equity_1d.csv"
    portfolio_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create valid DataFrame with enough rows
    from datetime import datetime
    data = {
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + pd.Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    }
    df = pd.DataFrame(data)
    df.to_csv(portfolio_file, index=False)
    
    result = check_portfolio("1d", output_dir=tmp_path)
    
    assert result.name == "portfolio"
    assert result.status == "ok"
    assert result.details is not None
    assert result.details["rows"] == 10
    assert result.details["equity_min"] > 0
    assert result.details["equity_max"] > 0


def test_aggregate_qa_status_no_files(tmp_path: Path):
    """Test aggregate_qa_status when no files exist (should return error)."""
    result = aggregate_qa_status("1d", output_dir=tmp_path)
    
    assert result["freq"] == "1d"
    assert result["overall_status"] == "error"
    assert len(result["checks"]) == 3
    
    # All checks should be errors
    for check in result["checks"]:
        assert check["status"] == "error"


def test_aggregate_qa_status_all_ok(tmp_path: Path):
    """Test aggregate_qa_status with all valid files (should return ok)."""
    # Create all required files
    from datetime import datetime
    
    # Prices
    price_file = tmp_path / "aggregates" / "daily.parquet"
    price_file.parent.mkdir(parents=True, exist_ok=True)
    prices_df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0)] * 3, utc=True),
        "symbol": ["AAPL", "MSFT", "AAPL"],
        "close": [100.0, 200.0, 101.0]
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
    orders_df.to_csv(orders_file, index=False)
    
    # Portfolio
    portfolio_file = tmp_path / "portfolio_equity_1d.csv"
    portfolio_df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2025, 11, 28, 14, 0, 0) + pd.Timedelta(days=i) for i in range(10)], utc=True),
        "equity": [10000.0 + i * 10.0 for i in range(10)]
    })
    # Ensure timestamp is written as string in CSV
    portfolio_df["timestamp"] = portfolio_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    portfolio_df.to_csv(portfolio_file, index=False)
    
    result = aggregate_qa_status("1d", output_dir=tmp_path)
    
    assert result["freq"] == "1d"
    assert result["overall_status"] == "ok"
    assert len(result["checks"]) == 3
    
    # All checks should be ok
    for check in result["checks"]:
        assert check["status"] == "ok", f"Check {check['name']} should be ok but got {check['status']}"

