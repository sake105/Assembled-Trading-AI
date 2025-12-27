"""Tests for metrics.json export functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.reports.metrics_export import export_metrics_json


def create_sample_equity_data() -> pd.DataFrame:
    """Create sample equity data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    equity_values = 10000.0 * (1.0 + (dates.day % 10) / 100.0)  # Simple pattern
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values,
    })


def create_sample_trades_data() -> pd.DataFrame:
    """Create sample trades data for testing."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    
    trades = []
    for i, date in enumerate(dates):
        trades.append({
            "timestamp": date,
            "symbol": "AAPL" if i % 2 == 0 else "MSFT",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 10.0 + i,
            "price": 100.0 + i * 0.5,
        })
    
    return pd.DataFrame(trades)


def test_export_metrics_json_basic(tmp_path: Path):
    """Test that metrics.json is written with correct structure."""
    # Create sample equity data
    equity = create_sample_equity_data()
    
    # Compute metrics
    metrics = compute_all_metrics(
        equity=equity,
        trades=None,
        start_capital=10000.0,
        freq="1d",
    )
    
    # Export metrics
    output_path = tmp_path / "metrics.json"
    export_metrics_json(metrics, output_path)
    
    # Verify file exists
    assert output_path.exists(), "metrics.json should be created"
    
    # Verify file content
    with output_path.open("r", encoding="utf-8") as f:
        metrics_dict = json.load(f)
    
    # Check that required keys are present
    required_keys = [
        "final_pf",
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sharpe",  # Alias
        "max_drawdown",
        "max_drawdown_pct",
        "start_date",
        "end_date",
        "periods",
        "start_capital",
        "end_equity",
    ]
    
    for key in required_keys:
        assert key in metrics_dict, f"Key '{key}' should be present in metrics.json"
    
    # Check that values are normalized (no NaN/inf)
    for key, value in metrics_dict.items():
        if value is not None:
            assert not (isinstance(value, float) and (pd.isna(value) or pd.isinf(value))), (
                f"Value for '{key}' should not be NaN or inf: {value}"
            )
    
    # Check that sharpe alias exists
    assert "sharpe" in metrics_dict, "Alias 'sharpe' should be present"
    assert metrics_dict["sharpe"] == metrics_dict.get("sharpe_ratio"), (
        "Alias 'sharpe' should match 'sharpe_ratio'"
    )


def test_export_metrics_json_with_trades(tmp_path: Path):
    """Test that metrics.json includes trade metrics when trades are provided."""
    # Create sample equity and trades data
    equity = create_sample_equity_data()
    trades = create_sample_trades_data()
    
    # Compute metrics
    metrics = compute_all_metrics(
        equity=equity,
        trades=trades,
        start_capital=10000.0,
        freq="1d",
    )
    
    # Export metrics
    output_path = tmp_path / "metrics.json"
    export_metrics_json(metrics, output_path)
    
    # Verify file content
    with output_path.open("r", encoding="utf-8") as f:
        metrics_dict = json.load(f)
    
    # Check that trade-related keys are present
    trade_keys = ["total_trades", "trades", "hit_rate", "profit_factor", "turnover"]
    
    for key in trade_keys:
        assert key in metrics_dict, f"Trade key '{key}' should be present in metrics.json"
    
    # Check that trades alias exists
    assert "trades" in metrics_dict, "Alias 'trades' should be present"
    assert metrics_dict["trades"] == metrics_dict.get("total_trades"), (
        "Alias 'trades' should match 'total_trades'"
    )
    
    # Check that total_trades is an integer or None
    if metrics_dict["total_trades"] is not None:
        assert isinstance(metrics_dict["total_trades"], int), (
            "total_trades should be an integer or None"
        )


def test_export_metrics_json_deterministic_keys(tmp_path: Path):
    """Test that metrics.json has deterministic key ordering."""
    # Create sample equity data
    equity = create_sample_equity_data()
    
    # Compute metrics
    metrics = compute_all_metrics(
        equity=equity,
        trades=None,
        start_capital=10000.0,
        freq="1d",
    )
    
    # Export metrics twice
    output_path1 = tmp_path / "metrics1.json"
    output_path2 = tmp_path / "metrics2.json"
    
    export_metrics_json(metrics, output_path1)
    export_metrics_json(metrics, output_path2)
    
    # Read both files
    with output_path1.open("r", encoding="utf-8") as f1, output_path2.open("r", encoding="utf-8") as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    # Files should be identical (deterministic key ordering)
    assert content1 == content2, "metrics.json should have deterministic key ordering"


def test_export_metrics_json_normalizes_nan_inf(tmp_path: Path):
    """Test that NaN and inf values are normalized to null."""
    # Create metrics with potentially NaN values (using a very short equity curve)
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
        "equity": [10000.0, 10100.0, 10050.0, 10150.0, 10200.0],
    })
    
    # Compute metrics (CAGR might be None for short series)
    metrics = compute_all_metrics(
        equity=equity,
        trades=None,
        start_capital=10000.0,
        freq="1d",
    )
    
    # Export metrics
    output_path = tmp_path / "metrics.json"
    export_metrics_json(metrics, output_path)
    
    # Verify file content (should not contain NaN or inf)
    with output_path.open("r", encoding="utf-8") as f:
        content = f.read()
        metrics_dict = json.load(f)
    
    # Check that JSON doesn't contain "NaN" or "Infinity" strings
    assert "NaN" not in content, "JSON should not contain 'NaN' string"
    assert "Infinity" not in content, "JSON should not contain 'Infinity' string"
    assert "-Infinity" not in content, "JSON should not contain '-Infinity' string"
    
    # Check that None values are represented as null in JSON
    for key, value in metrics_dict.items():
        if value is None:
            assert key in metrics_dict, f"Key '{key}' should be present (even if None)"
        elif isinstance(value, float):
            assert not (pd.isna(value) or pd.isinf(value)), (
                f"Value for '{key}' should not be NaN or inf: {value}"
            )


def test_export_metrics_json_timestamp_format(tmp_path: Path):
    """Test that timestamps are exported as ISO 8601 strings."""
    # Create sample equity data
    equity = create_sample_equity_data()
    
    # Compute metrics
    metrics = compute_all_metrics(
        equity=equity,
        trades=None,
        start_capital=10000.0,
        freq="1d",
    )
    
    # Export metrics
    output_path = tmp_path / "metrics.json"
    export_metrics_json(metrics, output_path)
    
    # Verify file content
    with output_path.open("r", encoding="utf-8") as f:
        metrics_dict = json.load(f)
    
    # Check that timestamps are ISO 8601 strings
    assert "start_date" in metrics_dict, "start_date should be present"
    assert "end_date" in metrics_dict, "end_date should be present"
    
    if metrics_dict["start_date"] is not None:
        assert isinstance(metrics_dict["start_date"], str), "start_date should be a string"
        # Should be ISO 8601 format (contains 'T' or has date format)
        assert "T" in metrics_dict["start_date"] or len(metrics_dict["start_date"]) >= 10, (
            "start_date should be ISO 8601 format"
        )
    
    if metrics_dict["end_date"] is not None:
        assert isinstance(metrics_dict["end_date"], str), "end_date should be a string"
        assert "T" in metrics_dict["end_date"] or len(metrics_dict["end_date"]) >= 10, (
            "end_date should be ISO 8601 format"
        )

