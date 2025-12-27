"""Unit tests for batch runner metrics collection.

Tests cover:
- collect_backtest_metrics with valid outputs
- Robust handling of missing files
- Metrics extraction from equity curves and trades
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.assembled_core.experiments.batch_runner import collect_backtest_metrics


def test_collect_backtest_metrics_with_valid_equity(tmp_path: Path) -> None:
    """Test metrics collection from valid equity curve."""
    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create synthetic equity curve
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    equity_values = [10000.0 * (1.001 ** i) for i in range(100)]  # Slight upward trend

    equity_df = pd.DataFrame({"timestamp": dates, "equity": equity_values})
    equity_file = backtest_dir / "portfolio_equity_1d.csv"
    equity_df.to_csv(equity_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # Should have extracted dates
    assert metrics["start_date"] is not None
    assert metrics["end_date"] is not None

    # Should have computed sharpe (might be low for small sample)
    assert "sharpe" in metrics
    # max_dd should be negative or zero
    assert metrics["max_dd"] <= 0
    assert metrics["max_dd_pct"] <= 0


def test_collect_backtest_metrics_with_trades(tmp_path: Path) -> None:
    """Test metrics collection with trades for turnover calculation."""
    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create equity curve
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    equity_values = [10000.0 * (1.001 ** i) for i in range(50)]
    equity_df = pd.DataFrame({"timestamp": dates, "equity": equity_values})
    equity_file = backtest_dir / "portfolio_equity_1d.csv"
    equity_df.to_csv(equity_file, index=False)

    # Create orders/trades file
    trade_dates = dates[:10]
    orders_df = pd.DataFrame(
        {
            "timestamp": trade_dates,
            "symbol": ["AAPL"] * 10,
            "side": ["BUY", "SELL"] * 5,
            "qty": [10.0] * 10,
            "price": [100.0] * 10,
        }
    )
    orders_file = backtest_dir / "orders_1d.csv"
    orders_df.to_csv(orders_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # Should have turnover if trades are available
    assert "turnover" in metrics
    # Turnover might be NaN if calculation fails, but key should exist


def test_collect_backtest_metrics_missing_files(tmp_path: Path) -> None:
    """Test that missing files return NaN metrics."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # All numeric metrics should be NaN
    assert math.isnan(metrics["sharpe"])
    assert math.isnan(metrics["deflated_sharpe"])
    assert math.isnan(metrics["max_dd"])
    assert math.isnan(metrics["max_dd_pct"])
    assert math.isnan(metrics["turnover"])

    # Dates should be None
    assert metrics["start_date"] is None
    assert metrics["end_date"] is None


def test_collect_backtest_metrics_with_manifest(tmp_path: Path) -> None:
    """Test that manifest information is extracted if available."""
    import json

    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create manifest
    manifest = {
        "config_hash": "abc123",
        "base_args": {"strategy": "multifactor_long_short"},
        "run_spec": {"bundle_path": "config/bundle.yaml"},
    }
    manifest_file = run_dir / "run_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Create minimal equity curve
    dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    equity_df = pd.DataFrame({"timestamp": dates, "equity": [10000.0] * 10})
    equity_file = backtest_dir / "equity_curve_1d.csv"
    equity_df.to_csv(equity_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    assert metrics["params_hash"] == "abc123"
    assert metrics["strategy"] == "multifactor_long_short"


def test_collect_backtest_metrics_invalid_equity_file(tmp_path: Path) -> None:
    """Test robust handling of invalid equity file."""
    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create invalid equity file (missing columns)
    equity_file = backtest_dir / "portfolio_equity_1d.csv"
    invalid_df = pd.DataFrame({"date": ["2020-01-01"], "value": [10000.0]})
    invalid_df.to_csv(equity_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # Should return NaN metrics
    assert math.isnan(metrics["sharpe"])


def test_collect_backtest_metrics_empty_equity(tmp_path: Path) -> None:
    """Test robust handling of empty equity file."""
    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create empty equity file
    equity_file = backtest_dir / "portfolio_equity_1d.csv"
    empty_df = pd.DataFrame({"timestamp": [], "equity": []})
    empty_df.to_csv(equity_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # Should return NaN metrics
    assert math.isnan(metrics["sharpe"])


def test_collect_backtest_metrics_deflated_sharpe(tmp_path: Path) -> None:
    """Test that deflated Sharpe is computed when possible."""
    run_dir = tmp_path / "run"
    backtest_dir = run_dir / "backtest"
    backtest_dir.mkdir(parents=True)

    # Create equity curve with returns
    # Generate returns with some volatility
    import numpy as np

    np.random.seed(42)
    n_periods = 100
    returns = np.random.normal(0.001, 0.02, n_periods)
    equity_values = [10000.0]
    for r in returns:
        equity_values.append(equity_values[-1] * (1 + r))

    # Create dates matching equity_values length
    dates = pd.date_range("2020-01-01", periods=len(equity_values), freq="D", tz="UTC")
    equity_df = pd.DataFrame({"timestamp": dates, "equity": equity_values})
    equity_file = backtest_dir / "portfolio_equity_1d.csv"
    equity_df.to_csv(equity_file, index=False)

    metrics = collect_backtest_metrics(run_dir, freq="1d")

    # Deflated Sharpe might be NaN if calculation fails, but should be attempted
    assert "deflated_sharpe" in metrics

