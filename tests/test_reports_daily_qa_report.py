"""Tests for daily QA report generation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.phase4

from src.assembled_core.qa.metrics import PerformanceMetrics, compute_all_metrics
from src.assembled_core.qa.qa_gates import QAGatesSummary, QAResult, evaluate_all_gates
from src.assembled_core.reports.daily_qa_report import (
    generate_qa_report,
    generate_qa_report_from_files,
)


@pytest.fixture
def synthetic_equity_positive() -> pd.DataFrame:
    """Create synthetic equity curve with positive returns."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D", tz="UTC")
    
    # Upward trending equity
    start_capital = 10000.0
    returns = pd.Series([0.001] * 252)  # 0.1% daily return
    equity_values = [start_capital]
    for ret in returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values[:-1],  # Remove last (one extra)
        "daily_return": returns
    })


@pytest.fixture
def synthetic_equity_negative() -> pd.DataFrame:
    """Create synthetic equity curve with negative returns."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    
    start_capital = 10000.0
    returns = pd.Series([-0.002] * 100)  # -0.2% daily return
    equity_values = [start_capital]
    for ret in returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values[:-1],
        "daily_return": returns
    })


@pytest.fixture
def synthetic_trades() -> pd.DataFrame:
    """Create synthetic trades DataFrame."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    
    trades = []
    for i, date in enumerate(dates[::5]):  # Every 5th day
        trades.append({
            "timestamp": date,
            "symbol": "AAPL",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 10.0,
            "price": 100.0 + i * 0.5
        })
    
    return pd.DataFrame(trades)


@pytest.fixture
def sample_metrics_positive(synthetic_equity_positive) -> PerformanceMetrics:
    """Create PerformanceMetrics from positive equity curve."""
    return compute_all_metrics(
        equity=synthetic_equity_positive,
        trades=None,
        start_capital=10000.0,
        freq="1d",
        risk_free_rate=0.0
    )


@pytest.fixture
def sample_metrics_negative(synthetic_equity_negative) -> PerformanceMetrics:
    """Create PerformanceMetrics from negative equity curve."""
    return compute_all_metrics(
        equity=synthetic_equity_negative,
        trades=None,
        start_capital=10000.0,
        freq="1d",
        risk_free_rate=0.0
    )


@pytest.fixture
def sample_gate_result_ok(sample_metrics_positive) -> QAGatesSummary:
    """Create QAGatesSummary with OK status."""
    return evaluate_all_gates(sample_metrics_positive)


@pytest.fixture
def sample_gate_result_block(sample_metrics_negative) -> QAGatesSummary:
    """Create QAGatesSummary with BLOCK status."""
    return evaluate_all_gates(sample_metrics_negative)


def test_generate_qa_report_creates_file(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that generate_qa_report creates a report file."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    # Check file exists
    assert report_path.exists()
    assert report_path.suffix == ".md"
    assert "qa_report" in report_path.name
    assert "test_strategy" in report_path.name
    assert "1d" in report_path.name


def test_generate_qa_report_contains_metrics(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report contains expected performance metrics."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check header
    assert "# QA Report:" in content
    assert "test_strategy" in content
    assert "1d" in content
    
    # Check performance metrics section
    assert "## Performance Metrics" in content
    assert "Final Performance Factor" in content
    assert "Total Return" in content
    assert "Sharpe Ratio" in content or "Sharpe Ratio:** N/A" in content
    assert "Max Drawdown" in content


def test_generate_qa_report_contains_qa_gates(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report contains QA gates section."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check QA gates section
    assert "## QA Gates" in content
    assert "Overall Status" in content
    assert "Gate Details" in content
    
    # Check status emoji
    assert "✅" in content or "⚠️" in content or "❌" in content
    
    # Check gate counts
    assert "Passed:" in content
    assert "Warnings:" in content
    assert "Blocked:" in content


def test_generate_qa_report_qa_status_ok(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report shows OK status correctly."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # If overall status is OK, should have ✅
    if sample_gate_result_ok.overall_result == QAResult.OK:
        assert "✅" in content or "OK" in content


def test_generate_qa_report_qa_status_block(tmp_path, sample_metrics_negative, sample_gate_result_block):
    """Test that report shows BLOCK status correctly."""
    report_path = generate_qa_report(
        metrics=sample_metrics_negative,
        gate_result=sample_gate_result_block,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # If overall status is BLOCK, should have ❌
    if sample_gate_result_block.overall_result == QAResult.BLOCK:
        assert "❌" in content or "BLOCK" in content


def test_generate_qa_report_contains_equity_curve(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report contains equity curve section when path provided."""
    equity_path = tmp_path / "equity_curve.csv"
    equity_path.write_text("timestamp,equity\n2023-01-01,10000.0\n")
    
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        equity_curve_path=equity_path,
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check equity curve section
    assert "## Equity Curve" in content
    assert "equity_curve.csv" in content or "Equity Curve File" in content


def test_generate_qa_report_contains_data_status(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report contains data status section."""
    data_start = pd.Timestamp("2023-01-01", tz="UTC")
    data_end = pd.Timestamp("2023-12-31", tz="UTC")
    
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        data_start_date=data_start,
        data_end_date=data_end,
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check data status section
    assert "## Data Status" in content
    assert "2023-01-01" in content
    assert "2023-12-31" in content
    assert "Frequency" in content
    assert "1d" in content


def test_generate_qa_report_contains_config(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that report contains configuration section when provided."""
    config_info = {
        "ema_fast": 20,
        "ema_slow": 60,
        "commission_bps": 0.5,
        "spread_w": 0.25,
        "impact_w": 0.5
    }
    
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        config_info=config_info,
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check configuration section
    assert "## Configuration" in content
    assert "ema_fast" in content
    assert "20" in content
    assert "commission_bps" in content


def test_generate_qa_report_no_gate_result(tmp_path, sample_metrics_positive):
    """Test that report can be generated without gate result."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=None,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Should still have metrics section
    assert "## Performance Metrics" in content
    # Should not have QA gates section
    assert "## QA Gates" not in content


def test_generate_qa_report_from_files(tmp_path, synthetic_equity_positive, synthetic_trades):
    """Test generate_qa_report_from_files convenience function."""
    # Write equity and trades to files
    equity_file = tmp_path / "portfolio_equity_1d.csv"
    synthetic_equity_positive.to_csv(equity_file, index=False)
    
    trades_file = tmp_path / "trades.csv"
    synthetic_trades.to_csv(trades_file, index=False)
    
    # Generate report
    report_path = generate_qa_report_from_files(
        freq="1d",
        strategy_name="test_strategy",
        equity_file=equity_file,
        trades_file=trades_file,
        start_capital=10000.0,
        config_info={"ema_fast": 20, "ema_slow": 60},
        output_dir=tmp_path
    )
    
    # Check file exists
    assert report_path.exists()
    assert report_path.suffix == ".md"
    
    # Check content
    content = report_path.read_text(encoding="utf-8")
    assert "## Performance Metrics" in content
    assert "## QA Gates" in content
    assert "test_strategy" in content


def test_generate_qa_report_from_files_default_equity():
    """
    Test generate_qa_report_from_files with default equity file path.
    
    In CI gibt es kein vorher erzeugtes portfolio_equity_1d.csv.
    Deshalb erstellt der Test hier eine kleine Dummy-Equity-Kurve
    genau an diesem Default-Pfad (OUTPUT_DIR).
    """
    from src.assembled_core.config import OUTPUT_DIR
    
    # OUTPUT_DIR sicherstellen (kann src/output/ oder output/ sein, je nach config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    equity_file = OUTPUT_DIR / "portfolio_equity_1d.csv"
    
    # kleine synthetische Equity-Kurve
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            "equity": [10000.0, 10050.0, 10090.0],
        }
    )
    df.to_csv(equity_file, index=False)
    
    # jetzt sollte die Funktion das Default-File finden
    report_path = generate_qa_report_from_files(
        freq="1d",
        strategy_name="test_strategy",
        equity_file=None,  # Use default
        start_capital=10000.0
    )
    
    assert report_path.exists()
    assert report_path.is_file()
    
    # Cleanup: Remove test file after test
    if equity_file.exists():
        equity_file.unlink()


def test_generate_qa_report_from_files_missing_equity(tmp_path):
    """Test that error is raised when equity file is missing."""
    with pytest.raises(FileNotFoundError, match="Equity file not found"):
        generate_qa_report_from_files(
            freq="1d",
            strategy_name="test_strategy",
            equity_file=tmp_path / "nonexistent.csv",
            start_capital=10000.0,
            output_dir=tmp_path
        )


def test_generate_qa_report_gate_table_format(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that gate details table is properly formatted."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check table format
    assert "| Gate | Status | Reason |" in content
    assert "|------|--------|--------|" in content
    
    # Check that gate names appear in table
    for gate_result in sample_gate_result_ok.gate_results:
        assert gate_result.gate_name in content


def test_generate_qa_report_period_info(tmp_path, sample_metrics_positive, sample_gate_result_ok):
    """Test that period information is correctly included."""
    report_path = generate_qa_report(
        metrics=sample_metrics_positive,
        gate_result=sample_gate_result_ok,
        strategy_name="test_strategy",
        freq="1d",
        output_dir=tmp_path
    )
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check period information
    assert "Start Date" in content
    assert "End Date" in content
    assert "Periods" in content
    assert "Start Capital" in content
    assert "End Equity" in content
    
    # Check that dates are formatted
    assert sample_metrics_positive.start_date.strftime("%Y-%m-%d") in content
    assert sample_metrics_positive.end_date.strftime("%Y-%m-%d") in content

