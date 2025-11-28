"""Tests for qa.qa_gates module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.metrics import PerformanceMetrics, compute_all_metrics
from src.assembled_core.qa.qa_gates import (
    QAGateResult,
    QAGatesSummary,
    QAResult,
    check_cagr,
    check_hit_rate,
    check_max_drawdown,
    check_profit_factor,
    check_sharpe_ratio,
    check_turnover,
    check_volatility,
    evaluate_all_gates,
)


@pytest.fixture
def equity_strong_positive() -> pd.DataFrame:
    """Equity curve with strong positive trend (should pass all gates)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong positive: ~0.3% daily return, 1.5% volatility
    np.random.seed(42)
    returns = np.random.normal(0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def equity_negative() -> pd.DataFrame:
    """Equity curve with negative trend (should block)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong negative: ~-0.3% daily return, 1.5% volatility
    np.random.seed(43)
    returns = np.random.normal(-0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def equity_sideways() -> pd.DataFrame:
    """Equity curve with sideways movement (should warn)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Sideways: ~0% daily return, 3% volatility
    np.random.seed(44)
    returns = np.random.normal(0.0, 0.03, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def trades_high_turnover() -> pd.DataFrame:
    """Trades with high turnover (should warn/block)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    trades = []
    for i, date in enumerate(dates):
        # Trade every day
        trades.append({
            "timestamp": date,
            "symbol": f"SYM{i % 5}",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 100.0,
            "price": 100.0 + i * 0.1
        })
    
    return pd.DataFrame(trades)


@pytest.fixture
def trades_low_turnover() -> pd.DataFrame:
    """Trades with low turnover (should pass)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    trades = []
    for i, date in enumerate(dates):
        trades.append({
            "timestamp": date,
            "symbol": "AAPL",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 10.0,
            "price": 100.0 + i * 0.5
        })
    
    return pd.DataFrame(trades)


@pytest.fixture
def metrics_good() -> PerformanceMetrics:
    """Good performance metrics (all gates should pass)."""
    return PerformanceMetrics(
        final_pf=1.20,
        total_return=0.20,
        cagr=0.10,  # 10% CAGR
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=0.5,
        max_drawdown=-10.0,
        max_drawdown_pct=-10.0,
        current_drawdown=-2.0,
        volatility=0.15,  # 15% volatility
        var_95=-500.0,
        hit_rate=0.60,  # 60% win rate
        profit_factor=2.0,
        avg_win=100.0,
        avg_loss=-50.0,
        turnover=20.0,  # 20x turnover
        total_trades=100,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        periods=252,
        start_capital=10000.0,
        end_equity=12000.0
    )


@pytest.fixture
def metrics_warning() -> PerformanceMetrics:
    """Warning-level performance metrics."""
    return PerformanceMetrics(
        final_pf=1.05,
        total_return=0.05,
        cagr=0.02,  # 2% CAGR (below min but above warning)
        sharpe_ratio=0.6,  # Below min but above warning
        sortino_ratio=0.8,
        calmar_ratio=0.2,
        max_drawdown=-18.0,  # Below warning threshold
        max_drawdown_pct=-18.0,
        current_drawdown=-5.0,
        volatility=0.28,  # Above warning threshold
        var_95=-800.0,
        hit_rate=0.45,  # Below min but above warning
        profit_factor=1.3,  # Below min but above warning
        avg_win=80.0,
        avg_loss=-60.0,
        turnover=35.0,  # Above warning threshold
        total_trades=50,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        periods=252,
        start_capital=10000.0,
        end_equity=10500.0
    )


@pytest.fixture
def metrics_block() -> PerformanceMetrics:
    """Block-level performance metrics."""
    return PerformanceMetrics(
        final_pf=0.95,
        total_return=-0.05,
        cagr=-0.05,  # Negative CAGR
        sharpe_ratio=0.3,  # Below warning threshold
        sortino_ratio=0.4,
        calmar_ratio=-0.1,
        max_drawdown=-25.0,  # Below block threshold
        max_drawdown_pct=-25.0,
        current_drawdown=-10.0,
        volatility=0.35,  # Above block threshold
        var_95=-1200.0,
        hit_rate=0.35,  # Below warning threshold
        profit_factor=1.0,  # Below warning threshold
        avg_win=50.0,
        avg_loss=-70.0,
        turnover=60.0,  # Above block threshold
        total_trades=30,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        periods=252,
        start_capital=10000.0,
        end_equity=9500.0
    )


@pytest.fixture
def metrics_no_trades() -> PerformanceMetrics:
    """Metrics without trade data."""
    return PerformanceMetrics(
        final_pf=1.15,
        total_return=0.15,
        cagr=0.08,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        calmar_ratio=0.4,
        max_drawdown=-12.0,
        max_drawdown_pct=-12.0,
        current_drawdown=-3.0,
        volatility=0.18,
        var_95=-600.0,
        hit_rate=None,
        profit_factor=None,
        avg_win=None,
        avg_loss=None,
        turnover=None,
        total_trades=None,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        periods=252,
        start_capital=10000.0,
        end_equity=11500.0
    )


@pytest.mark.smoke
def test_metrics_strong_positive_scenario(equity_strong_positive):
    """Test metrics computation for strong positive scenario."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Strong positive should have:
    assert metrics.final_pf > 1.5  # At least 50% return
    assert metrics.total_return > 0.5
    assert metrics.cagr is not None
    assert metrics.cagr > 0.3  # At least 30% CAGR
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio > 1.0  # Good Sharpe
    assert metrics.max_drawdown_pct > -20.0  # Reasonable drawdown
    assert metrics.volatility is not None
    assert metrics.volatility < 0.30  # Not too volatile


@pytest.mark.smoke
def test_metrics_negative_scenario(equity_negative):
    """Test metrics computation for negative scenario."""
    metrics = compute_all_metrics(
        equity=equity_negative,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Negative should have:
    assert metrics.final_pf < 1.0  # Loss
    assert metrics.total_return < 0
    assert metrics.cagr is not None
    assert metrics.cagr < 0  # Negative CAGR
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio < 0.5  # Poor Sharpe
    assert metrics.max_drawdown_pct < -20.0  # Large drawdown


@pytest.mark.smoke
def test_metrics_sideways_scenario(equity_sideways):
    """Test metrics computation for sideways scenario."""
    metrics = compute_all_metrics(
        equity=equity_sideways,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Sideways should have:
    assert abs(metrics.total_return) < 0.2  # Small total return
    assert metrics.volatility is not None
    assert metrics.volatility > 0.25  # High volatility
    if metrics.sharpe_ratio is not None:
        assert metrics.sharpe_ratio < 1.0  # Low Sharpe


@pytest.mark.smoke
def test_metrics_high_turnover_scenario(equity_strong_positive, trades_high_turnover):
    """Test metrics computation with high turnover."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=trades_high_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    # High turnover should be reflected:
    assert metrics.turnover is not None
    assert metrics.turnover > 30.0  # High turnover
    assert metrics.total_trades == len(trades_high_turnover)


@pytest.mark.smoke
def test_gates_strong_positive_scenario(equity_strong_positive):
    """Test QA gates for strong positive scenario (should pass)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Strong positive should pass most gates
    assert summary.overall_result in [QAResult.OK, QAResult.WARNING]  # May have warnings for trade metrics
    assert summary.blocked_gates == 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result == QAResult.OK
    
    cagr_gate = next(g for g in summary.gate_results if g.gate_name == "cagr")
    assert cagr_gate.result == QAResult.OK
    
    max_dd_gate = next(g for g in summary.gate_results if g.gate_name == "max_drawdown")
    assert max_dd_gate.result == QAResult.OK


@pytest.mark.smoke
def test_gates_negative_scenario(equity_negative):
    """Test QA gates for negative scenario (should block)."""
    metrics = compute_all_metrics(
        equity=equity_negative,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Negative should block
    assert summary.overall_result == QAResult.BLOCK
    assert summary.blocked_gates > 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result == QAResult.BLOCK
    
    cagr_gate = next(g for g in summary.gate_results if g.gate_name == "cagr")
    assert cagr_gate.result == QAResult.BLOCK
    
    max_dd_gate = next(g for g in summary.gate_results if g.gate_name == "max_drawdown")
    assert max_dd_gate.result == QAResult.BLOCK


@pytest.mark.smoke
def test_gates_sideways_scenario(equity_sideways):
    """Test QA gates for sideways scenario (should warn)."""
    metrics = compute_all_metrics(
        equity=equity_sideways,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Sideways should warn
    assert summary.overall_result == QAResult.WARNING
    assert summary.warning_gates > 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result in [QAResult.WARNING, QAResult.BLOCK]
    
    volatility_gate = next(g for g in summary.gate_results if g.gate_name == "volatility")
    assert volatility_gate.result in [QAResult.WARNING, QAResult.BLOCK]


@pytest.mark.smoke
def test_gates_high_turnover_scenario(equity_strong_positive, trades_high_turnover):
    """Test QA gates with high turnover (should warn/block)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=trades_high_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # High turnover should trigger warning or block
    turnover_gate = next(g for g in summary.gate_results if g.gate_name == "turnover")
    assert turnover_gate.result in [QAResult.WARNING, QAResult.BLOCK]
    
    # Overall result should reflect turnover issue
    assert summary.overall_result in [QAResult.WARNING, QAResult.BLOCK]


@pytest.mark.smoke
def test_gates_low_turnover_scenario(equity_strong_positive, trades_low_turnover):
    """Test QA gates with low turnover (should pass)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=trades_low_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Low turnover should pass
    turnover_gate = next(g for g in summary.gate_results if g.gate_name == "turnover")
    assert turnover_gate.result == QAResult.OK


@pytest.mark.unit
def test_check_sharpe_ratio_ok(metrics_good):
    """Test Sharpe ratio gate with good metrics."""
    result = check_sharpe_ratio(metrics_good)
    
    assert result.gate_name == "sharpe_ratio"
    assert result.result == QAResult.OK
    assert "meets minimum threshold" in result.reason.lower()
    assert result.details is not None
    assert result.details["sharpe_ratio"] == 1.5


@pytest.mark.unit
def test_check_sharpe_ratio_warning(metrics_warning):
    """Test Sharpe ratio gate with warning metrics."""
    result = check_sharpe_ratio(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "below minimum threshold" in result.reason.lower()
    assert result.details["sharpe_ratio"] == 0.6


@pytest.mark.unit
def test_check_sharpe_ratio_block(metrics_block):
    """Test Sharpe ratio gate with block metrics."""
    result = check_sharpe_ratio(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "below warning threshold" in result.reason.lower()
    assert result.details["sharpe_ratio"] == 0.3


@pytest.mark.unit
def test_check_sharpe_ratio_none():
    """Test Sharpe ratio gate with None Sharpe."""
    metrics = PerformanceMetrics(
        final_pf=1.0,
        total_return=0.0,
        cagr=None,
        sharpe_ratio=None,
        sortino_ratio=None,
        calmar_ratio=None,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        current_drawdown=0.0,
        volatility=None,
        var_95=None,
        hit_rate=None,
        profit_factor=None,
        avg_win=None,
        avg_loss=None,
        turnover=None,
        total_trades=None,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-01-02"),
        periods=2,
        start_capital=10000.0,
        end_equity=10000.0
    )
    
    result = check_sharpe_ratio(metrics)
    
    assert result.result == QAResult.WARNING
    assert "cannot be computed" in result.reason.lower()


@pytest.mark.unit
def test_check_max_drawdown_ok(metrics_good):
    """Test max drawdown gate with good metrics."""
    result = check_max_drawdown(metrics_good)
    
    assert result.result == QAResult.OK
    assert "within acceptable limits" in result.reason.lower()


@pytest.mark.unit
def test_check_max_drawdown_warning(metrics_warning):
    """Test max drawdown gate with warning metrics."""
    result = check_max_drawdown(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "exceeds warning threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_max_drawdown_block(metrics_block):
    """Test max drawdown gate with block metrics."""
    result = check_max_drawdown(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "exceeds limit" in result.reason.lower()


@pytest.mark.unit
def test_check_turnover_ok(metrics_good):
    """Test turnover gate with good metrics."""
    result = check_turnover(metrics_good)
    
    assert result.result == QAResult.OK
    assert "within acceptable limits" in result.reason.lower()


@pytest.mark.unit
def test_check_turnover_warning(metrics_warning):
    """Test turnover gate with warning metrics."""
    result = check_turnover(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "exceeds warning threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_turnover_block(metrics_block):
    """Test turnover gate with block metrics."""
    result = check_turnover(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "exceeds maximum limit" in result.reason.lower()


@pytest.mark.unit
def test_check_turnover_none(metrics_no_trades):
    """Test turnover gate with None turnover."""
    result = check_turnover(metrics_no_trades)
    
    assert result.result == QAResult.WARNING
    assert "cannot be computed" in result.reason.lower()


@pytest.mark.unit
def test_check_cagr_ok(metrics_good):
    """Test CAGR gate with good metrics."""
    result = check_cagr(metrics_good)
    
    assert result.result == QAResult.OK
    assert "meets minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_cagr_warning(metrics_warning):
    """Test CAGR gate with warning metrics."""
    result = check_cagr(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "below minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_cagr_block(metrics_block):
    """Test CAGR gate with block metrics."""
    result = check_cagr(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "below warning threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_volatility_ok(metrics_good):
    """Test volatility gate with good metrics."""
    result = check_volatility(metrics_good)
    
    assert result.result == QAResult.OK
    assert "within acceptable limits" in result.reason.lower()


@pytest.mark.unit
def test_check_volatility_warning(metrics_warning):
    """Test volatility gate with warning metrics."""
    result = check_volatility(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "exceeds warning threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_volatility_block(metrics_block):
    """Test volatility gate with block metrics."""
    result = check_volatility(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "exceeds maximum limit" in result.reason.lower()


@pytest.mark.unit
def test_check_hit_rate_ok(metrics_good):
    """Test hit rate gate with good metrics."""
    result = check_hit_rate(metrics_good)
    
    assert result.result == QAResult.OK
    assert "meets minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_hit_rate_warning(metrics_warning):
    """Test hit rate gate with warning metrics."""
    result = check_hit_rate(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "below minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_hit_rate_block(metrics_block):
    """Test hit rate gate with block metrics."""
    result = check_hit_rate(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "below warning threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_hit_rate_none(metrics_no_trades):
    """Test hit rate gate with None hit rate."""
    result = check_hit_rate(metrics_no_trades)
    
    assert result.result == QAResult.WARNING
    assert "cannot be computed" in result.reason.lower()


@pytest.mark.unit
def test_check_profit_factor_ok(metrics_good):
    """Test profit factor gate with good metrics."""
    result = check_profit_factor(metrics_good)
    
    assert result.result == QAResult.OK
    assert "meets minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_profit_factor_warning(metrics_warning):
    """Test profit factor gate with warning metrics."""
    result = check_profit_factor(metrics_warning)
    
    assert result.result == QAResult.WARNING
    assert "below minimum threshold" in result.reason.lower()


@pytest.mark.unit
def test_check_profit_factor_block(metrics_block):
    """Test profit factor gate with block metrics."""
    result = check_profit_factor(metrics_block)
    
    assert result.result == QAResult.BLOCK
    assert "below warning threshold" in result.reason.lower()


@pytest.mark.smoke
def test_evaluate_all_gates_ok(metrics_good):
    """Test evaluate_all_gates with good metrics."""
    summary = evaluate_all_gates(metrics_good)
    
    assert isinstance(summary, QAGatesSummary)
    assert summary.overall_result == QAResult.OK
    assert summary.passed_gates > 0
    assert summary.blocked_gates == 0
    assert len(summary.gate_results) == 7  # 7 gates


@pytest.mark.smoke
def test_evaluate_all_gates_warning(metrics_warning):
    """Test evaluate_all_gates with warning metrics."""
    summary = evaluate_all_gates(metrics_warning)
    
    assert summary.overall_result == QAResult.WARNING
    assert summary.warning_gates > 0
    assert summary.blocked_gates == 0


@pytest.mark.smoke
def test_evaluate_all_gates_block(metrics_block):
    """Test evaluate_all_gates with block metrics."""
    summary = evaluate_all_gates(metrics_block)
    
    assert summary.overall_result == QAResult.BLOCK
    assert summary.blocked_gates > 0


@pytest.mark.smoke
def test_evaluate_all_gates_custom_config(metrics_good):
    """Test evaluate_all_gates with custom configuration."""
    custom_config = {
        "sharpe": {"min": 2.0, "warning": 1.5},
        "max_drawdown": {"max": -15.0, "warning": -10.0},
        "turnover": {"max": 30.0, "warning": 20.0}
    }
    
    summary = evaluate_all_gates(metrics_good, gate_config=custom_config)
    
    assert isinstance(summary, QAGatesSummary)
    # With stricter thresholds, some gates might now fail
    assert summary.overall_result in [QAResult.OK, QAResult.WARNING, QAResult.BLOCK]


@pytest.mark.smoke
def test_evaluate_all_gates_no_trades(metrics_no_trades):
    """Test evaluate_all_gates without trade data."""
    summary = evaluate_all_gates(metrics_no_trades)
    
    assert isinstance(summary, QAGatesSummary)
    # Trade-based gates should return WARNING (cannot compute)
    trade_gates = [r for r in summary.gate_results if r.gate_name in ["turnover", "hit_rate", "profit_factor"]]
    assert all(r.result == QAResult.WARNING for r in trade_gates)


@pytest.fixture
def equity_strong_positive() -> pd.DataFrame:
    """Equity curve with strong positive trend (should pass all gates)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong positive: ~0.3% daily return, 1.5% volatility
    np.random.seed(42)
    returns = np.random.normal(0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def equity_negative() -> pd.DataFrame:
    """Equity curve with negative trend (should block)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong negative: ~-0.3% daily return, 1.5% volatility
    np.random.seed(43)
    returns = np.random.normal(-0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def equity_sideways() -> pd.DataFrame:
    """Equity curve with sideways movement (should warn)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Sideways: ~0% daily return, 3% volatility
    np.random.seed(44)
    returns = np.random.normal(0.0, 0.03, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def trades_high_turnover() -> pd.DataFrame:
    """Trades with high turnover (should warn/block)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    trades = []
    for i, date in enumerate(dates):
        # Trade every day
        trades.append({
            "timestamp": date,
            "symbol": f"SYM{i % 5}",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 100.0,
            "price": 100.0 + i * 0.1
        })
    
    return pd.DataFrame(trades)


@pytest.fixture
def trades_low_turnover() -> pd.DataFrame:
    """Trades with low turnover (should pass)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    trades = []
    for i, date in enumerate(dates):
        trades.append({
            "timestamp": date,
            "symbol": "AAPL",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 10.0,
            "price": 100.0 + i * 0.5
        })
    
    return pd.DataFrame(trades)


@pytest.mark.smoke
def test_gates_strong_positive_scenario(equity_strong_positive):
    """Test QA gates for strong positive scenario (should pass)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Strong positive should pass most gates
    assert summary.overall_result in [QAResult.OK, QAResult.WARNING]  # May have warnings for trade metrics
    assert summary.blocked_gates == 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result == QAResult.OK
    
    cagr_gate = next(g for g in summary.gate_results if g.gate_name == "cagr")
    assert cagr_gate.result == QAResult.OK
    
    max_dd_gate = next(g for g in summary.gate_results if g.gate_name == "max_drawdown")
    assert max_dd_gate.result == QAResult.OK


@pytest.mark.smoke
def test_gates_negative_scenario(equity_negative):
    """Test QA gates for negative scenario (should block)."""
    metrics = compute_all_metrics(
        equity=equity_negative,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Negative should block
    assert summary.overall_result == QAResult.BLOCK
    assert summary.blocked_gates > 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result == QAResult.BLOCK
    
    cagr_gate = next(g for g in summary.gate_results if g.gate_name == "cagr")
    assert cagr_gate.result == QAResult.BLOCK
    
    max_dd_gate = next(g for g in summary.gate_results if g.gate_name == "max_drawdown")
    assert max_dd_gate.result == QAResult.BLOCK


@pytest.mark.smoke
def test_gates_sideways_scenario(equity_sideways):
    """Test QA gates for sideways scenario (should warn)."""
    metrics = compute_all_metrics(
        equity=equity_sideways,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Sideways should warn
    assert summary.overall_result == QAResult.WARNING
    assert summary.warning_gates > 0
    
    # Check specific gates
    sharpe_gate = next(g for g in summary.gate_results if g.gate_name == "sharpe_ratio")
    assert sharpe_gate.result in [QAResult.WARNING, QAResult.BLOCK]
    
    volatility_gate = next(g for g in summary.gate_results if g.gate_name == "volatility")
    assert volatility_gate.result in [QAResult.WARNING, QAResult.BLOCK]


@pytest.mark.smoke
def test_gates_high_turnover_scenario(equity_strong_positive, trades_high_turnover):
    """Test QA gates with high turnover (should warn/block)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=trades_high_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # High turnover should trigger warning or block
    turnover_gate = next(g for g in summary.gate_results if g.gate_name == "turnover")
    assert turnover_gate.result in [QAResult.WARNING, QAResult.BLOCK]
    
    # Overall result should reflect turnover issue
    assert summary.overall_result in [QAResult.WARNING, QAResult.BLOCK]


@pytest.mark.smoke
def test_gates_low_turnover_scenario(equity_strong_positive, trades_low_turnover):
    """Test QA gates with low turnover (should pass)."""
    metrics = compute_all_metrics(
        equity=equity_strong_positive,
        trades=trades_low_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    summary = evaluate_all_gates(metrics)
    
    # Low turnover should pass
    turnover_gate = next(g for g in summary.gate_results if g.gate_name == "turnover")
    assert turnover_gate.result == QAResult.OK
