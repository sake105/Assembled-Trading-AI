# src/assembled_core/api/models.py
"""Pydantic models for FastAPI endpoints (future implementation).

These models represent the data structures that will be exposed via the FastAPI backend,
derived from the current file-based Sprint-9 pipeline outputs.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Enums
# ============================================================================

class SignalType(str, Enum):
    """Trading signal type."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class OrderSide(str, Enum):
    """Order side (SAFE-Bridge compatible)."""
    BUY = "BUY"
    SELL = "SELL"


class Frequency(str, Enum):
    """Trading frequency."""
    FIVE_MIN = "5min"
    DAILY = "1d"


class QaStatusEnum(str, Enum):
    """QA/QC status values."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


# ============================================================================
# Signal Models
# ============================================================================

class Signal(BaseModel):
    """Individual trading signal from EMA crossover strategy.
    
    Derived from: internal signal generation in pipeline.signals.compute_ema_signals()
    (currently not persisted, but could be saved to output/signals_{freq}.parquet)
    """
    timestamp: datetime = Field(..., description="Signal timestamp (UTC)")
    symbol: str = Field(..., description="Ticker symbol")
    signal_type: SignalType = Field(..., description="BUY, SELL, or NEUTRAL")
    price: float = Field(..., description="Close price at signal time")
    ema_fast: Optional[float] = Field(None, description="Fast EMA value")
    ema_slow: Optional[float] = Field(None, description="Slow EMA value")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-28T14:30:00Z",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "price": 278.58,
                "ema_fast": 279.2,
                "ema_slow": 277.8
            }
        }
    )


# ============================================================================
# Order Models
# ============================================================================

class OrderPreview(BaseModel):
    """SAFE-Bridge style order preview.
    
    Derived from: output/orders_{freq}.csv
    Schema: timestamp, symbol, side, qty, price
    """
    timestamp: datetime = Field(..., description="Order timestamp (UTC)")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    qty: float = Field(..., gt=0, description="Quantity (always positive)")
    price: float = Field(..., gt=0, description="Order price")
    notional: float = Field(..., description="Notional value (qty * price)")
    
    @classmethod
    def from_csv_row(cls, row: dict) -> OrderPreview:
        """Create from CSV row (helper for file-based backend)."""
        return cls(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            qty=float(row["qty"]),
            price=float(row["price"]),
            notional=float(row["qty"]) * float(row["price"])
        )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-28T14:30:00Z",
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 1.0,
                "price": 278.58,
                "notional": 278.58
            }
        }
    )


# ============================================================================
# Portfolio Models
# ============================================================================

class PortfolioSnapshot(BaseModel):
    """Current portfolio state snapshot.
    
    Derived from:
    - output/portfolio_equity_{freq}.csv (latest row)
    - output/portfolio_report.md (metrics)
    - output/orders_{freq}.csv (position calculation)
    """
    timestamp: datetime = Field(..., description="Snapshot timestamp (UTC)")
    equity: float = Field(..., description="Total portfolio equity")
    cash: float = Field(..., description="Cash position")
    positions: dict[str, float] = Field(..., description="Symbol -> quantity mapping")
    performance_factor: float = Field(..., alias="pf", description="Final PF from portfolio_report.md")
    sharpe: Optional[float] = Field(None, description="Sharpe ratio")
    total_trades: int = Field(..., description="Total number of trades")
    start_capital: float = Field(..., description="Starting capital")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-28T16:00:00Z",
                "equity": 10050.25,
                "cash": 9721.67,
                "positions": {"AAPL": 1.0, "MSFT": 0.0},
                "pf": 1.0050,
                "sharpe": 0.1566,
                "total_trades": 2,
                "start_capital": 10000.0
            }
        }
    )


class EquityPoint(BaseModel):
    """Single point in equity curve.
    
    Derived from: output/equity_curve_{freq}.csv or output/portfolio_equity_{freq}.csv
    Schema: timestamp, equity
    """
    timestamp: datetime = Field(..., description="Timestamp (UTC)")
    equity: float = Field(..., description="Portfolio equity at this timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-28T14:30:00Z",
                "equity": 10000.0
            }
        }
    )


# ============================================================================
# Risk Models
# ============================================================================

class RiskMetrics(BaseModel):
    """Risk metrics for portfolio.
    
    Derived from:
    - output/equity_curve_{freq}.csv (drawdown calculation)
    - output/portfolio_equity_{freq}.csv (volatility)
    - output/performance_report_{freq}.md (Sharpe)
    """
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio from performance_report")
    max_drawdown: float = Field(..., description="Maximum drawdown (negative value)")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown in percent")
    volatility: Optional[float] = Field(None, description="Volatility (annualized)")
    current_drawdown: float = Field(..., description="Current drawdown from peak")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sharpe_ratio": 0.1566,
                "max_drawdown": -50.25,
                "max_drawdown_pct": -0.5025,
                "volatility": 0.15,
                "current_drawdown": -25.0,
                "var_95": -100.0
            }
        }
    )


# ============================================================================
# QA/QC Models
# ============================================================================

class QaCheck(BaseModel):
    """Individual QA/QC check result.
    
    Derived from: QC checks on output/aggregates/{freq}.parquet
    """
    check_name: str = Field(..., description="Name of the check")
    status: QaStatusEnum = Field(..., description="OK, WARNING, or ERROR")
    message: str = Field(..., description="Check result message")
    details: Optional[dict] = Field(None, description="Additional check details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "check_name": "schema_validation",
                "status": "ok",
                "message": "Schema correct: symbol, timestamp, close",
                "details": {"columns": ["symbol", "timestamp", "close"]}
            }
        }
    )


class QaStatus(BaseModel):
    """Overall QA/QC status for pipeline outputs.
    
    Derived from: QC checks on all pipeline outputs
    """
    overall_status: QaStatusEnum = Field(..., description="Overall status")
    timestamp: datetime = Field(..., description="QC check timestamp")
    checks: list[QaCheck] = Field(..., description="List of individual checks")
    summary: dict[str, int] = Field(..., description="Summary: ok_count, warning_count, error_count")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_status": "ok",
                "timestamp": "2025-11-28T16:00:00Z",
                "checks": [
                    {
                        "check_name": "schema_validation",
                        "status": "ok",
                        "message": "Schema correct"
                    }
                ],
                "summary": {"ok": 5, "warning": 0, "error": 0}
            }
        }
    )


# ============================================================================
# Response Wrappers
# ============================================================================

class SignalsResponse(BaseModel):
    """Response for signals endpoint."""
    frequency: Frequency = Field(..., description="Trading frequency")
    signals: list[Signal] = Field(..., description="List of signals")
    count: int = Field(..., description="Total number of signals")
    first_timestamp: Optional[datetime] = Field(None, description="First signal timestamp")
    last_timestamp: Optional[datetime] = Field(None, description="Last signal timestamp")


class OrdersResponse(BaseModel):
    """Response for orders endpoint."""
    frequency: Frequency = Field(..., description="Trading frequency")
    orders: list[OrderPreview] = Field(..., description="List of orders")
    count: int = Field(..., description="Total number of orders")
    total_notional: float = Field(..., description="Total notional value")
    first_timestamp: Optional[datetime] = Field(None, description="First order timestamp")
    last_timestamp: Optional[datetime] = Field(None, description="Last order timestamp")


class EquityCurveResponse(BaseModel):
    """Response for equity curve endpoint."""
    frequency: Frequency = Field(..., description="Trading frequency")
    points: list[EquityPoint] = Field(..., description="Equity curve points")
    count: int = Field(..., description="Number of points")
    start_equity: float = Field(..., description="Starting equity")
    end_equity: float = Field(..., description="Ending equity")


# ============================================================================
# QA Performance & Gates Models
# ============================================================================

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response.
    
    Derived from: qa.metrics.PerformanceMetrics
    Source: run_manifest_{freq}.json (qa_metrics) or computed from equity/trades
    """
    # Performance
    final_pf: float = Field(..., description="Final Performance Factor")
    total_return: float = Field(..., description="Total Return")
    cagr: Optional[float] = Field(None, description="Compound Annual Growth Rate")
    
    # Risk-Adjusted Returns
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe Ratio (annualized)")
    sortino_ratio: Optional[float] = Field(None, description="Sortino Ratio (annualized)")
    calmar_ratio: Optional[float] = Field(None, description="Calmar Ratio")
    
    # Risk Metrics
    max_drawdown: float = Field(..., description="Maximum drawdown (absolute, negative value)")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown (percent, negative value)")
    current_drawdown: float = Field(..., description="Current drawdown (absolute)")
    volatility: Optional[float] = Field(None, description="Volatility (annualized)")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    
    # Trade Metrics
    hit_rate: Optional[float] = Field(None, description="Win Rate")
    profit_factor: Optional[float] = Field(None, description="Profit Factor")
    avg_win: Optional[float] = Field(None, description="Average win per trade")
    avg_loss: Optional[float] = Field(None, description="Average loss per trade")
    turnover: Optional[float] = Field(None, description="Portfolio turnover (annualized)")
    total_trades: Optional[int] = Field(None, description="Total number of trades")
    
    # Metadata
    start_date: datetime = Field(..., description="First timestamp")
    end_date: datetime = Field(..., description="Last timestamp")
    periods: int = Field(..., description="Number of periods")
    start_capital: float = Field(..., description="Starting capital")
    end_equity: float = Field(..., description="Ending equity")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "final_pf": 1.1250,
                "total_return": 0.1250,
                "cagr": 0.1523,
                "sharpe_ratio": 1.2345,
                "sortino_ratio": 1.5678,
                "calmar_ratio": 0.8765,
                "max_drawdown": -150.25,
                "max_drawdown_pct": -15.025,
                "current_drawdown": -50.0,
                "volatility": 0.18,
                "var_95": -200.0,
                "hit_rate": 0.55,
                "profit_factor": 1.75,
                "avg_win": 25.0,
                "avg_loss": -15.0,
                "turnover": 2.5,
                "total_trades": 100,
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-12-31T00:00:00Z",
                "periods": 252,
                "start_capital": 10000.0,
                "end_equity": 11250.0
            }
        }
    )


class QAGateResultModel(BaseModel):
    """Individual QA gate result.
    
    Derived from: qa.qa_gates.QAGateResult
    """
    gate_name: str = Field(..., description="Name of the gate (e.g., 'sharpe_ratio', 'max_drawdown')")
    result: str = Field(..., description="Gate result: 'OK', 'WARNING', or 'BLOCK'")
    reason: str = Field(..., description="Human-readable reason for the result")
    details: Optional[dict[str, Any]] = Field(None, description="Additional details (e.g., actual value, threshold)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gate_name": "sharpe_ratio",
                "result": "OK",
                "reason": "Sharpe ratio 1.2345 meets minimum threshold 1.00",
                "details": {
                    "sharpe_ratio": 1.2345,
                    "min_sharpe": 1.0,
                    "warning_sharpe": 0.5
                }
            }
        }
    )


class QAGatesSummaryResponse(BaseModel):
    """QA gates summary response.
    
    Derived from: qa.qa_gates.QAGatesSummary
    Source: run_manifest_{freq}.json (qa_gate_result) or computed from metrics
    """
    overall_result: str = Field(..., description="Overall result: 'OK', 'WARNING', or 'BLOCK'")
    counts: dict[str, int] = Field(..., description="Gate counts: {'ok': N, 'warning': M, 'block': K}")
    gate_results: list[QAGateResultModel] = Field(..., description="List of individual gate results")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_result": "OK",
                "counts": {
                    "ok": 5,
                    "warning": 1,
                    "block": 0
                },
                "gate_results": [
                    {
                        "gate_name": "sharpe_ratio",
                        "result": "OK",
                        "reason": "Sharpe ratio 1.2345 meets minimum threshold 1.00",
                        "details": {"sharpe_ratio": 1.2345, "min_sharpe": 1.0}
                    }
                ]
            }
        }
    )

