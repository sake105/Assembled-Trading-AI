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


class SignalsResponse(BaseModel):
    """Signals response with list of signals and summary."""
    frequency: str = Field(..., description="Trading frequency")
    signals: list[Signal] = Field(..., description="List of signals")
    count: int = Field(..., description="Total number of signals")
    first_timestamp: Optional[datetime] = Field(None, description="First signal timestamp")
    last_timestamp: Optional[datetime] = Field(None, description="Last signal timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frequency": "1d",
                "signals": [
                    {
                        "timestamp": "2025-11-28T14:30:00Z",
                        "symbol": "AAPL",
                        "signal_type": "BUY",
                        "price": 278.58,
                        "ema_fast": 279.2,
                        "ema_slow": 277.8
                    }
                ],
                "count": 1,
                "first_timestamp": "2025-11-28T14:30:00Z",
                "last_timestamp": "2025-11-28T14:30:00Z"
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
                "qty": 10.0,
                "price": 278.58,
                "notional": 2785.80
            }
        }
    )


class OrdersResponse(BaseModel):
    """Orders response with list of orders and summary.
    
    Response from GET /api/v1/orders/{freq} endpoint.
    """
    frequency: str = Field(..., description="Trading frequency ('1d' or '5min')")
    orders: list[OrderPreview] = Field(..., description="List of orders")
    count: int = Field(..., description="Total number of orders")
    total_notional: float = Field(..., description="Total notional value of all orders")
    first_timestamp: Optional[datetime] = Field(None, description="First order timestamp")
    last_timestamp: Optional[datetime] = Field(None, description="Last order timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frequency": "1d",
                "orders": [
                    {
                        "timestamp": "2025-11-28T14:30:00Z",
                        "symbol": "AAPL",
                        "side": "BUY",
                        "qty": 10.0,
                        "price": 278.58,
                        "notional": 2785.80
                    }
                ],
                "count": 1,
                "total_notional": 2785.80,
                "first_timestamp": "2025-11-28T14:30:00Z",
                "last_timestamp": "2025-11-28T14:30:00Z"
            }
        }
    )


# ============================================================================
# Portfolio Models
# ============================================================================

class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot at a given point in time.
    
    Derived from:
    - output/portfolio_report.md (metrics)
    - output/portfolio_equity_{freq}.csv (equity curve)
    - output/orders_{freq}.csv (current positions inferred from orders)
    """
    timestamp: datetime = Field(..., description="Snapshot timestamp (UTC)")
    total_equity: float = Field(..., description="Total portfolio equity")
    cash: Optional[float] = Field(None, description="Cash balance")
    positions: dict[str, float] = Field(..., description="Position sizes by symbol (quantity)")
    performance_factor: float = Field(..., alias="pf", description="Final PF from portfolio_report.md")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-28T14:30:00Z",
                "total_equity": 11250.50,
                "cash": 250.50,
                "positions": {"AAPL": 10.0, "MSFT": 5.0},
                "pf": 1.12505
            }
        }
    )


# ============================================================================
# Performance Models
# ============================================================================

class PerformanceSummary(BaseModel):
    """Simplified performance summary.
    
    Derived from:
    - output/performance_report_{freq}.md (Sharpe)
    - output/portfolio_report.md (PF, trades)
    """
    performance_factor: float = Field(..., alias="pf", description="Final PF from portfolio_report.md")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio from performance_report")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pf": 1.1250,
                "sharpe_ratio": 1.2345
            }
        }
    )


# ============================================================================
# Risk Models
# ============================================================================

class RiskMetrics(BaseModel):
    """Risk metrics summary.
    
    Derived from: portfolio equity curve and risk_metrics module.
    """
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio (annualized)")
    max_drawdown: float = Field(..., description="Maximum drawdown (absolute, negative value)")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown (percent, negative value)")
    volatility: Optional[float] = Field(None, description="Volatility (annualized)")
    current_drawdown: float = Field(..., description="Current drawdown (absolute)")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sharpe_ratio": 1.2345,
                "max_drawdown": -150.25,
                "max_drawdown_pct": -15.025,
                "volatility": 0.18,
                "current_drawdown": -50.0,
                "var_95": -200.0
            }
        }
    )


# ============================================================================
# QA Models
# ============================================================================

class QaCheck(BaseModel):
    """Individual QA check result."""
    check_name: str = Field(..., description="Name of the check")
    status: QaStatusEnum = Field(..., description="Check status")
    message: str = Field(..., description="Check message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional check details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "check_name": "data_quality",
                "status": "ok",
                "message": "All data checks passed",
                "details": {"missing_values": 0, "outliers": 0}
            }
        }
    )


class QaStatus(BaseModel):
    """QA/Health check status."""
    overall_status: QaStatusEnum = Field(..., description="Overall QA status")
    timestamp: datetime = Field(..., description="Status timestamp (UTC)")
    checks: list[QaCheck] = Field(..., description="List of individual checks")
    summary: dict[str, int] = Field(..., description="Summary counts: {'ok': N, 'warning': M, 'error': K}")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_status": "ok",
                "timestamp": "2025-11-28T14:30:00Z",
                "checks": [
                    {
                        "check_name": "data_quality",
                        "status": "ok",
                        "message": "All data checks passed",
                        "details": {}
                    }
                ],
                "summary": {"ok": 5, "warning": 0, "error": 0}
            }
        }
    )


# ============================================================================
# Performance Metrics Models
# ============================================================================

class EquityPoint(BaseModel):
    """Single equity curve point."""
    timestamp: datetime = Field(..., description="Timestamp")
    equity: float = Field(..., description="Equity value")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2023-01-01T00:00:00Z",
                "equity": 10000.0
            }
        }
    )


class EquityCurveResponse(BaseModel):
    """Equity curve response."""
    frequency: str = Field(..., description="Trading frequency")
    points: list[EquityPoint] = Field(..., description="List of equity curve points")
    count: int = Field(..., description="Number of points")
    start_equity: float = Field(..., description="Starting equity")
    end_equity: float = Field(..., description="Ending equity")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frequency": "1d",
                "points": [
                    {"timestamp": "2023-01-01T00:00:00Z", "equity": 10000.0},
                    {"timestamp": "2023-01-02T00:00:00Z", "equity": 10100.0}
                ],
                "count": 2,
                "start_equity": 10000.0,
                "end_equity": 11250.0
            }
        }
    )


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


# ============================================================================
# Monitoring Models
# ============================================================================

class QAStatusSummary(BaseModel):
    """Simplified QA status summary for monitoring.
    
    Provides a quick overview of QA gate results and key metrics.
    """
    overall_result: str = Field(..., description="Overall QA gate result: 'OK', 'WARNING', or 'BLOCK'")
    gate_counts: dict[str, int] = Field(..., description="Gate counts: {'ok': N, 'warning': M, 'block': K}")
    key_metrics: dict[str, Optional[float]] = Field(
        ...,
        description="Key performance metrics: sharpe_ratio, max_drawdown_pct, total_return, cagr"
    )
    last_updated: Optional[datetime] = Field(None, description="Timestamp of last QA evaluation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_result": "OK",
                "gate_counts": {"ok": 5, "warning": 1, "block": 0},
                "key_metrics": {
                    "sharpe_ratio": 1.2345,
                    "max_drawdown_pct": -15.025,
                    "total_return": 0.1250,
                    "cagr": 0.1523
                },
                "last_updated": "2025-11-28T14:30:00Z"
            }
        }
    )


class RiskStatusSummary(BaseModel):
    """Simplified risk status summary for monitoring.
    
    Provides a quick overview of risk metrics from the last portfolio report.
    """
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio (annualized)")
    max_drawdown_pct: Optional[float] = Field(None, description="Maximum drawdown (percent, negative value)")
    volatility: Optional[float] = Field(None, description="Volatility (annualized)")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    current_drawdown: Optional[float] = Field(None, description="Current drawdown (absolute)")
    last_updated: Optional[datetime] = Field(None, description="Timestamp of last risk evaluation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sharpe_ratio": 1.2345,
                "max_drawdown_pct": -15.025,
                "volatility": 0.18,
                "var_95": -200.0,
                "current_drawdown": -50.0,
                "last_updated": "2025-11-28T14:30:00Z"
            }
        }
    )


class FeatureDriftItem(BaseModel):
    """Single feature drift item."""
    feature: str = Field(..., description="Feature name")
    psi: float = Field(..., description="Population Stability Index")
    drift_flag: str = Field(..., description="Drift severity: 'NONE', 'MODERATE', or 'SEVERE'")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature": "ta_ema_20",
                "psi": 0.25,
                "drift_flag": "MODERATE"
            }
        }
    )


class DriftStatusSummary(BaseModel):
    """Drift status summary for monitoring.
    
    Provides a quick overview of feature drift detection results.
    """
    overall_severity: str = Field(..., description="Worst-case drift severity: 'NONE', 'MODERATE', or 'SEVERE'")
    features_with_drift: list[FeatureDriftItem] = Field(
        ...,
        description="Top features with drift (sorted by PSI, descending)"
    )
    total_features_checked: int = Field(..., description="Total number of features checked")
    last_updated: Optional[datetime] = Field(None, description="Timestamp of last drift analysis")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_severity": "MODERATE",
                "features_with_drift": [
                    {
                        "feature": "ta_ema_20",
                        "psi": 0.25,
                        "drift_flag": "MODERATE"
                    },
                    {
                        "feature": "insider_net_buy",
                        "psi": 0.15,
                        "drift_flag": "NONE"
                    }
                ],
                "total_features_checked": 10,
                "last_updated": "2025-11-28T14:30:00Z"
            }
        }
    )


# ============================================================================
# Paper Trading Models
# ============================================================================

class PaperOrderRequest(BaseModel):
    """Paper trading order request.
    
    Used for submitting orders to the paper trading engine.
    """
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Order quantity (must be positive)")
    price: Optional[float] = Field(None, gt=0, description="Order price (optional, None = market order)")
    client_order_id: Optional[str] = Field(None, description="Optional client-provided order ID")
    route: Optional[str] = Field(None, description="Order route (e.g., 'PAPER', 'IBKR'). Default: 'PAPER'")
    source: Optional[str] = Field(None, description="Order source (e.g., 'CLI', 'API', 'BACKTEST', 'DASHBOARD')")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0,
                "client_order_id": "client-order-123",
                "route": "PAPER",
                "source": "API"
            }
        }
    )


class PaperOrderResponse(BaseModel):
    """Paper trading order response.
    
    Response after submitting an order to the paper trading engine.
    """
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (None for market orders)")
    status: str = Field(..., description="Order status: NEW, FILLED, or REJECTED")
    reason: Optional[str] = Field(None, description="Reason for rejection (if status is REJECTED)")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID (if provided)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "order_id": "order-abc123",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0,
                "status": "FILLED",
                "reason": None,
                "client_order_id": "client-order-123"
            }
        }
    )


class PaperPosition(BaseModel):
    """Paper trading position.
    
    Current position in a symbol from the paper trading engine.
    """
    symbol: str = Field(..., description="Ticker symbol")
    quantity: float = Field(..., description="Position quantity (positive = long, negative = short)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "quantity": 6.0
            }
        }
    )


class PaperResetResponse(BaseModel):
    """Paper trading reset response.
    
    Response after resetting the paper trading engine.
    """
    status: str = Field(..., description="Reset status (always 'ok')")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok"
            }
        }
    )


# ============================================================================
# OMS (Order Management System) Models
# ============================================================================

class OmsOrderView(BaseModel):
    """OMS order view for blotter display.
    
    Represents an order in the OMS blotter view with all relevant details.
    """
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (None for market orders)")
    status: str = Field(..., description="Order status: NEW, FILLED, or REJECTED")
    route: Optional[str] = Field(None, description="Order route (e.g., 'PAPER', 'IBKR', etc.)")
    source: Optional[str] = Field(None, description="Order source (e.g., 'CLI', 'API', 'BACKTEST')")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    created_at: datetime = Field(..., description="Order creation timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "order_id": "order-abc123",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0,
                "status": "FILLED",
                "route": "PAPER",
                "source": "API",
                "client_order_id": "client-order-123",
                "created_at": "2025-01-15T10:30:00Z"
            }
        }
    )


class OmsExecution(BaseModel):
    """OMS execution (fill) representation.
    
    Represents a single execution/fill of an order in the OMS.
    For OMS-Light, each FILLED order is treated as a single execution.
    """
    exec_id: str = Field(..., description="Unique execution identifier")
    order_id: str = Field(..., description="Order ID this execution belongs to")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="BUY or SELL")
    quantity: float = Field(..., description="Execution quantity")
    price: Optional[float] = Field(None, description="Execution price")
    timestamp: datetime = Field(..., description="Execution timestamp")
    route: Optional[str] = Field(None, description="Order route")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "exec_id": "EXEC-order-abc123",
                "order_id": "order-abc123",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0,
                "timestamp": "2025-01-15T10:30:00Z",
                "route": "PAPER"
            }
        }
    )


class OmsRoute(BaseModel):
    """OMS route configuration.
    
    Represents a routing destination for orders (e.g., paper trading, broker APIs).
    """
    route_id: str = Field(..., description="Route identifier (e.g., 'PAPER', 'IBKR')")
    description: str = Field(..., description="Human-readable route description")
    is_default: bool = Field(False, description="Whether this is the default route")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route_id": "PAPER",
                "description": "Internal paper trading route",
                "is_default": True
            }
        }
    )
