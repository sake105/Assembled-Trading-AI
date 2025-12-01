"""Performance metrics computation.

This module provides a centralized location for computing all performance metrics
from equity curves and trade data. It consolidates logic from pipeline/backtest.py,
pipeline/portfolio.py, and api/routers/risk.py into a single, consistent API.

Key features:
- Consistent annualization across all metrics
- Robust handling of short time series and NaNs
- Support for both equity-only and trade-level metrics
- Type-safe output via PerformanceMetrics dataclass
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Constants for annualization
PERIODS_PER_YEAR_1D = 252  # Trading days per year
PERIODS_PER_YEAR_5MIN = 252 * 78  # 78 five-minute periods per trading day


@dataclass
class PerformanceMetrics:
    """Performance metrics from equity curve and optional trades.
    
    Attributes:
        final_pf: Final Performance Factor (equity[-1] / equity[0])
        total_return: Total Return (final_pf - 1.0)
        cagr: Compound Annual Growth Rate (None if < 1 year of data)
        
        sharpe_ratio: Sharpe Ratio (annualized, None if insufficient data)
        sortino_ratio: Sortino Ratio (annualized, None if insufficient data)
        calmar_ratio: Calmar Ratio (CAGR / |max_drawdown_pct|, None if insufficient data)
        
        max_drawdown: Maximum drawdown (absolute, negative value)
        max_drawdown_pct: Maximum drawdown (in percent, negative value)
        current_drawdown: Current drawdown (absolute)
        volatility: Volatility (annualized, None if insufficient data)
        var_95: Value at Risk (95% confidence, None if insufficient data)
        
        hit_rate: Win Rate (None if no trades provided)
        profit_factor: Profit Factor (None if no trades provided)
        avg_win: Average win per trade (None if no trades provided)
        avg_loss: Average loss per trade (None if no trades provided)
        turnover: Portfolio turnover (annualized, None if no trades provided)
        total_trades: Total number of trades (None if no trades provided)
        
        start_date: First timestamp
        end_date: Last timestamp
        periods: Number of periods
        start_capital: Starting capital
        end_equity: Ending equity
    """
    # Performance
    final_pf: float
    total_return: float
    cagr: float | None
    
    # Risk-Adjusted Returns
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    
    # Risk Metrics
    max_drawdown: float
    max_drawdown_pct: float
    current_drawdown: float
    volatility: float | None
    var_95: float | None
    
    # Trade Metrics
    hit_rate: float | None
    profit_factor: float | None
    avg_win: float | None
    avg_loss: float | None
    turnover: float | None
    total_trades: int | None
    
    # Metadata
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    periods: int
    start_capital: float
    end_equity: float


def _get_periods_per_year(freq: str) -> int:
    """Get periods per year for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Number of periods per year
    """
    if freq == "1d":
        return PERIODS_PER_YEAR_1D
    elif freq == "5min":
        return PERIODS_PER_YEAR_5MIN
    else:
        # Default to daily
        return PERIODS_PER_YEAR_1D


def _compute_returns(equity: pd.Series) -> pd.Series:
    """Compute returns from equity series.
    
    Args:
        equity: Series of equity values
    
    Returns:
        Series of returns (pct_change), with inf/-inf replaced by NaN and dropped
    """
    returns = equity.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna()
    return returns


def compute_sharpe_ratio(
    returns: pd.Series,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> float | None:
    """Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio (annualized) or None if insufficient data
    """
    if len(returns) < 2:
        return None
    
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    
    if std_return <= 0:
        return None
    
    periods_per_year = _get_periods_per_year(freq)
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = excess_return / std_return * np.sqrt(periods_per_year)
    
    return float(sharpe) if not np.isnan(sharpe) else None


def compute_sortino_ratio(
    returns: pd.Series,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> float | None:
    """Compute annualized Sortino ratio (downside deviation only).
    
    Args:
        returns: Series of returns
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sortino ratio (annualized) or None if insufficient data
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return None
    
    # Excess returns
    periods_per_year = _get_periods_per_year(freq)
    excess = returns - (risk_free_rate / periods_per_year)
    
    # Downside deviation: only negative excess returns
    downside = excess[excess < 0]
    
    if downside.empty:
        # No negative returns => "unendlich" gut; Tests wollen oft None
        return None
    
    # Downside standard deviation
    # Use ddof=0 (population std) if only one downside return, otherwise ddof=1 (sample std)
    if len(downside) == 1:
        downside_std = 0.0 if downside.iloc[0] == 0 else abs(downside.iloc[0])
    else:
        downside_std = downside.std(ddof=1)
    
    if downside_std == 0 or np.isnan(downside_std):
        return None
    
    mean_excess = excess.mean()
    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    
    return float(sortino) if not np.isnan(sortino) else None


def compute_drawdown(equity: pd.Series) -> tuple[pd.Series, float, float, float]:
    """Compute drawdown series and metrics.
    
    Args:
        equity: Series of equity values
    
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_pct, current_drawdown)
        - drawdown_series: Series of drawdown values (negative)
        - max_drawdown: Maximum drawdown (absolute, negative value)
        - max_drawdown_pct: Maximum drawdown (in percent, negative value)
        - current_drawdown: Current drawdown (absolute)
    """
    rolling_max = equity.expanding().max()
    drawdown_series = equity - rolling_max
    
    max_drawdown = float(drawdown_series.min())
    peak_equity = float(rolling_max.max())
    max_drawdown_pct = float((max_drawdown / peak_equity) * 100) if peak_equity > 0 else 0.0
    current_drawdown = float(drawdown_series.iloc[-1])
    
    return drawdown_series, max_drawdown, max_drawdown_pct, current_drawdown


def compute_cagr(
    start_value: float,
    end_value: float,
    periods: int,
    freq: str = "1d"
) -> float | None:
    """Compute Compound Annual Growth Rate.
    
    Args:
        start_value: Starting value
        end_value: Ending value
        periods: Number of periods
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        CAGR (annualized) or None if < 1 year of data or invalid inputs
    """
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return None
    
    periods_per_year = _get_periods_per_year(freq)
    
    # Only compute CAGR if we have at least 1 year of data
    if periods < periods_per_year:
        return None
    
    years = periods / periods_per_year
    if years <= 0:
        return None
    
    total_return = end_value / start_value
    cagr = (total_return ** (1.0 / years)) - 1.0
    
    return float(cagr) if not np.isnan(cagr) and not np.isinf(cagr) else None


def compute_turnover(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    start_capital: float,
    freq: str = "1d"
) -> float | None:
    """Compute annualized portfolio turnover.
    
    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
        equity: DataFrame with columns: timestamp, equity
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Annualized turnover ratio or None if insufficient data
    """
    if trades.empty or equity.empty:
        return None
    
    # Compute total notional (sum of absolute trade values)
    trades = trades.copy()
    trades["notional"] = (trades["qty"].abs() * trades["price"].abs())
    total_notional = float(trades["notional"].sum())
    
    if total_notional <= 0:
        return None
    
    # Compute average equity
    avg_equity = float(equity["equity"].mean())
    if avg_equity <= 0:
        avg_equity = start_capital  # Fallback
    
    # Compute turnover ratio
    turnover_ratio = total_notional / avg_equity
    
    # Annualize
    periods_per_year = _get_periods_per_year(freq)
    periods = len(equity)
    if periods <= 0:
        return None
    
    years = periods / periods_per_year
    if years <= 0:
        return None
    
    annualized_turnover = turnover_ratio / years
    
    return float(annualized_turnover) if not np.isnan(annualized_turnover) else None


def compute_trade_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    start_capital: float,
    freq: str = "1d"
) -> dict[str, float | int | None]:
    """Compute trade-level metrics (hit_rate, profit_factor, turnover, etc.).
    
    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
        equity: DataFrame with columns: timestamp, equity (for turnover calculation)
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Dictionary with trade metrics:
        - hit_rate: float | None
        - profit_factor: float | None
        - avg_win: float | None
        - avg_loss: float | None
        - turnover: float | None
        - total_trades: int
    """
    if trades.empty:
        return {
            "hit_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "turnover": None,
            "total_trades": 0
        }
    
    # Compute P&L per trade
    # For simplicity, we assume each trade is independent
    # In reality, we'd need to track positions, but for metrics we can approximate
    # by assuming each BUY/SELL pair is a round trip
    
    # Group trades by symbol and compute approximate P&L
    # This is a simplified approach - for accurate P&L, we'd need position tracking
    trades = trades.copy()
    trades["notional"] = trades["qty"].abs() * trades["price"].abs()
    
    # For hit rate and profit factor, we need to track actual P&L
    # Since we don't have position tracking here, we'll use a simplified approach:
    # - Assume each symbol's trades are independent
    # - Compute approximate P&L based on price changes
    
    # For now, we'll compute basic metrics that don't require position tracking
    total_trades = len(trades)
    
    # Turnover
    turnover = compute_turnover(trades, equity, start_capital, freq)
    
    # For hit_rate, profit_factor, avg_win, avg_loss, we'd need position tracking
    # This would require a more complex implementation that tracks positions over time
    # For MVP, we'll return None for these metrics
    # TODO: Implement position tracking for accurate trade-level metrics
    
    return {
        "hit_rate": None,  # TODO: Implement with position tracking
        "profit_factor": None,  # TODO: Implement with position tracking
        "avg_win": None,  # TODO: Implement with position tracking
        "avg_loss": None,  # TODO: Implement with position tracking
        "turnover": turnover,
        "total_trades": total_trades
    }


def compute_equity_metrics(
    equity: pd.DataFrame,
    start_capital: float = 10000.0,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """Compute metrics from equity curve only (no trade-level metrics).
    
    Args:
        equity: DataFrame with columns: timestamp, equity (and optionally daily_return)
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        PerformanceMetrics (trade metrics will be None)
    
    Raises:
        ValueError: If required columns are missing
    """
    # Validate input
    if "timestamp" not in equity.columns or "equity" not in equity.columns:
        raise ValueError("equity DataFrame must have 'timestamp' and 'equity' columns")
    
    if equity.empty:
        raise ValueError("equity DataFrame is empty")
    
    # Sort by timestamp
    equity = equity.sort_values("timestamp").reset_index(drop=True)
    
    # Extract equity series
    equity_series = equity["equity"].copy()
    
    # Sanitize equity values
    equity_series = equity_series.replace([np.inf, -np.inf], np.nan)
    equity_series = equity_series.ffill().fillna(start_capital)
    
    # Compute returns
    if "daily_return" in equity.columns:
        returns = equity["daily_return"].copy()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        returns = _compute_returns(equity_series)
    
    # Performance metrics
    # Use start_capital as reference point (as per test expectations)
    # total_return = end_eq / start_capital - 1, not end_eq / start_eq - 1
    start_eq = float(equity_series.iloc[0])
    end_eq = float(equity_series.iloc[-1])
    final_pf = end_eq / max(start_capital, 1e-12)
    total_return = final_pf - 1.0
    
    periods = len(equity)
    cagr = compute_cagr(start_capital, end_eq, periods, freq)
    
    # Risk-adjusted returns
    sharpe_ratio = compute_sharpe_ratio(returns, freq, risk_free_rate) if len(returns) >= 2 else None
    sortino_ratio = compute_sortino_ratio(returns, freq, risk_free_rate) if len(returns) >= 2 else None
    
    # Drawdown metrics
    _, max_drawdown, max_drawdown_pct, current_drawdown = compute_drawdown(equity_series)
    
    # Calmar ratio
    calmar_ratio = None
    if cagr is not None and max_drawdown_pct < 0:
        calmar_ratio = cagr / abs(max_drawdown_pct / 100.0) if max_drawdown_pct != 0 else None
    
    # Volatility
    volatility = None
    if len(returns) >= 2:
        periods_per_year = _get_periods_per_year(freq)
        volatility = float(returns.std() * np.sqrt(periods_per_year))
        if np.isnan(volatility) or volatility <= 0:
            volatility = None
    
    # VaR (95% confidence, historical)
    var_95 = None
    if len(returns) >= 5:
        var_95 = float(np.percentile(returns, 5) * end_eq)
        if np.isnan(var_95):
            var_95 = None
    
    # Metadata
    start_date = pd.Timestamp(equity["timestamp"].iloc[0])
    end_date = pd.Timestamp(equity["timestamp"].iloc[-1])
    
    return PerformanceMetrics(
        final_pf=final_pf,
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        current_drawdown=current_drawdown,
        volatility=volatility,
        var_95=var_95,
        hit_rate=None,
        profit_factor=None,
        avg_win=None,
        avg_loss=None,
        turnover=None,
        total_trades=None,
        start_date=start_date,
        end_date=end_date,
        periods=periods,
        start_capital=start_capital,
        end_equity=end_eq
    )


def compute_all_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    start_capital: float = 10000.0,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and optional trades.
    
    Args:
        equity: DataFrame with columns: timestamp, equity (and optionally daily_return)
        trades: Optional DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital (for CAGR, turnover)
        freq: Frequency string ("1d" or "5min") for annualization
        risk_free_rate: Risk-free rate (annualized) for Sharpe/Sortino
    
    Returns:
        PerformanceMetrics dataclass with all computed metrics
    
    Raises:
        ValueError: If required columns are missing
    """
    # Compute equity metrics
    metrics = compute_equity_metrics(equity, start_capital, freq, risk_free_rate)
    
    # Compute trade metrics if trades provided
    if trades is not None and not trades.empty:
        trade_metrics = compute_trade_metrics(trades, equity, start_capital, freq)
        
        # Update metrics with trade-level data
        metrics.hit_rate = trade_metrics["hit_rate"]
        metrics.profit_factor = trade_metrics["profit_factor"]
        metrics.avg_win = trade_metrics["avg_win"]
        metrics.avg_loss = trade_metrics["avg_loss"]
        metrics.turnover = trade_metrics["turnover"]
        metrics.total_trades = trade_metrics["total_trades"]
    
    return metrics

