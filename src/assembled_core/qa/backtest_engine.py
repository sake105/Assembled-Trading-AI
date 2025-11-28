"""Portfolio-level backtest engine.

This module provides a unified backtest engine that orchestrates the complete backtest workflow:
1. Load price data (OHLCV)
2. Compute technical analysis features
3. Generate trading signals
4. Compute target positions (position sizing)
5. Generate orders
6. Simulate equity curve (with or without costs)
7. Compute performance metrics

The engine is designed to be flexible and composable:
- Accepts custom signal functions (callable)
- Accepts custom position sizing functions (callable)
- Supports both cost-free and cost-aware simulation
- Returns equity curve and optional trade list

Example usage:
    >>> from src.assembled_core.data.prices_ingest import load_eod_prices
    >>> from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
    >>> from src.assembled_core.portfolio.position_sizing import compute_target_positions
    >>> 
    >>> # Load prices
    >>> prices = load_eod_prices(freq="1d")
    >>> 
    >>> # Define signal function
    >>> def my_signal_fn(prices_df):
    ...     return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)
    >>> 
    >>> # Define position sizing function
    >>> def my_sizing_fn(signals_df, capital):
    ...     return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
    >>> 
    >>> # Run backtest
    >>> result = run_portfolio_backtest(
    ...     prices=prices,
    ...     signal_fn=my_signal_fn,
    ...     position_sizing_fn=my_sizing_fn,
    ...     start_capital=10000.0,
    ...     commission_bps=0.0,
    ...     spread_w=0.25,
    ...     impact_w=0.5,
    ...     include_trades=True
    ... )
    >>> 
    >>> equity = result["equity"]
    >>> metrics = result["metrics"]
    >>> trades = result["trades"]  # Optional

Zukünftige Integration:
- Nutzt pipeline.backtest.simulate_equity für kostenfreie Simulation
- Nutzt pipeline.portfolio.simulate_with_costs für kostenbewusste Simulation
- Nutzt execution.order_generation für Order-Generierung
- Erweitert um Walk-Forward-Analyse, Monte-Carlo-Simulation, etc.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity
from src.assembled_core.pipeline.portfolio import simulate_with_costs


@dataclass
class BacktestResult:
    """Result of a portfolio-level backtest.
    
    Attributes:
        equity: DataFrame with columns: date, timestamp, equity, daily_return
            Equity curve over time with daily returns
            - date: Date (date object)
            - timestamp: Timestamp (pd.Timestamp, UTC)
            - equity: Portfolio equity value
            - daily_return: Daily return (pct_change of equity)
        metrics: Dictionary with performance metrics:
            - final_pf: Final performance factor (equity[-1] / equity[0])
            - sharpe: Sharpe ratio
            - trades: Number of trades executed
            - Additional metrics may be present
        trades: Optional DataFrame with columns: timestamp, symbol, side, qty, price
            List of all trades executed during backtest
            Only present if include_trades=True in run_portfolio_backtest
        signals: Optional DataFrame with columns: timestamp, symbol, direction, score
            All signals generated during backtest
            Only present if include_signals=True in run_portfolio_backtest
        target_positions: Optional DataFrame with columns: symbol, target_weight, target_qty
            Target positions computed at each rebalancing point
            Only present if include_targets=True in run_portfolio_backtest
    """
    equity: pd.DataFrame
    metrics: dict[str, float | int]
    trades: pd.DataFrame | None = None
    signals: pd.DataFrame | None = None
    target_positions: pd.DataFrame | None = None


def run_portfolio_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    start_capital: float = 10000.0,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    cost_model: CostModel | None = None,
    include_costs: bool = True,
    include_trades: bool = False,
    include_signals: bool = False,
    include_targets: bool = False,
    rebalance_freq: str = "1d",
    compute_features: bool = True,
    feature_config: dict[str, Any] | None = None
) -> BacktestResult:
    """Run a portfolio-level backtest with configurable signal and position sizing functions.
    
    This is the main entry point for portfolio-level backtesting. It orchestrates:
    1. Feature computation (optional)
    2. Signal generation (via signal_fn)
    3. Position sizing (via position_sizing_fn)
    4. Order generation
    5. Equity simulation (with or without costs)
    6. Performance metrics computation
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close (and optionally open, high, low, volume)
            Price data for backtesting. Must be sorted by symbol, then timestamp.
        signal_fn: Callable that takes prices DataFrame and returns signals DataFrame
            Input: DataFrame with columns: timestamp, symbol, close, ... (features if compute_features=True)
            Output: DataFrame with columns: timestamp, symbol, direction, score
            Example: signals.rules_trend.generate_trend_signals_from_prices
        position_sizing_fn: Callable that takes signals DataFrame and capital, returns target positions
            Input: (signals_df: pd.DataFrame, total_capital: float)
            Output: DataFrame with columns: symbol, target_weight, target_qty
            Example: portfolio.position_sizing.compute_target_positions
        start_capital: Starting capital (default: 10000.0)
        commission_bps: Commission in basis points (default: from cost_model or get_default_cost_model)
        spread_w: Spread weight (default: from cost_model or get_default_cost_model)
        impact_w: Market impact weight (default: from cost_model or get_default_cost_model)
        cost_model: Optional CostModel instance. If provided, overrides individual cost parameters.
        include_costs: If True, use cost-aware simulation (pipeline.portfolio.simulate_with_costs)
            If False, use cost-free simulation (pipeline.backtest.simulate_equity)
        include_trades: If True, include trades DataFrame in result (default: False)
        include_signals: If True, include signals DataFrame in result (default: False)
        include_targets: If True, include target_positions DataFrame in result (default: False)
        rebalance_freq: Rebalancing frequency string ("1d" or "5min") for order generation (default: "1d")
        compute_features: If True, compute TA features before signal generation (default: True)
        feature_config: Optional dict with feature configuration:
            - ma_windows: tuple[int, ...] = (20, 50, 200)
            - atr_window: int = 14
            - rsi_window: int = 14
            - include_rsi: bool = True
    
    Returns:
        BacktestResult with:
        - equity: DataFrame with columns: date, timestamp, equity, daily_return
        - metrics: Dictionary with performance metrics (final_pf, sharpe, trades, ...)
        - trades: Optional DataFrame with all trades (if include_trades=True)
        - signals: Optional DataFrame with all signals (if include_signals=True)
        - target_positions: Optional DataFrame with target positions (if include_targets=True)
    
    Raises:
        ValueError: If required columns are missing in prices
        KeyError: If signal_fn or position_sizing_fn return invalid DataFrames
    
    Example:
        >>> # Simple trend-following backtest
        >>> from src.assembled_core.data.prices_ingest import load_eod_prices
        >>> from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
        >>> from src.assembled_core.portfolio.position_sizing import compute_target_positions
        >>> 
        >>> prices = load_eod_prices(freq="1d")
        >>> 
        >>> def signal_fn(prices_df):
        ...     return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)
        >>> 
        >>> def sizing_fn(signals_df, capital):
        ...     return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
        >>> 
        >>> result = run_portfolio_backtest(
        ...     prices=prices,
        ...     signal_fn=signal_fn,
        ...     position_sizing_fn=sizing_fn,
        ...     start_capital=10000.0,
        ...     include_costs=True,
        ...     include_trades=True
        ... )
        >>> 
        >>> print(f"Final PF: {result.metrics['final_pf']:.4f}")
        >>> print(f"Sharpe: {result.metrics['sharpe']:.4f}")
        >>> print(f"Trades: {result.metrics['trades']}")
    """
    # Validate input
    required_cols = ["timestamp", "symbol", "close"]
    missing = [c for c in required_cols if c not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns in prices: {missing}")
    
    # Ensure prices are sorted
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    
    # Step 1: Compute features (optional)
    if compute_features:
        config = feature_config or {}
        prices_with_features = add_all_features(
            prices,
            ma_windows=config.get("ma_windows", (20, 50, 200)),
            atr_window=config.get("atr_window", 14),
            rsi_window=config.get("rsi_window", 14),
            include_rsi=config.get("include_rsi", True)
        )
    else:
        prices_with_features = prices.copy()
    
    # Step 2: Generate signals
    signals = signal_fn(prices_with_features)
    
    # Validate signals
    required_signal_cols = ["timestamp", "symbol", "direction"]
    missing_signal = [c for c in required_signal_cols if c not in signals.columns]
    if missing_signal:
        raise KeyError(f"signal_fn must return DataFrame with columns: {required_signal_cols}. Missing: {missing_signal}")
    
    signals = signals.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Step 3: Compute target positions (group by timestamp for rebalancing)
    all_targets = []
    all_orders = []
    current_positions = pd.DataFrame(columns=["symbol", "qty"])
    
    # Group signals by timestamp for rebalancing
    for timestamp, signal_group in signals.groupby("timestamp"):
        # Compute target positions for this timestamp
        targets = position_sizing_fn(signal_group, start_capital)
        
        # Generate orders to transition from current to target positions
        orders = generate_orders_from_targets(
            target_positions=targets,
            current_positions=current_positions,
            timestamp=timestamp,
            prices=prices[prices["timestamp"] == timestamp] if len(prices[prices["timestamp"] == timestamp]) > 0 else None
        )
        
        # Update current positions (simple: assume all orders execute at order price)
        if not orders.empty:
            # Build position updates from orders
            position_updates = {}
            for _, order in orders.iterrows():
                symbol = order["symbol"]
                side = order["side"]
                qty = order["qty"]
                if symbol not in position_updates:
                    position_updates[symbol] = 0.0
                if side == "BUY":
                    position_updates[symbol] += qty
                elif side == "SELL":
                    position_updates[symbol] -= qty
            
            # Apply updates to current_positions
            for symbol, delta in position_updates.items():
                if current_positions.empty or symbol not in current_positions["symbol"].values:
                    # Add new position
                    new_row = pd.DataFrame({"symbol": [symbol], "qty": [delta]})
                    current_positions = pd.concat([current_positions, new_row], ignore_index=True)
                else:
                    # Update existing position
                    idx = current_positions[current_positions["symbol"] == symbol].index[0]
                    current_positions.loc[idx, "qty"] += delta
            
            # Remove zero positions (optional, for cleanliness)
            current_positions = current_positions[current_positions["qty"].abs() > 1e-6].reset_index(drop=True)
        
        # Store targets and orders
        if include_targets and not targets.empty:
            targets_copy = targets.copy()
            targets_copy["timestamp"] = timestamp
            all_targets.append(targets_copy)
        
        if not orders.empty:
            all_orders.append(orders)
    
    # Combine all orders
    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
        orders_df = orders_df.sort_values("timestamp").reset_index(drop=True)
    else:
        orders_df = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    # Step 4: Simulate equity
    # Get cost parameters
    if cost_model is not None:
        commission_bps = commission_bps if commission_bps is not None else cost_model.commission_bps
        spread_w = spread_w if spread_w is not None else cost_model.spread_w
        impact_w = impact_w if impact_w is not None else cost_model.impact_w
    else:
        default_costs = get_default_cost_model()
        commission_bps = commission_bps if commission_bps is not None else default_costs.commission_bps
        spread_w = spread_w if spread_w is not None else default_costs.spread_w
        impact_w = impact_w if impact_w is not None else default_costs.impact_w
    
    if include_costs:
        equity, metrics = simulate_with_costs(
            orders=orders_df,
            start_capital=start_capital,
            commission_bps=commission_bps,
            spread_w=spread_w,
            impact_w=impact_w,
            freq=rebalance_freq
        )
        # Add trades count to metrics
        metrics["trades"] = len(orders_df)
    else:
        equity = simulate_equity(prices, orders_df, start_capital)
        metrics = compute_metrics(equity)
        metrics["trades"] = len(orders_df)
    
    # Step 5: Enhance equity DataFrame with daily_return
    # Ensure equity has timestamp column (rename if needed)
    if "timestamp" in equity.columns:
        equity = equity.copy()
        # Add date column (date part of timestamp)
        equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
        # Compute daily return
        equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
        # Ensure columns are in correct order: date, timestamp, equity, daily_return
        equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()
    elif "date" in equity.columns:
        # If already has date, add daily_return
        equity = equity.copy()
        equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
        # Ensure columns are in correct order: date, equity, daily_return
        if "timestamp" not in equity.columns:
            equity = equity[["date", "equity", "daily_return"]].copy()
        else:
            equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()
    else:
        # Fallback: create date from index or use timestamp
        equity = equity.copy()
        if equity.index.dtype == "datetime64[ns]":
            equity["date"] = equity.index.date
            equity["timestamp"] = equity.index
        else:
            # Try to infer from timestamp column
            if "timestamp" in equity.columns:
                equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
            else:
                # Last resort: use row number as date surrogate
                equity["date"] = pd.date_range(start="2000-01-01", periods=len(equity), freq="D").date
                equity["timestamp"] = pd.to_datetime(equity["date"])
        equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
        equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()
    
    # Step 6: Build result
    result = BacktestResult(
        equity=equity,
        metrics=metrics,
        trades=orders_df if include_trades else None,
        signals=signals if include_signals else None,
        target_positions=pd.concat(all_targets, ignore_index=True) if include_targets and all_targets else None
    )
    
    return result
