"""Scenario engine for risk analysis.

This module provides functionality to apply various risk scenarios to price data
and equity curves. Scenarios can simulate market crashes, volatility spikes,
shipping blockades, and other stress events.

Supported Scenarios:
- equity_crash: Apply a percentage price decline starting at a specific date
  - shock_magnitude: Percentage change (e.g., -0.20 for -20% crash)
  - Prices are scaled down by the shock_magnitude at shock_start
  - Subsequent prices in the shock period follow the new level
  
- vol_spike: Increase volatility (returns variance) in a time window
  - shock_magnitude: Multiplier on return volatility (e.g., 2.0 for 2x volatility)
  - Returns in the shock period are multiplied by the volatility multiplier
  - Prices are reconstructed from modified returns
  
- shipping_blockade: Apply stronger shocks to symbols with shipping exposure
  - shock_magnitude: Base shock magnitude (e.g., -0.10 for -10%)
  - Symbols with "SHIP" in their name receive an additional 50% of base shock
  - Future: Will integrate with shipping_features to identify shipping-exposed symbols

Usage:
    >>> from src.assembled_core.qa.scenario_engine import Scenario, apply_scenario_to_prices
    >>> from datetime import datetime, timezone
    >>> 
    >>> # Equity crash scenario
    >>> scenario = Scenario(
    ...     name="Market Crash 2024",
    ...     shock_type="equity_crash",
    ...     shock_magnitude=-0.20,  # -20% crash
    ...     shock_start=datetime(2024, 3, 15, tzinfo=timezone.utc),
    ...     shock_end=datetime(2024, 3, 20, tzinfo=timezone.utc)
    ... )
    >>> shocked_prices = apply_scenario_to_prices(prices_df, scenario)
    >>> 
    >>> # Volatility spike scenario
    >>> vol_scenario = Scenario(
    ...     name="Volatility Spike",
    ...     shock_type="vol_spike",
    ...     shock_magnitude=2.0,  # 2x volatility
    ...     shock_start=datetime(2024, 3, 15, tzinfo=timezone.utc)
    ... )
    >>> shocked_prices = apply_scenario_to_prices(prices_df, vol_scenario)
    >>> 
    >>> # Run scenario on equity and compare risk metrics
    >>> from src.assembled_core.qa.scenario_engine import run_scenario_on_equity
    >>> results = run_scenario_on_equity(equity_series, scenario, freq="1d")
    >>> print(f"Baseline VaR: {results['baseline_metrics']['var_95']}")
    >>> print(f"Shocked VaR: {results['shocked_metrics']['var_95']}")

Note:
    - If shock_start is None, scenario applies to entire series
    - If shock_end is None, scenario applies from shock_start to end of series
    - Empty price DataFrames are handled gracefully (returned unchanged)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """Definition of a risk scenario.
    
    Attributes:
        name: Human-readable name for the scenario
        shock_type: Type of shock to apply:
            - "equity_crash": Percentage price decline
            - "vol_spike": Volatility increase (multiplier on returns)
            - "shipping_blockade": Stronger shock for shipping-exposed symbols
        shock_magnitude: Magnitude of the shock:
            - For equity_crash: percentage change (e.g., -0.20 for -20%)
            - For vol_spike: multiplier on return volatility (e.g., 2.0 for 2x)
            - For shipping_blockade: additional shock magnitude for shipping symbols
        shock_start: Start datetime of the shock (None = apply to entire series)
        shock_end: End datetime of the shock (None = apply to entire series or until end)
    """
    name: str
    shock_type: Literal["equity_crash", "vol_spike", "shipping_blockade"]
    shock_magnitude: float
    shock_start: datetime | None = None
    shock_end: datetime | None = None


def apply_scenario_to_prices(
    prices: pd.DataFrame,
    scenario: Scenario
) -> pd.DataFrame:
    """Apply a risk scenario to price data.
    
    This function modifies price data according to the specified scenario.
    The original DataFrame is not modified; a copy is returned.
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close
            (and optionally open, high, low, volume)
        scenario: Scenario definition
    
    Returns:
        DataFrame with modified prices according to the scenario
    
    Raises:
        ValueError: If required columns are missing or scenario type is unknown
    
    Example:
        >>> scenario = Scenario(
        ...     name="Crash",
        ...     shock_type="equity_crash",
        ...     shock_magnitude=-0.20,
        ...     shock_start=datetime(2024, 3, 15, tzinfo=timezone.utc)
        ... )
        >>> shocked_prices = apply_scenario_to_prices(prices_df, scenario)
    """
    if prices.empty:
        return prices.copy()
    
    # Validate required columns
    required_cols = ["timestamp", "symbol", "close"]
    missing_cols = [c for c in required_cols if c not in prices.columns]
    if missing_cols:
        raise ValueError(f"Prices DataFrame missing required columns: {', '.join(missing_cols)}")
    
    # Make a copy to avoid modifying original
    shocked_prices = prices.copy()
    
    # Ensure timestamp is datetime
    shocked_prices["timestamp"] = pd.to_datetime(shocked_prices["timestamp"], utc=True)
    # Store original order for restoration
    shocked_prices["_original_order"] = range(len(shocked_prices))
    shocked_prices = shocked_prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Determine shock window
    if scenario.shock_start is None:
        shock_start = shocked_prices["timestamp"].min()
    else:
        shock_start = pd.to_datetime(scenario.shock_start, utc=True)
    
    if scenario.shock_end is None:
        shock_end = shocked_prices["timestamp"].max()
    else:
        shock_end = pd.to_datetime(scenario.shock_end, utc=True)
    
    # Apply scenario based on type
    if scenario.shock_type == "equity_crash":
        shocked_prices = _apply_equity_crash(shocked_prices, scenario, shock_start, shock_end)
    elif scenario.shock_type == "vol_spike":
        shocked_prices = _apply_vol_spike(shocked_prices, scenario, shock_start, shock_end)
    elif scenario.shock_type == "shipping_blockade":
        shocked_prices = _apply_shipping_blockade(shocked_prices, scenario, shock_start, shock_end)
    else:
        raise ValueError(f"Unknown scenario type: {scenario.shock_type}")
    
    # Restore original order using stored order column
    if "_original_order" in shocked_prices.columns:
        shocked_prices = shocked_prices.sort_values("_original_order").reset_index(drop=True)
        shocked_prices = shocked_prices.drop(columns=["_original_order"])
    else:
        # Fallback: sort by timestamp, symbol to ensure consistent order
        shocked_prices = shocked_prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    return shocked_prices


def _apply_equity_crash(
    prices: pd.DataFrame,
    scenario: Scenario,
    shock_start: pd.Timestamp,
    shock_end: pd.Timestamp
) -> pd.DataFrame:
    """Apply equity crash scenario: reduce prices by percentage.
    
    For each symbol, prices are scaled down by shock_magnitude starting at shock_start.
    The crash is applied once at shock_start, and subsequent prices follow the new level.
    """
    shocked_prices = prices.copy()
    
    # Create mask for shock period
    mask = (shocked_prices["timestamp"] >= shock_start) & (shocked_prices["timestamp"] <= shock_end)
    
    # For each symbol, find the price just before shock_start (baseline)
    for symbol in shocked_prices["symbol"].unique():
        symbol_mask = shocked_prices["symbol"] == symbol
        symbol_data = shocked_prices[symbol_mask].copy()
        
        # Find baseline price (last price before shock_start)
        baseline = symbol_data[symbol_data["timestamp"] < shock_start]
        if baseline.empty:
            # If no baseline, use first price in shock period
            baseline_price = symbol_data[symbol_data["timestamp"] >= shock_start]["close"].iloc[0] if len(symbol_data[symbol_data["timestamp"] >= shock_start]) > 0 else symbol_data["close"].iloc[0]
        else:
            baseline_price = baseline["close"].iloc[-1]
        
        # Apply crash: scale prices in shock period
        shock_mask = symbol_mask & mask
        if shock_mask.any():
            # Calculate shocked price at start of shock period
            shocked_baseline = baseline_price * (1 + scenario.shock_magnitude)
            
            # For prices in shock period, scale them relative to the shocked baseline
            # We maintain the relative movements within the shock period
            shock_prices = shocked_prices.loc[shock_mask, "close"].copy()
            if len(shock_prices) > 0:
                # Scale all prices in shock period by the same factor
                price_ratio = shocked_baseline / baseline_price
                shocked_prices.loc[shock_mask, "close"] = shock_prices * price_ratio
                
                # Also update other price columns if present
                for col in ["open", "high", "low"]:
                    if col in shocked_prices.columns:
                        shocked_prices.loc[shock_mask, col] = shocked_prices.loc[shock_mask, col] * price_ratio
    
    return shocked_prices


def _apply_vol_spike(
    prices: pd.DataFrame,
    scenario: Scenario,
    shock_start: pd.Timestamp,
    shock_end: pd.Timestamp
) -> pd.DataFrame:
    """Apply volatility spike scenario: increase return volatility.
    
    Returns in the shock period are multiplied by the volatility multiplier,
    effectively increasing the variance of price movements.
    """
    shocked_prices = prices.copy()
    
    # Create mask for shock period
    mask = (shocked_prices["timestamp"] >= shock_start) & (shocked_prices["timestamp"] <= shock_end)
    
    # For each symbol, compute returns and apply volatility multiplier
    for symbol in shocked_prices["symbol"].unique():
        symbol_mask = shocked_prices["symbol"] == symbol
        symbol_data = shocked_prices[symbol_mask].copy().sort_values("timestamp").reset_index(drop=True)
        symbol_indices = shocked_prices[symbol_mask].index
        
        # Compute log returns
        symbol_data["log_return"] = np.log(symbol_data["close"] / symbol_data["close"].shift(1))
        symbol_data["log_return"] = symbol_data["log_return"].fillna(0.0)
        
        # Apply volatility multiplier to returns in shock period
        symbol_shock_mask = (symbol_data["timestamp"] >= shock_start) & (symbol_data["timestamp"] <= shock_end)
        if symbol_shock_mask.any():
            # Multiply returns in shock period by volatility multiplier
            symbol_data.loc[symbol_shock_mask, "log_return"] = symbol_data.loc[symbol_shock_mask, "log_return"] * scenario.shock_magnitude
            
            # Reconstruct prices from modified returns
            symbol_data["close"] = symbol_data["close"].iloc[0] * np.exp(symbol_data["log_return"].cumsum())
            
            # Update shocked_prices
            shocked_prices.loc[symbol_indices, "close"] = symbol_data["close"].values
            
            # Also update other price columns if present (simplified: scale by same ratio)
            for col in ["open", "high", "low"]:
                if col in shocked_prices.columns:
                    # Compute ratio of new close to old close for each row
                    old_close = prices.loc[symbol_indices, "close"].values
                    new_close = symbol_data["close"].values
                    close_ratio = new_close / old_close
                    shocked_prices.loc[symbol_indices, col] = prices.loc[symbol_indices, col].values * close_ratio
    
    return shocked_prices


def _apply_shipping_blockade(
    prices: pd.DataFrame,
    scenario: Scenario,
    shock_start: pd.Timestamp,
    shock_end: pd.Timestamp
) -> pd.DataFrame:
    """Apply shipping blockade scenario: stronger shock for shipping-exposed symbols.
    
    This is a placeholder implementation. Symbols with "SHIP" in their name
    (or other shipping indicators) receive an additional shock on top of the base shock.
    
    Future: Integrate with shipping_features to identify shipping-exposed symbols.
    """
    shocked_prices = prices.copy()
    
    # Identify shipping-exposed symbols (placeholder: symbols with "SHIP" in name)
    shipping_symbols = [s for s in shocked_prices["symbol"].unique() if "SHIP" in str(s).upper()]
    
    # Apply base equity crash to all symbols
    base_scenario = Scenario(
        name=f"{scenario.name}_base",
        shock_type="equity_crash",
        shock_magnitude=scenario.shock_magnitude,
        shock_start=scenario.shock_start,
        shock_end=scenario.shock_end
    )
    shocked_prices = _apply_equity_crash(shocked_prices, base_scenario, shock_start, shock_end)
    
    # Apply additional shock to shipping symbols
    if shipping_symbols:
        additional_shock = Scenario(
            name=f"{scenario.name}_shipping_additional",
            shock_type="equity_crash",
            shock_magnitude=scenario.shock_magnitude * 0.5,  # Additional 50% of base shock
            shock_start=scenario.shock_start,
            shock_end=scenario.shock_end
        )
        
        # Apply additional shock only to shipping symbols
        shipping_mask = shocked_prices["symbol"].isin(shipping_symbols)
        shipping_prices = shocked_prices[shipping_mask].copy()
        shipping_shocked = _apply_equity_crash(shipping_prices, additional_shock, shock_start, shock_end)
        shocked_prices.loc[shipping_mask, "close"] = shipping_shocked["close"].values
    
    return shocked_prices


def apply_scenario_to_equity(
    equity: pd.Series | pd.DataFrame,
    scenario: Scenario
) -> pd.Series:
    """Apply a risk scenario to an equity curve.
    
    This is a simplified version that applies scenarios directly to equity values.
    For equity_crash, it scales equity down by the shock magnitude.
    For vol_spike, it increases the volatility of equity returns.
    
    Args:
        equity: Series of equity values, or DataFrame with 'equity' column
        scenario: Scenario definition
    
    Returns:
        Series of modified equity values
    
    Example:
        >>> equity = pd.Series([10000, 10100, 10200, 10300])
        >>> scenario = Scenario(
        ...     name="Crash",
        ...     shock_type="equity_crash",
        ...     shock_magnitude=-0.20
        ... )
        >>> shocked_equity = apply_scenario_to_equity(equity, scenario)
    """
    # Extract equity series if DataFrame provided
    if isinstance(equity, pd.DataFrame):
        if "equity" not in equity.columns:
            raise ValueError("DataFrame must have 'equity' column")
        equity_series = equity["equity"].copy()
    else:
        equity_series = equity.copy()
    
    if equity_series.empty:
        return equity_series
    
    # For equity curves, we primarily support equity_crash
    if scenario.shock_type == "equity_crash":
        # Simple: scale equity down by shock_magnitude starting at shock_start
        shocked_equity = equity_series.copy()
        
        if scenario.shock_start is not None:
            # Convert to datetime index if needed
            if not isinstance(equity_series.index, pd.DatetimeIndex):
                # Try to infer from timestamp column if DataFrame
                if isinstance(equity, pd.DataFrame) and "timestamp" in equity.columns:
                    shocked_equity.index = pd.to_datetime(equity["timestamp"], utc=True)
                else:
                    # Use integer index and assume daily frequency
                    shocked_equity.index = pd.date_range(start="2000-01-01", periods=len(shocked_equity), freq="D")
            
            shock_start = pd.to_datetime(scenario.shock_start, utc=True)
            shock_end = pd.to_datetime(scenario.shock_end, utc=True) if scenario.shock_end else shocked_equity.index.max()
            
            mask = (shocked_equity.index >= shock_start) & (shocked_equity.index <= shock_end)
            if mask.any():
                # Scale equity in shock period
                shocked_equity.loc[mask] = shocked_equity.loc[mask] * (1 + scenario.shock_magnitude)
        else:
            # Apply to entire series
            shocked_equity = shocked_equity * (1 + scenario.shock_magnitude)
        
        return shocked_equity
    elif scenario.shock_type == "vol_spike":
        # For vol_spike, increase volatility of returns
        returns = equity_series.pct_change().fillna(0.0)
        
        if scenario.shock_start is not None:
            if not isinstance(equity_series.index, pd.DatetimeIndex):
                if isinstance(equity, pd.DataFrame) and "timestamp" in equity.columns:
                    returns.index = pd.to_datetime(equity["timestamp"], utc=True)
                else:
                    returns.index = pd.date_range(start="2000-01-01", periods=len(returns), freq="D")
            
            shock_start = pd.to_datetime(scenario.shock_start, utc=True)
            shock_end = pd.to_datetime(scenario.shock_end, utc=True) if scenario.shock_end else returns.index.max()
            
            mask = (returns.index >= shock_start) & (returns.index <= shock_end)
            if mask.any():
                returns.loc[mask] = returns.loc[mask] * scenario.shock_magnitude
        
        # Reconstruct equity from modified returns
        shocked_equity = equity_series.iloc[0] * (1 + returns).cumprod()
        return shocked_equity
    else:
        # For other scenario types, return original (or implement as needed)
        return equity_series


def run_scenario_on_equity(
    equity: pd.Series | pd.DataFrame,
    scenario: Scenario,
    freq: str = "1d"
) -> dict[str, float]:
    """Run a scenario on equity and compute risk metrics before/after.
    
    This helper function applies a scenario to an equity curve and computes
    portfolio risk metrics for both the baseline and shocked equity.
    
    Args:
        equity: Series of equity values, or DataFrame with 'equity' column
        scenario: Scenario definition
        freq: Trading frequency ("1d" or "5min") for risk metric computation
    
    Returns:
        Dictionary with:
        - baseline_metrics: Risk metrics for original equity
        - shocked_metrics: Risk metrics for shocked equity
        - delta_metrics: Difference between shocked and baseline metrics
    
    Example:
        >>> from src.assembled_core.qa.scenario_engine import Scenario, run_scenario_on_equity
        >>> 
        >>> scenario = Scenario(
        ...     name="Crash",
        ...     shock_type="equity_crash",
        ...     shock_magnitude=-0.20
        ... )
        >>> 
        >>> results = run_scenario_on_equity(equity_series, scenario)
        >>> print(f"Baseline VaR: {results['baseline_metrics']['var_95']}")
        >>> print(f"Shocked VaR: {results['shocked_metrics']['var_95']}")
    """
    from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics
    
    # Compute baseline metrics
    baseline_metrics = compute_portfolio_risk_metrics(equity, freq=freq)
    
    # Apply scenario
    shocked_equity = apply_scenario_to_equity(equity, scenario)
    
    # Compute shocked metrics
    shocked_metrics = compute_portfolio_risk_metrics(shocked_equity, freq=freq)
    
    # Compute deltas (where both are not None)
    delta_metrics = {}
    for key in baseline_metrics:
        baseline_val = baseline_metrics[key]
        shocked_val = shocked_metrics[key]
        
        if baseline_val is not None and shocked_val is not None:
            if isinstance(baseline_val, (int, float)) and isinstance(shocked_val, (int, float)):
                delta_metrics[key] = shocked_val - baseline_val
            else:
                delta_metrics[key] = None
        else:
            delta_metrics[key] = None
    
    return {
        "baseline_metrics": baseline_metrics,
        "shocked_metrics": shocked_metrics,
        "delta_metrics": delta_metrics
    }

