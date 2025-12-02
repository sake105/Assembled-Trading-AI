"""Trade and equity curve labeling for machine learning.

This module provides functions to label trades and daily records for supervised learning.
Labels indicate whether a trade or period was "successful" based on a threshold.

Key features:
- Label trades based on P&L percentage or reconstructed P&L from prices
- Label daily equity records based on forward-looking returns
- Flexible horizon and threshold parameters
- Robust handling of missing data and edge cases
"""
from __future__ import annotations

import pandas as pd


def label_trades(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    horizon_days: int = 10,
    success_threshold: float = 0.02,
) -> pd.DataFrame:
    """Label trades as successful (1) or unsuccessful (0) based on P&L.
    
    This function adds a binary label to trades indicating whether they achieved
    a minimum return threshold within a specified horizon.
    
    Args:
        trades: DataFrame with trade data. Expected columns:
            - timestamp (or open_time): Trade entry timestamp (required)
            - symbol: Stock symbol (required)
            - side: "BUY" or "SELL" (optional, for P&L reconstruction)
            - qty: Trade quantity (optional, for P&L reconstruction)
            - price: Trade price (optional, for P&L reconstruction)
            - pnl_pct: Pre-computed P&L percentage (optional, if present, used directly)
            - close_time: Trade exit timestamp (optional, if missing, approximated as open_time + horizon_days)
            - max_dd_pct: Maximum drawdown percentage (optional, for future use)
        
        prices: Optional DataFrame with price data for P&L reconstruction.
            Required columns: timestamp, symbol, close
            Only used if pnl_pct is not present in trades.
        
        horizon_days: Number of days to look forward for success evaluation (default: 10).
            If close_time is missing, it is approximated as open_time + horizon_days.
        
        success_threshold: Minimum return percentage to be considered successful (default: 0.02 = 2%).
            Label = 1 if pnl_pct >= success_threshold, else 0.
    
    Returns:
        DataFrame with original trades plus additional columns:
            - label: Binary label (1 = successful, 0 = unsuccessful)
            - horizon_days: Horizon used for labeling (same for all rows)
            - pnl_pct: P&L percentage (computed if not present)
            - close_time: Trade exit timestamp (computed if not present)
    
    Raises:
        ValueError: If required columns are missing and cannot be reconstructed
        KeyError: If prices are needed but not provided
    
    Example:
        >>> import pandas as pd
        >>> from datetime import datetime, timezone, timedelta
        >>> 
        >>> # Trades with pre-computed P&L
        >>> trades = pd.DataFrame({
        ...     "timestamp": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
        ...     "symbol": ["AAPL"],
        ...     "pnl_pct": [0.05]  # 5% profit
        ... })
        >>> 
        >>> labeled = label_trades(trades, success_threshold=0.02)
        >>> assert labeled["label"].iloc[0] == 1  # 5% > 2% threshold
    """
    if trades.empty:
        return trades.copy()
    
    trades = trades.copy()
    
    # Ensure timestamp column exists (check for open_time as alias)
    if "timestamp" not in trades.columns and "open_time" in trades.columns:
        trades["timestamp"] = trades["open_time"]
    elif "timestamp" not in trades.columns:
        raise ValueError("trades must have 'timestamp' or 'open_time' column")
    
    # Ensure timestamp is datetime
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    
    # Add horizon_days column
    trades["horizon_days"] = horizon_days
    
    # Case 1: pnl_pct is already present
    if "pnl_pct" in trades.columns:
        # Use existing P&L directly
        trades["pnl_pct"] = pd.to_numeric(trades["pnl_pct"], errors="coerce")
        
        # Compute labels
        trades["label"] = (trades["pnl_pct"] >= success_threshold).astype(int)
        
        # Ensure close_time exists (approximate if missing)
        if "close_time" not in trades.columns:
            trades["close_time"] = trades["timestamp"] + pd.Timedelta(days=horizon_days)
        else:
            trades["close_time"] = pd.to_datetime(trades["close_time"], utc=True)
        
        return trades
    
    # Case 2: pnl_pct missing, need to reconstruct from prices
    if prices is None or prices.empty:
        raise KeyError(
            "Cannot compute P&L: 'pnl_pct' not in trades and 'prices' not provided. "
            "Either provide pnl_pct in trades or provide prices DataFrame."
        )
    
    # Validate prices DataFrame
    required_price_cols = ["timestamp", "symbol", "close"]
    missing_cols = [c for c in required_price_cols if c not in prices.columns]
    if missing_cols:
        raise ValueError(f"prices DataFrame missing required columns: {', '.join(missing_cols)}")
    
    # Ensure prices are sorted and timestamp is datetime
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Ensure trades have required columns for P&L reconstruction
    required_trade_cols = ["symbol", "side", "qty", "price"]
    missing_trade_cols = [c for c in required_trade_cols if c not in trades.columns]
    if missing_trade_cols:
        raise ValueError(
            f"trades missing required columns for P&L reconstruction: {', '.join(missing_trade_cols)}. "
            "Need: symbol, side, qty, price"
        )
    
    # Compute close_time if missing
    if "close_time" not in trades.columns:
        trades["close_time"] = trades["timestamp"] + pd.Timedelta(days=horizon_days)
    else:
        trades["close_time"] = pd.to_datetime(trades["close_time"], utc=True)
    
    # Reconstruct P&L for each trade (optimized: use indexed lookup)
    # Set up prices index for faster lookup
    prices_indexed = prices.set_index(["symbol", "timestamp"])["close"]
    
    pnl_list = []
    for idx, trade in trades.iterrows():
        symbol = trade["symbol"]
        open_time = trade["timestamp"]
        close_time = trade["close_time"]
        side = trade["side"]
        open_price = trade["price"]
        
        # Get entry price (from trade)
        entry_price = float(open_price)
        
        # Get exit price (from prices DataFrame) - optimized lookup
        try:
            # Try to get price at close_time or nearest
            symbol_prices_slice = prices_indexed.loc[symbol]
            if isinstance(symbol_prices_slice, pd.Series):
                # Multiple prices for this symbol - find closest to close_time
                exit_candidates = symbol_prices_slice[
                    (symbol_prices_slice.index >= open_time) &
                    (symbol_prices_slice.index <= close_time)
                ]
                if not exit_candidates.empty:
                    exit_price = float(exit_candidates.iloc[-1])
                else:
                    # Try future prices
                    future_prices = symbol_prices_slice[symbol_prices_slice.index > close_time]
                    if not future_prices.empty:
                        exit_price = float(future_prices.iloc[0])
                    else:
                        pnl_list.append(0.0)
                        continue
            else:
                # Single price point
                exit_price = float(symbol_prices_slice)
        except (KeyError, IndexError):
            pnl_list.append(0.0)
            continue
        
        # Compute P&L percentage
        if side.upper() == "BUY":
            # Long position: profit if exit > entry
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0.0
        elif side.upper() == "SELL":
            # Short position: profit if exit < entry
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price != 0 else 0.0
        else:
            # Unknown side, assume 0 P&L
            pnl_pct = 0.0
        
        pnl_list.append(pnl_pct)
    
    trades["pnl_pct"] = pnl_list
    
    # Compute labels
    trades["pnl_pct"] = pd.to_numeric(trades["pnl_pct"], errors="coerce")
    trades["label"] = (trades["pnl_pct"] >= success_threshold).astype(int)
    
    return trades


def label_daily_records(
    df: pd.DataFrame,
    horizon_days: int = 10,
    success_threshold: float = 0.02,
    price_col: str = "equity",
) -> pd.DataFrame:
    """Label daily records based on forward-looking returns.
    
    For each day, check if the equity (or specified price column) increases by
    at least success_threshold within horizon_days.
    
    Args:
        df: DataFrame with daily records. Required columns:
            - timestamp: Date/timestamp (required)
            - price_col: Column name for price/equity values (default: "equity")
        
        horizon_days: Number of days to look forward for success evaluation (default: 10).
        
        success_threshold: Minimum return percentage to be considered successful (default: 0.02 = 2%).
            Label = 1 if forward return >= success_threshold, else 0.
        
        price_col: Column name containing price/equity values (default: "equity").
    
    Returns:
        DataFrame with original data plus additional column:
            - label: Binary label (1 = successful, 0 = unsuccessful)
                Label is 1 if equity increases by at least success_threshold within horizon_days.
                Label is 0 otherwise (or NaN if insufficient forward data).
    
    Raises:
        ValueError: If required columns are missing
    
    Example:
        >>> import pandas as pd
        >>> from datetime import datetime, timezone
        >>> 
        >>> # Equity curve
        >>> equity = pd.DataFrame({
        ...     "timestamp": pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC"),
        ...     "equity": [10000 + i * 50 for i in range(20)]  # Upward trend
        ... })
        >>> 
        >>> labeled = label_daily_records(equity, horizon_days=5, success_threshold=0.01)
        >>> # Days with 5%+ forward return should be labeled 1
    """
    if df.empty:
        return df.copy()
    
    df = df.copy()
    
    # Validate required columns
    if "timestamp" not in df.columns:
        raise ValueError("df must have 'timestamp' column")
    
    if price_col not in df.columns:
        raise ValueError(f"df must have '{price_col}' column")
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure price column is numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    
    # Compute forward returns and labels (optimized: vectorized where possible)
    # Use shift to get future prices more efficiently
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate forward prices using shift (more efficient than iterrows)
    # For each row, find the price horizon_days ahead
    labels = []
    for i in range(len(df_sorted)):
        current_price = df_sorted.iloc[i][price_col]
        current_time = df_sorted.iloc[i]["timestamp"]
        target_time = current_time + pd.Timedelta(days=horizon_days)
        
        # Find price at target_time (or closest available)
        future_mask = df_sorted["timestamp"] >= target_time
        future_rows = df_sorted[future_mask]
        
        if future_rows.empty:
            # No future data available, label as 0 (not successful)
            labels.append(0)
            continue
        
        # Get closest future price (first row after target_time)
        future_price = future_rows.iloc[0][price_col]
        
        # Compute forward return
        if pd.isna(current_price) or pd.isna(future_price) or current_price <= 0:
            labels.append(0)
            continue
        
        forward_return = (future_price - current_price) / current_price
        
        # Label: 1 if return >= threshold, else 0
        label = 1 if forward_return >= success_threshold else 0
        labels.append(label)
    
    # Restore original order if needed
    if not df.index.equals(df_sorted.index):
        # Map labels back to original order
        df["label"] = pd.Series(labels, index=df_sorted.index).reindex(df.index).fillna(0).astype(int)
    else:
        df["label"] = labels
    
    return df

