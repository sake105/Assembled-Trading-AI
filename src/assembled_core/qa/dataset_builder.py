"""ML Dataset Builder for Trading Features and Labels.

This module provides functions to build machine learning datasets from backtest results
by combining features (TA, Insider, Shipping, etc.) with trade labels.

Key features:
- Extracts features from prices_with_features DataFrame
- Labels trades using qa.labeling.label_trades
- Joins trades with features to create ML-ready dataset
- Supports filtering by feature prefixes
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.logging_utils import get_logger
from src.assembled_core.qa.labeling import label_trades

logger = get_logger(__name__)


def build_ml_dataset_from_backtest(
    prices_with_features: pd.DataFrame,
    trades: pd.DataFrame,
    label_horizon_days: int = 10,
    success_threshold: float = 0.02,
    feature_prefixes: tuple[str, ...] = ("ta_", "insider_", "congress_", "shipping_", "news_"),
) -> pd.DataFrame:
    """Build ML dataset from backtest results by joining features with labeled trades.
    
    This function:
    1. Labels trades using label_trades() from qa.labeling
    2. Extracts feature columns from prices_with_features based on feature_prefixes
    3. Joins trades (with labels) with features on symbol + timestamp
    4. Returns a flat table ready for ML training
    
    Args:
        prices_with_features: DataFrame with price data and computed features.
            Required columns: timestamp, symbol, close
            Feature columns should match feature_prefixes (e.g., "insider_net_buy_20d", "ma_20", etc.)
        
        trades: DataFrame with trade data. Expected columns:
            - timestamp (or open_time): Trade entry timestamp (required)
            - symbol: Stock symbol (required)
            - side: "BUY" or "SELL" (required for P&L reconstruction if pnl_pct missing)
            - qty: Trade quantity (required for P&L reconstruction if pnl_pct missing)
            - price: Trade price (required for P&L reconstruction if pnl_pct missing)
            - pnl_pct: Pre-computed P&L percentage (optional, if present, used directly)
        
        label_horizon_days: Number of days to look forward for success evaluation (default: 10).
            Passed to label_trades().
        
        success_threshold: Minimum return percentage to be considered successful (default: 0.02 = 2%).
            Passed to label_trades().
        
        feature_prefixes: Tuple of feature prefixes to extract (default: all event feature prefixes).
            Features whose names start with any of these prefixes will be included.
            Note: TA features (ma_*, atr_*, rsi_*, log_return) don't have a "ta_" prefix,
            but can be included by adding specific column names or using a custom filter.
    
    Returns:
        DataFrame with columns:
            - label: Binary label (1 = successful, 0 = unsuccessful)
            - timestamp (or open_time): Trade entry timestamp
            - symbol: Stock symbol
            - All feature columns matching feature_prefixes
            - Optional: pnl_pct, horizon_days, close_time (from label_trades)
        
        Sorted by timestamp, then symbol.
        Only includes trades that could be matched with features.
    
    Raises:
        ValueError: If required columns are missing
        KeyError: If prices are needed for P&L reconstruction but not provided
    
    Example:
        >>> import pandas as pd
        >>> from datetime import datetime, timezone
        >>> 
        >>> # Prices with features
        >>> prices = pd.DataFrame({
        ...     "timestamp": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
        ...     "symbol": ["AAPL"],
        ...     "close": [100.0],
        ...     "ma_20": [99.0],
        ...     "insider_net_buy_20d": [1000.0],
        ...     "shipping_congestion_score_7d": [30.0],
        ... })
        >>> 
        >>> # Trades
        >>> trades = pd.DataFrame({
        ...     "timestamp": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
        ...     "symbol": ["AAPL"],
        ...     "pnl_pct": [0.05],
        ... })
        >>> 
        >>> dataset = build_ml_dataset_from_backtest(
        ...     prices_with_features=prices,
        ...     trades=trades,
        ...     feature_prefixes=("ma_", "insider_", "shipping_")
        ... )
        >>> 
        >>> assert "label" in dataset.columns
        >>> assert "ma_20" in dataset.columns
        >>> assert "insider_net_buy_20d" in dataset.columns
    """
    if prices_with_features.empty:
        logger.warning("prices_with_features is empty, returning empty dataset")
        return pd.DataFrame()
    
    if trades.empty:
        logger.warning("trades is empty, returning empty dataset")
        return pd.DataFrame()
    
    # Validate required columns
    required_price_cols = ["timestamp", "symbol"]
    missing_price_cols = [c for c in required_price_cols if c not in prices_with_features.columns]
    if missing_price_cols:
        raise ValueError(f"prices_with_features missing required columns: {', '.join(missing_price_cols)}")
    
    # Ensure timestamps are datetime (work on copy to avoid modifying original)
    prices_with_features = prices_with_features.copy()
    prices_with_features["timestamp"] = pd.to_datetime(prices_with_features["timestamp"], utc=True)
    # Sort once for efficient processing
    if not prices_with_features.index.is_monotonic_increasing or prices_with_features["symbol"].is_monotonic_increasing is False:
        prices_with_features = prices_with_features.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Label trades
    logger.debug(f"Labeling {len(trades)} trades...")
    labeled_trades = label_trades(
        trades=trades,
        prices=prices_with_features if "pnl_pct" not in trades.columns else None,
        horizon_days=label_horizon_days,
        success_threshold=success_threshold,
    )
    
    # Ensure labeled_trades has timestamp column (may be open_time)
    if "timestamp" not in labeled_trades.columns and "open_time" in labeled_trades.columns:
        labeled_trades["timestamp"] = labeled_trades["open_time"]
    elif "timestamp" not in labeled_trades.columns:
        raise ValueError("labeled_trades must have 'timestamp' or 'open_time' column")
    
    labeled_trades["timestamp"] = pd.to_datetime(labeled_trades["timestamp"], utc=True)
    
    # Extract feature columns
    # Get all columns that start with any of the feature_prefixes
    feature_cols = []
    
    # TA feature patterns (they don't have "ta_" prefix)
    ta_patterns = ["ma_", "atr_", "rsi_", "log_return"]
    
    # Check if TA features should be included
    # Include if "ta_" is in prefixes OR if any TA pattern is explicitly in prefixes
    include_ta = "ta_" in feature_prefixes
    if not include_ta:
        # Check if any prefix is a TA pattern
        include_ta = any(
            any(prefix.startswith(pattern) for pattern in ta_patterns)
            for prefix in feature_prefixes
        )
    
    for col in prices_with_features.columns:
        # Skip non-feature columns (standard price/volume columns)
        if col in ["timestamp", "symbol", "date", "open", "high", "low", "close", "volume"]:
            continue
        
        # Check if column name starts with any prefix
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            feature_cols.append(col)
        
        # Include TA features if requested (either via "ta_" prefix or explicit patterns)
        if include_ta and any(col.startswith(pattern) for pattern in ta_patterns):
            if col not in feature_cols:
                feature_cols.append(col)
    
    if not feature_cols:
        logger.warning(f"No feature columns found matching prefixes: {feature_prefixes}")
        logger.warning("Available columns:", list(prices_with_features.columns))
    
    logger.debug(f"Extracted {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    # Select columns for join: timestamp, symbol, and all feature columns
    join_cols = ["timestamp", "symbol"] + feature_cols
    prices_subset = prices_with_features[join_cols]
    
    # Join labeled_trades with features
    # Use merge on symbol + timestamp (with tolerance for exact match)
    dataset = labeled_trades.merge(
        prices_subset,
        on=["symbol", "timestamp"],
        how="inner",  # Only keep trades that have matching features
        suffixes=("", "_feature")
    )
    
    # Remove duplicate columns (if any)
    if dataset.columns.duplicated().any():
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    
    # Sort by timestamp, then symbol (always sort after merge for consistency)
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    logger.info(f"Built ML dataset: {len(dataset)} rows, {len(feature_cols)} features, label distribution: {dataset['label'].value_counts().to_dict()}")
    
    return dataset


def save_ml_dataset(df: pd.DataFrame, path: Path | str) -> None:
    """Save ML dataset to Parquet file.
    
    Args:
        df: DataFrame with ML dataset (from build_ml_dataset_from_backtest)
        path: Path to output Parquet file
    
    Raises:
        ValueError: If DataFrame is empty
        IOError: If file cannot be written
    
    Example:
        >>> from pathlib import Path
        >>> 
        >>> dataset = build_ml_dataset_from_backtest(...)
        >>> save_ml_dataset(dataset, Path("output/ml_dataset.parquet"))
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, index=False)
    
    logger.info(f"Saved ML dataset to {path} ({len(df)} rows, {len(df.columns)} columns)")

