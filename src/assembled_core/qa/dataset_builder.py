"""ML Dataset Builder for Trading Features and Labels.

This module provides functions to build machine learning datasets from backtest results
by combining features (TA, Insider, Shipping, etc.) with trade labels.

Key features:
- Extracts features from prices_with_features DataFrame
- Labels trades using qa.labeling.label_trades
- Joins trades with features to create ML-ready dataset
- Supports filtering by feature prefixes

Sprint 7.1 additions:
- build_ml_dataset_for_strategy(): High-level function to build datasets directly from strategy names
- export_ml_dataset(): Export function supporting both Parquet and CSV formats
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
    feature_prefixes: tuple[str, ...] = (
        "ta_",
        "insider_",
        "congress_",
        "shipping_",
        "news_",
    ),
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
    missing_price_cols = [
        c for c in required_price_cols if c not in prices_with_features.columns
    ]
    if missing_price_cols:
        raise ValueError(
            f"prices_with_features missing required columns: {', '.join(missing_price_cols)}"
        )

    # Ensure timestamps are datetime (work on copy to avoid modifying original)
    prices_with_features = prices_with_features.copy()
    prices_with_features["timestamp"] = pd.to_datetime(
        prices_with_features["timestamp"], utc=True
    )
    # Sort once for efficient processing
    if (
        not prices_with_features.index.is_monotonic_increasing
        or prices_with_features["symbol"].is_monotonic_increasing is False
    ):
        prices_with_features = prices_with_features.sort_values(
            ["symbol", "timestamp"]
        ).reset_index(drop=True)

    # Label trades
    logger.debug(f"Labeling {len(trades)} trades...")
    labeled_trades = label_trades(
        trades=trades,
        prices=prices_with_features if "pnl_pct" not in trades.columns else None,
        horizon_days=label_horizon_days,
        success_threshold=success_threshold,
    )

    # Ensure labeled_trades has timestamp column (may be open_time)
    if (
        "timestamp" not in labeled_trades.columns
        and "open_time" in labeled_trades.columns
    ):
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
        if col in [
            "timestamp",
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            continue

        # Check if column name starts with any prefix
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            feature_cols.append(col)

        # Include TA features if requested (either via "ta_" prefix or explicit patterns)
        if include_ta and any(col.startswith(pattern) for pattern in ta_patterns):
            if col not in feature_cols:
                feature_cols.append(col)

    if not feature_cols:
        logger.warning(
            f"No feature columns found matching prefixes: {feature_prefixes}"
        )
        logger.warning("Available columns:", list(prices_with_features.columns))

    logger.debug(
        f"Extracted {len(feature_cols)} feature columns: {feature_cols[:5]}..."
    )

    # Select columns for join: timestamp, symbol, and all feature columns
    join_cols = ["timestamp", "symbol"] + feature_cols
    prices_subset = prices_with_features[join_cols]

    # Join labeled_trades with features
    # Use merge on symbol + timestamp (with tolerance for exact match)
    dataset = labeled_trades.merge(
        prices_subset,
        on=["symbol", "timestamp"],
        how="inner",  # Only keep trades that have matching features
        suffixes=("", "_feature"),
    )

    # Remove duplicate columns (if any)
    if dataset.columns.duplicated().any():
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    # Sort by timestamp, then symbol (always sort after merge for consistency)
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    logger.info(
        f"Built ML dataset: {len(dataset)} rows, {len(feature_cols)} features, label distribution: {dataset['label'].value_counts().to_dict()}"
    )

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

    logger.info(
        f"Saved ML dataset to {path} ({len(df)} rows, {len(df.columns)} columns)"
    )


def build_ml_dataset_for_strategy(
    strategy_name: str,
    start_date: str,
    end_date: str,
    universe: list[str] | None = None,
    universe_file: Path | str | None = None,
    label_params: dict | None = None,
    price_file: Path | str | None = None,
    freq: str = "1d",
) -> pd.DataFrame:
    """Build ML dataset for a specific strategy by loading data, generating signals, and labeling.

    This function provides a high-level interface to build ML datasets directly from strategy names.
    It handles the complete pipeline: data loading → feature computation → signal generation → labeling.

    Args:
        strategy_name: Strategy name ("trend_baseline" or "event_insider_shipping")
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        universe: Optional list of symbols to include. If None, uses universe_file or default watchlist.
        universe_file: Optional path to universe file (text file with one symbol per line).
            If both universe and universe_file are None, uses default watchlist.
        label_params: Optional dictionary with labeling parameters:
            - horizon_days: int (default: 10)
            - threshold_pct: float (default: 0.05)
            - label_type: str (default: "binary_absolute")
            - benchmark_prices: pd.DataFrame | None (default: None)
        price_file: Optional explicit path to price file. If None, loads from default location.
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        DataFrame with ML-ready dataset:
            - Index: (timestamp, symbol) or flat index
            - Columns:
                - label: Binary or multi-class label
                - realized_return: Actual return achieved
                - entry_price, exit_price: Entry and exit prices
                - All feature columns (with consistent prefix, e.g., feat_* or ta_*, insider_*, etc.)
                - Signal metadata (direction, score, etc.)

        Sorted by timestamp, then symbol.

    Raises:
        ValueError: If strategy_name is unknown or required data is missing
        FileNotFoundError: If price file or universe file not found

    Example:
        >>> from pathlib import Path
        >>>
        >>> dataset = build_ml_dataset_for_strategy(
        ...     strategy_name="trend_baseline",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ...     universe=["AAPL", "MSFT", "GOOGL"],
        ...     label_params={
        ...         "horizon_days": 10,
        ...         "threshold_pct": 0.05,
        ...         "label_type": "binary_absolute"
        ...     }
        ... )
        >>>
        >>> assert "label" in dataset.columns
        >>> assert len(dataset) > 0
    """
    # Default label parameters
    if label_params is None:
        label_params = {}

    horizon_days = label_params.get("horizon_days", 10)
    threshold_pct = label_params.get("threshold_pct", 0.05)
    label_type = label_params.get("label_type", "binary_absolute")
    benchmark_prices = label_params.get("benchmark_prices", None)

    logger.info(f"Building ML dataset for strategy: {strategy_name}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(
        f"Label params: horizon={horizon_days}d, threshold={threshold_pct:.2%}, type={label_type}"
    )

    # Load price data
    from src.assembled_core.data.prices_ingest import (
        load_eod_prices,
        load_eod_prices_for_universe,
    )
    from src.assembled_core.config.settings import get_settings

    settings = get_settings()

    if price_file:
        logger.info(f"Loading prices from explicit file: {price_file}")
        prices = load_eod_prices(price_file=price_file, freq=freq)
    elif universe_file:
        logger.info(f"Loading prices for universe file: {universe_file}")
        prices = load_eod_prices_for_universe(
            universe_file=Path(universe_file), data_dir=settings.data_dir, freq=freq
        )
    elif universe:
        logger.info(f"Loading prices for {len(universe)} symbols")
        prices = load_eod_prices(price_file=None, symbols=universe, freq=freq)
    else:
        # Default: use watchlist
        logger.info("Loading prices for default universe (watchlist.txt)")
        prices = load_eod_prices_for_universe(
            universe_file=settings.watchlist_file, data_dir=settings.data_dir, freq=freq
        )

    if prices.empty:
        raise ValueError("No price data loaded")

    # Filter by date range
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)

    prices = prices[
        (prices["timestamp"] >= start_dt) & (prices["timestamp"] <= end_dt)
    ].copy()

    if prices.empty:
        raise ValueError(f"No price data in date range {start_date} to {end_date}")

    logger.info(
        f"Loaded {len(prices)} price rows for {prices['symbol'].nunique()} symbols"
    )

    # Compute features
    logger.info("Computing features...")
    from src.assembled_core.features.ta_features import add_all_features

    has_ohlc = all(col in prices.columns for col in ["high", "low", "open"])
    if has_ohlc:
        prices_with_features = add_all_features(
            prices,
            ma_windows=(20, 50, 200),
            atr_window=14,
            rsi_window=14,
            include_rsi=True,
        )
    else:
        from src.assembled_core.features.ta_features import (
            add_log_returns,
            add_moving_averages,
        )

        prices_with_features = add_log_returns(prices.copy())
        prices_with_features = add_moving_averages(
            prices_with_features, windows=(20, 50, 200)
        )

    # Add event features if event strategy
    if strategy_name == "event_insider_shipping":
        logger.info("Adding event features (insider, shipping)...")
        from src.assembled_core.features.insider_features import add_insider_features
        from src.assembled_core.features.shipping_features import add_shipping_features
        from src.assembled_core.data.insider_ingest import load_insider_sample
        from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample

        insider_events = load_insider_sample()
        shipping_events = load_shipping_sample()

        prices_with_features = add_insider_features(
            prices_with_features, insider_events
        )
        prices_with_features = add_shipping_features(
            prices_with_features, shipping_events
        )

    # Generate signals
    logger.info(f"Generating signals for strategy: {strategy_name}")

    if strategy_name == "trend_baseline":
        from src.assembled_core.signals.rules_trend import (
            generate_trend_signals_from_prices,
        )
        from src.assembled_core.ema_config import get_default_ema_config

        ema_config = get_default_ema_config(freq)
        signals = generate_trend_signals_from_prices(
            prices_with_features, ma_fast=ema_config.fast, ma_slow=ema_config.slow
        )

    elif strategy_name == "event_insider_shipping":
        from src.assembled_core.signals.rules_event_insider_shipping import (
            generate_event_signals,
        )

        signals = generate_event_signals(prices_with_features)

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Supported: trend_baseline, event_insider_shipping"
        )

    if signals.empty:
        logger.warning("No signals generated")
        return pd.DataFrame()

    # Filter to LONG signals only (for labeling)
    if "direction" in signals.columns:
        long_signals = signals[signals["direction"] == "LONG"].copy()
    else:
        long_signals = signals.copy()

    if long_signals.empty:
        logger.warning("No LONG signals to label")
        return pd.DataFrame()

    logger.info(f"Generated {len(long_signals)} LONG signals")

    # Generate labels
    logger.info("Generating labels...")
    from src.assembled_core.qa.labeling import generate_trade_labels

    labeled_signals = generate_trade_labels(
        prices=prices_with_features,
        signals=long_signals,
        horizon_days=horizon_days,
        threshold_pct=threshold_pct,
        label_type=label_type,
        benchmark_prices=benchmark_prices,
    )

    if labeled_signals.empty:
        logger.warning("No labeled signals after labeling")
        return pd.DataFrame()

    # Extract features and join with labeled signals
    logger.info("Extracting features and building final dataset...")

    # Get feature columns (exclude standard price/volume columns)
    feature_cols = [
        col
        for col in prices_with_features.columns
        if col
        not in ["timestamp", "symbol", "date", "open", "high", "low", "close", "volume"]
        and not col.startswith("_")
    ]

    # Rename features with consistent prefix (optional: can be configured)
    # For now, keep original names but ensure they're identifiable
    feature_subset = prices_with_features[["timestamp", "symbol"] + feature_cols].copy()

    # Join labeled signals with features
    dataset = labeled_signals.merge(
        feature_subset,
        on=["timestamp", "symbol"],
        how="inner",
        suffixes=("", "_feature"),
    )

    # Remove duplicate columns
    if dataset.columns.duplicated().any():
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    # Sort by timestamp, then symbol
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    logger.info(f"Built ML dataset: {len(dataset)} rows, {len(feature_cols)} features")
    if "label" in dataset.columns:
        label_counts = dataset["label"].value_counts().to_dict()
        logger.info(f"Label distribution: {label_counts}")

    return dataset


def export_ml_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",
) -> None:
    """Export ML dataset to file (Parquet or CSV).

    Args:
        df: DataFrame with ML dataset
        output_path: Path to output file
        format: Export format ("parquet" or "csv"), default "parquet"

    Raises:
        ValueError: If DataFrame is empty or format is invalid
        IOError: If file cannot be written

    Example:
        >>> export_ml_dataset(dataset, "output/ml_dataset.parquet", format="parquet")
        >>> export_ml_dataset(dataset, "output/ml_dataset.csv", format="csv")
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "parquet":
        df.to_parquet(output_path, index=False)
        logger.info(
            f"Exported ML dataset to Parquet: {output_path} ({len(df)} rows, {len(df.columns)} columns)"
        )
    elif format.lower() == "csv":
        df.to_csv(output_path, index=False)
        logger.info(
            f"Exported ML dataset to CSV: {output_path} ({len(df)} rows, {len(df.columns)} columns)"
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: 'parquet', 'csv'")
