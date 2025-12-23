"""Factor Analysis and Information Coefficient (IC) Engine.

This module implements Phase C, Sprints C1 and C2 from the Advanced Analytics & Factor Labs roadmap.
It provides tools for evaluating factor effectiveness:

Phase C1 - IC Engine:
- Forward Returns Computation
- Cross-Sectional IC (Information Coefficient)
- Rank-IC (Spearman correlation)
- IC Aggregation and Summary Statistics (IC-IR, hit ratio, etc.)

Phase C2 - Factor Portfolio Returns:
- Quantile-based portfolio returns
- Long/Short portfolio returns
- Portfolio performance metrics (Sharpe, t-stat, drawdown)
- Deflated Sharpe Ratio (multiple testing adjustment)

The IC Engine measures how well factors predict future returns:
- IC = correlation(factor, forward_return) at each timestamp (cross-sectional)
- IC-IR = mean(IC) / std(IC) (Information Ratio)
- Hit Ratio = percentage of days with positive IC

The Portfolio Engine measures actual portfolio returns when investing based on factor values:
- Quantile portfolios: Sort by factor, invest in quantiles
- Long/Short: Top quantile - Bottom quantile
- Performance metrics: Sharpe, t-stat, drawdown, win ratio

Integration:
- Works with factors from Phase A (ta_factors_core, ta_liquidity_vol_factors)
- Designed for factor research and evaluation
- Compatible with backtest engine and research workflows
- Primary use: Research notebooks, factor evaluation, factor selection

Usage:
    # Add forward returns to price data
    prices_with_returns = add_forward_returns(prices, horizon_days=21)

    # Compute IC for factors
    ic_df = compute_factor_ic(
        df_with_factors,
        factor_cols=["returns_12m", "trend_strength_200", "rv_20"],
        fwd_return_col="fwd_return_21d"
    )

    # Summarize IC statistics
    summary = summarize_factor_ic(ic_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_forward_returns(
    prices: pd.DataFrame,
    horizon_days: int | list[int] = 1,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    col_name: str | None = None,
    return_type: str = "log",
) -> pd.DataFrame:
    """
    Add forward returns column(s) to price or factor DataFrame.

    Computes returns looking forward from current time point (no look-ahead bias).
    Used for factor evaluation: we want to predict these forward returns.

    This function works with both pure price DataFrames and factor DataFrames
    (that already contain factor columns but also have a price column).

    Args:
        prices: DataFrame with price data (panel format: multiple symbols over time)
            Required columns: timestamp_col, group_col, price_col
            Can also be a factor DataFrame that contains price_col
        horizon_days: Number of days forward (default: 1)
            Can be a single integer or a list of integers (e.g., [20, 60, 252])
            For multiple horizons, creates columns: fwd_ret_{horizon} for each horizon
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        col_name: Optional explicit column name for forward returns (only used if horizon_days is int)
            (default: None, auto-generates "fwd_return_{horizon_days}d" or "fwd_ret_{horizon}")
        return_type: Type of return calculation (default: "log")
            - "log": log returns (ln(price[t+h] / price[t]))
            - "simple": simple returns (price[t+h] / price[t] - 1)

    Returns:
        DataFrame with original columns plus new forward return column(s):
        - Single horizon: fwd_return_{horizon_days}d (or col_name if provided)
        - Multiple horizons: fwd_ret_{horizon} for each horizon in horizons list

        Forward returns are computed per symbol (group_col).
        Last horizon_days rows per symbol will have NaN (no future data available).
        For multiple horizons, each horizon is handled independently.

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid

    Example:
        >>> # Single horizon
        >>> df_with_returns = add_forward_returns(prices, horizon_days=20)
        >>> # Creates column: fwd_return_20d
        >>>
        >>> # Multiple horizons
        >>> df_with_returns = add_forward_returns(prices, horizon_days=[20, 60, 252])
        >>> # Creates columns: fwd_ret_20, fwd_ret_60, fwd_ret_252
    """
    # Validate input
    required_cols = [timestamp_col, group_col, price_col]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(prices.columns)}"
        )

    if prices.empty:
        raise ValueError("Input DataFrame is empty")

    result = prices.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)

    # Sort by group and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # Handle single horizon vs. multiple horizons
    if isinstance(horizon_days, int):
        horizons = [horizon_days]
        use_custom_name = col_name is not None
    else:
        horizons = list(horizon_days)
        use_custom_name = False  # Multiple horizons always use auto-generated names
        if not horizons:
            raise ValueError(
                "horizon_days must be a positive integer or a non-empty list of integers"
            )

    # Compute forward returns per symbol
    grouped_price = result.groupby(group_col, group_keys=False)[price_col]

    # For each horizon, compute forward returns
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon}")

        # Forward price: shift(-horizon) to look forward
        forward_price = grouped_price.shift(-horizon)
        current_price = result[price_col]

        # Compute returns based on return_type
        if return_type == "log":
            forward_return = np.log(forward_price / current_price)
        elif return_type == "simple":
            forward_return = (forward_price / current_price) - 1.0
        else:
            raise ValueError(
                f"Invalid return_type: {return_type}. Supported: 'log', 'simple'"
            )

        # Set column name
        if len(horizons) == 1 and use_custom_name:
            # Single horizon with custom name
            col_name_final = col_name
        elif len(horizons) == 1:
            # Single horizon, auto-generate name
            col_name_final = f"fwd_return_{horizon}d"
        else:
            # Multiple horizons, use short format
            col_name_final = f"fwd_ret_{horizon}"

        result[col_name_final] = forward_return.astype("float64")

    if len(horizons) == 1:
        logger.info(
            f"Added forward returns column '{col_name_final}' (horizon: {horizons[0]} days, type: {return_type})"
        )
    else:
        logger.info(
            f"Added forward returns columns for {len(horizons)} horizons: {horizons} (type: {return_type})"
        )

    return result


def compute_factor_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    fwd_return_col: str = "fwd_return_1d",
    group_col: str = "timestamp",
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute cross-sectional Information Coefficient (IC) per timestamp and factor.

    IC measures how well a factor predicts forward returns at each point in time.
    It is computed as the correlation between factor values and forward returns
    across all symbols at each timestamp (cross-sectional correlation).

    Args:
        df: DataFrame with factor and forward return data (panel format)
            Required columns: group_col (typically "timestamp"), symbol (or group_col),
            factor_cols, fwd_return_col
            Format: multiple symbols per timestamp, factors and forward returns per row
        factor_cols: List of factor column names to compute IC for
        fwd_return_col: Column name for forward returns (default: "fwd_return_1d")
        group_col: Column name for grouping (default: "timestamp")
            IC is computed separately for each value of group_col
        method: Correlation method (default: "pearson")
            - "pearson": Pearson correlation (linear)
            - "spearman": Spearman rank correlation

    Returns:
        DataFrame with columns:
        - {group_col}: Grouping value (e.g., timestamp)
        - factor: Factor column name
        - ic: Correlation value (IC) for that factor at that timestamp
        - count: Number of symbols used in correlation calculation

        One row per (group_col, factor) combination.
        Sorted by group_col, then factor.

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or no valid data
    """
    # Validate input
    required_cols = [group_col, fwd_return_col]

    # Check if "symbol" column exists (used for cross-sectional grouping)
    symbol_col = "symbol" if "symbol" in df.columns else None
    if symbol_col is None:
        # Try to infer: if group_col is "timestamp", then we need another column for symbols
        if group_col == "timestamp":
            # Look for a column that looks like symbols
            potential_symbol_cols = [
                col
                for col in df.columns
                if col not in [group_col, fwd_return_col] + factor_cols
            ]
            if potential_symbol_cols:
                symbol_col = potential_symbol_cols[0]
                logger.warning(
                    f"Assuming '{symbol_col}' is the symbol column. Please ensure this is correct."
                )
            else:
                raise KeyError(
                    "Could not identify symbol column. Please ensure DataFrame has a 'symbol' column "
                    "or specify the correct column structure."
                )

    required_cols.append(symbol_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(df.columns)}"
        )

    missing_factors = [col for col in factor_cols if col not in df.columns]
    if missing_factors:
        raise KeyError(
            f"Missing factor columns: {', '.join(missing_factors)}. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    result_df = df.copy()

    # Ensure group_col is datetime if it's a timestamp
    if group_col == "timestamp" and not pd.api.types.is_datetime64_any_dtype(
        result_df[group_col]
    ):
        result_df[group_col] = pd.to_datetime(result_df[group_col], utc=True)

    # Sort by group and symbol
    result_df = result_df.sort_values([group_col, symbol_col]).reset_index(drop=True)

    ic_data = []

    # For each timestamp (group), compute IC for each factor
    for group_value in sorted(result_df[group_col].unique()):
        group_data = result_df[result_df[group_col] == group_value].copy()

        # Filter rows with valid forward return
        valid_mask = group_data[fwd_return_col].notna()
        group_data_valid = group_data[valid_mask]

        if len(group_data_valid) < 3:  # Need at least 3 points for correlation
            continue

        # For each factor, compute correlation with forward return
        for factor_col in factor_cols:
            # Filter rows with valid factor value
            factor_valid_mask = group_data_valid[factor_col].notna()
            factor_data = group_data_valid[factor_valid_mask]

            if len(factor_data) < 3:
                continue

            # Extract factor values and forward returns
            factor_values = factor_data[factor_col].values
            forward_returns = factor_data[fwd_return_col].values

            # Compute correlation using pandas (no scipy dependency)
            if method == "pearson":
                # Pearson correlation: linear correlation
                factor_series = pd.Series(factor_values)
                return_series = pd.Series(forward_returns)
                corr = factor_series.corr(return_series, method="pearson")
            elif method == "spearman":
                # Spearman correlation: rank correlation (implemented manually to avoid scipy dependency)
                # Convert to ranks (handle ties by using average rank)
                factor_series = pd.Series(factor_values)
                return_series = pd.Series(forward_returns)

                # Compute ranks (handling ties by averaging)
                factor_ranks = factor_series.rank(method="average")
                return_ranks = return_series.rank(method="average")

                # Pearson correlation of ranks = Spearman correlation
                corr = factor_ranks.corr(return_ranks, method="pearson")
            else:
                raise ValueError(
                    f"Invalid method: {method}. Supported: 'pearson', 'spearman'"
                )

            # Handle NaN/inf
            if pd.isna(corr) or np.isinf(corr):
                continue

            ic_data.append(
                {
                    group_col: group_value,
                    "factor": factor_col,
                    "ic": float(corr),
                    "count": len(factor_data),
                }
            )

    if not ic_data:
        logger.warning("No IC data computed. Check input data and factor columns.")
        return pd.DataFrame(columns=[group_col, "factor", "ic", "count"])

    ic_df = pd.DataFrame(ic_data)

    # Sort by group_col, then factor
    ic_df = ic_df.sort_values([group_col, "factor"]).reset_index(drop=True)

    logger.info(
        f"Computed IC for {len(factor_cols)} factor(s) across {ic_df[group_col].nunique()} "
        f"{group_col}(s), total {len(ic_df)} IC values"
    )

    return ic_df


def compute_factor_rank_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    fwd_return_col: str = "fwd_return_1d",
    group_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute cross-sectional Rank-IC (Spearman correlation) per timestamp and factor.

    Rank-IC uses Spearman rank correlation instead of Pearson correlation.
    It is more robust to outliers and captures monotonic relationships.

    This is a convenience wrapper around compute_factor_ic with method="spearman".

    Note: This is the legacy function that works with compute_factor_ic().
    For the new API, use compute_rank_ic() which works with compute_ic().

    Args:
        df: DataFrame with factor and forward return data (panel format)
            Required columns: group_col, symbol, factor_cols, fwd_return_col
        factor_cols: List of factor column names to compute Rank-IC for
        fwd_return_col: Column name for forward returns (default: "fwd_return_1d")
        group_col: Column name for grouping (default: "timestamp")

    Returns:
        DataFrame with columns: group_col, factor, ic (Rank-IC), count
        Same format as compute_factor_ic, but IC values are from Spearman correlation.
    """
    return compute_factor_ic(
        df=df,
        factor_cols=factor_cols,
        fwd_return_col=fwd_return_col,
        group_col=group_col,
        method="spearman",
    )


# Legacy alias for backward compatibility
compute_rank_ic_legacy = compute_factor_rank_ic


def summarize_factor_ic(
    ic_df: pd.DataFrame,
    group_col: str = "timestamp",
    ic_col: str = "ic",
) -> pd.DataFrame:
    """
    Aggregate IC time-series to factor-level summary statistics.

    Computes summary statistics for each factor across all timestamps:
    - Mean IC
    - Standard deviation of IC
    - IC-IR (Information Ratio) = mean_ic / std_ic
    - Hit ratio (percentage of days with positive IC)
    - Optional: t-statistic for testing if mean IC is significantly different from zero

    Args:
        ic_df: DataFrame from compute_factor_ic() or compute_rank_ic()
            Required columns: factor, ic_col (default: "ic")
        group_col: Column name for grouping (default: "timestamp")
            Used only for counting unique groups
        ic_col: Column name for IC values (default: "ic")

    Returns:
        DataFrame with columns:
        - factor: Factor name
        - mean_ic: Mean IC across all timestamps
        - std_ic: Standard deviation of IC
        - ic_ir: Information Ratio (mean_ic / std_ic)
        - hit_ratio: Percentage of timestamps with positive IC (0.0 to 1.0)
        - count: Number of timestamps with valid IC values
        - min_ic: Minimum IC value
        - max_ic: Maximum IC value

        One row per factor, sorted by ic_ir (descending).

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty
    """
    # Validate input
    required_cols = ["factor", ic_col]
    missing_cols = [col for col in required_cols if col not in ic_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(ic_df.columns)}"
        )

    if ic_df.empty:
        raise ValueError("Input DataFrame is empty")

    summary_data = []

    # Group by factor and compute summary statistics
    for factor in ic_df["factor"].unique():
        factor_ic = ic_df[ic_df["factor"] == factor][ic_col].dropna()

        if len(factor_ic) == 0:
            continue

        mean_ic = float(factor_ic.mean())
        std_ic = float(factor_ic.std())

        # IC-IR: Information Ratio (mean IC / std IC)
        ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        # Hit ratio: percentage of positive IC values
        hit_ratio = (
            float((factor_ic > 0).sum() / len(factor_ic)) if len(factor_ic) > 0 else 0.0
        )

        # Min/Max IC
        min_ic = float(factor_ic.min())
        max_ic = float(factor_ic.max())

        # Count
        count = len(factor_ic)

        summary_data.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "hit_ratio": hit_ratio,
                "count": count,
                "min_ic": min_ic,
                "max_ic": max_ic,
            }
        )

    if not summary_data:
        logger.warning("No summary statistics computed. Check input IC DataFrame.")
        return pd.DataFrame(
            columns=[
                "factor",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "count",
                "min_ic",
                "max_ic",
            ]
        )

    summary_df = pd.DataFrame(summary_data)

    # Sort by IC-IR (descending) to rank factors
    summary_df = summary_df.sort_values("ic_ir", ascending=False).reset_index(drop=True)

    logger.info(f"Summarized IC statistics for {len(summary_df)} factor(s)")

    return summary_df


def run_factor_report(
    prices: pd.DataFrame,
    factor_set: str = "core",
    fwd_horizon_days: int = 5,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> dict[str, pd.DataFrame]:
    """
    High-level factor report workflow.

    This function orchestrates the complete factor analysis pipeline:
    1. Compute factors based on factor_set
    2. Add forward returns
    3. Compute IC and Rank-IC
    4. Generate summary statistics

    Args:
        prices: DataFrame with price data (panel format: multiple symbols over time)
            Required columns: timestamp_col, symbol_col, price_col
            Optional columns: open, high, low, volume (for enhanced features)
        factor_set: Which factors to compute (default: "core")
            - "core": Core TA/Price factors (multi-horizon returns, trend strength, reversal)
            - "vol_liquidity": Volatility and liquidity factors (realized vol, vol-of-vol, turnover)
            - "all": Combination of both core and vol_liquidity
        fwd_horizon_days: Forward return horizon in days (default: 5)
        timestamp_col: Column name for timestamp (default: "timestamp")
        symbol_col: Column name for symbol (default: "symbol")
        price_col: Column name for price (default: "close")

    Returns:
        Dictionary with keys:
        - "factors": DataFrame with computed factors (original prices + factor columns)
        - "ic": DataFrame with IC values per timestamp and factor
        - "rank_ic": DataFrame with Rank-IC values per timestamp and factor
        - "summary_ic": DataFrame with aggregated IC statistics (mean, std, IC-IR, etc.)
        - "summary_rank_ic": DataFrame with aggregated Rank-IC statistics

    Raises:
        KeyError: If required columns are missing
        ValueError: If factor_set is invalid or DataFrame is empty
    """
    # Validate input
    required_cols = [timestamp_col, symbol_col, price_col]
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(prices.columns)}"
        )

    if prices.empty:
        raise ValueError("Input DataFrame is empty")

    if factor_set not in ["core", "vol_liquidity", "all"]:
        raise ValueError(
            f"Invalid factor_set: {factor_set}. Supported: 'core', 'vol_liquidity', 'all'"
        )

    result_df = prices.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)

    # Sort by symbol and timestamp
    result_df = result_df.sort_values([symbol_col, timestamp_col]).reset_index(
        drop=True
    )

    logger.info(
        f"Starting factor report: factor_set={factor_set}, fwd_horizon={fwd_horizon_days}d, "
        f"symbols={result_df[symbol_col].nunique()}, rows={len(result_df)}"
    )

    # Step 1: Compute factors based on factor_set
    factor_columns = []

    if factor_set in ["core", "all"]:
        logger.info("Computing core TA/Price factors...")
        from src.assembled_core.features.ta_factors_core import build_core_ta_factors

        result_df = build_core_ta_factors(
            result_df,
            price_col=price_col,
            group_col=symbol_col,
            timestamp_col=timestamp_col,
        )

        # Collect core factor columns
        core_factor_patterns = [
            "returns_1m",
            "returns_3m",
            "returns_6m",
            "returns_12m",
            "momentum_12m_excl_1m",
            "trend_strength_20",
            "trend_strength_50",
            "trend_strength_200",
            "reversal_1d",
            "reversal_2d",
            "reversal_3d",
        ]

        for pattern in core_factor_patterns:
            if pattern in result_df.columns:
                factor_columns.append(pattern)

    if factor_set in ["vol_liquidity", "all"]:
        logger.info("Computing volatility and liquidity factors...")
        from src.assembled_core.features.ta_liquidity_vol_factors import (
            add_realized_volatility,
            add_vol_of_vol,
            add_turnover_and_liquidity_proxies,
        )

        # Add realized volatility
        result_df = add_realized_volatility(
            result_df,
            price_col=price_col,
            group_col=symbol_col,
            timestamp_col=timestamp_col,
            windows=[20, 60],
        )

        # Add vol-of-vol (auto-detects rv_* columns)
        result_df = add_vol_of_vol(
            result_df,
            rv_cols=None,  # Auto-detect rv_* columns
            group_col=symbol_col,
            timestamp_col=timestamp_col,
        )

        # Add turnover and liquidity proxies (if volume column exists)
        if "volume" in result_df.columns:
            result_df = add_turnover_and_liquidity_proxies(
                result_df,
                volume_col="volume",
                freefloat_col=None,
                group_col=symbol_col,
                timestamp_col=timestamp_col,
            )

        # Collect vol/liquidity factor columns
        vol_factor_patterns = [
            "rv_20",
            "rv_60",
            "vov_rv_20",
            "vov_rv_60",
            "volume_zscore",
            "spread_proxy",
        ]

        for pattern in vol_factor_patterns:
            if pattern in result_df.columns:
                factor_columns.append(pattern)

    logger.info(
        f"Computed {len(factor_columns)} factors: {factor_columns[:5]}{'...' if len(factor_columns) > 5 else ''}"
    )

    # Step 2: Add forward returns
    fwd_return_col = f"fwd_return_{fwd_horizon_days}d"
    result_df = add_forward_returns(
        result_df,
        horizon_days=fwd_horizon_days,
        price_col=price_col,
        group_col=symbol_col,
        timestamp_col=timestamp_col,
        col_name=fwd_return_col,
    )

    logger.info(f"Added forward returns: {fwd_return_col}")

    # Step 3: Compute IC and Rank-IC
    logger.info("Computing IC and Rank-IC...")

    ic_df = compute_factor_ic(
        result_df,
        factor_cols=factor_columns,
        fwd_return_col=fwd_return_col,
        group_col=timestamp_col,
        method="pearson",
    )

    rank_ic_df = compute_factor_rank_ic(
        result_df,
        factor_cols=factor_columns,
        fwd_return_col=fwd_return_col,
        group_col=timestamp_col,
    )

    logger.info(f"Computed IC: {len(ic_df)} values, Rank-IC: {len(rank_ic_df)} values")

    # Step 4: Summarize IC statistics
    logger.info("Summarizing IC statistics...")

    # Handle empty IC DataFrames gracefully
    if ic_df.empty:
        logger.warning("No IC data computed. Summary will be empty.")
        summary_ic_df = pd.DataFrame(
            columns=[
                "factor",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "count",
                "min_ic",
                "max_ic",
            ]
        )
    else:
        summary_ic_df = summarize_factor_ic(ic_df, group_col=timestamp_col)

    if rank_ic_df.empty:
        logger.warning("No Rank-IC data computed. Summary will be empty.")
        summary_rank_ic_df = pd.DataFrame(
            columns=[
                "factor",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "count",
                "min_ic",
                "max_ic",
            ]
        )
    else:
        summary_rank_ic_df = summarize_factor_ic(rank_ic_df, group_col=timestamp_col)

    logger.info(f"Factor report completed: {len(summary_ic_df)} factors summarized")

    return {
        "factors": result_df,
        "ic": ic_df,
        "rank_ic": rank_ic_df,
        "summary_ic": summary_ic_df,
        "summary_rank_ic": summary_rank_ic_df,
    }


def compute_ic(
    factor_df: pd.DataFrame,
    forward_returns_col: str,
    group_col: str = "symbol",
    method: str = "pearson",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute cross-sectional Information Coefficient (IC) per timestamp and factor.

    This function computes IC for all factor columns in the DataFrame and returns
    a wide-format DataFrame with timestamp as index and ic_<factor_name> as columns.

    Args:
        factor_df: DataFrame with factor and forward return data (panel format)
            Required columns: timestamp_col, group_col, forward_returns_col
            Plus any number of factor columns (e.g., factor_x, factor_y, returns_12m, etc.)
        forward_returns_col: Column name for forward returns (e.g., "fwd_return_5d")
        group_col: Column name for grouping symbols (default: "symbol")
            IC is computed cross-sectionally across all symbols at each timestamp
        method: Correlation method (default: "pearson")
            - "pearson": Pearson correlation (linear)
            - "spearman": Spearman rank correlation
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with:
        - Index: timestamp (sorted)
        - Columns: ic_<factor_name> for each factor column found in factor_df
        - Values: IC (correlation) for that factor at that timestamp

        Example:
            timestamp              ic_returns_12m  ic_trend_strength_200  ic_rv_20
            2020-01-01 00:00:00+00:00  0.15           0.08                   0.05
            2020-01-02 00:00:00+00:00  0.18           0.10                   0.06
            ...

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or no valid data

    Note:
        - Automatically detects all factor columns (excludes timestamp_col, group_col, forward_returns_col)
        - Handles MultiIndex by resetting to regular columns if needed
        - Robust to NaNs: rows with NaN in factor or forward return are ignored for that timestamp/factor
        - Requires at least 3 valid observations per timestamp/factor for correlation
    """
    # Validate input
    required_cols = [timestamp_col, group_col, forward_returns_col]
    missing_cols = [col for col in required_cols if col not in factor_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(factor_df.columns)}"
        )

    if factor_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Handle MultiIndex: reset to regular columns if needed
    result_df = factor_df.copy()
    if isinstance(result_df.index, pd.MultiIndex):
        result_df = result_df.reset_index()
        logger.info("Reset MultiIndex to regular columns for IC computation")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)

    # Auto-detect factor columns (exclude metadata columns)
    exclude_cols = {timestamp_col, group_col, forward_returns_col}
    factor_cols = [col for col in result_df.columns if col not in exclude_cols]

    if not factor_cols:
        raise ValueError(
            f"No factor columns found. Excluded columns: {exclude_cols}. "
            f"Available columns: {list(result_df.columns)}"
        )

    logger.info(
        f"Auto-detected {len(factor_cols)} factor columns: {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}"
    )

    # Sort by timestamp and group
    result_df = result_df.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    # Compute IC for each timestamp and factor
    ic_data = {}

    for timestamp in sorted(result_df[timestamp_col].unique()):
        timestamp_data = result_df[result_df[timestamp_col] == timestamp].copy()

        # Filter rows with valid forward return
        valid_mask = timestamp_data[forward_returns_col].notna()
        timestamp_data_valid = timestamp_data[valid_mask]

        if len(timestamp_data_valid) < 3:  # Need at least 3 points for correlation
            continue

        # Initialize row for this timestamp
        ic_row = {}

        # For each factor, compute correlation with forward return
        for factor_col in factor_cols:
            # Filter rows with valid factor value
            factor_valid_mask = timestamp_data_valid[factor_col].notna()
            factor_data = timestamp_data_valid[factor_valid_mask]

            if len(factor_data) < 3:
                ic_row[f"ic_{factor_col}"] = np.nan
                continue

            # Extract factor values and forward returns
            factor_values = factor_data[factor_col].values
            forward_returns = factor_data[forward_returns_col].values

            # Compute correlation
            if method == "pearson":
                factor_series = pd.Series(factor_values)
                return_series = pd.Series(forward_returns)
                corr = factor_series.corr(return_series, method="pearson")
            elif method == "spearman":
                # Spearman correlation: rank correlation
                factor_series = pd.Series(factor_values)
                return_series = pd.Series(forward_returns)

                # Compute ranks (handling ties by averaging)
                factor_ranks = factor_series.rank(method="average")
                return_ranks = return_series.rank(method="average")

                # Pearson correlation of ranks = Spearman correlation
                corr = factor_ranks.corr(return_ranks, method="pearson")
            else:
                raise ValueError(
                    f"Invalid method: {method}. Supported: 'pearson', 'spearman'"
                )

            # Handle NaN/inf
            if pd.isna(corr) or np.isinf(corr):
                ic_row[f"ic_{factor_col}"] = np.nan
            else:
                ic_row[f"ic_{factor_col}"] = float(corr)

        # Only add row if at least one valid IC was computed
        if any(not pd.isna(v) for v in ic_row.values()):
            ic_data[timestamp] = ic_row

    if not ic_data:
        logger.warning("No IC data computed. Check input data and factor columns.")
        # Return empty DataFrame with expected structure
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([], name=timestamp_col))
        for factor_col in factor_cols:
            empty_df[f"ic_{factor_col}"] = []
        return empty_df

    # Convert to DataFrame with timestamp as index
    ic_df = pd.DataFrame.from_dict(ic_data, orient="index")
    ic_df.index.name = timestamp_col
    ic_df = ic_df.sort_index()

    logger.info(
        f"Computed IC for {len(factor_cols)} factor(s) across {len(ic_df)} "
        f"timestamp(s), method={method}"
    )

    return ic_df


def compute_rank_ic(
    factor_df: pd.DataFrame,
    forward_returns_col: str,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute cross-sectional Rank-IC (Spearman correlation) per timestamp and factor.

    This is a convenience wrapper around compute_ic() with method="spearman".
    Rank-IC uses Spearman rank correlation, which is more robust to outliers
    and captures monotonic relationships better than Pearson correlation.

    Args:
        factor_df: DataFrame with factor and forward return data (panel format)
            Required columns: timestamp_col, group_col, forward_returns_col
            Plus any number of factor columns
        forward_returns_col: Column name for forward returns (e.g., "fwd_return_5d")
        group_col: Column name for grouping symbols (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with timestamp as index and ic_<factor_name> columns (Rank-IC values).
        Same format as compute_ic(), but IC values are from Spearman correlation.

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or no valid data
    """
    return compute_ic(
        factor_df=factor_df,
        forward_returns_col=forward_returns_col,
        group_col=group_col,
        method="spearman",
        timestamp_col=timestamp_col,
    )


def summarize_ic_series(
    ic_df: pd.DataFrame,
    ic_col_prefix: str = "ic_",
) -> pd.DataFrame:
    """
    Summarize IC time-series to factor-level statistics.

    Computes summary statistics for each factor's IC time-series:
    - Mean IC
    - Standard deviation of IC
    - IC-IR (Information Ratio) = mean_IC / std_IC
    - Hit ratio (percentage of days with positive IC)
    - 5% and 95% quantiles
    - Min/Max IC values
    - Count of valid observations

    Args:
        ic_df: DataFrame with IC time-series (from compute_ic() or compute_rank_ic())
            Expected format: Index = timestamp, Columns = ic_<factor_name>
            Or: Columns include timestamp and ic_<factor_name>
        ic_col_prefix: Prefix for IC columns (default: "ic_")
            Used to identify factor columns (e.g., "ic_returns_12m" -> factor "returns_12m")

    Returns:
        DataFrame with columns:
        - factor: Factor name (extracted from column name by removing ic_col_prefix)
        - mean_ic: Mean IC across all timestamps
        - std_ic: Standard deviation of IC
        - ic_ir: Information Ratio (mean_ic / std_ic)
        - hit_ratio: Percentage of timestamps with positive IC (0.0 to 1.0)
        - q05: 5% quantile of IC values
        - q95: 95% quantile of IC values
        - min_ic: Minimum IC value
        - max_ic: Maximum IC value
        - count: Number of timestamps with valid IC values

        One row per factor, sorted by ic_ir (descending).

    Raises:
        ValueError: If DataFrame is empty or no IC columns found

    Note:
        - Handles both index-based (timestamp as index) and column-based (timestamp as column) formats
        - Automatically extracts factor names from column names
        - Robust to NaNs: only valid IC values are used for statistics
    """
    if ic_df.empty:
        raise ValueError("Input IC DataFrame is empty")

    # Handle different input formats
    result_df = ic_df.copy()

    # Check if timestamp is in index or columns
    if isinstance(result_df.index, pd.DatetimeIndex):
        # Timestamp is in index - use as is
        timestamp_col = None
    elif "timestamp" in result_df.columns:
        # Timestamp is in column - set as index
        timestamp_col = "timestamp"
        result_df = result_df.set_index(timestamp_col)
    else:
        # Try to infer timestamp column
        potential_timestamp_cols = [
            col
            for col in result_df.columns
            if "time" in col.lower() or "date" in col.lower()
        ]
        if potential_timestamp_cols:
            timestamp_col = potential_timestamp_cols[0]
            result_df = result_df.set_index(timestamp_col)
            logger.warning(f"Using '{timestamp_col}' as timestamp index")
        else:
            # Assume first column is timestamp or use index
            if len(result_df.columns) > 0:
                timestamp_col = result_df.columns[0]
                result_df = result_df.set_index(timestamp_col)
                logger.warning(
                    f"Using first column '{timestamp_col}' as timestamp index"
                )
            else:
                raise ValueError("Could not identify timestamp column or index")

    # Find IC columns (columns starting with ic_col_prefix)
    ic_columns = [col for col in result_df.columns if col.startswith(ic_col_prefix)]

    if not ic_columns:
        raise ValueError(
            f"No IC columns found with prefix '{ic_col_prefix}'. "
            f"Available columns: {list(result_df.columns)}"
        )

    logger.info(f"Found {len(ic_columns)} IC columns to summarize")

    summary_data = []

    # For each IC column, compute summary statistics
    for ic_col in ic_columns:
        # Extract factor name (remove prefix)
        factor_name = (
            ic_col[len(ic_col_prefix) :] if ic_col.startswith(ic_col_prefix) else ic_col
        )

        # Get IC values (drop NaN)
        ic_values = result_df[ic_col].dropna()

        if len(ic_values) == 0:
            logger.warning(f"No valid IC values for factor '{factor_name}'. Skipping.")
            continue

        # Compute statistics
        mean_ic = float(ic_values.mean())
        std_ic = float(ic_values.std())

        # IC-IR: Information Ratio (mean IC / std IC)
        ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        # Hit ratio: percentage of positive IC values
        hit_ratio = (
            float((ic_values > 0).sum() / len(ic_values)) if len(ic_values) > 0 else 0.0
        )

        # Quantiles
        q05 = float(ic_values.quantile(0.05))
        q95 = float(ic_values.quantile(0.95))

        # Min/Max
        min_ic = float(ic_values.min())
        max_ic = float(ic_values.max())

        # Count
        count = len(ic_values)

        summary_data.append(
            {
                "factor": factor_name,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "hit_ratio": hit_ratio,
                "q05": q05,
                "q95": q95,
                "min_ic": min_ic,
                "max_ic": max_ic,
                "count": count,
            }
        )

    if not summary_data:
        logger.warning("No summary statistics computed. Check input IC DataFrame.")
        return pd.DataFrame(
            columns=[
                "factor",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "q05",
                "q95",
                "min_ic",
                "max_ic",
                "count",
            ]
        )

    summary_df = pd.DataFrame(summary_data)

    # Sort by IC-IR (descending) to rank factors
    summary_df = summary_df.sort_values("ic_ir", ascending=False).reset_index(drop=True)

    logger.info(f"Summarized IC statistics for {len(summary_df)} factor(s)")

    return summary_df


def compute_rolling_ic(
    ic_df: pd.DataFrame,
    window: int = 60,
    ic_col_prefix: str = "ic_",
) -> pd.DataFrame:
    """
    Compute rolling average and rolling IR (Information Ratio) for IC time-series.

    This function computes rolling statistics to analyze IC stability over time:
    - Rolling mean IC (average IC over rolling window)
    - Rolling IR (rolling mean / rolling std)

    Useful for:
    - Identifying periods when factors were more/less predictive
    - Analyzing factor stability over different market regimes
    - Detecting factor decay or improvement over time

    Args:
        ic_df: DataFrame with IC time-series (from compute_ic() or compute_rank_ic())
            Expected format: Index = timestamp, Columns = ic_<factor_name>
            Or: Columns include timestamp and ic_<factor_name>
        window: Rolling window size in periods (default: 60)
            For daily data, window=60 means 60-day rolling window
        ic_col_prefix: Prefix for IC columns (default: "ic_")
            Used to identify factor columns

    Returns:
        DataFrame with same index as ic_df and columns:
        - rolling_mean_<factor_name>: Rolling mean IC for each factor
        - rolling_ir_<factor_name>: Rolling IR (rolling_mean / rolling_std) for each factor

        Example:
            timestamp              rolling_mean_returns_12m  rolling_ir_returns_12m
            2020-01-01 00:00:00+00:00  0.15                     1.50
            2020-01-02 00:00:00+00:00  0.16                     1.55
            ...

    Raises:
        ValueError: If DataFrame is empty or no IC columns found

    Note:
        - Handles both index-based (timestamp as index) and column-based (timestamp as column) formats
        - Rolling statistics require at least `window` observations
        - First `window-1` rows will have NaN for rolling statistics
        - Robust to NaNs: uses pandas rolling with min_periods
    """
    if ic_df.empty:
        raise ValueError("Input IC DataFrame is empty")

    # Handle different input formats
    result_df = ic_df.copy()

    # Check if timestamp is in index or columns
    if isinstance(result_df.index, pd.DatetimeIndex):
        # Timestamp is in index - use as is
        timestamp_col = None
    elif "timestamp" in result_df.columns:
        # Timestamp is in column - set as index
        timestamp_col = "timestamp"
        result_df = result_df.set_index(timestamp_col)
    else:
        # Try to infer timestamp column
        potential_timestamp_cols = [
            col
            for col in result_df.columns
            if "time" in col.lower() or "date" in col.lower()
        ]
        if potential_timestamp_cols:
            timestamp_col = potential_timestamp_cols[0]
            result_df = result_df.set_index(timestamp_col)
            logger.warning(f"Using '{timestamp_col}' as timestamp index")
        else:
            # Assume first column is timestamp or use index
            if len(result_df.columns) > 0:
                timestamp_col = result_df.columns[0]
                result_df = result_df.set_index(timestamp_col)
                logger.warning(
                    f"Using first column '{timestamp_col}' as timestamp index"
                )
            else:
                raise ValueError("Could not identify timestamp column or index")

    # Ensure index is sorted
    result_df = result_df.sort_index()

    # Find IC columns (columns starting with ic_col_prefix)
    ic_columns = [col for col in result_df.columns if col.startswith(ic_col_prefix)]

    if not ic_columns:
        raise ValueError(
            f"No IC columns found with prefix '{ic_col_prefix}'. "
            f"Available columns: {list(result_df.columns)}"
        )

    logger.info(
        f"Computing rolling IC statistics for {len(ic_columns)} factor(s), window={window}"
    )

    rolling_df = pd.DataFrame(index=result_df.index)

    # For each IC column, compute rolling statistics
    for ic_col in ic_columns:
        # Extract factor name (remove prefix)
        factor_name = (
            ic_col[len(ic_col_prefix) :] if ic_col.startswith(ic_col_prefix) else ic_col
        )

        # Rolling mean
        rolling_mean = (
            result_df[ic_col]
            .rolling(window=window, min_periods=min(5, window // 4))
            .mean()
        )
        rolling_df[f"rolling_mean_{factor_name}"] = rolling_mean

        # Rolling std
        rolling_std = (
            result_df[ic_col]
            .rolling(window=window, min_periods=min(5, window // 4))
            .std()
        )

        # Rolling IR (rolling_mean / rolling_std)
        rolling_ir = rolling_mean / rolling_std
        rolling_ir = rolling_ir.replace(
            [np.inf, -np.inf], np.nan
        )  # Handle division by zero
        rolling_df[f"rolling_ir_{factor_name}"] = rolling_ir

    logger.info(f"Computed rolling IC statistics for {len(ic_columns)} factor(s)")

    return rolling_df


def example_factor_analysis_workflow(
    prices_df: pd.DataFrame,
    factor_df: pd.DataFrame | None = None,
    horizons: list[int] = [20],
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    return_type: str = "log",
) -> dict[str, pd.DataFrame]:
    """
    Example workflow for factor analysis with forward returns.

    This function demonstrates a simple workflow for factor analysis:
    1. Add forward returns to prices or factor DataFrame
    2. Compute IC for all factors
    3. Summarize IC statistics
    4. Compute rolling IC statistics

    Args:
        prices_df: DataFrame with price data (panel format)
            Required columns: timestamp_col, group_col, price_col
        factor_df: Optional DataFrame with factor columns (if None, uses prices_df)
            If provided, must have same structure as prices_df (timestamp, group_col, price_col)
            plus additional factor columns
        horizons: List of forward return horizons in days (default: [20])
            Creates columns: fwd_ret_{horizon} for each horizon
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        return_type: Type of return calculation (default: "log")
            - "log": log returns
            - "simple": simple returns

    Returns:
        Dictionary with keys:
        - "data_with_returns": DataFrame with original data plus forward returns
        - "ic": DataFrame with IC values per timestamp and factor (for first horizon)
        - "rank_ic": DataFrame with Rank-IC values per timestamp and factor (for first horizon)
        - "summary_ic": DataFrame with aggregated IC statistics
        - "summary_rank_ic": DataFrame with aggregated Rank-IC statistics
        - "rolling_ic": DataFrame with rolling IC statistics (for first horizon)

    Example:
        >>> from src.assembled_core.qa.factor_analysis import example_factor_analysis_workflow
        >>> from src.assembled_core.features.ta_factors_core import build_core_ta_factors
        >>>
        >>> # Build factors
        >>> factors = build_core_ta_factors(prices)
        >>>
        >>> # Run example workflow
        >>> results = example_factor_analysis_workflow(
        ...     prices_df=prices,
        ...     factor_df=factors,
        ...     horizons=[20, 60]
        ... )
        >>>
        >>> # Access results
        >>> print(results["summary_ic"])
        >>> print(results["rolling_ic"])
    """
    # Use factor_df if provided, otherwise use prices_df
    if factor_df is not None:
        # Validate that factor_df has required columns
        required_cols = [timestamp_col, group_col, price_col]
        missing_cols = [col for col in required_cols if col not in factor_df.columns]
        if missing_cols:
            raise KeyError(
                f"factor_df missing required columns: {', '.join(missing_cols)}. "
                f"Available columns: {list(factor_df.columns)}"
            )
        data_df = factor_df.copy()
    else:
        data_df = prices_df.copy()

    # Step 1: Add forward returns
    logger.info(f"Adding forward returns for horizons: {horizons}")
    data_with_returns = add_forward_returns(
        data_df,
        horizon_days=horizons,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        return_type=return_type,
    )

    # Step 2: Identify factor columns (exclude metadata and forward return columns)
    exclude_cols = {timestamp_col, group_col, price_col}
    # Also exclude forward return columns
    forward_return_cols = [
        col
        for col in data_with_returns.columns
        if col.startswith("fwd_ret") or col.startswith("fwd_return")
    ]
    exclude_cols.update(forward_return_cols)

    factor_cols = [col for col in data_with_returns.columns if col not in exclude_cols]

    if not factor_cols:
        logger.warning(
            "No factor columns found. Returning data with forward returns only."
        )
        return {
            "data_with_returns": data_with_returns,
            "ic": pd.DataFrame(),
            "rank_ic": pd.DataFrame(),
            "summary_ic": pd.DataFrame(),
            "summary_rank_ic": pd.DataFrame(),
            "rolling_ic": pd.DataFrame(),
        }

    logger.info(
        f"Found {len(factor_cols)} factor columns: {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}"
    )

    # Step 3: Compute IC and Rank-IC (use first horizon for IC computation)
    first_horizon = horizons[0]
    if len(horizons) == 1:
        fwd_return_col = f"fwd_return_{first_horizon}d"
    else:
        fwd_return_col = f"fwd_ret_{first_horizon}"

    if fwd_return_col not in data_with_returns.columns:
        # Try alternative naming
        potential_cols = [
            col
            for col in data_with_returns.columns
            if str(first_horizon) in col and "fwd" in col.lower()
        ]
        if potential_cols:
            fwd_return_col = potential_cols[0]
        else:
            raise ValueError(
                f"Could not find forward return column for horizon {first_horizon}"
            )

    logger.info(
        f"Computing IC and Rank-IC using forward return column: {fwd_return_col}"
    )

    ic_df = compute_ic(
        data_with_returns,
        forward_returns_col=fwd_return_col,
        group_col=group_col,
        method="pearson",
        timestamp_col=timestamp_col,
    )

    rank_ic_df = compute_rank_ic(
        data_with_returns,
        forward_returns_col=fwd_return_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
    )

    # Step 4: Summarize IC statistics
    logger.info("Summarizing IC statistics...")
    summary_ic_df = summarize_ic_series(ic_df) if not ic_df.empty else pd.DataFrame()
    summary_rank_ic_df = (
        summarize_ic_series(rank_ic_df) if not rank_ic_df.empty else pd.DataFrame()
    )

    # Step 5: Compute rolling IC statistics
    logger.info("Computing rolling IC statistics...")
    rolling_ic_df = (
        compute_rolling_ic(ic_df, window=60) if not ic_df.empty else pd.DataFrame()
    )

    logger.info("Factor analysis workflow completed successfully")

    return {
        "data_with_returns": data_with_returns,
        "ic": ic_df,
        "rank_ic": rank_ic_df,
        "summary_ic": summary_ic_df,
        "summary_rank_ic": summary_rank_ic_df,
        "rolling_ic": rolling_ic_df,
    }


# ============================================================================
# Phase C2: Factor Portfolio Returns & Ranking
# ============================================================================


def build_factor_portfolio_returns(
    data: pd.DataFrame,
    factor_cols: str | list[str],
    forward_returns_col: str,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    quantiles: int = 5,
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Build factor portfolio returns based on quantile sorting.

    For each factor and each timestamp:
    1. Sort all symbols by factor value
    2. Divide into quantiles (e.g., 5 → quintiles)
    3. Compute equal-weighted portfolio return per quantile

    This function implements Phase C2 from the Advanced Analytics & Factor Labs roadmap.
    It complements IC-based evaluation (C1) with portfolio-based evaluation.

    Args:
        data: Panel DataFrame with columns:
            - timestamp_col: Timestamp (datetime)
            - group_col: Symbol/group identifier (string)
            - factor_cols: One or more factor columns to evaluate
            - forward_returns_col: Forward return column (e.g., "fwd_return_20d")
        factor_cols: Column name(s) of factor(s) to rank by
            Can be a single string or list of strings
        forward_returns_col: Column name of forward returns
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        quantiles: Number of quantiles (default: 5, i.e., quintiles)
        min_obs: Minimum number of valid observations per timestamp/factor
            Days with fewer than min_obs valid observations are skipped (default: 10)

    Returns:
        DataFrame in long format with columns:
        - timestamp: Timestamp
        - factor: Factor name
        - quantile: Quantile number (1, 2, ..., quantiles)
        - mean_return: Equal-weighted portfolio return for this quantile
        - n: Number of symbols in this quantile

        Example:
            timestamp              factor          quantile  mean_return  n
            2020-01-01 00:00:00+00:00  returns_12m     1        0.01        10
            2020-01-01 00:00:00+00:00  returns_12m     2        0.02        10
            ...
            2020-01-01 00:00:00+00:00  returns_12m     5        0.05        10

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid

    Note:
        - Quantile 1 = lowest factor values (bottom quantile)
        - Quantile quantiles = highest factor values (top quantile)
        - Days with insufficient data (< min_obs) are skipped
        - NaN values in factor or forward return are excluded from quantile assignment
    """
    # Validate input
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    # Normalize factor_cols to list
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]

    # Validate columns
    required_cols = {timestamp_col, group_col, forward_returns_col}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    missing_factors = set(factor_cols) - set(data.columns)
    if missing_factors:
        raise KeyError(f"Missing factor columns: {missing_factors}")

    # Make a copy to avoid modifying original
    df = data[[timestamp_col, group_col, forward_returns_col] + factor_cols].copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Sort by timestamp, then by group_col for consistent processing
    df = df.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    results = []

    # Process each timestamp
    for timestamp, group_df in df.groupby(timestamp_col):
        # Process each factor
        for factor_col in factor_cols:
            # Filter valid observations (non-NaN in both factor and forward return)
            valid_mask = (
                group_df[factor_col].notna() & group_df[forward_returns_col].notna()
            )
            valid_df = group_df[valid_mask].copy()

            # Skip if insufficient observations
            if len(valid_df) < min_obs:
                continue

            # Sort by factor value (ascending: lowest = Q1, highest = Qquantiles)
            valid_df = valid_df.sort_values(factor_col).reset_index(drop=True)

            # Assign quantiles (1-based: 1 = lowest, quantiles = highest)
            # Use qcut for equal-sized groups, but handle edge cases
            try:
                valid_df["quantile"] = (
                    pd.qcut(
                        valid_df[factor_col].rank(method="first"),
                        q=quantiles,
                        labels=False,
                        duplicates="drop",
                    )
                    + 1
                )  # Convert to 1-based
            except ValueError:
                # If qcut fails (e.g., too many duplicates), use manual binning
                n_valid = len(valid_df)
                quantile_size = n_valid / quantiles
                valid_df["quantile"] = (
                    (np.arange(n_valid) / quantile_size).astype(int) + 1
                ).clip(1, quantiles)

            # Compute mean return per quantile
            quantile_returns = (
                valid_df.groupby("quantile")[forward_returns_col]
                .agg(["mean", "count"])
                .reset_index()
            )
            quantile_returns.columns = ["quantile", "mean_return", "n"]

            # Add metadata
            quantile_returns["timestamp"] = timestamp
            quantile_returns["factor"] = factor_col

            # Reorder columns
            quantile_returns = quantile_returns[
                ["timestamp", "factor", "quantile", "mean_return", "n"]
            ]

            results.append(quantile_returns)

    if not results:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=["timestamp", "factor", "quantile", "mean_return", "n"]
        )

    # Combine all results
    result_df = pd.concat(results, ignore_index=True)

    # Ensure quantile is integer
    result_df["quantile"] = result_df["quantile"].astype(int)

    # Sort by timestamp, factor, quantile
    result_df = result_df.sort_values(
        [timestamp_col, "factor", "quantile"]
    ).reset_index(drop=True)

    logger.info(
        f"Built factor portfolio returns for {len(factor_cols)} factor(s), "
        f"{result_df[timestamp_col].nunique()} timestamps, "
        f"{quantiles} quantiles"
    )

    return result_df


def build_long_short_portfolio_returns(
    portfolios_df: pd.DataFrame,
    low_quantile: int = 1,
    high_quantile: int | None = None,
    timestamp_col: str = "timestamp",
    quantile_col: str = "quantile",
    return_col: str = "mean_return",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """
    Build Long/Short portfolio returns from quantile portfolio returns.

    For each factor and timestamp:
    - Long: Top quantile (high_quantile, default: highest quantile)
    - Short: Bottom quantile (low_quantile, default: 1)
    - Long/Short Return = High Quantile Return - Low Quantile Return

    Args:
        portfolios_df: DataFrame from build_factor_portfolio_returns()
            Must have columns: timestamp_col, factor_col, quantile_col, return_col
        low_quantile: Bottom quantile for short position (default: 1)
        high_quantile: Top quantile for long position (default: None = highest available)
        timestamp_col: Column name for timestamp (default: "timestamp")
        quantile_col: Column name for quantile (default: "quantile")
        return_col: Column name for portfolio return (default: "mean_return")
        factor_col: Column name for factor (default: "factor")

    Returns:
        DataFrame with columns:
        - timestamp: Timestamp
        - factor: Factor name
        - ls_return: Long/Short return (high_quantile - low_quantile)
        - gross_exposure: Gross exposure (2.0 for long/short, 1.0 for long-only)
        - n_long: Number of symbols in long quantile
        - n_short: Number of symbols in short quantile

        Example:
            timestamp              factor          ls_return  gross_exposure  n_long  n_short
            2020-01-01 00:00:00+00:00  returns_12m     0.04        2.0         10     10
            ...

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid
    """
    # Validate input
    if portfolios_df.empty:
        raise ValueError("Input DataFrame is empty")

    required_cols = {timestamp_col, factor_col, quantile_col, return_col}
    missing_cols = required_cols - set(portfolios_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Determine high_quantile if not provided
    if high_quantile is None:
        high_quantile = int(portfolios_df[quantile_col].max())

    # Validate quantiles
    if low_quantile < 1 or high_quantile < low_quantile:
        raise ValueError(
            f"Invalid quantiles: low={low_quantile}, high={high_quantile}. "
            f"Must satisfy 1 <= low <= high"
        )

    results = []

    # Process each timestamp and factor combination
    for (timestamp, factor), group_df in portfolios_df.groupby(
        [timestamp_col, factor_col]
    ):
        # Get returns for low and high quantiles
        low_row = group_df[group_df[quantile_col] == low_quantile]
        high_row = group_df[group_df[quantile_col] == high_quantile]

        # Skip if either quantile is missing
        if low_row.empty or high_row.empty:
            continue

        # Extract values
        low_return = low_row[return_col].iloc[0]
        high_return = high_row[return_col].iloc[0]

        # Get n values if available
        n_col = "n" if "n" in group_df.columns else None
        n_long = (
            int(high_row[n_col].iloc[0])
            if n_col and not high_row[n_col].isna().iloc[0]
            else None
        )
        n_short = (
            int(low_row[n_col].iloc[0])
            if n_col and not low_row[n_col].isna().iloc[0]
            else None
        )

        # Compute Long/Short return
        ls_return = high_return - low_return

        # Build result row
        result_row = {
            timestamp_col: timestamp,
            factor_col: factor,
            "ls_return": ls_return,
            "gross_exposure": 2.0,  # Long + Short
            "n_long": n_long,
            "n_short": n_short,
        }

        results.append(result_row)

    if not results:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=[
                timestamp_col,
                factor_col,
                "ls_return",
                "gross_exposure",
                "n_long",
                "n_short",
            ]
        )

    # Combine results
    result_df = pd.DataFrame(results)

    # Sort by timestamp, factor
    result_df = result_df.sort_values([timestamp_col, factor_col]).reset_index(
        drop=True
    )

    logger.info(
        f"Built Long/Short portfolio returns for {result_df[factor_col].nunique()} factor(s), "
        f"{result_df[timestamp_col].nunique()} timestamps"
    )

    return result_df


def summarize_factor_portfolios(
    ls_returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    timestamp_col: str = "timestamp",
    factor_col: str = "factor",
    return_col: str = "ls_return",
) -> pd.DataFrame:
    """
    Summarize factor portfolio performance metrics.

    Aggregates Long/Short returns per factor and computes:
    - Annualized return and volatility
    - Sharpe Ratio
    - t-statistic (for mean return significance)
    - Win ratio (percentage of positive returns)
    - Maximum drawdown (simple rolling max drawdown on cumulative returns)

    Args:
        ls_returns_df: DataFrame from build_long_short_portfolio_returns()
            Must have columns: timestamp_col, factor_col, return_col
        risk_free_rate: Risk-free rate (annualized, default: 0.0)
        periods_per_year: Trading periods per year (default: 252 for daily)
        timestamp_col: Column name for timestamp (default: "timestamp")
        factor_col: Column name for factor (default: "factor")
        return_col: Column name for Long/Short return (default: "ls_return")

    Returns:
        DataFrame with one row per factor, columns:
        - factor: Factor name
        - annualized_return: Mean return (annualized)
        - annualized_vol: Standard deviation of returns (annualized)
        - sharpe: Sharpe Ratio (annualized)
        - t_stat: t-statistic for mean return (H0: mean = 0)
        - p_value: p-value for t-test (two-sided)
        - win_ratio: Percentage of positive returns (0.0 to 1.0)
        - max_drawdown: Maximum drawdown (negative value)
        - n_periods: Number of periods
        - n_positive: Number of positive returns
        - n_negative: Number of negative returns

        Sorted by Sharpe Ratio (descending).

        Example:
            factor          annualized_return  annualized_vol  sharpe  t_stat  ...
            returns_12m     0.15               0.12            1.25    3.45   ...
            trend_strength_200  0.10           0.10            1.00    2.50   ...

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid
    """
    # Validate input
    if ls_returns_df.empty:
        raise ValueError("Input DataFrame is empty")

    required_cols = {timestamp_col, factor_col, return_col}
    missing_cols = required_cols - set(ls_returns_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is datetime
    df = ls_returns_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Sort by timestamp
    df = df.sort_values([timestamp_col, factor_col]).reset_index(drop=True)

    results = []

    # Process each factor
    for factor, factor_df in df.groupby(factor_col):
        # Extract returns
        returns = factor_df[return_col].dropna()

        if len(returns) < 2:
            # Insufficient data
            continue

        # Basic statistics
        mean_return = float(returns.mean())
        std_return = float(returns.std())
        n_periods = len(returns)

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_vol = std_return * np.sqrt(periods_per_year)

        # Sharpe Ratio
        if annualized_vol > 0:
            sharpe = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe = np.nan

        # t-statistic (H0: mean = 0)
        if std_return > 0 and n_periods > 1:
            t_stat = mean_return / (std_return / np.sqrt(n_periods))
            # Simple p-value approximation (two-sided t-test)
            # For large n, t ~ N(0,1), so p ≈ 2 * (1 - Φ(|t|))
            # We'll use a simple approximation
            abs_t = abs(t_stat)
            # For large n, use normal approximation
            if n_periods > 30:
                # Approximate p-value using normal CDF
                # p = 2 * (1 - norm.cdf(abs_t))
                # Without scipy, we use a simple approximation
                # For |t| > 3, p is very small; for |t| < 1, p is large
                if abs_t > 3:
                    p_value = 0.002  # Very significant
                elif abs_t > 2:
                    p_value = 0.05  # Significant
                elif abs_t > 1:
                    p_value = 0.3  # Not significant
                else:
                    p_value = 0.5  # Not significant
            else:
                # For small n, use conservative estimate
                p_value = 0.1 if abs_t > 2 else 0.5
        else:
            t_stat = np.nan
            p_value = np.nan

        # Win ratio
        n_positive = int((returns > 0).sum())
        n_negative = int((returns < 0).sum())
        win_ratio = n_positive / n_periods if n_periods > 0 else 0.0

        # Maximum drawdown (simple rolling max drawdown)
        # Compute cumulative returns
        cum_returns = (1 + returns).cumprod()
        # Running maximum
        running_max = cum_returns.expanding().max()
        # Drawdown
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdown.min())  # Negative value

        # Compute deflated Sharpe (B4)
        # n_tests: Conservative estimate = number of factors being tested
        # This will be set later when we know the total number of factors
        deflated_sharpe = None
        if sharpe is not None and not np.isnan(sharpe) and n_periods >= 2:
            # For now, we'll compute it with n_tests=1 (conservative)
            # The actual n_tests will be set after we know the total number of factors
            # This is a placeholder that will be updated later
            try:
                from src.assembled_core.qa.metrics import deflated_sharpe_ratio

                # Use n_tests=1 as placeholder (will be updated after we know total factor count)
                deflated_sharpe = deflated_sharpe_ratio(
                    sharpe_annual=sharpe,
                    n_obs=n_periods,
                    n_tests=1,  # Placeholder, will be updated
                    skew=0.0,  # Default: assume normal
                    kurtosis=3.0,  # Default: assume normal
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute deflated Sharpe for factor {factor}: {e}"
                )
                deflated_sharpe = None

        # Build result row
        result_row = {
            factor_col: factor,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe": sharpe,
            "deflated_sharpe": deflated_sharpe,  # B4: Deflated Sharpe (n_tests will be updated)
            "t_stat": t_stat,
            "p_value": p_value,
            "win_ratio": win_ratio,
            "max_drawdown": max_drawdown,
            "n_periods": n_periods,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

        results.append(result_row)

    if not results:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=[
                factor_col,
                factor_col,
                "annualized_return",
                "annualized_vol",
                "sharpe",
                "deflated_sharpe",  # B4: Added deflated_sharpe column
                "t_stat",
                "p_value",
                "win_ratio",
                "max_drawdown",
                "n_periods",
                "n_positive",
                "n_negative",
            ]
        )

    # Combine results
    result_df = pd.DataFrame(results)

    # B4: Update deflated_sharpe with correct n_tests (number of factors tested)
    if not result_df.empty and "deflated_sharpe" in result_df.columns:
        n_tests_actual = len(result_df)  # Number of factors being tested
        if n_tests_actual > 1:
            # Recompute deflated Sharpe with correct n_tests
            from src.assembled_core.qa.metrics import deflated_sharpe_ratio

            for idx, row in result_df.iterrows():
                sharpe_val = row.get("sharpe")
                n_periods_val = row.get("n_periods", 0)
                if (
                    sharpe_val is not None
                    and not np.isnan(sharpe_val)
                    and n_periods_val >= 2
                ):
                    try:
                        result_df.at[idx, "deflated_sharpe"] = deflated_sharpe_ratio(
                            sharpe_annual=sharpe_val,
                            n_obs=n_periods_val,
                            n_tests=n_tests_actual,
                            skew=0.0,
                            kurtosis=3.0,
                        )
                    except (ValueError, TypeError, AttributeError):
                        # Keep existing value (or NaN) if stats computation fails
                        pass

    # Sort by Sharpe Ratio (descending)
    result_df = result_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    logger.info(f"Summarized factor portfolios for {len(result_df)} factor(s)")

    return result_df


def compute_deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int = 1,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) to adjust for multiple testing.

    The Deflated Sharpe Ratio adjusts the observed Sharpe Ratio for:
    - Multiple testing (False Discovery Rate)
    - Non-normal return distributions (skewness, kurtosis)

    Formula (simplified, based on Bailey et al. 2014):
        DSR = (SR - E[SR]) / std(SR)
        where:
        - E[SR] = expected Sharpe under null (accounting for multiple testing)
        - std(SR) = standard deviation of Sharpe (accounting for distribution)

    This is a simplified implementation. For full accuracy, see:
    Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
    Correcting for selection bias, backtest overfitting and non-normality.
    Journal of Portfolio Management, 40(5), 94-107.

    Args:
        sharpe: Observed Sharpe Ratio
        n_obs: Number of observations (time periods)
        n_trials: Number of factors/trials tested (for multiple testing adjustment)
            Default: 1 (no multiple testing adjustment)
        skew: Skewness of returns (default: 0.0, assumes normal)
        kurt: Kurtosis of returns (default: 3.0, assumes normal)
            Excess kurtosis = kurt - 3.0

    Returns:
        Deflated Sharpe Ratio (float)
        - Positive DSR indicates significant Sharpe after adjustment
        - Negative DSR indicates Sharpe may be due to luck/multiple testing

    Note:
        - This is a simplified implementation using pure numpy
        - For production use, consider using scipy.stats.norm.cdf if available
        - The formula assumes independence between trials (may not hold for correlated factors)
    """
    if n_obs < 2:
        return np.nan

    if np.isnan(sharpe) or np.isinf(sharpe):
        return np.nan

    # Expected Sharpe under null (multiple testing adjustment)
    # For n_trials independent tests, expected max Sharpe ≈ sqrt(2 * log(n_trials))
    # This is a simplified approximation
    if n_trials > 1:
        # Expected maximum Sharpe under null (Bonferroni-like adjustment)
        # E[max_SR] ≈ sqrt(2 * log(n_trials)) / sqrt(n_obs)
        # But we use a simpler approximation
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials)) / np.sqrt(n_obs)
    else:
        expected_max_sharpe = 0.0

    # Standard deviation of Sharpe (distribution adjustment)
    # For normal returns: std(SR) ≈ sqrt((1 + SR^2/2) / n_obs)
    # For non-normal: adjust for skewness and kurtosis
    excess_kurt = kurt - 3.0
    variance_term = (
        1.0 + (sharpe**2 / 2.0) + (skew * sharpe) + (excess_kurt * sharpe**2 / 4.0)
    )
    std_sharpe = np.sqrt(variance_term / n_obs)

    # Deflated Sharpe Ratio
    if std_sharpe > 0:
        dsr = (sharpe - expected_max_sharpe) / std_sharpe
    else:
        dsr = np.nan

    return float(dsr)
