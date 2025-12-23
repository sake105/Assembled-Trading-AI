"""Factor Report Workflow Module.

This module provides high-level workflows for generating factor analysis reports.
It integrates Phase A factors (ta_factors_core, ta_liquidity_vol_factors) with
Phase C1 factor analysis tools (IC computation, summary statistics).

Usage:
    from src.assembled_core.qa.factor_report import run_factor_report

    report = run_factor_report(
        prices=prices_df,
        factor_set="core",
        fwd_horizon_days=5
    )

    print(report["summary_ic"])
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Import Phase A factor modules
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
    add_vol_of_vol,
)

# Import Phase C1 factor analysis tools
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_factor_ic,
    compute_rank_ic,
    summarize_factor_ic,
)


def run_factor_report(
    prices: pd.DataFrame,
    factor_set: Literal["core", "vol_liquidity", "all"] = "core",
    fwd_horizon_days: int = 5,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """
    Run a comprehensive factor analysis report.

    This high-level function:
    1. Builds factors based on the specified factor_set
    2. Adds forward returns
    3. Computes IC and Rank-IC for all factors
    4. Generates summary statistics

    Args:
        prices: DataFrame with price data (panel format: multiple symbols over time)
            Required columns: timestamp_col, group_col, price_col
            Optional columns: high, low, volume (for enhanced factors)
        factor_set: Which factors to compute (default: "core")
            - "core": Core TA/Price factors (returns, momentum, trend strength, reversal)
            - "vol_liquidity": Volatility and liquidity factors (RV, Vol-of-Vol, turnover)
            - "all": Combination of core + vol_liquidity factors
        fwd_horizon_days: Forward return horizon in days (default: 5)
        price_col: Column name for price data (default: "close")
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        Dictionary with keys:
        - "factors": DataFrame with all computed factors (original columns + factor columns)
        - "ic": DataFrame with IC values per timestamp and factor
        - "rank_ic": DataFrame with Rank-IC values per timestamp and factor
        - "summary_ic": DataFrame with summary statistics for IC (sorted by IC-IR)
        - "summary_rank_ic": DataFrame with summary statistics for Rank-IC (sorted by IC-IR)

    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrame is empty or invalid factor_set
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

    if factor_set not in ["core", "vol_liquidity", "all"]:
        raise ValueError(
            f"Invalid factor_set: {factor_set}. Supported: 'core', 'vol_liquidity', 'all'"
        )

    logger.info(
        f"Starting factor report: factor_set={factor_set}, "
        f"fwd_horizon_days={fwd_horizon_days}, "
        f"symbols={prices[group_col].nunique()}, "
        f"periods={prices[timestamp_col].nunique()}"
    )

    # Sort prices by symbol and timestamp
    result_df = prices.copy()
    result_df = result_df.sort_values([group_col, timestamp_col]).reset_index(drop=True)

    # Step 1: Build factors based on factor_set
    if factor_set == "core":
        logger.info("Computing core TA/Price factors...")
        result_df = build_core_ta_factors(
            result_df,
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Identify factor columns (common patterns from build_core_ta_factors)
        factor_patterns = [
            "returns_",
            "momentum_",
            "trend_strength_",
            "reversal_",
        ]

    elif factor_set == "vol_liquidity":
        logger.info("Computing volatility and liquidity factors...")

        # Add realized volatility
        result_df = add_realized_volatility(
            result_df,
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Add volatility of volatility
        result_df = add_vol_of_vol(
            result_df,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Add turnover and liquidity proxies
        result_df = add_turnover_and_liquidity_proxies(
            result_df,
            volume_col="volume" if "volume" in result_df.columns else None,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Identify factor columns
        factor_patterns = [
            "rv_",
            "vov_",
            "turnover",
            "volume_zscore",
            "spread_proxy",
        ]

    else:  # factor_set == "all"
        logger.info("Computing all factors (core + vol_liquidity)...")

        # Build core factors first
        result_df = build_core_ta_factors(
            result_df,
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Add volatility and liquidity factors
        result_df = add_realized_volatility(
            result_df,
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        result_df = add_vol_of_vol(
            result_df,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        result_df = add_turnover_and_liquidity_proxies(
            result_df,
            volume_col="volume" if "volume" in result_df.columns else None,
            group_col=group_col,
            timestamp_col=timestamp_col,
        )

        # Identify factor columns (both sets)
        factor_patterns = [
            "returns_",
            "momentum_",
            "trend_strength_",
            "reversal_",
            "rv_",
            "vov_",
            "turnover",
            "volume_zscore",
            "spread_proxy",
        ]

    # Step 2: Add forward returns
    fwd_return_col = f"fwd_return_{fwd_horizon_days}d"
    logger.info(f"Adding forward returns (horizon: {fwd_horizon_days} days)...")
    result_df = add_forward_returns(
        result_df,
        horizon_days=fwd_horizon_days,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        col_name=fwd_return_col,
    )

    # Step 3: Identify factor columns (all columns matching patterns, excluding original and forward return)
    original_cols = set(prices.columns)
    exclude_cols = {timestamp_col, group_col, price_col, fwd_return_col}
    if "open" in original_cols:
        exclude_cols.add("open")
    if "high" in original_cols:
        exclude_cols.add("high")
    if "low" in original_cols:
        exclude_cols.add("low")
    if "volume" in original_cols:
        exclude_cols.add("volume")

    # Find all columns that match factor patterns
    factor_cols = []
    for col in result_df.columns:
        if col in exclude_cols:
            continue
        # Check if column matches any pattern
        if any(pattern in col for pattern in factor_patterns):
            factor_cols.append(col)

    # Also include columns that are clearly factors (not original price columns)
    for col in result_df.columns:
        if col in exclude_cols or col in factor_cols:
            continue
        # If column is not in original columns and is numeric, it might be a factor
        if col not in original_cols and result_df[col].dtype in [
            "float64",
            "float32",
            "int64",
            "int32",
        ]:
            factor_cols.append(col)

    # Remove duplicates and sort
    factor_cols = sorted(list(set(factor_cols)))

    if not factor_cols:
        logger.warning("No factor columns identified. Check factor_set and input data.")
        # Return empty DataFrames
        empty_ic = pd.DataFrame(columns=[timestamp_col, "factor", "ic", "count"])
        empty_summary = pd.DataFrame(
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
        return {
            "factors": result_df,
            "ic": empty_ic,
            "rank_ic": empty_ic,
            "summary_ic": empty_summary,
            "summary_rank_ic": empty_summary,
        }

    logger.info(
        f"Identified {len(factor_cols)} factor columns: {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}"
    )

    # Step 4: Compute IC and Rank-IC
    logger.info("Computing IC and Rank-IC...")
    ic_df = compute_factor_ic(
        result_df,
        factor_cols=factor_cols,
        fwd_return_col=fwd_return_col,
        group_col=timestamp_col,
        method="pearson",
    )

    rank_ic_df = compute_rank_ic(
        result_df,
        factor_cols=factor_cols,
        fwd_return_col=fwd_return_col,
        group_col=timestamp_col,
    )

    # Step 5: Summarize IC statistics
    logger.info("Generating summary statistics...")
    if ic_df.empty:
        logger.warning("No IC data available, returning empty summary.")
        empty_summary = pd.DataFrame(
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
        summary_ic = empty_summary
        summary_rank_ic = empty_summary
    else:
        summary_ic = summarize_factor_ic(ic_df)
        summary_rank_ic = summarize_factor_ic(rank_ic_df)

    logger.info(
        f"Factor report completed: {len(factor_cols)} factors, "
        f"{len(ic_df)} IC values, {len(summary_ic)} summarized factors"
    )

    return {
        "factors": result_df,
        "ic": ic_df,
        "rank_ic": rank_ic_df,
        "summary_ic": summary_ic,
        "summary_rank_ic": summary_rank_ic,
    }
