"""Factor Ranking Module.

This module provides functionality to consolidate factor analysis results from
multiple IC summaries, Rank-IC summaries, and portfolio summaries into a single
ranking table for easy comparison and selection of top factors.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_factor_ranking(
    ic_summary_paths: list[Path],
    rank_ic_summary_paths: list[Path],
    portfolio_summary_paths: list[Path] | None = None,
) -> pd.DataFrame:
    """Build a consolidated factor ranking table from multiple analysis outputs.
    
    This function reads IC summaries, Rank-IC summaries, and optionally portfolio
    summaries from multiple factor analysis runs and merges them into a single
    ranking table. Factors are ranked by a combined score that incorporates
    both IC-based metrics and portfolio performance metrics.
    
    Args:
        ic_summary_paths: List of paths to IC summary CSV files.
            Expected columns: factor, mean_ic, std_ic, ic_ir, hit_ratio, count, ...
        rank_ic_summary_paths: List of paths to Rank-IC summary CSV files.
            Expected columns: factor, mean_ic, std_ic, ic_ir, hit_ratio, count, ...
            Note: mean_ic in Rank-IC files represents mean_rank_ic.
        portfolio_summary_paths: Optional list of paths to portfolio summary CSV files.
            Expected columns: factor, annualized_return, sharpe, max_drawdown, win_ratio, n_periods, ...
            If None, portfolio metrics will not be included.
    
    Returns:
        DataFrame with columns:
        - factor_name: Factor name
        - mean_ic: Mean IC (Pearson correlation)
        - ic_ir: IC Information Ratio
        - hit_ratio: IC hit ratio (percentage of positive IC values)
        - mean_rank_ic: Mean Rank-IC (Spearman correlation)
        - rank_ic_ir: Rank-IC Information Ratio
        - ls_sharpe: Long/Short portfolio Sharpe ratio
        - ls_deflated_sharpe: Long/Short portfolio deflated Sharpe ratio
        - ls_annualized_return: Long/Short portfolio annualized return
        - ls_max_drawdown: Long/Short portfolio maximum drawdown
        - ls_win_ratio: Long/Short portfolio win ratio
        - n_periods_ic: Number of periods used for IC calculation
        - n_periods_portfolio: Number of periods used for portfolio calculation
        - combined_score: Weighted combination of ic_ir and ls_deflated_sharpe
        
        Sorted by combined_score (descending).
    
    Raises:
        FileNotFoundError: If any of the specified paths do not exist
        ValueError: If CSV files have unexpected structure
    """
    # Validate input paths
    all_paths = ic_summary_paths + rank_ic_summary_paths
    if portfolio_summary_paths:
        all_paths.extend(portfolio_summary_paths)
    
    missing_paths = [p for p in all_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing files: {', '.join(str(p) for p in missing_paths)}"
        )
    
    # Read and merge IC summaries
    ic_dfs = []
    for path in ic_summary_paths:
        try:
            df = pd.read_csv(path)
            if "factor" not in df.columns:
                logger.warning(f"IC summary {path} missing 'factor' column. Skipping.")
                continue
            ic_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read IC summary {path}: {e}. Skipping.")
            continue
    
    if not ic_dfs:
        raise ValueError("No valid IC summary files found.")
    
    # Merge IC summaries (average values if same factor appears in multiple files)
    ic_merged = pd.concat(ic_dfs, ignore_index=True)
    ic_grouped = ic_merged.groupby("factor", as_index=False).agg({
        "mean_ic": "mean",
        "std_ic": "mean",
        "ic_ir": "mean",
        "hit_ratio": "mean",
        "count": "sum",  # Sum counts across runs
    })
    
    # Read and merge Rank-IC summaries
    rank_ic_dfs = []
    for path in rank_ic_summary_paths:
        try:
            df = pd.read_csv(path)
            if "factor" not in df.columns:
                logger.warning(f"Rank-IC summary {path} missing 'factor' column. Skipping.")
                continue
            rank_ic_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read Rank-IC summary {path}: {e}. Skipping.")
            continue
    
    if not rank_ic_dfs:
        logger.warning("No valid Rank-IC summary files found. Rank-IC metrics will be missing.")
        rank_ic_grouped = pd.DataFrame(columns=["factor", "mean_rank_ic", "rank_ic_ir"])
    else:
        rank_ic_merged = pd.concat(rank_ic_dfs, ignore_index=True)
        # Rename columns to distinguish from IC
        rank_ic_merged = rank_ic_merged.rename(columns={
            "mean_ic": "mean_rank_ic",
            "ic_ir": "rank_ic_ir",
        })
        rank_ic_grouped = rank_ic_merged.groupby("factor", as_index=False).agg({
            "mean_rank_ic": "mean",
            "std_ic": "mean",  # Keep for reference, but we'll use rank_ic_ir
            "rank_ic_ir": "mean",
            "hit_ratio": "mean",
            "count": "sum",
        })
    
    # Read and merge portfolio summaries
    if portfolio_summary_paths:
        portfolio_dfs = []
        for path in portfolio_summary_paths:
            try:
                df = pd.read_csv(path)
                if "factor" not in df.columns:
                    logger.warning(f"Portfolio summary {path} missing 'factor' column. Skipping.")
                    continue
                portfolio_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read portfolio summary {path}: {e}. Skipping.")
                continue
        
        if not portfolio_dfs:
            logger.warning("No valid portfolio summary files found. Portfolio metrics will be missing.")
            portfolio_grouped = pd.DataFrame(columns=["factor"])
        else:
            portfolio_merged = pd.concat(portfolio_dfs, ignore_index=True)
            # Aggregate portfolio metrics (average for most, sum for n_periods)
            agg_dict = {
                "annualized_return": "mean",
                "sharpe": "mean",
                "max_drawdown": "mean",
                "win_ratio": "mean",
                "n_periods": "sum",  # Sum periods across runs
            }
            
            # Check if deflated_sharpe exists
            if "deflated_sharpe" in portfolio_merged.columns:
                agg_dict["deflated_sharpe"] = "mean"
            
            portfolio_grouped = portfolio_merged.groupby("factor", as_index=False).agg(agg_dict)
    else:
        portfolio_grouped = pd.DataFrame(columns=["factor"])
    
    # Merge all summaries on factor name
    ranking_df = ic_grouped[["factor", "mean_ic", "ic_ir", "hit_ratio", "count"]].copy()
    ranking_df = ranking_df.rename(columns={"count": "n_periods_ic"})
    
    # Add Rank-IC metrics
    if not rank_ic_grouped.empty and "mean_rank_ic" in rank_ic_grouped.columns:
        ranking_df = ranking_df.merge(
            rank_ic_grouped[["factor", "mean_rank_ic", "rank_ic_ir"]],
            on="factor",
            how="left"
        )
    else:
        ranking_df["mean_rank_ic"] = np.nan
        ranking_df["rank_ic_ir"] = np.nan
    
    # Add portfolio metrics
    if not portfolio_grouped.empty and "sharpe" in portfolio_grouped.columns:
        portfolio_cols = ["factor", "annualized_return", "sharpe", "max_drawdown", "win_ratio"]
        if "deflated_sharpe" in portfolio_grouped.columns:
            portfolio_cols.append("deflated_sharpe")
        if "n_periods" in portfolio_grouped.columns:
            portfolio_cols.append("n_periods")
        
        ranking_df = ranking_df.merge(
            portfolio_grouped[portfolio_cols],
            on="factor",
            how="left"
        )
        
        # Rename portfolio columns with "ls_" prefix
        rename_map = {
            "annualized_return": "ls_annualized_return",
            "sharpe": "ls_sharpe",
            "deflated_sharpe": "ls_deflated_sharpe",
            "max_drawdown": "ls_max_drawdown",
            "win_ratio": "ls_win_ratio",
            "n_periods": "n_periods_portfolio",
        }
        ranking_df = ranking_df.rename(columns=rename_map)
    else:
        ranking_df["ls_annualized_return"] = np.nan
        ranking_df["ls_sharpe"] = np.nan
        ranking_df["ls_deflated_sharpe"] = np.nan
        ranking_df["ls_max_drawdown"] = np.nan
        ranking_df["ls_win_ratio"] = np.nan
        ranking_df["n_periods_portfolio"] = np.nan
    
    # Rename factor column to factor_name
    ranking_df = ranking_df.rename(columns={"factor": "factor_name"})
    
    # Compute combined_score
    # Weighted combination: 60% ic_ir, 40% ls_deflated_sharpe (if available)
    # Normalize both to 0-1 range for fair comparison
    ranking_df["combined_score"] = np.nan
    
    # Normalize ic_ir (assume range roughly -2 to +2, but use actual min/max)
    ic_ir_values = ranking_df["ic_ir"].dropna()
    if len(ic_ir_values) > 0:
        ic_ir_min = ic_ir_values.min()
        ic_ir_max = ic_ir_values.max()
        ic_ir_range = ic_ir_max - ic_ir_min
        if ic_ir_range > 1e-10:
            ranking_df["ic_ir_normalized"] = (ranking_df["ic_ir"] - ic_ir_min) / ic_ir_range
        else:
            ranking_df["ic_ir_normalized"] = 0.5  # All same value
    else:
        ranking_df["ic_ir_normalized"] = 0.0
    
    # Normalize ls_deflated_sharpe (if available)
    dsr_values = ranking_df["ls_deflated_sharpe"].dropna()
    if len(dsr_values) > 0:
        dsr_min = dsr_values.min()
        dsr_max = dsr_values.max()
        dsr_range = dsr_max - dsr_min
        if dsr_range > 1e-10:
            ranking_df["dsr_normalized"] = (ranking_df["ls_deflated_sharpe"] - dsr_min) / dsr_range
        else:
            ranking_df["dsr_normalized"] = 0.5
    else:
        ranking_df["dsr_normalized"] = 0.0
    
    # Compute combined_score
    # If both metrics available: 60% ic_ir, 40% dsr
    # If only ic_ir: 100% ic_ir
    # If only dsr: 100% dsr
    has_ic_ir = ranking_df["ic_ir"].notna()
    has_dsr = ranking_df["ls_deflated_sharpe"].notna()
    
    # Both available
    both_mask = has_ic_ir & has_dsr
    ranking_df.loc[both_mask, "combined_score"] = (
        0.6 * ranking_df.loc[both_mask, "ic_ir_normalized"] +
        0.4 * ranking_df.loc[both_mask, "dsr_normalized"]
    )
    
    # Only ic_ir
    only_ic_mask = has_ic_ir & ~has_dsr
    ranking_df.loc[only_ic_mask, "combined_score"] = ranking_df.loc[only_ic_mask, "ic_ir_normalized"]
    
    # Only dsr
    only_dsr_mask = ~has_ic_ir & has_dsr
    ranking_df.loc[only_dsr_mask, "combined_score"] = ranking_df.loc[only_dsr_mask, "dsr_normalized"]
    
    # Drop normalization columns
    ranking_df = ranking_df.drop(columns=["ic_ir_normalized", "dsr_normalized"], errors="ignore")
    
    # Sort by combined_score (descending)
    ranking_df = ranking_df.sort_values("combined_score", ascending=False, na_last=True).reset_index(drop=True)
    
    logger.info(f"Built factor ranking table with {len(ranking_df)} factors")
    
    return ranking_df

