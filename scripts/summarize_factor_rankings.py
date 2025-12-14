"""Summarize Factor Rankings from Multiple Analysis Outputs.

This script automatically finds all IC summary, Rank-IC summary, and portfolio
summary CSV files in the output/factor_analysis/ directory and generates a
consolidated factor ranking table.

Usage:
    python scripts/summarize_factor_rankings.py [--output-dir OUTPUT_DIR]

The script will:
1. Find all *_ic_summary.csv files
2. Find all *_rank_ic_summary.csv files
3. Find all *_portfolio_summary.csv files
4. Merge them into a single ranking table
5. Save as CSV and Markdown report
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.factor_ranking import build_factor_ranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_summary_files(output_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Find all IC, Rank-IC, and portfolio summary CSV files.
    
    Args:
        output_dir: Directory to search for summary files
    
    Returns:
        Tuple of (ic_summary_paths, rank_ic_summary_paths, portfolio_summary_paths)
    """
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return [], [], []
    
    ic_summary_paths = sorted(output_dir.glob("*_ic_summary.csv"))
    rank_ic_summary_paths = sorted(output_dir.glob("*_rank_ic_summary.csv"))
    portfolio_summary_paths = sorted(output_dir.glob("*_portfolio_summary.csv"))
    
    logger.info(f"Found {len(ic_summary_paths)} IC summary files")
    logger.info(f"Found {len(rank_ic_summary_paths)} Rank-IC summary files")
    logger.info(f"Found {len(portfolio_summary_paths)} portfolio summary files")
    
    return ic_summary_paths, rank_ic_summary_paths, portfolio_summary_paths


def write_ranking_report(
    ranking_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write ranking table to CSV and Markdown files.
    
    Args:
        ranking_df: Factor ranking DataFrame
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    csv_path = output_dir / "factor_ranking_overview.csv"
    ranking_df.to_csv(csv_path, index=False)
    logger.info(f"Saved ranking table to {csv_path}")
    
    # Write Markdown report
    md_path = output_dir / "factor_ranking_overview.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Factor Ranking Overview\n\n")
        f.write("This report consolidates factor analysis results from multiple runs.\n\n")
        
        f.write("## Combined Score Explanation\n\n")
        f.write("The `combined_score` is a weighted combination of:\n\n")
        f.write("- **60% IC-IR (Information Ratio)**: Measures the consistency of a factor's ")
        f.write("predictive power. Higher values indicate more stable factor performance.\n")
        f.write("- **40% Deflated Sharpe Ratio**: Measures the risk-adjusted return of the ")
        f.write("Long/Short portfolio, adjusted for multiple testing bias.\n\n")
        f.write("If only one metric is available, it is used at 100% weight.\n\n")
        f.write("Both metrics are normalized to a 0-1 range before combination.\n\n")
        
        f.write("## Top 20 Factors\n\n")
        
        # Display top 20 factors
        top_20 = ranking_df.head(20).copy()
        
        # Select relevant columns for display
        display_cols = [
            "factor_name",
            "combined_score",
            "ic_ir",
            "ls_deflated_sharpe",
            "mean_ic",
            "hit_ratio",
            "ls_sharpe",
            "ls_annualized_return",
            "ls_max_drawdown",
            "ls_win_ratio",
        ]
        
        # Only include columns that exist
        display_cols = [col for col in display_cols if col in top_20.columns]
        
        # Format numeric columns for display
        display_df = top_20[display_cols].copy()
        for col in display_df.columns:
            if col == "factor_name":
                continue
            if display_df[col].dtype in ["float64", "float32"]:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        # Write table
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Full Ranking Table\n\n")
        f.write(f"Total factors analyzed: {len(ranking_df)}\n\n")
        f.write("The complete ranking table is available in CSV format: ")
        f.write(f"`{csv_path.name}`\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write("### IC Metrics\n\n")
        if "mean_ic" in ranking_df.columns:
            mean_ic_stats = ranking_df["mean_ic"].describe()
            f.write(f"- Mean IC: {mean_ic_stats['mean']:.4f}\n")
            f.write(f"- Median IC: {mean_ic_stats['50%']:.4f}\n")
            f.write(f"- Max IC: {mean_ic_stats['max']:.4f}\n")
            f.write(f"- Min IC: {mean_ic_stats['min']:.4f}\n\n")
        
        if "ic_ir" in ranking_df.columns:
            ic_ir_stats = ranking_df["ic_ir"].describe()
            f.write(f"- Mean IC-IR: {ic_ir_stats['mean']:.4f}\n")
            f.write(f"- Median IC-IR: {ic_ir_stats['50%']:.4f}\n")
            f.write(f"- Max IC-IR: {ic_ir_stats['max']:.4f}\n\n")
        
        f.write("### Portfolio Metrics\n\n")
        if "ls_sharpe" in ranking_df.columns:
            sharpe_stats = ranking_df["ls_sharpe"].dropna().describe()
            if len(sharpe_stats) > 0:
                f.write(f"- Mean Sharpe: {sharpe_stats['mean']:.4f}\n")
                f.write(f"- Median Sharpe: {sharpe_stats['50%']:.4f}\n")
                f.write(f"- Max Sharpe: {sharpe_stats['max']:.4f}\n\n")
        
        if "ls_deflated_sharpe" in ranking_df.columns:
            dsr_stats = ranking_df["ls_deflated_sharpe"].dropna().describe()
            if len(dsr_stats) > 0:
                f.write(f"- Mean Deflated Sharpe: {dsr_stats['mean']:.4f}\n")
                f.write(f"- Median Deflated Sharpe: {dsr_stats['50%']:.4f}\n")
                f.write(f"- Max Deflated Sharpe: {dsr_stats['max']:.4f}\n\n")
    
    logger.info(f"Saved ranking report to {md_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize factor rankings from multiple analysis outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "factor_analysis",
        help="Directory containing factor analysis outputs (default: output/factor_analysis)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Factor Ranking Summary")
    logger.info("=" * 80)
    logger.info(f"Searching in: {args.output_dir}")
    logger.info("")
    
    # Find summary files
    ic_paths, rank_ic_paths, portfolio_paths = find_summary_files(args.output_dir)
    
    if not ic_paths and not rank_ic_paths:
        logger.error("No IC or Rank-IC summary files found. Cannot generate ranking.")
        return 1
    
    if not ic_paths:
        logger.warning("No IC summary files found. Using only Rank-IC summaries.")
    
    if not rank_ic_paths:
        logger.warning("No Rank-IC summary files found. Using only IC summaries.")
    
    # Build ranking table
    try:
        ranking_df = build_factor_ranking(
            ic_summary_paths=ic_paths,
            rank_ic_summary_paths=rank_ic_paths,
            portfolio_summary_paths=portfolio_paths if portfolio_paths else None,
        )
        
        logger.info(f"Built ranking table with {len(ranking_df)} factors")
        logger.info("")
        
        # Display top 10
        logger.info("Top 10 Factors by Combined Score:")
        top_10 = ranking_df.head(10)
        for idx, row in top_10.iterrows():
            logger.info(
                f"  {idx + 1:2d}. {row['factor_name']:40s} "
                f"Score: {row['combined_score']:.4f} "
                f"(IC-IR: {row.get('ic_ir', 'N/A'):.4f if pd.notna(row.get('ic_ir')) else 'N/A'}, "
                f"DSR: {row.get('ls_deflated_sharpe', 'N/A'):.4f if pd.notna(row.get('ls_deflated_sharpe')) else 'N/A'})"
            )
        logger.info("")
        
        # Write reports
        write_ranking_report(ranking_df, args.output_dir)
        
        logger.info("=" * 80)
        logger.info("Factor ranking summary completed successfully!")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"Failed to build factor ranking: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

