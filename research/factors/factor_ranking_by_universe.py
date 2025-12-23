"""Factor Ranking by Universe Research Script.

This script analyzes factor rankings across different universes (e.g., AI/Tech vs. Macro-ETFs)
by grouping factor analysis outputs by universe and factor set.

It reads all factor analysis output files, extracts universe and factor_set information
from filenames, and creates ranking tables for each (universe, factor_set) combination.

Usage:
    python research/factors/factor_ranking_by_universe.py [--output-dir OUTPUT_DIR] [--plots]

Output:
    - output/factor_analysis/FACTOR_RANKING_BY_UNIVERSE.md: Consolidated ranking report
    - output/factor_analysis/plots/*.png: Optional bar plots (if --plots is set)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.factor_ranking import build_factor_ranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_universe_from_filename(filename: str) -> str | None:
    """Extract universe identifier from filename.

    Since factor analysis output files don't directly contain universe information
    in their names (format: factor_analysis_{factor_set}_{horizon}d_{freq}_...),
    this function attempts to infer the universe from:
    1. Common universe patterns in the filename (if manually added)
    2. Symbols file patterns in parent directory paths

    Known universes:
    - macro_world_etfs
    - universe_ai_tech
    - healthcare_biotech
    - energy_resources_cyclicals
    - defense_security_aero
    - consumer_financial_misc

    Args:
        filename: Filename or full path to check

    Returns:
        Universe identifier if found, None otherwise
    """
    filename_lower = filename.lower()
    path_lower = str(filename).lower() if isinstance(filename, Path) else filename_lower

    # Common universe patterns (order matters - more specific first)
    universe_patterns = [
        ("macro_world_etfs", r"macro.*world.*etf|macro_world_etfs"),
        ("universe_ai_tech", r"universe.*ai.*tech|ai.*tech.*universe|universe_ai_tech"),
        (
            "healthcare_biotech",
            r"healthcare.*biotech|biotech.*healthcare|healthcare_biotech",
        ),
        (
            "energy_resources_cyclicals",
            r"energy.*resource|resource.*energy|energy_resources",
        ),
        (
            "defense_security_aero",
            r"defense.*security|security.*defense|defense_security",
        ),
        (
            "consumer_financial_misc",
            r"consumer.*financial|financial.*consumer|consumer_financial",
        ),
    ]

    for universe_id, pattern in universe_patterns:
        if re.search(pattern, filename_lower) or re.search(pattern, path_lower):
            return universe_id

    # Try to extract from symbols-file pattern in path
    # (e.g., config/macro_world_etfs_tickers.txt or macro_world_etfs_tickers)
    symbols_file_match = re.search(r"([a-z_]+)_tickers", path_lower)
    if symbols_file_match:
        universe_candidate = symbols_file_match.group(1)
        # Validate it's a known universe
        known_universes = [u[0] for u in universe_patterns]
        if universe_candidate in known_universes:
            return universe_candidate

    return None


def extract_factor_set_from_filename(filename: str) -> str | None:
    """Extract factor_set identifier from filename.

    Expected pattern: factor_analysis_{factor_set}_{horizon}d_{freq}_...

    Args:
        filename: Filename to check

    Returns:
        Factor set identifier if found, None otherwise
    """
    # Pattern: factor_analysis_{factor_set}_{horizon}d_{freq}_
    match = re.search(r"factor_analysis_([^_]+(?:_[^_]+)*?)_(\d+)d_", filename)
    if match:
        return match.group(1)

    return None


def group_files_by_universe_and_factor_set(
    output_dir: Path,
) -> dict[tuple[str, str], dict[str, list[Path]]]:
    """Group summary files by (universe, factor_set) combination.

    Note: Since factor analysis output filenames don't directly contain universe
    information (format: factor_analysis_{factor_set}_{horizon}d_{freq}_...),
    this function attempts to extract universe from:
    1. Filename/path patterns (if manually added)
    2. Subdirectory names (e.g., output/factor_analysis/macro_world_etfs/...)

    If universe cannot be determined, files are skipped with a warning.

    Args:
        output_dir: Directory containing factor analysis outputs

    Returns:
        Dictionary mapping (universe, factor_set) tuples to dictionaries with:
        - 'ic_summary': List of IC summary file paths
        - 'rank_ic_summary': List of Rank-IC summary file paths
        - 'portfolio_summary': List of portfolio summary file paths
    """
    grouped = defaultdict(
        lambda: {
            "ic_summary": [],
            "rank_ic_summary": [],
            "portfolio_summary": [],
        }
    )

    # Find all summary files (including in subdirectories)
    ic_files = list(output_dir.rglob("*_ic_summary.csv"))
    rank_ic_files = list(output_dir.rglob("*_rank_ic_summary.csv"))
    portfolio_files = list(output_dir.rglob("*_portfolio_summary.csv"))

    logger.info(f"Found {len(ic_files)} IC summary files")
    logger.info(f"Found {len(rank_ic_files)} Rank-IC summary files")
    logger.info(f"Found {len(portfolio_files)} portfolio summary files")

    # Helper function to extract universe from file path
    def get_universe_from_path(file_path: Path) -> str | None:
        # Try filename first
        universe = extract_universe_from_filename(file_path.name)
        if universe:
            return universe

        # Try parent directory names
        for parent in file_path.parents:
            parent_universe = extract_universe_from_filename(parent.name)
            if parent_universe:
                return parent_universe

        # Try full path
        return extract_universe_from_filename(str(file_path))

    # Group IC files
    for file_path in ic_files:
        universe = get_universe_from_path(file_path)
        factor_set = extract_factor_set_from_filename(file_path.name)

        if not universe or not factor_set:
            logger.warning(
                f"Could not extract universe/factor_set from {file_path.name}. "
                f"Skipping. (Hint: Consider organizing files in subdirectories named after universes)"
            )
            continue

        grouped[(universe, factor_set)]["ic_summary"].append(file_path)

    # Group Rank-IC files
    for file_path in rank_ic_files:
        universe = get_universe_from_path(file_path)
        factor_set = extract_factor_set_from_filename(file_path.name)

        if not universe or not factor_set:
            continue

        grouped[(universe, factor_set)]["rank_ic_summary"].append(file_path)

    # Group portfolio files
    for file_path in portfolio_files:
        universe = get_universe_from_path(file_path)
        factor_set = extract_factor_set_from_filename(file_path.name)

        if not universe or not factor_set:
            continue

        grouped[(universe, factor_set)]["portfolio_summary"].append(file_path)

    logger.info(
        f"Grouped files into {len(grouped)} (universe, factor_set) combinations"
    )

    return dict(grouped)


def create_ranking_by_universe(
    grouped_files: dict[tuple[str, str], dict[str, list[Path]]],
) -> pd.DataFrame:
    """Create consolidated ranking table grouped by universe and factor_set.

    Args:
        grouped_files: Dictionary from group_files_by_universe_and_factor_set()

    Returns:
        DataFrame with columns: universe, factor_set, factor_name, and all ranking metrics
    """
    all_rankings = []

    for (universe, factor_set), files in grouped_files.items():
        logger.info(
            f"Building ranking for universe={universe}, factor_set={factor_set}"
        )

        ic_paths = files.get("ic_summary", [])
        rank_ic_paths = files.get("rank_ic_summary", [])
        portfolio_paths = files.get("portfolio_summary", [])

        if not ic_paths and not rank_ic_paths:
            logger.warning(
                f"No IC or Rank-IC files for ({universe}, {factor_set}). Skipping."
            )
            continue

        try:
            ranking_df = build_factor_ranking(
                ic_summary_paths=ic_paths,
                rank_ic_summary_paths=rank_ic_paths,
                portfolio_summary_paths=portfolio_paths if portfolio_paths else None,
            )

            # Add universe and factor_set columns
            ranking_df["universe"] = universe
            ranking_df["factor_set"] = factor_set

            # Reorder columns: universe, factor_set, factor_name first
            cols = ["universe", "factor_set", "factor_name"] + [
                col
                for col in ranking_df.columns
                if col not in ["universe", "factor_set", "factor_name"]
            ]
            ranking_df = ranking_df[cols]

            all_rankings.append(ranking_df)

        except Exception as e:
            logger.error(
                f"Failed to build ranking for ({universe}, {factor_set}): {e}",
                exc_info=True,
            )
            continue

    if not all_rankings:
        logger.warning(
            "No rankings generated. Check if summary files exist and have correct naming."
        )
        return pd.DataFrame()

    consolidated = pd.concat(all_rankings, ignore_index=True)
    logger.info(f"Created consolidated ranking with {len(consolidated)} factor entries")

    return consolidated


def write_universe_ranking_report(
    consolidated_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write ranking report grouped by universe and factor_set.

    Args:
        consolidated_df: Consolidated ranking DataFrame
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "FACTOR_RANKING_BY_UNIVERSE.md"

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Factor Ranking by Universe\n\n")
        f.write(
            "This report shows factor rankings grouped by universe and factor set.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(f"Total factor entries: {len(consolidated_df)}\n")
        f.write(f"Universes analyzed: {consolidated_df['universe'].nunique()}\n")
        f.write(f"Factor sets analyzed: {consolidated_df['factor_set'].nunique()}\n\n")

        # Group by universe
        for universe in sorted(consolidated_df["universe"].unique()):
            universe_data = consolidated_df[consolidated_df["universe"] == universe]

            f.write(f"## Universe: {universe}\n\n")

            # Group by factor_set within universe
            for factor_set in sorted(universe_data["factor_set"].unique()):
                factor_set_data = universe_data[
                    universe_data["factor_set"] == factor_set
                ].copy()

                f.write(f"### Factor Set: {factor_set}\n\n")

                # Get top 5 factors
                top_5 = factor_set_data.head(5).copy()

                # Select display columns
                display_cols = [
                    "factor_name",
                    "combined_score",
                    "ic_ir",
                    "mean_ic",
                    "hit_ratio",
                    "ls_deflated_sharpe",
                    "ls_sharpe",
                    "ls_annualized_return",
                ]
                display_cols = [col for col in display_cols if col in top_5.columns]

                # Format numeric columns
                display_df = top_5[display_cols].copy()
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

                # Add summary statistics
                if "combined_score" in factor_set_data.columns:
                    stats = factor_set_data["combined_score"].describe()
                    f.write(f"**Summary:** Top score: {stats['max']:.4f}, ")
                    f.write(f"Median: {stats['50%']:.4f}, ")
                    f.write(f"Mean: {stats['mean']:.4f}\n\n")

                f.write("---\n\n")

    logger.info(f"Saved universe ranking report to {md_path}")


def create_plots(
    consolidated_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create bar plots for top factors by universe.

    Args:
        consolidated_df: Consolidated ranking DataFrame
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        logger.warning("matplotlib not available. Skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create plot for each universe
    for universe in sorted(consolidated_df["universe"].unique()):
        universe_data = consolidated_df[consolidated_df["universe"] == universe]

        # Get top 10 factors by combined_score
        top_10 = universe_data.nlargest(10, "combined_score")

        if top_10.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar plot: factor_name vs combined_score
        factor_names = top_10["factor_name"].values
        scores = top_10["combined_score"].values

        bars = ax.barh(range(len(factor_names)), scores, color="steelblue")
        ax.set_yticks(range(len(factor_names)))
        ax.set_yticklabels(factor_names)
        ax.set_xlabel("Combined Score")
        ax.set_title(f"Top 10 Factors by Combined Score - {universe}")
        ax.invert_yaxis()  # Highest score at top
        ax.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        # Save plot
        plot_filename = f"top_factors_{universe.lower().replace(' ', '_')}.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot to {plot_path}")

    logger.info(
        f"Created {len(consolidated_df['universe'].unique())} plots in {plots_dir}"
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze factor rankings by universe and factor set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "factor_analysis",
        help="Directory containing factor analysis outputs (default: output/factor_analysis)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate bar plots for top factors (requires matplotlib)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Factor Ranking by Universe Analysis")
    logger.info("=" * 80)
    logger.info(f"Searching in: {args.output_dir}")
    logger.info("")

    if not args.output_dir.exists():
        logger.error(f"Output directory does not exist: {args.output_dir}")
        return 1

    # Group files by universe and factor_set
    grouped_files = group_files_by_universe_and_factor_set(args.output_dir)

    if not grouped_files:
        logger.error("No files found with recognizable universe/factor_set patterns.")
        logger.info(
            "Expected filename pattern: factor_analysis_{factor_set}_{horizon}d_{freq}_..."
        )
        return 1

    logger.info(f"Found {len(grouped_files)} (universe, factor_set) combinations:")
    for (universe, factor_set), files in sorted(grouped_files.items()):
        logger.info(
            f"  - {universe} / {factor_set}: "
            f"{len(files['ic_summary'])} IC, "
            f"{len(files['rank_ic_summary'])} Rank-IC, "
            f"{len(files['portfolio_summary'])} Portfolio files"
        )
    logger.info("")

    # Create consolidated ranking
    consolidated_df = create_ranking_by_universe(grouped_files)

    if consolidated_df.empty:
        logger.error("Failed to create any rankings. Check input files.")
        return 1

    # Write report
    write_universe_ranking_report(consolidated_df, args.output_dir)

    # Create plots if requested
    if args.plots:
        create_plots(consolidated_df, args.output_dir)

    # Display summary
    logger.info("=" * 80)
    logger.info("Summary by Universe:")
    logger.info("=" * 80)
    for universe in sorted(consolidated_df["universe"].unique()):
        universe_data = consolidated_df[consolidated_df["universe"] == universe]
        logger.info(f"\n{universe}:")
        for factor_set in sorted(universe_data["factor_set"].unique()):
            factor_set_data = universe_data[universe_data["factor_set"] == factor_set]
            top_factor = factor_set_data.iloc[0] if not factor_set_data.empty else None
            if top_factor is not None:
                logger.info(
                    f"  {factor_set}: "
                    f"Top factor = {top_factor['factor_name']} "
                    f"(Score: {top_factor['combined_score']:.4f})"
                )

    logger.info("")
    logger.info("=" * 80)
    logger.info("Factor ranking by universe analysis completed successfully!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
