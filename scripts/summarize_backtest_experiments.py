"""Summarize backtest experiment runs.

This script reads experiment runs from the experiments directory, filters by tags,
and generates a summary table with key performance metrics.

Usage:
    python scripts/summarize_backtest_experiments.py --experiments-root experiments --filter-tag "altdata" --output-csv output/summaries/altdata_backtests.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tabulate import tabulate


def extract_universe_from_name(name: str) -> str:
    """Extract universe name from experiment name.
    
    Examples:
        "trend_baseline_ai_tech_2000_2025" -> "ai_tech"
        "trend_baseline_healthcare_2000_2025" -> "healthcare"
        "baseline_v1" -> "unknown"
    """
    parts = name.lower().split("_")
    
    # Look for known universe identifiers
    universe_keywords = [
        "ai_tech", "healthcare", "energy", "defense", "consumer",
        "biotech", "cyclicals", "security", "aero", "financial", "misc"
    ]
    
    for keyword in universe_keywords:
        if keyword in parts:
            # Try to find the full universe name
            idx = parts.index(keyword)
            if idx > 0 and idx < len(parts) - 1:
                # Check if previous part is also part of universe name
                potential = "_".join(parts[idx-1:idx+1])
                if potential in ["ai_tech", "healthcare_biotech", "energy_resources_cyclicals", 
                                "defense_security_aero", "consumer_financial_misc"]:
                    return potential.replace("_", " ").title()
            return keyword.replace("_", " ").title()
    
    return "unknown"


def load_experiment_run(run_dir: Path) -> Optional[dict]:
    """Load experiment run metadata and metrics.
    
    Args:
        run_dir: Path to experiment run directory
    
    Returns:
        Dictionary with run summary, or None if failed
    """
    run_json = run_dir / "run.json"
    metrics_csv = run_dir / "metrics.csv"
    
    if not run_json.exists():
        return None
    
    try:
        # Load run metadata
        with open(run_json, "r", encoding="utf-8") as f:
            run_data = json.load(f)
        
        # Load metrics
        metrics_data = {}
        if metrics_csv.exists():
            try:
                metrics_df = pd.read_csv(metrics_csv)
                
                # Extract scalar metrics (latest value for time-series metrics)
                if len(metrics_df) > 0:
                    # Get latest values for each metric
                    for metric_name in metrics_df["metric_name"].unique():
                        metric_rows = metrics_df[metrics_df["metric_name"] == metric_name]
                        if len(metric_rows) > 0:
                            # Use last value
                            latest_row = metric_rows.iloc[-1]
                            try:
                                metrics_data[metric_name] = float(latest_row["metric_value"])
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                print(f"Warning: Failed to load metrics from {metrics_csv}: {e}")
        
        # Extract key metrics
        summary = {
            "run_id": run_data.get("run_id", run_dir.name),
            "name": run_data.get("name", "unknown"),
            "created_at": run_data.get("created_at", ""),
            "status": run_data.get("status", "unknown"),
            "tags": ",".join(run_data.get("tags", [])),
            "universe": extract_universe_from_name(run_data.get("name", "")),
            # Metrics
            "cagr": metrics_data.get("cagr"),
            "sharpe_ratio": metrics_data.get("sharpe_ratio"),
            "max_drawdown_pct": metrics_data.get("max_drawdown_pct"),
            "total_return": metrics_data.get("total_return"),
            "final_pf": metrics_data.get("final_pf"),
            "total_trades": metrics_data.get("total_trades"),
            # Config info
            "strategy": run_data.get("config", {}).get("strategy", "unknown"),
            "freq": run_data.get("config", {}).get("freq", "unknown"),
            "symbol_count": run_data.get("config", {}).get("symbol_count"),
            "start_capital": run_data.get("config", {}).get("start_capital"),
        }
        
        # Add symbols if available
        config = run_data.get("config", {})
        if "symbols" in config:
            symbols_list = config["symbols"]
            if isinstance(symbols_list, list):
                summary["symbols"] = ",".join(symbols_list[:5]) + ("..." if len(symbols_list) > 5 else "")
            else:
                summary["symbols"] = str(symbols_list)
        else:
            summary["symbols"] = ""
        
        return summary
        
    except Exception as e:
        print(f"Warning: Failed to load run from {run_dir}: {e}")
        return None


def summarize_experiments(
    experiments_root: Path,
    filter_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize all experiment runs in the experiments directory.
    
    Args:
        experiments_root: Root directory containing experiment runs
        filter_tag: Optional tag to filter runs (e.g., "altdata")
    
    Returns:
        DataFrame with one row per experiment run
    """
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_root}")
    
    runs = []
    
    # Find all run directories
    for run_dir in sorted(experiments_root.iterdir()):
        if not run_dir.is_dir():
            continue
        
        run_summary = load_experiment_run(run_dir)
        if run_summary is None:
            continue
        
        # Filter by tag if specified
        if filter_tag:
            tags = run_summary.get("tags", "").lower()
            if filter_tag.lower() not in tags:
                continue
        
        runs.append(run_summary)
    
    if not runs:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(runs)
    
    # Sort by created_at (newest first)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values("created_at", ascending=False)
    
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Print summary table to stdout.
    
    Args:
        df: DataFrame with experiment summaries
    """
    if df.empty:
        print("No experiment runs found matching criteria.")
        return
    
    # Select columns for display
    display_cols = [
        "universe",
        "name",
        "cagr",
        "sharpe_ratio",
        "max_drawdown_pct",
        "total_trades",
        "status",
        "created_at",
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    
    # Format DataFrame for display
    display_df = df[available_cols].copy()
    
    # Format percentages and dates
    if "cagr" in display_df.columns:
        display_df["cagr"] = display_df["cagr"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    if "max_drawdown_pct" in display_df.columns:
        display_df["max_drawdown_pct"] = display_df["max_drawdown_pct"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )
    if "sharpe_ratio" in display_df.columns:
        display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "N/A"
        )
    
    print("=" * 100)
    print(f"Backtest Experiment Summary ({len(df)} runs)")
    print("=" * 100)
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    print("=" * 100)
    
    # Universe aggregation (if multiple runs per universe)
    if "universe" in df.columns and "cagr" in df.columns:
        print("\n" + "=" * 100)
        print("Universe Aggregation")
        print("=" * 100)
        
        universe_stats = []
        for universe in df["universe"].unique():
            if universe == "unknown":
                continue
            
            universe_runs = df[df["universe"] == universe]
            if len(universe_runs) == 0:
                continue
            
            cagr_values = universe_runs["cagr"].dropna()
            sharpe_values = universe_runs["sharpe_ratio"].dropna()
            dd_values = universe_runs["max_drawdown_pct"].dropna()
            
            stats = {
                "universe": universe,
                "runs": len(universe_runs),
                "avg_cagr": cagr_values.mean() if len(cagr_values) > 0 else None,
                "median_cagr": cagr_values.median() if len(cagr_values) > 0 else None,
                "avg_sharpe": sharpe_values.mean() if len(sharpe_values) > 0 else None,
                "avg_drawdown": dd_values.mean() if len(dd_values) > 0 else None,
            }
            universe_stats.append(stats)
        
        if universe_stats:
            stats_df = pd.DataFrame(universe_stats)
            
            # Format
            if "avg_cagr" in stats_df.columns:
                stats_df["avg_cagr"] = stats_df["avg_cagr"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
            if "median_cagr" in stats_df.columns:
                stats_df["median_cagr"] = stats_df["median_cagr"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
            if "avg_sharpe" in stats_df.columns:
                stats_df["avg_sharpe"] = stats_df["avg_sharpe"].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
            if "avg_drawdown" in stats_df.columns:
                stats_df["avg_drawdown"] = stats_df["avg_drawdown"].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                )
            
            print(tabulate(stats_df, headers="keys", tablefmt="grid", showindex=False))
            print("=" * 100)


def main() -> None:
    """Main entry point for experiment summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize backtest experiment runs from experiments directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Summarize all runs
    python scripts/summarize_backtest_experiments.py
    
    # Filter by tag
    python scripts/summarize_backtest_experiments.py --filter-tag "altdata"
    
    # Save to CSV
    python scripts/summarize_backtest_experiments.py \\
      --filter-tag "altdata" \\
      --output-csv output/experiment_summaries/altdata_backtests.csv
        """,
    )
    
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing experiment runs (default: experiments)",
    )
    
    parser.add_argument(
        "--filter-tag",
        type=str,
        default=None,
        help="Optional tag to filter runs (e.g., 'altdata', 'ai_tech'). Only runs with this tag are included.",
    )
    
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save summary as CSV file",
    )
    
    args = parser.parse_args()
    
    try:
        # Summarize experiments
        df = summarize_experiments(args.experiments_root, filter_tag=args.filter_tag)
        
        if df.empty:
            print(f"No experiment runs found in {args.experiments_root}")
            if args.filter_tag:
                print(f"(filtered by tag: {args.filter_tag})")
            exit(0)
        
        # Print summary table
        print_summary_table(df)
        
        # Save to CSV if requested
        if args.output_csv:
            output_dir = args.output_csv.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            print(f"\nSummary saved to: {args.output_csv}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()

