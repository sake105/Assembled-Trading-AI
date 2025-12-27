#!/usr/bin/env python
"""
Leaderboard Tool - Rank and display best runs from batch backtest results.

Reads summary.csv from batch output directory and ranks runs by various metrics.

Example:
    python scripts/leaderboard.py --batch-output output/batch_backtests/my_batch --sort-by sharpe --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_batch_summary(batch_output_dir: Path) -> pd.DataFrame:
    """Load batch summary CSV.
    
    Args:
        batch_output_dir: Path to batch output directory (contains summary.csv)
        
    Returns:
        DataFrame with batch summary data
        
    Raises:
        FileNotFoundError: If summary.csv does not exist
        ValueError: If CSV is empty or invalid
    """
    summary_csv = batch_output_dir / "summary.csv"
    
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"summary.csv not found in {batch_output_dir}. "
            f"Ensure this is a valid batch output directory."
        )
    
    df = pd.read_csv(summary_csv)
    
    if df.empty:
        raise ValueError(f"summary.csv is empty in {batch_output_dir}")
    
    return df


def rank_runs(
    df: pd.DataFrame,
    sort_by: str = "sharpe",
    top_k: int = 20,
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank runs by specified metric.
    
    Args:
        df: Summary DataFrame
        sort_by: Column name to sort by ("sharpe", "total_return", "final_pf")
        top_k: Number of top runs to return
        ascending: Sort ascending (default: False = descending)
        
    Returns:
        Sorted DataFrame with top_k rows
        
    Raises:
        ValueError: If sort_by column is missing or invalid
    """
    # Validate sort_by column
    valid_sort_columns = ["sharpe", "total_return", "final_pf", "max_drawdown_pct", "cagr", "trades"]
    
    if sort_by not in df.columns:
        available_cols = [col for col in valid_sort_columns if col in df.columns]
        raise ValueError(
            f"Column '{sort_by}' not found in summary.csv. "
            f"Available columns: {', '.join(df.columns)}. "
            f"Valid sort columns: {', '.join(available_cols)}"
        )
    
    # Handle ascending sort (e.g., max_drawdown_pct should be sorted ascending for best = smallest)
    if sort_by == "max_drawdown_pct":
        ascending = True  # Best drawdown is smallest (least negative)
    
    # Sort by specified column (handle NaN values - put them at the end)
    df_sorted = df.sort_values(
        by=sort_by,
        ascending=ascending,
        na_last=True,  # Put NaN values at the end
    )
    
    # Return top_k rows
    return df_sorted.head(top_k).copy()


def format_leaderboard_table(df: pd.DataFrame, sort_by: str) -> str:
    """Format leaderboard as a table string.
    
    Args:
        df: Sorted DataFrame with runs
        sort_by: Column used for sorting (for display)
        
    Returns:
        Formatted table string
    """
    # Select columns to display
    display_columns = ["run_id", "status", "runtime_sec"]
    
    # Add metric columns if they exist
    metric_columns = ["final_pf", "total_return", "cagr", "sharpe", "max_drawdown_pct", "trades"]
    for col in metric_columns:
        if col in df.columns:
            display_columns.append(col)
    
    # Select and format DataFrame
    display_df = df[display_columns].copy()
    
    # Format numeric columns
    if "runtime_sec" in display_df.columns:
        display_df["runtime_sec"] = display_df["runtime_sec"].apply(
            lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
        )
    
    # Format metric columns (2-4 decimal places depending on metric)
    format_map = {
        "final_pf": lambda x: f"{x:.4f}",
        "total_return": lambda x: f"{x:.2%}",
        "cagr": lambda x: f"{x:.2%}",
        "sharpe": lambda x: f"{x:.4f}",
        "max_drawdown_pct": lambda x: f"{x:.2f}%",
        "trades": lambda x: f"{x:.0f}",
    }
    
    for col, fmt_fn in format_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: fmt_fn(x) if pd.notna(x) else "N/A"
            )
    
    # Use tabulate if available, otherwise simple string representation
    if tabulate:
        table_str = tabulate(
            display_df,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".4f",
        )
    else:
        # Fallback: simple string representation
        table_str = display_df.to_string(index=False)
    
    return table_str


def export_leaderboard_json(df: pd.DataFrame, output_path: Path) -> None:
    """Export leaderboard to JSON.
    
    Args:
        df: Sorted DataFrame with runs
        output_path: Path to output JSON file
    """
    # Convert DataFrame to list of dicts (records format)
    records = df.to_dict(orient="records")
    
    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Leaderboard exported to {output_path}", file=sys.stderr)


def get_best_run_config(
    df: pd.DataFrame,
    sort_by: str = "sharpe",
    batch_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract best run configuration from summary DataFrame.
    
    Finds the best run (sorted by sort_by) with status=="success" and extracts
    the run configuration fields needed for a reproducible rerun.
    
    Args:
        df: Summary DataFrame
        sort_by: Column name to sort by (for finding "best" run)
        batch_output_dir: Optional batch output directory (to load full manifest if needed)
        
    Returns:
        Dictionary with run configuration fields:
        - id: Run ID
        - strategy: Strategy name
        - freq: Trading frequency ("1d" or "5min")
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - universe: Universe file path (optional)
        - start_capital: Starting capital (optional)
        - use_factor_store: Use factor store flag (optional)
        - factor_store_root: Factor store root (optional)
        - factor_group: Factor group name (optional)
        
    Raises:
        ValueError: If no successful runs found or required fields missing
    """
    # Filter to successful runs only
    if "status" in df.columns:
        successful_df = df[df["status"] == "success"].copy()
    else:
        successful_df = df.copy()
    
    if successful_df.empty:
        raise ValueError(
            "No successful runs found in summary. Cannot export best run config."
        )
    
    # Rank successful runs (need to ensure sort_by column exists)
    if sort_by not in successful_df.columns:
        raise ValueError(
            f"Sort column '{sort_by}' not found in summary. "
            f"Available columns: {', '.join(successful_df.columns)}"
        )
    
    ranked_df = rank_runs(successful_df, sort_by=sort_by, top_k=1)
    
    if ranked_df.empty:
        raise ValueError("Failed to rank successful runs")
    
    best_run = ranked_df.iloc[0]
    
    # Extract configuration fields (with defaults where appropriate)
    config: dict[str, Any] = {}
    
    # Required fields (check in best_run, which is a Series from ranked_df)
    required_fields = ["run_id", "strategy", "freq"]
    for field in required_fields:
        if field not in ranked_df.columns:
            raise ValueError(
                f"Required field '{field}' not found in summary.csv. "
                f"Available columns: {', '.join(df.columns)}"
            )
        value = best_run[field]
        if pd.isna(value):
            raise ValueError(f"Required field '{field}' is missing (NaN) for best run")
        # Convert pandas types to Python types
        config[field] = str(value)
    
    # Use "id" instead of "run_id" for consistency with RunConfig
    if "run_id" in config:
        config["id"] = config.pop("run_id")
    
    # Optional fields (with sensible defaults) - will be loaded from manifest if available
    optional_fields_with_defaults = {
        "universe": None,
        "start_capital": 100000.0,
        "use_factor_store": False,
        "factor_store_root": None,
        "factor_group": None,
    }
    
    # Initialize optional fields with defaults
    for field, default in optional_fields_with_defaults.items():
        config[field] = default
    
    # Try to load configuration from manifest (primary source for complete config)
    run_id = config.get("id")
    manifest_loaded = False
    if batch_output_dir and run_id:
        manifest_path = batch_output_dir / run_id / "run_manifest.json"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                # Extract params from manifest (primary source for config)
                params = manifest.get("params", {})
                if params:
                    manifest_loaded = True
                    # Load all config fields from manifest params
                    for field in ["start_date", "end_date", "universe", "start_capital", "use_factor_store", "factor_store_root", "factor_group"]:
                        if field in params:
                            value = params[field]
                            if value not in (None, ""):
                                config[field] = value
                    
                    # Convert types appropriately
                    if "start_capital" in config:
                        try:
                            config["start_capital"] = float(config["start_capital"])
                        except (ValueError, TypeError):
                            config["start_capital"] = 100000.0
                    
                    if "use_factor_store" in config:
                        try:
                            config["use_factor_store"] = bool(config["use_factor_store"])
                        except (ValueError, TypeError):
                            config["use_factor_store"] = False
                
                # Also extract seed from manifest if available
                if "seed" in manifest:
                    config["seed"] = int(manifest["seed"])
            except IOError:
                # If manifest file cannot be read, try to get dates from summary.csv if available
                # Log warning but don't fail - we can try CSV fallback
                pass
            except json.JSONDecodeError:
                # Invalid JSON in manifest - try CSV fallback
                pass
            except (KeyError, ValueError):
                # Missing required keys or invalid values - try CSV fallback
                pass
    
    # Fallback: Try to get dates from summary.csv if not loaded from manifest
    if not manifest_loaded or "start_date" not in config or config["start_date"] is None:
        date_fields = ["start_date", "end_date"]
        for field in date_fields:
            if field not in config or config[field] is None:
                if field in ranked_df.columns:
                    value = best_run[field]
                    if pd.notna(value):
                        config[field] = str(value)
    
    # Ensure required fields are present
    if "start_date" not in config or config["start_date"] is None:
        raise ValueError(
            "start_date is required but not found in summary or manifest. "
            "Ensure batch_output_dir points to a valid batch output directory with run manifests."
        )
    if "end_date" not in config or config["end_date"] is None:
        raise ValueError(
            "end_date is required but not found in summary or manifest. "
            "Ensure batch_output_dir points to a valid batch output directory with run manifests."
        )
    
    return config


def export_best_run_config_yaml(
    df: pd.DataFrame,
    sort_by: str,
    output_path: Path,
    batch_output_dir: Path | None = None,
) -> None:
    """Export best run configuration as YAML file.
    
    Extracts the best successful run (sorted by sort_by) and writes its
    configuration as a YAML file compatible with batch runner config format.
    
    Args:
        df: Summary DataFrame
        sort_by: Column name used for sorting (to find "best" run)
        output_path: Path to output YAML file
        batch_output_dir: Optional batch output directory (to load manifest for additional fields)
        
    Raises:
        ValueError: If no successful runs found or YAML library not available
        RuntimeError: If YAML export fails
    """
    if yaml is None:
        raise RuntimeError(
            "YAML export requires PyYAML. Install via 'pip install pyyaml'"
        )
    
    # Get best run config
    config = get_best_run_config(df, sort_by=sort_by, batch_output_dir=batch_output_dir)
    
    # Build YAML structure compatible with batch runner
    # The batch runner expects a list of runs, so we wrap it
    # Remove None values to keep YAML clean (optional fields)
    clean_config = {k: v for k, v in config.items() if v is not None}
    
    yaml_data = {
        "runs": [clean_config]
    }
    
    # Write YAML
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to create output directory for YAML export: {output_path.parent}. "
            f"Error: {exc}"
        ) from exc
    
    try:
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except IOError as exc:
        raise RuntimeError(
            f"Failed to write YAML file: {output_path}. "
            f"Error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error while writing YAML file: {output_path}. "
            f"Error: {exc}"
        ) from exc
    
    print(f"Best run config exported to {output_path}", file=sys.stderr)
    print(f"  Run ID: {config.get('id', 'N/A')}", file=sys.stderr)
    print(f"  Strategy: {config.get('strategy', 'N/A')}", file=sys.stderr)
    
    # Find the metric value for this run
    run_id = config.get("id")
    if run_id and "run_id" in df.columns:
        run_row = df[df["run_id"] == run_id]
        if not run_row.empty and sort_by in run_row.columns:
            metric_value = run_row[sort_by].iloc[0]
            print(f"  Sort metric ({sort_by}): {metric_value}", file=sys.stderr)


def main() -> int:
    """Main entry point for leaderboard tool."""
    parser = argparse.ArgumentParser(
        description="Rank and display best runs from batch backtest results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--batch-output",
        type=Path,
        required=True,
        help="Path to batch output directory (contains summary.csv)",
    )
    
    parser.add_argument(
        "--sort-by",
        type=str,
        default="sharpe",
        choices=["sharpe", "total_return", "final_pf", "max_drawdown_pct", "cagr", "trades"],
        help="Metric to sort by (default: sharpe)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top runs to display (default: 20)",
    )
    
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: Export leaderboard to JSON file",
    )
    
    parser.add_argument(
        "--export-best",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: Export best run configuration as YAML file (for reproducible reruns)",
    )
    
    args = parser.parse_args()
    
    # Validate batch output directory
    if not args.batch_output.exists():
        print(f"Error: Batch output directory does not exist: {args.batch_output}", file=sys.stderr)
        return 1
    
    if not args.batch_output.is_dir():
        print(f"Error: Batch output path is not a directory: {args.batch_output}", file=sys.stderr)
        return 1
    
    # Load summary
    try:
        df = load_batch_summary(args.batch_output)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    
    # Rank runs
    try:
        ranked_df = rank_runs(df, sort_by=args.sort_by, top_k=args.top_k)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    
    # Print table
    table_str = format_leaderboard_table(ranked_df, args.sort_by)
    print(f"\nTop {len(ranked_df)} runs (sorted by {args.sort_by}):\n")
    print(table_str)
    
    # Export JSON if requested
    if args.json:
        try:
            export_leaderboard_json(ranked_df, args.json)
        except Exception as exc:
            print(f"Warning: Failed to export JSON: {exc}", file=sys.stderr)
            return 1
    
    # Export best run config if requested
    if args.export_best:
        try:
            export_best_run_config_yaml(
                df,  # Use full df, not ranked_df, so we can filter to successful
                sort_by=args.sort_by,
                output_path=args.export_best,
                batch_output_dir=args.batch_output,
            )
        except ValueError as exc:
            print(f"Error: Failed to export best run config: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error: Failed to export best run config: {exc}", file=sys.stderr)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
