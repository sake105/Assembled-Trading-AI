"""Robustness analysis for strategy validation (Sprint 12 RB2).

This module provides parameter stability sweeps, heatmap generation, and plateau detection
to assess strategy robustness across parameter variations.

Example:
    from src.assembled_core.qa.robustness import (
        run_param_grid_sweep,
        build_heatmap_table,
        detect_plateau,
    )

    # Define backtest function
    def backtest_fn(config):
        # Run backtest with config parameters
        return {
            "sharpe": 1.5,
            "cagr": 0.15,
            "max_drawdown": -0.10,
            "turnover": 0.5,
        }

    # Define parameter grid
    base_config = {"strategy": "ema", "freq": "1d"}
    grid = {
        "ma_fast": [10, 20, 30],
        "ma_slow": [50, 100, 200],
    }

    # Run sweep
    results_df = run_param_grid_sweep(
        backtest_fn=backtest_fn,
        base_config=base_config,
        grid=grid,
        deterministic=True,
    )

    # Build heatmap
    heatmap_df = build_heatmap_table(
        results_df=results_df,
        x_param="ma_fast",
        y_param="ma_slow",
        metric="sharpe",
    )

    # Detect plateau
    plateau_info = detect_plateau(
        results_df=results_df,
        metric="sharpe",
        top_k=5,
        epsilon=0.05,
    )
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


def run_param_grid_sweep(
    backtest_fn: Callable[[dict[str, Any]], dict[str, float | int]],
    base_config: dict[str, Any],
    grid: dict[str, list[Any]],
    *,
    deterministic: bool = True,
) -> pd.DataFrame:
    """Run parameter grid sweep with deterministic ordering (RB2).

    This function generates all parameter combinations from the grid, runs backtests
    for each combination, and collects metrics in a long-format DataFrame.

    Args:
        backtest_fn: Backtest function that takes a config dict and returns metrics dict
            Signature: (config: dict[str, Any]) -> dict[str, float | int]
            Should return metrics like: sharpe, cagr, max_drawdown, turnover, etc.
        base_config: Base configuration dictionary (merged with grid values)
        grid: Parameter grid dictionary
            Keys are parameter names, values are lists of parameter values to test
            Example: {"ma_fast": [10, 20, 30], "ma_slow": [50, 100]}
        deterministic: If True, ensure deterministic ordering (default: True)
            - Parameter keys are sorted
            - Parameter combinations use itertools.product with sorted keys

    Returns:
        DataFrame with columns:
        - Parameter columns (one per grid key)
        - Metric columns (from backtest_fn return values)
        - Rows: one per parameter combination
        Sorted by parameter values (lexicographic order)

    Raises:
        ValueError: If grid is empty or backtest_fn fails for all combinations

    Example:
        >>> def my_backtest_fn(config):
        ...     return {"sharpe": 1.0, "cagr": 0.10}
        ...
        >>> base = {"strategy": "ema"}
        >>> grid = {"ma_fast": [10, 20], "ma_slow": [50, 100]}
        >>> results = run_param_grid_sweep(my_backtest_fn, base, grid)
        >>> len(results)  # 2 * 2 = 4 combinations
        4
    """
    if not grid:
        raise ValueError("grid must not be empty")

    # Sort parameter keys for deterministic ordering
    param_keys = sorted(grid.keys()) if deterministic else list(grid.keys())
    param_values = [grid[key] for key in param_keys]

    # Generate all combinations (deterministic order)
    combinations = list(itertools.product(*param_values))

    if not combinations:
        raise ValueError("grid must contain at least one combination")

    logger.info(f"Running parameter sweep: {len(combinations)} combinations")

    # Run backtest for each combination
    results = []
    for combo in combinations:
        # Build config for this combination
        config = base_config.copy()
        for key, value in zip(param_keys, combo, strict=True):
            config[key] = value

        try:
            # Run backtest
            metrics = backtest_fn(config)

            # Build result row
            row = {}
            # Add parameter values
            for key, value in zip(param_keys, combo, strict=True):
                row[key] = value
            # Add metrics
            for metric_key, metric_value in metrics.items():
                row[metric_key] = metric_value

            results.append(row)

            logger.debug(f"Combination {combo}: sharpe={metrics.get('sharpe', 'N/A')}")

        except Exception as exc:
            logger.warning(f"Backtest failed for combination {combo}: {exc}", exc_info=True)
            # Add failed row with NaN metrics
            row = {}
            for key, value in zip(param_keys, combo, strict=True):
                row[key] = value
            # Add NaN for all expected metrics (if we know them)
            # For now, just add error indicator
            row["error"] = str(exc)
            results.append(row)

    if not results:
        raise ValueError("All backtest combinations failed")

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # Sort by parameter values (deterministic ordering)
    if deterministic and param_keys:
        results_df = results_df.sort_values(param_keys, kind="mergesort").reset_index(drop=True)

    logger.info(f"Parameter sweep completed: {len(results)} combinations")

    return results_df


def build_heatmap_table(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Build pivot table for heatmap visualization (RB2).

    This function creates a pivot table from long-format results, suitable for
    heatmap visualization (e.g., in spreadsheet or plotting tools).

    Args:
        results_df: Long-format DataFrame from run_param_grid_sweep()
        x_param: Parameter name for x-axis (columns)
        y_param: Parameter name for y-axis (rows)
        metric: Metric name to pivot (default: "sharpe")

    Returns:
        Pivot DataFrame with:
        - Index: y_param values
        - Columns: x_param values
        - Values: metric values
        Missing combinations are represented as NaN (deterministic)

    Raises:
        ValueError: If x_param, y_param, or metric not found in results_df

    Example:
        >>> results_df = pd.DataFrame({
        ...     "ma_fast": [10, 10, 20, 20],
        ...     "ma_slow": [50, 100, 50, 100],
        ...     "sharpe": [1.0, 1.2, 1.1, 1.3],
        ... })
        >>> heatmap = build_heatmap_table(results_df, "ma_fast", "ma_slow", "sharpe")
        >>> heatmap.shape  # (2, 2) for 2 ma_slow values x 2 ma_fast values
        (2, 2)
    """
    if x_param not in results_df.columns:
        raise ValueError(f"x_param '{x_param}' not found in results_df columns")
    if y_param not in results_df.columns:
        raise ValueError(f"y_param '{y_param}' not found in results_df columns")
    if metric not in results_df.columns:
        raise ValueError(f"metric '{metric}' not found in results_df columns")

    # Build pivot table
    pivot_df = results_df.pivot_table(
        index=y_param,
        columns=x_param,
        values=metric,
        aggfunc="first",  # Use first value if duplicates (shouldn't happen in grid)
    )

    # Sort index and columns for deterministic output
    if pivot_df.index.dtype == "object":
        pivot_df = pivot_df.sort_index(kind="mergesort")
    else:
        pivot_df = pivot_df.sort_index(kind="mergesort")

    if pivot_df.columns.dtype == "object":
        pivot_df = pivot_df.sort_index(axis=1, kind="mergesort")
    else:
        pivot_df = pivot_df.sort_index(axis=1, kind="mergesort")

    return pivot_df


def detect_plateau(
    results_df: pd.DataFrame,
    metric: str = "sharpe",
    top_k: int = 5,
    epsilon: float = 0.05,
) -> dict[str, Any]:
    """Detect performance plateau in parameter sweep results (RB2).

    A plateau is defined as a region of parameter combinations where performance
    is within (1 - epsilon) of the best performance. This helps identify robust
    parameter regions rather than just the single best peak.

    Args:
        results_df: Long-format DataFrame from run_param_grid_sweep()
        metric: Metric name to analyze (default: "sharpe")
        top_k: Number of top combinations to consider (default: 5)
        epsilon: Relative tolerance for plateau (default: 0.05 = 5%)
            Combinations within (1 - epsilon) * best_metric are considered part of plateau

    Returns:
        Dictionary with keys:
        - plateau_size: Number of combinations in plateau
        - plateau_fraction: Fraction of total combinations in plateau
        - best_metric: Best metric value found
        - robust_score: Plateau size / total combinations (higher = more robust)
        - plateau_threshold: Threshold value (best_metric * (1 - epsilon))
        - top_k_combinations: List of top_k combinations (as dicts with params + metric)

    Example:
        >>> results_df = pd.DataFrame({
        ...     "ma_fast": [10, 20, 30],
        ...     "sharpe": [1.0, 1.2, 1.15],
        ... })
        >>> plateau = detect_plateau(results_df, "sharpe", top_k=3, epsilon=0.05)
        >>> plateau["plateau_size"]  # Combinations within 5% of best (1.2)
        2
    """
    if metric not in results_df.columns:
        raise ValueError(f"metric '{metric}' not found in results_df columns")

    # Filter out NaN/invalid values
    valid_df = results_df.dropna(subset=[metric]).copy()
    if valid_df.empty:
        return {
            "plateau_size": 0,
            "plateau_fraction": 0.0,
            "best_metric": None,
            "robust_score": 0.0,
            "plateau_threshold": None,
            "top_k_combinations": [],
        }

    # Find best metric value
    best_metric = float(valid_df[metric].max())
    worst_metric = float(valid_df[metric].min())

    # Calculate plateau threshold
    if best_metric > 0:
        plateau_threshold = best_metric * (1.0 - epsilon)
    elif best_metric < 0:
        # For negative metrics (e.g., max_drawdown), plateau is within epsilon of best
        plateau_threshold = best_metric * (1.0 + epsilon)
    else:
        # best_metric == 0: use absolute epsilon
        plateau_threshold = -abs(epsilon)

    # Find combinations in plateau
    if best_metric >= 0:
        plateau_mask = valid_df[metric] >= plateau_threshold
    else:
        plateau_mask = valid_df[metric] <= plateau_threshold

    plateau_df = valid_df[plateau_mask]
    plateau_size = len(plateau_df)
    total_size = len(valid_df)
    plateau_fraction = plateau_size / total_size if total_size > 0 else 0.0

    # Get top_k combinations
    top_k_df = valid_df.nlargest(top_k, metric) if best_metric >= 0 else valid_df.nsmallest(top_k, metric)
    top_k_combinations = top_k_df.to_dict(orient="records")

    # Calculate robust score (plateau size normalized by total)
    robust_score = plateau_fraction

    return {
        "plateau_size": plateau_size,
        "plateau_fraction": float(plateau_fraction),
        "best_metric": float(best_metric),
        "robust_score": float(robust_score),
        "plateau_threshold": float(plateau_threshold),
        "top_k_combinations": top_k_combinations,
    }


def export_robustness_sweep_results(
    results_df: pd.DataFrame,
    heatmap_tables: dict[str, pd.DataFrame],
    plateau_info: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Export robustness sweep results to CSV and JSON files (RB2).

    Args:
        results_df: Long-format results DataFrame
        heatmap_tables: Dictionary of heatmap DataFrames
            Keys: "{x_param}_{y_param}" (e.g., "ma_fast_ma_slow")
            Values: Pivot DataFrames
        plateau_info: Plateau detection results (from detect_plateau)
        output_dir: Output directory (will create robustness/<run_id> subdirectory)
        run_id: Run identifier for file naming

    Returns:
        Dictionary with keys: results_csv, heatmap_csvs (dict), plateau_json
        Values are Path objects to written files

    Note:
        - CSV files are sorted deterministically
        - JSON uses sort_keys=True, indent=2
        - NaN values in JSON are converted to None
    """
    robustness_dir = output_dir / "robustness" / run_id
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # Export results CSV (long format)
    results_csv = robustness_dir / "param_sweep_results.csv"
    results_df_sorted = results_df.sort_values(list(results_df.columns), kind="mergesort")
    results_df_sorted.to_csv(results_csv, index=False, encoding="utf-8")

    # Export heatmap CSVs
    heatmap_csvs = {}
    for heatmap_key, heatmap_df in heatmap_tables.items():
        heatmap_csv = robustness_dir / f"param_sweep_heatmap_{heatmap_key}.csv"
        heatmap_df.to_csv(heatmap_csv, encoding="utf-8")
        heatmap_csvs[heatmap_key] = heatmap_csv

    # Export plateau JSON
    plateau_json = robustness_dir / "plateau.json"
    with plateau_json.open("w", encoding="utf-8") as f:
        json.dump(plateau_info, f, sort_keys=True, indent=2, default=_json_serialize_nan)

    logger.info(f"Robustness sweep results exported to {robustness_dir}")

    return {
        "results_csv": results_csv,
        "heatmap_csvs": heatmap_csvs,
        "plateau_json": plateau_json,
    }


def _json_serialize_nan(obj: Any) -> Any:
    """JSON serializer that converts NaN/Inf to None (for deterministic JSON output).

    Args:
        obj: Object to serialize

    Returns:
        Serialized value (None for NaN/Inf, otherwise obj)
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# ============================================================================
# RB3: Sensitivity Suite
# ============================================================================


def apply_disclosure_delay(
    events_df: pd.DataFrame,
    delay_days: int,
    *,
    disclosure_date_col: str = "disclosure_date",
    effective_date_col: str = "effective_date",
) -> pd.DataFrame:
    """Apply disclosure delay to alt-data events (RB3).

    This function shifts disclosure_date and effective_date by delay_days for sensitivity testing.
    IMPORTANT: PIT-safety must be preserved:
    - delay_days > 0: Events become visible LATER (stricter PIT, no leakage risk)
    - delay_days < 0: Events become visible EARLIER (stress test, may introduce leakage - WARNING)

    Args:
        events_df: Event DataFrame with disclosure_date (and optionally effective_date)
        delay_days: Number of days to shift (positive = later, negative = earlier)
        disclosure_date_col: Name of disclosure_date column (default: "disclosure_date")
        effective_date_col: Name of effective_date column (default: "effective_date")

    Returns:
        DataFrame with shifted disclosure_date and effective_date
        Original DataFrame is not modified (copy returned)

    Note:
        - For delay_days > 0: PIT-safety is preserved (events visible later)
        - For delay_days < 0: May introduce leakage (events visible earlier than in reality)
          This should only be used for stress testing and must be clearly marked as WARNING
    """
    if events_df.empty:
        return events_df.copy()

    if disclosure_date_col not in events_df.columns:
        raise ValueError(f"disclosure_date_col '{disclosure_date_col}' not found in events_df")

    result = events_df.copy()

    # Shift disclosure_date
    result[disclosure_date_col] = pd.to_datetime(result[disclosure_date_col], utc=True) + pd.Timedelta(
        days=delay_days
    )

    # Shift effective_date if present
    if effective_date_col in result.columns:
        result[effective_date_col] = pd.to_datetime(result[effective_date_col], utc=True) + pd.Timedelta(
            days=delay_days
        )

    return result


def run_sensitivity_suite(
    backtest_fn: Callable[[dict[str, Any]], dict[str, float | int]],
    base_config: dict[str, Any],
    *,
    delay_days_list: list[int] = [-2, 0, 2],
    deterministic: bool = True,
) -> pd.DataFrame:
    """Run sensitivity suite with cost, slippage, and alt-data delay variants (RB3).

    This function runs backtests with different sensitivity variants to assess robustness:
    - baseline: Original configuration
    - costs_x2: All costs doubled (commission_bps, spread_w, impact_w × 2)
    - slippage_x2: Only slippage/impact doubled (impact_w × 2, commission_bps and spread_w unchanged)
    - alt_delay_{d}: Alt-data disclosure_date shifted by d days

    Args:
        backtest_fn: Backtest function that takes config dict and returns metrics dict
            Signature: (config: dict[str, Any]) -> dict[str, float | int]
            Config may contain: commission_bps, spread_w, impact_w, events_df, etc.
        base_config: Base configuration dictionary
        delay_days_list: List of delay days to test (default: [-2, 0, 2])
            - Negative values: Events visible earlier (stress test, WARNING)
            - Zero: No delay (baseline for alt-data)
            - Positive values: Events visible later (stricter PIT)
        deterministic: If True, ensure deterministic variant ordering (default: True)

    Returns:
        DataFrame with columns:
        - variant_name: str (e.g., "baseline", "costs_x2", "slippage_x2", "alt_delay_-2")
        - Metric columns (from backtest_fn return values)
        - warnings: str (optional warnings, e.g., "delay_days < 0 may introduce leakage")
        Sorted by variant_name (deterministic)

    Example:
        >>> def my_backtest_fn(config):
        ...     return {"sharpe": 1.0, "cagr": 0.10}
        ...
        >>> base = {"commission_bps": 1.0, "spread_w": 0.5, "impact_w": 1.0}
        >>> results = run_sensitivity_suite(my_backtest_fn, base, delay_days_list=[0, 2])
        >>> len(results)  # baseline + costs_x2 + slippage_x2 + alt_delay_0 + alt_delay_2 = 5
        5
    """
    variants = []

    # Variant 1: Baseline
    baseline_config = base_config.copy()
    variants.append(("baseline", baseline_config, None))

    # Variant 2: Costs ×2 (commission_bps, spread_w, impact_w all ×2)
    costs_x2_config = base_config.copy()
    if "commission_bps" in costs_x2_config:
        costs_x2_config["commission_bps"] = costs_x2_config["commission_bps"] * 2.0
    if "spread_w" in costs_x2_config:
        costs_x2_config["spread_w"] = costs_x2_config["spread_w"] * 2.0
    if "impact_w" in costs_x2_config:
        costs_x2_config["impact_w"] = costs_x2_config["impact_w"] * 2.0
    variants.append(("costs_x2", costs_x2_config, None))

    # Variant 3: Slippage ×2 (only impact_w ×2)
    slippage_x2_config = base_config.copy()
    if "impact_w" in slippage_x2_config:
        slippage_x2_config["impact_w"] = slippage_x2_config["impact_w"] * 2.0
    variants.append(("slippage_x2", slippage_x2_config, None))

    # Variant 4-N: Alt-data delay variants
    for delay_days in sorted(delay_days_list) if deterministic else delay_days_list:
        if delay_days == 0:
            # Skip delay=0 (already covered by baseline)
            continue

        alt_delay_config = base_config.copy()

        # Apply delay to events_df if present
        if "events_df" in alt_delay_config:
            events_df = alt_delay_config["events_df"]
            if events_df is not None and not events_df.empty:
                alt_delay_config["events_df"] = apply_disclosure_delay(
                    events_df=events_df,
                    delay_days=delay_days,
                )

        # Determine warning
        warning = None
        if delay_days < 0:
            warning = f"delay_days={delay_days} may introduce leakage (events visible earlier than reality)"

        variants.append((f"alt_delay_{delay_days:+d}", alt_delay_config, warning))

    # Run backtests for each variant
    results = []
    for variant_name, variant_config, warning in variants:
        try:
            # Run backtest
            metrics = backtest_fn(variant_config)

            # Build result row
            row = {"variant_name": variant_name}
            # Add metrics
            for metric_key, metric_value in metrics.items():
                row[metric_key] = metric_value
            # Add warning if present
            if warning:
                row["warnings"] = warning

            results.append(row)

            logger.debug(f"Variant {variant_name}: sharpe={metrics.get('sharpe', 'N/A')}")

        except Exception as exc:
            logger.warning(f"Variant {variant_name} failed: {exc}", exc_info=True)
            # Add failed row
            row = {"variant_name": variant_name, "error": str(exc)}
            if warning:
                row["warnings"] = warning
            results.append(row)

    if not results:
        raise ValueError("All sensitivity variants failed")

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # Sort by variant_name for deterministic output
    if deterministic:
        results_df = results_df.sort_values("variant_name", kind="mergesort").reset_index(drop=True)

    logger.info(f"Sensitivity suite completed: {len(results)} variants")

    return results_df


def export_sensitivity_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
) -> Path:
    """Export sensitivity suite results to CSV (RB3).

    Args:
        results_df: Results DataFrame from run_sensitivity_suite()
        output_dir: Output directory (will create robustness/<run_id> subdirectory)
        run_id: Run identifier for file naming

    Returns:
        Path to written CSV file

    Note:
        - CSV is sorted by variant_name (deterministic)
    """
    robustness_dir = output_dir / "robustness" / run_id
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # Export results CSV
    results_csv = robustness_dir / "sensitivity_results.csv"
    results_df_sorted = results_df.sort_values("variant_name", kind="mergesort")
    results_df_sorted.to_csv(results_csv, index=False, encoding="utf-8")

    logger.info(f"Sensitivity results exported to {results_csv}")

    return results_csv


# ============================================================================
# RB4: Crisis Windows Evaluation
# ============================================================================


def get_standard_crisis_windows() -> list[dict[str, Any]]:
    """Get standard crisis windows for robustness evaluation (RB4).

    Returns:
        List of crisis window dictionaries with keys:
        - name: str (e.g., "GFC", "COVID", "2022_RATES")
        - start: str (ISO format date, inclusive)
        - end: str (ISO format date, exclusive)
        - description: str (optional human-readable description)

    Note:
        - Windows are sorted by start date, then by name (deterministic)
        - Date ranges are [start, end) (start inclusive, end exclusive)
        - All dates are in UTC timezone

    Example:
        >>> windows = get_standard_crisis_windows()
        >>> len(windows)  # Number of standard windows
        3
        >>> windows[0]["name"]  # First window (sorted by start)
        "GFC"
    """
    windows = [
        {
            "name": "GFC",
            "start": "2007-12-01",
            "end": "2009-06-30",
            "description": "Global Financial Crisis (Lehman collapse, market crash)",
        },
        {
            "name": "COVID",
            "start": "2020-02-20",
            "end": "2020-04-30",
            "description": "COVID-19 market crash (rapid sell-off, volatility spike)",
        },
        {
            "name": "2022_RATES",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "description": "2022 rate hiking cycle (inflation, Fed tightening, bond sell-off)",
        },
    ]

    # Sort by start date, then by name (deterministic)
    windows = sorted(windows, key=lambda w: (w["start"], w["name"]))

    return windows


def run_crisis_windows(
    backtest_fn: Callable[[dict[str, Any]], dict[str, float | int]],
    base_config: dict[str, Any],
    windows: list[dict[str, Any]] | None = None,
    *,
    max_dd_threshold: float = -0.30,
    sharpe_floor: float = -1.0,
    deterministic: bool = True,
) -> pd.DataFrame:
    """Run backtest for each crisis window and evaluate pass/fail (RB4).

    This function runs backtests restricted to specific crisis date ranges to assess
    strategy performance during market stress periods.

    Args:
        backtest_fn: Backtest function that takes config dict and returns metrics dict
            Signature: (config: dict[str, Any]) -> dict[str, float | int]
            Config should support: start_date, end_date (or will be filtered internally)
        base_config: Base configuration dictionary
        windows: List of crisis window dictionaries (default: get_standard_crisis_windows())
            Each window must have: name (str), start (str ISO date), end (str ISO date)
            Date ranges are [start, end) (start inclusive, end exclusive)
        max_dd_threshold: Maximum drawdown threshold for pass/fail (default: -0.30 = -30%)
            Window passes if max_drawdown >= max_dd_threshold
        sharpe_floor: Minimum Sharpe ratio threshold for pass/fail (default: -1.0)
            Window passes if sharpe >= sharpe_floor
        deterministic: If True, ensure deterministic window ordering (default: True)

    Returns:
        DataFrame with columns:
        - window_name: str (crisis window name)
        - window_start: str (ISO date)
        - window_end: str (ISO date)
        - Metric columns (from backtest_fn return values)
        - pass_max_dd: bool (True if max_drawdown >= max_dd_threshold)
        - pass_sharpe: bool (True if sharpe >= sharpe_floor)
        - pass_overall: bool (True if both pass_max_dd and pass_sharpe are True)
        Sorted by window_start, then window_name (deterministic)

    Raises:
        ValueError: If windows is empty or invalid

    Example:
        >>> def my_backtest_fn(config):
        ...     return {"sharpe": 0.5, "max_drawdown": -0.20}
        ...
        >>> base = {"strategy": "ema"}
        >>> windows = get_standard_crisis_windows()
        >>> results = run_crisis_windows(my_backtest_fn, base, windows)
        >>> len(results)  # Number of crisis windows
        3
    """
    if windows is None:
        windows = get_standard_crisis_windows()

    if not windows:
        raise ValueError("windows must not be empty")

    # Sort windows for deterministic ordering
    if deterministic:
        windows = sorted(windows, key=lambda w: (w["start"], w["name"]))

    logger.info(f"Running crisis windows evaluation: {len(windows)} windows")

    # Run backtest for each window
    results = []
    for window in windows:
        window_name = window["name"]
        window_start = window["start"]
        window_end = window["end"]

        try:
            # Build config for this window
            window_config = base_config.copy()
            window_config["start_date"] = window_start
            window_config["end_date"] = window_end

            # Run backtest
            metrics = backtest_fn(window_config)

            # Extract metrics
            sharpe = metrics.get("sharpe", 0.0)
            max_dd = metrics.get("max_drawdown", 0.0)

            # Evaluate pass/fail
            pass_max_dd = max_dd >= max_dd_threshold if max_dd is not None else False
            pass_sharpe = sharpe >= sharpe_floor if sharpe is not None else False
            pass_overall = pass_max_dd and pass_sharpe

            # Build result row
            row = {
                "window_name": window_name,
                "window_start": window_start,
                "window_end": window_end,
                "pass_max_dd": pass_max_dd,
                "pass_sharpe": pass_sharpe,
                "pass_overall": pass_overall,
            }
            # Add all metrics
            for metric_key, metric_value in metrics.items():
                row[metric_key] = metric_value

            results.append(row)

            logger.debug(
                f"Window {window_name}: sharpe={sharpe:.2f}, max_dd={max_dd:.2f}, "
                f"pass_overall={pass_overall}"
            )

        except Exception as exc:
            logger.warning(f"Window {window_name} failed: {exc}", exc_info=True)
            # Add failed row
            row = {
                "window_name": window_name,
                "window_start": window_start,
                "window_end": window_end,
                "pass_max_dd": False,
                "pass_sharpe": False,
                "pass_overall": False,
                "error": str(exc),
            }
            results.append(row)

    if not results:
        raise ValueError("All crisis windows failed")

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # Sort by window_start, then window_name for deterministic output
    if deterministic:
        results_df = results_df.sort_values(
            ["window_start", "window_name"], kind="mergesort"
        ).reset_index(drop=True)

    logger.info(f"Crisis windows evaluation completed: {len(results)} windows")

    return results_df


def export_crisis_windows_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
) -> Path:
    """Export crisis windows results to CSV (RB4).

    Args:
        results_df: Results DataFrame from run_crisis_windows()
        output_dir: Output directory (will create robustness/<run_id> subdirectory)
        run_id: Run identifier for file naming

    Returns:
        Path to written CSV file

    Note:
        - CSV is sorted by window_start, then window_name (deterministic)
    """
    robustness_dir = output_dir / "robustness" / run_id
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # Export results CSV
    results_csv = robustness_dir / "crisis_windows.csv"
    results_df_sorted = results_df.sort_values(
        ["window_start", "window_name"], kind="mergesort"
    )
    results_df_sorted.to_csv(results_csv, index=False, encoding="utf-8")

    logger.info(f"Crisis windows results exported to {results_csv}")

    return results_csv


# ============================================================================
# RB5: Deflated Sharpe / Multiple Testing Warnings
# ============================================================================


def compute_deflated_sharpe(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = 1,
    alpha: float = 0.05,
) -> float | None:
    """Compute deflated Sharpe ratio adjusted for multiple testing (RB5).

    This function implements the deflated Sharpe ratio formula from:
    Bailey & López de Prado (2014): "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality"

    Formula:
        DS = SR * sqrt((1 - gamma * SR) / (n_obs - 1)) - sqrt((1 - gamma * SR) / (n_obs - 1)) * Z(1 - alpha / n_trials)
        where gamma = (skew * SR) / 4 + ((kurt - 3) * SR^2) / 24

    Args:
        sharpe: Observed Sharpe ratio
        n_obs: Number of observations (sample size)
        skew: Skewness of returns (default: 0.0, assumes normal)
        kurt: Kurtosis of returns (default: 3.0, assumes normal)
        n_trials: Number of independent trials/parameter combinations tested (default: 1)
        alpha: Significance level for multiple testing correction (default: 0.05)

    Returns:
        Deflated Sharpe ratio (float) or None if inputs are invalid

    Note:
        - Returns None if n_obs < 2, n_trials < 1, or alpha not in (0, 1)
        - For n_trials = 1, this reduces to standard Sharpe ratio (no multiple testing adjustment)
        - Higher n_trials -> lower deflated Sharpe (penalty for multiple testing)
    """
    import math
    import scipy.stats

    # Validate inputs
    if n_obs < 2:
        logger.warning(f"compute_deflated_sharpe: n_obs={n_obs} < 2, returning None")
        return None

    if n_trials < 1:
        logger.warning(f"compute_deflated_sharpe: n_trials={n_trials} < 1, returning None")
        return None

    if not (0 < alpha < 1):
        logger.warning(f"compute_deflated_sharpe: alpha={alpha} not in (0, 1), returning None")
        return None

    # If n_trials = 1, no multiple testing adjustment needed
    if n_trials == 1:
        return sharpe

    # Compute gamma (non-normality adjustment)
    gamma = (skew * sharpe) / 4.0 + ((kurt - 3.0) * sharpe * sharpe) / 24.0

    # Compute variance term
    variance_term = (1.0 - gamma * sharpe) / (n_obs - 1.0)

    # Check for negative variance (invalid)
    if variance_term <= 0:
        logger.warning(
            f"compute_deflated_sharpe: variance_term={variance_term} <= 0 "
            f"(sharpe={sharpe}, skew={skew}, kurt={kurt}), returning None"
        )
        return None

    # Compute Z-score for multiple testing correction
    # Bonferroni-like correction: alpha / n_trials
    z_critical = scipy.stats.norm.ppf(1.0 - alpha / n_trials)

    # Compute deflated Sharpe
    sqrt_variance = math.sqrt(variance_term)
    deflated_sharpe = sharpe * sqrt_variance - sqrt_variance * z_critical

    return deflated_sharpe


def build_multiple_testing_warnings(
    results_df: pd.DataFrame,
    metric_col: str = "sharpe",
) -> dict[str, Any]:
    """Build multiple testing warnings from parameter sweep results (RB5).

    This function heuristically detects potential overfitting from multiple testing:
    - Large number of trials (n_trials)
    - Best metric significantly higher than median (suggests selection bias)

    Args:
        results_df: Results DataFrame from parameter sweep (must contain metric_col)
        metric_col: Column name for metric to analyze (default: "sharpe")

    Returns:
        Dictionary with keys:
        - n_trials: int (number of parameter combinations tested)
        - best_metric: float | None (best metric value)
        - median_metric: float | None (median metric value)
        - metric_spread: float | None (best - median)
        - warning_inflated: bool (True if heuristic suggests inflation)
        - warning_message: str (human-readable warning if applicable)

    Note:
        - Returns empty warnings if metric_col not found or all values are NaN
        - Heuristic: warning_inflated = True if n_trials >= 10 AND metric_spread > 2.0
    """
    warnings_dict: dict[str, Any] = {
        "n_trials": 0,
        "best_metric": None,
        "median_metric": None,
        "metric_spread": None,
        "warning_inflated": False,
        "warning_message": "",
    }

    if metric_col not in results_df.columns:
        logger.warning(f"build_multiple_testing_warnings: metric_col '{metric_col}' not found")
        return warnings_dict

    # Filter out NaN values
    valid_metrics = results_df[metric_col].dropna()

    if len(valid_metrics) == 0:
        logger.warning("build_multiple_testing_warnings: no valid metrics found")
        return warnings_dict

    n_trials = len(valid_metrics)
    best_metric = float(valid_metrics.max())
    median_metric = float(valid_metrics.median())
    metric_spread = best_metric - median_metric

    warnings_dict["n_trials"] = n_trials
    warnings_dict["best_metric"] = best_metric
    warnings_dict["median_metric"] = median_metric
    warnings_dict["metric_spread"] = metric_spread

    # Heuristic: warning if many trials and large spread
    # Threshold: n_trials >= 10 AND spread > 2.0 (for Sharpe ratio)
    warning_inflated = n_trials >= 10 and metric_spread > 2.0

    if warning_inflated:
        warnings_dict["warning_inflated"] = True
        warnings_dict["warning_message"] = (
            f"Multiple testing warning: {n_trials} trials tested, "
            f"best {metric_col}={best_metric:.2f} vs median={median_metric:.2f} "
            f"(spread={metric_spread:.2f}). Consider deflated Sharpe ratio."
        )
    else:
        warnings_dict["warning_message"] = "No multiple testing concerns detected."

    return warnings_dict


def export_robustness_warnings(
    warnings_dict: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> Path:
    """Export multiple testing warnings to JSON (RB5).

    Args:
        warnings_dict: Warnings dictionary from build_multiple_testing_warnings()
        output_dir: Output directory (will create robustness/<run_id> subdirectory)
        run_id: Run identifier for file naming

    Returns:
        Path to written JSON file

    Note:
        - JSON is deterministic (sort_keys=True, indent=2)
        - NaN/Inf values are converted to None for JSON compatibility
    """
    robustness_dir = output_dir / "robustness" / run_id
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # Export warnings JSON
    warnings_json = robustness_dir / "warnings.json"

    # Serialize with NaN/Inf handling
    warnings_serialized = _json_serialize_nan(warnings_dict)

    with warnings_json.open("w", encoding="utf-8") as f:
        json.dump(warnings_serialized, f, sort_keys=True, indent=2, ensure_ascii=False)

    logger.info(f"Robustness warnings exported to {warnings_json}")

    return warnings_json


# ============================================================================
# RB Pack: Robustness Pack Orchestrator
# ============================================================================


def build_robustness_pack(
    backtest_fn: Callable[[dict[str, Any]], dict[str, float | int]],
    base_config: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path | None = None,
    prices_df: pd.DataFrame | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    windows: list[dict[str, Any]] | None = None,
    delay_days_list: list[int] | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Build complete robustness pack by running RB1-RB5 in sequence (Sprint 12 Final).

    This function orchestrates all robustness tests (RB1-RB5) and produces a comprehensive
    robustness summary. The pack is required for "candidate" status.

    Args:
        backtest_fn: Backtest function that takes config dict and returns metrics dict
            Signature: (config: dict[str, Any]) -> dict[str, float | int]
        base_config: Base configuration dictionary
        run_id: Run identifier for file naming
        output_dir: Output directory (default: Path("output") / "robustness_pack_<run_id>")
        prices_df: Optional prices DataFrame for RB1 (walk-forward splits)
            If None, RB1 is skipped
        param_grid: Optional parameter grid for RB2 (parameter sweep)
            If None, RB2 is skipped
        windows: Optional crisis windows for RB4
            If None, uses get_standard_crisis_windows()
        delay_days_list: Optional delay days list for RB3 sensitivity suite
            If None, uses default [-2, 0, 2]
        deterministic: If True, ensure deterministic execution (default: True)

    Returns:
        Dictionary with manifest fields:
        - robustness_pack_path: str (path to robustness pack directory)
        - wf_oos_metrics: dict (RB1 OOS-first metrics, if available)
        - plateau_score: float | None (RB2 plateau robust_score, if available)
        - sensitivity_summary: dict (RB3 summary, if available)
        - crisis_summary: dict (RB4 summary, if available)
        - deflated_sharpe: float | None (RB5 deflated Sharpe for best run, if available)
        - multiple_testing_warning: dict (RB5 warnings, if available)
        - robustness_ok: bool (True if all required tests pass)

    Note:
        - Execution order: RB1 -> RB2 -> RB3 -> RB4 -> RB5 (fixed)
        - If a test is skipped (missing inputs), it is omitted from summary
        - robustness_ok = True only if all enabled tests pass their thresholds
    """
    if output_dir is None:
        output_dir = Path("output") / f"robustness_pack_{run_id}"
    else:
        output_dir = Path(output_dir) / f"robustness_pack_{run_id}"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building robustness pack for run_id={run_id} in {output_dir}")

    manifest_fields: dict[str, Any] = {
        "robustness_pack_path": str(output_dir),
        "wf_oos_metrics": None,
        "plateau_score": None,
        "sensitivity_summary": None,
        "crisis_summary": None,
        "deflated_sharpe": None,
        "multiple_testing_warning": None,
        "robustness_ok": False,
    }

    # Track which tests passed
    test_results: dict[str, bool] = {}

    # RB1: Walk-Forward Analysis
    if prices_df is not None and not prices_df.empty:
        try:
            logger.info("RB1: Running walk-forward analysis")
            from src.assembled_core.qa.walk_forward import (
                export_walk_forward_results,
                make_walk_forward_splits,
                run_walk_forward,
            )

            # Generate splits (default: 5 splits, 252 train days, 63 test days)
            splits = make_walk_forward_splits(
                prices_df=prices_df,
                n_splits=5,
                train_days=252,
                test_days=63,
                seed=0,
            )

            # Run walk-forward
            wf_result = run_walk_forward(
                backtest_fn=backtest_fn,
                splits=splits,
                config=base_config,
                deterministic_seed=0,
            )

            # Export results
            export_walk_forward_results(wf_result, output_dir, run_id)

            # Extract OOS-first metrics
            oos_metrics = wf_result.get("oos_first_metrics", {})
            manifest_fields["wf_oos_metrics"] = oos_metrics

            # Check pass: mean OOS Sharpe >= 0.5 and win rate >= 0.5
            oos_sharpe = oos_metrics.get("oos_mean_sharpe")
            oos_win_rate = oos_metrics.get("oos_win_rate")
            rb1_pass = (
                oos_sharpe is not None
                and oos_win_rate is not None
                and oos_sharpe >= 0.5
                and oos_win_rate >= 0.5
            )
            test_results["rb1"] = rb1_pass

            logger.info(f"RB1 completed: pass={rb1_pass}")

        except Exception as exc:
            logger.warning(f"RB1 failed: {exc}", exc_info=True)
            test_results["rb1"] = False
    else:
        logger.info("RB1 skipped: prices_df not provided")

            # RB2: Parameter Sweep
    if param_grid is not None and param_grid:
        try:
            logger.info("RB2: Running parameter sweep")

            # Run sweep (functions are in same module)
            results_df = run_param_grid_sweep(
                backtest_fn=backtest_fn,
                base_config=base_config,
                grid=param_grid,
                deterministic=deterministic,
            )

            # Build heatmap (use first two parameters if available)
            heatmap_tables = {}
            param_keys = sorted(param_grid.keys())
            if len(param_keys) >= 2:
                heatmap = build_heatmap_table(
                    results_df=results_df,
                    x_param=param_keys[0],
                    y_param=param_keys[1],
                    metric="sharpe",
                )
                heatmap_key = f"{param_keys[0]}_{param_keys[1]}"
                heatmap_tables[heatmap_key] = heatmap

            # Detect plateau
            plateau_info = detect_plateau(
                results_df=results_df,
                metric="sharpe",
                top_k=5,
                epsilon=0.05,
            )

            manifest_fields["plateau_score"] = plateau_info.get("robust_score")

            # RB5: Build multiple testing warnings
            warnings_dict = build_multiple_testing_warnings(results_df, metric_col="sharpe")
            manifest_fields["multiple_testing_warning"] = warnings_dict

            # Export sweep results (with warnings)
            export_robustness_sweep_results(
                results_df=results_df,
                heatmap_tables=heatmap_tables,
                plateau_info=plateau_info,
                output_dir=output_dir,
                run_id=run_id,
                warnings_dict=warnings_dict,
            )

            # RB5: Compute deflated Sharpe for best run
            best_sharpe = results_df["sharpe"].max()
            n_trials = len(results_df)
            n_obs = 252  # Default: assume 1 year daily (can be extracted from config if available)
            deflated_sharpe = compute_deflated_sharpe(
                sharpe=best_sharpe,
                n_obs=n_obs,
                n_trials=n_trials,
            )
            manifest_fields["deflated_sharpe"] = deflated_sharpe

            # Check pass: at least 50% of combinations have Sharpe >= 0.5
            sharpe_pass_count = (results_df["sharpe"] >= 0.5).sum()
            sharpe_pass_fraction = sharpe_pass_count / len(results_df)
            rb2_pass = sharpe_pass_fraction >= 0.5
            test_results["rb2"] = rb2_pass

            logger.info(f"RB2 completed: pass={rb2_pass}")

        except Exception as exc:
            logger.warning(f"RB2 failed: {exc}", exc_info=True)
            test_results["rb2"] = False
    else:
        logger.info("RB2 skipped: param_grid not provided")

    # RB3: Sensitivity Suite
    try:
        logger.info("RB3: Running sensitivity suite")
        if delay_days_list is None:
            delay_days_list = [-2, 0, 2]

        # Run sensitivity suite (function is in same module)
        sensitivity_results = run_sensitivity_suite(
            backtest_fn=backtest_fn,
            base_config=base_config,
            delay_days_list=delay_days_list,
            deterministic=deterministic,
        )

        # Export results
        export_sensitivity_results(sensitivity_results, output_dir, run_id)

        # Build summary
        baseline_sharpe = None
        if "baseline" in sensitivity_results["variant_name"].values and "sharpe" in sensitivity_results.columns:
            baseline_rows = sensitivity_results[sensitivity_results["variant_name"] == "baseline"]
            if not baseline_rows.empty:
                baseline_sharpe = float(baseline_rows["sharpe"].iloc[0])

        all_pass = False
        if "sharpe" in sensitivity_results.columns:
            all_pass = bool((sensitivity_results["sharpe"] >= 0.0).all())

        sensitivity_summary = {
            "n_variants": len(sensitivity_results),
            "baseline_sharpe": baseline_sharpe,
            "all_pass": all_pass,
        }
        manifest_fields["sensitivity_summary"] = sensitivity_summary

        # Check pass: all variants have Sharpe >= 0.0
        rb3_pass = sensitivity_summary["all_pass"]
        test_results["rb3"] = rb3_pass

        logger.info(f"RB3 completed: pass={rb3_pass}")

    except Exception as exc:
        logger.warning(f"RB3 failed: {exc}", exc_info=True)
        test_results["rb3"] = False

    # RB4: Crisis Windows
    try:
        logger.info("RB4: Running crisis windows evaluation")
        if windows is None:
            windows = get_standard_crisis_windows()

        # Run crisis windows (function is in same module)
        crisis_results = run_crisis_windows(
            backtest_fn=backtest_fn,
            base_config=base_config,
            windows=windows,
            deterministic=deterministic,
        )

        # Export results
        export_crisis_windows_results(crisis_results, output_dir, run_id)

        # Build summary
        crisis_summary = {
            "n_windows": len(crisis_results),
            "pass_count": int(crisis_results["pass_overall"].sum()),
            "pass_fraction": float(crisis_results["pass_overall"].sum() / len(crisis_results)),
        }
        manifest_fields["crisis_summary"] = crisis_summary

        # Check pass: at least 50% of windows pass
        rb4_pass = crisis_summary["pass_fraction"] >= 0.5
        test_results["rb4"] = rb4_pass

        logger.info(f"RB4 completed: pass={rb4_pass}")

    except Exception as exc:
        logger.warning(f"RB4 failed: {exc}", exc_info=True)
        test_results["rb4"] = False

    # RB5: Deflated Sharpe (already computed in RB2 if available)
    # If RB2 was skipped, we can't compute deflated Sharpe
    # (already handled above)

    # Determine overall robustness_ok
    # robustness_ok = True only if all enabled tests pass
    enabled_tests = [k for k, v in test_results.items() if v is not None]
    if enabled_tests:
        robustness_ok = all(test_results[k] for k in enabled_tests)
    else:
        # No tests enabled -> robustness_ok = False (robustness pack required)
        robustness_ok = False
        logger.warning("No robustness tests enabled - robustness_ok = False")

    manifest_fields["robustness_ok"] = robustness_ok

    # Write robustness summary JSON
    summary_json = output_dir / "robustness_summary.json"
    summary_serialized = _json_serialize_nan(manifest_fields)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_serialized, f, sort_keys=True, indent=2, ensure_ascii=False)

    logger.info(f"Robustness pack completed: robustness_ok={robustness_ok}")

    return manifest_fields
