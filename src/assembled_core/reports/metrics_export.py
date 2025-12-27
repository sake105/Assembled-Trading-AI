"""Metrics export utilities for backtest results.

This module provides functions to export performance metrics as JSON files,
preferred over parsing Markdown reports for batch runs and automated analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.assembled_core.qa.metrics import PerformanceMetrics


def export_metrics_json(
    metrics: PerformanceMetrics,
    output_path: Path,
) -> Path:
    """Export performance metrics to JSON file.
    
    This function serializes PerformanceMetrics to a JSON file with:
    - Deterministic key ordering (sorted alphabetically)
    - Normalized float values (NaN/inf converted to null)
    - Type-safe conversion (timestamps to ISO strings)
    
    Args:
        metrics: PerformanceMetrics instance to export
        output_path: Path to output JSON file (will be created if parent dir doesn't exist)
        
    Returns:
        Path to written JSON file
        
    Example:
        >>> from src.assembled_core.qa.metrics import compute_all_metrics
        >>> from src.assembled_core.reports.metrics_export import export_metrics_json
        >>> 
        >>> metrics = compute_all_metrics(equity_df, trades_df, start_capital=10000.0, freq="1d")
        >>> export_metrics_json(metrics, Path("output/metrics.json"))
    """
    # Convert metrics to dict with normalized values
    metrics_dict = _metrics_to_dict(metrics)
    
    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except (IOError, OSError) as exc:
        raise RuntimeError(f"Failed to create output directory for metrics JSON: {output_path.parent}") from exc
    
    # Write JSON with deterministic formatting (sorted keys, indented)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
    except (IOError, OSError) as exc:
        raise RuntimeError(f"Failed to write metrics JSON to {output_path}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to serialize metrics to JSON: {output_path}") from exc
    
    return output_path


def _metrics_to_dict(metrics: PerformanceMetrics) -> dict[str, float | int | str | None]:
    """Convert PerformanceMetrics to dictionary with normalized values.
    
    This function:
    - Converts all fields to JSON-serializable types
    - Normalizes float values (NaN/inf -> null)
    - Converts timestamps to ISO 8601 strings
    - Sorts keys deterministically
    
    Args:
        metrics: PerformanceMetrics instance
        
    Returns:
        Dictionary with normalized metric values (ready for JSON serialization)
    """
    result: dict[str, float | int | str | None] = {}
    
    # Performance metrics
    result["final_pf"] = _normalize_float(metrics.final_pf)
    result["total_return"] = _normalize_float(metrics.total_return)
    result["cagr"] = _normalize_float(metrics.cagr)
    
    # Risk-adjusted returns
    result["sharpe_ratio"] = _normalize_float(metrics.sharpe_ratio)
    result["sortino_ratio"] = _normalize_float(metrics.sortino_ratio)
    result["calmar_ratio"] = _normalize_float(metrics.calmar_ratio)
    
    # Risk metrics
    result["max_drawdown"] = _normalize_float(metrics.max_drawdown)
    result["max_drawdown_pct"] = _normalize_float(metrics.max_drawdown_pct)
    result["current_drawdown"] = _normalize_float(metrics.current_drawdown)
    result["volatility"] = _normalize_float(metrics.volatility)
    result["var_95"] = _normalize_float(metrics.var_95)
    
    # Trade metrics
    result["hit_rate"] = _normalize_float(metrics.hit_rate)
    result["profit_factor"] = _normalize_float(metrics.profit_factor)
    result["avg_win"] = _normalize_float(metrics.avg_win)
    result["avg_loss"] = _normalize_float(metrics.avg_loss)
    result["turnover"] = _normalize_float(metrics.turnover)
    result["total_trades"] = metrics.total_trades if metrics.total_trades is not None else None
    
    # Metadata
    result["start_date"] = metrics.start_date.isoformat() if metrics.start_date is not None else None
    result["end_date"] = metrics.end_date.isoformat() if metrics.end_date is not None else None
    result["periods"] = metrics.periods
    result["start_capital"] = _normalize_float(metrics.start_capital)
    result["end_equity"] = _normalize_float(metrics.end_equity)
    
    # Aliases for compatibility with batch runner (uses different key names)
    # These match the keys used in collect_backtest_metrics
    if "sharpe_ratio" in result:
        result["sharpe"] = result["sharpe_ratio"]  # Alias for batch runner compatibility
    if "total_trades" in result:
        result["trades"] = result["total_trades"]  # Alias for batch runner compatibility
    
    return result


def _normalize_float(value: float | None) -> float | None:
    """Normalize float value for JSON serialization.
    
    Converts:
    - NaN -> None (null in JSON)
    - inf/-inf -> None (null in JSON)
    - None -> None
    - Valid float -> float (preserved)
    
    Args:
        value: Float value (or None)
        
    Returns:
        Normalized value (float or None)
    """
    if value is None:
        return None
    
    # Check for NaN
    if math.isnan(value):
        return None
    
    # Check for inf/-inf
    if math.isinf(value):
        return None
    
    return float(value)

