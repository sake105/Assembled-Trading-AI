"""Export ML Alpha Factors from Factor Panels.

This script generates ML alpha factors by training ML models on factor panels
and merging predictions back into the panel as ml_alpha_{model_type}_{horizon}d columns.

Usage:
    python research/ml/export_ml_alpha_factor.py \
      --factor-panel-file output/factor_panels/factor_panel_ai_tech_core_20d_1d.parquet \
      --label-col fwd_return_20d \
      --model-type ridge \
      --model-param alpha=0.1 \
      --n-splits 5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    run_time_series_cv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_model_params(param_strings: list[str]) -> dict[str, Any]:
    """
    Parse model parameter strings (key=value) into dictionary.
    
    Args:
        param_strings: List of strings like "alpha=0.1", "max_depth=5"
        
    Returns:
        Dictionary with parsed parameters (values converted to appropriate types)
    """
    params: dict[str, Any] = {}
    
    for param_str in param_strings:
        if "=" not in param_str:
            logger.warning(f"Invalid parameter format: {param_str}. Expected key=value. Skipping.")
            continue
        
        key, value_str = param_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Try to convert to appropriate type
        try:
            # Try int
            value = int(value_str)
        except ValueError:
            try:
                # Try float
                value = float(value_str)
            except ValueError:
                # Check for boolean
                if value_str.lower() in ("true", "false"):
                    value = value_str.lower() == "true"
                else:
                    # Keep as string
                    value = value_str
        
        params[key] = value
    
    return params


def export_ml_alpha_factor(
    factor_panel_file: Path,
    label_col: str,
    model_type: str,
    model_params: dict[str, Any] | None = None,
    n_splits: int = 5,
    test_start: str | None = None,
    test_end: str | None = None,
    output_dir: Path | None = None,
    column_name: str | None = None,
) -> Path:
    """
    Export ML alpha factor by training model and merging predictions back into panel.
    
    Args:
        factor_panel_file: Path to factor panel file (Parquet or CSV)
        label_col: Name of label column (e.g., "fwd_return_20d")
        model_type: Model type ("linear", "ridge", "lasso", "random_forest")
        model_params: Model hyperparameters (dict)
        n_splits: Number of CV splits (default: 5)
        test_start: Optional test start date (YYYY-MM-DD)
        test_end: Optional test end date (YYYY-MM-DD)
        output_dir: Output directory (default: output/ml_alpha_factors)
        column_name: Optional custom column name (default: auto-generated)
        
    Returns:
        Path to saved ML alpha factor panel file
    """
    # Resolve paths
    if not factor_panel_file.is_absolute():
        factor_panel_file = ROOT / factor_panel_file
    
    if not factor_panel_file.exists():
        raise FileNotFoundError(f"Factor panel file not found: {factor_panel_file}")
    
    if output_dir is None:
        output_dir = ROOT / "output" / "ml_alpha_factors"
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("ML Alpha Factor Export")
    logger.info(f"Factor Panel: {factor_panel_file}")
    logger.info(f"Label Column: {label_col}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Load factor panel
    logger.info("Loading factor panel...")
    try:
        if factor_panel_file.suffix == ".parquet":
            factor_panel_df = pd.read_parquet(factor_panel_file)
        elif factor_panel_file.suffix == ".csv":
            factor_panel_df = pd.read_csv(factor_panel_file)
            # Try to parse timestamp column if present
            if "timestamp" in factor_panel_df.columns:
                factor_panel_df["timestamp"] = pd.to_datetime(
                    factor_panel_df["timestamp"], utc=True, errors="coerce"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {factor_panel_file.suffix}. "
                f"Supported formats: .parquet, .csv"
            )
        
        logger.info(f"Loaded factor panel: {len(factor_panel_df)} rows, {len(factor_panel_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load factor panel: {e}", exc_info=True)
        raise
    
    # Validate label column
    if label_col not in factor_panel_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in factor panel. "
            f"Available columns: {list(factor_panel_df.columns)}"
        )
    
    # Check for sklearn availability
    try:
        from sklearn.linear_model import LinearRegression  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )
    
    # Parse test dates
    test_start_ts = None
    test_end_ts = None
    
    if test_start:
        try:
            test_start_ts = pd.to_datetime(test_start, utc=True)
        except Exception as e:
            raise ValueError(f"Invalid test_start date format: {test_start}. Error: {e}")
    
    if test_end:
        try:
            test_end_ts = pd.to_datetime(test_end, utc=True)
        except Exception as e:
            raise ValueError(f"Invalid test_end date format: {test_end}. Error: {e}")
    
    # Build experiment config
    experiment_config = MLExperimentConfig(
        label_col=label_col,
        feature_cols=None,  # Auto-detect
        test_start=test_start_ts,
        test_end=test_end_ts,
        n_splits=n_splits,
        train_size=None,  # Expanding window
        standardize=True,
        min_train_samples=100,
    )
    
    # Build model config
    model_params = model_params or {}
    model_name = f"{model_type}_{label_col}"
    
    model_cfg = MLModelConfig(
        name=model_name,
        model_type=model_type,  # type: ignore
        params=model_params,
    )
    
    # Run time-series CV
    logger.info("Running time-series cross-validation...")
    try:
        cv_result = run_time_series_cv(
            factor_panel_df=factor_panel_df,
            experiment=experiment_config,
            model_cfg=model_cfg,
        )
    except Exception as e:
        logger.error(f"CV failed: {e}", exc_info=True)
        raise
    
    predictions_df = cv_result["predictions_df"]
    global_metrics = cv_result["global_metrics"]
    
    logger.info(
        f"CV completed: {len(predictions_df)} predictions, "
        f"global R² = {global_metrics.get('r2', 0):.4f}"
    )
    
    # Validate predictions_df structure
    if predictions_df.empty:
        raise ValueError("No predictions generated from CV. Check data quality and configuration.")
    
    # Ensure timestamp and symbol are present in predictions_df
    if "timestamp" not in predictions_df.columns or "symbol" not in predictions_df.columns:
        logger.warning(
            "predictions_df missing timestamp/symbol columns. "
            "Attempting to reconstruct from index..."
        )
        # If index contains MultiIndex with timestamp/symbol, extract
        if isinstance(predictions_df.index, pd.MultiIndex):
            if len(predictions_df.index.names) >= 2:
                timestamp_name = predictions_df.index.names[0] if "timestamp" in str(predictions_df.index.names[0]) else None
                symbol_name = predictions_df.index.names[1] if "symbol" in str(predictions_df.index.names[1]) else None
                if timestamp_name and symbol_name:
                    predictions_df = predictions_df.reset_index()
        else:
            # Try to match with original panel index
            if len(predictions_df) == len(factor_panel_df):
                # Assume same order and index alignment
                predictions_df["timestamp"] = factor_panel_df.loc[predictions_df.index, "timestamp"].values
                predictions_df["symbol"] = factor_panel_df.loc[predictions_df.index, "symbol"].values
            else:
                raise ValueError(
                    "Cannot reconstruct timestamp/symbol for predictions_df. "
                    "predictions_df should contain timestamp and symbol columns."
                )
    
    # Generate column name
    if column_name is None:
        # Extract horizon from label_col (e.g., "fwd_return_20d" -> "20d")
        horizon_match = re.search(r"(\d+)d", label_col)
        if horizon_match:
            horizon_str = horizon_match.group(1) + "d"
        else:
            horizon_str = label_col.replace("fwd_return_", "")
        
        column_name = f"ml_alpha_{model_type}_{horizon_str}"
    
    logger.info(f"ML alpha column name: {column_name}")
    
    # Prepare predictions for merge
    # Only keep timestamp, symbol, and y_pred
    predictions_for_merge = predictions_df[["timestamp", "symbol", "y_pred"]].copy()
    predictions_for_merge = predictions_for_merge.rename(columns={"y_pred": column_name})
    
    # Merge predictions back into original panel
    logger.info("Merging predictions back into factor panel...")
    
    # Ensure timestamp columns are timezone-aware and compatible
    if "timestamp" in factor_panel_df.columns:
        if factor_panel_df["timestamp"].dtype == "object":
            factor_panel_df["timestamp"] = pd.to_datetime(factor_panel_df["timestamp"], utc=True, errors="coerce")
        elif not hasattr(factor_panel_df["timestamp"].dtype, "tz") or factor_panel_df["timestamp"].dtype.tz is None:
            # Make timezone-aware if not already
            factor_panel_df["timestamp"] = pd.to_datetime(factor_panel_df["timestamp"], utc=True)
    
    if "timestamp" in predictions_for_merge.columns:
        if predictions_for_merge["timestamp"].dtype == "object":
            predictions_for_merge["timestamp"] = pd.to_datetime(predictions_for_merge["timestamp"], utc=True, errors="coerce")
        elif not hasattr(predictions_for_merge["timestamp"].dtype, "tz") or predictions_for_merge["timestamp"].dtype.tz is None:
            predictions_for_merge["timestamp"] = pd.to_datetime(predictions_for_merge["timestamp"], utc=True)
    
    # Merge on timestamp and symbol
    ml_alpha_panel_df = factor_panel_df.merge(
        predictions_for_merge,
        on=["timestamp", "symbol"],
        how="left",  # Left join preserves all rows from original panel
    )
    
    # Log statistics
    n_total = len(ml_alpha_panel_df)
    n_with_predictions = ml_alpha_panel_df[column_name].notna().sum()
    pct_with_predictions = (n_with_predictions / n_total * 100) if n_total > 0 else 0
    
    logger.info(f"Merge completed:")
    logger.info(f"  Total rows in panel: {n_total}")
    logger.info(f"  Rows with ML alpha predictions: {n_with_predictions} ({pct_with_predictions:.1f}%)")
    logger.info(f"  Rows without predictions (training samples): {n_total - n_with_predictions}")
    
    # Log basic metrics
    if global_metrics:
        r2 = global_metrics.get("r2")
        mse = global_metrics.get("mse")
        mae = global_metrics.get("mae")
        
        logger.info(f"Model performance (global metrics):")
        if r2 is not None:
            logger.info(f"  R²: {r2:.4f}")
        if mse is not None:
            logger.info(f"  MSE: {mse:.6f}")
        if mae is not None:
            logger.info(f"  MAE: {mae:.6f}")
    
    # Save output files
    logger.info("Saving output files...")
    
    # Generate output filename
    # Extract model type and horizon for filename
    horizon_match = re.search(r"(\d+)d", label_col)
    if horizon_match:
        horizon_str = horizon_match.group(1) + "d"
    else:
        horizon_str = label_col.replace("fwd_return_", "")
    
    output_filename = f"ml_alpha_panel_{model_type}_{horizon_str}.parquet"
    output_path = output_dir / output_filename
    
    # Save ML alpha panel
    ml_alpha_panel_df.to_parquet(output_path, index=False)
    logger.info(f"Saved ML alpha panel to {output_path}")
    
    # Optionally save raw predictions separately
    predictions_output_path = output_dir / f"ml_alpha_predictions_{model_type}_{horizon_str}.parquet"
    predictions_df.to_parquet(predictions_output_path, index=False)
    logger.info(f"Saved raw predictions to {predictions_output_path}")
    
    # Save metadata JSON
    metadata = {
        "model_config": {
            "name": model_cfg.name,
            "model_type": model_cfg.model_type,
            "params": model_cfg.params,
        },
        "experiment_config": {
            "label_col": experiment_config.label_col,
            "n_splits": experiment_config.n_splits,
            "train_size": experiment_config.train_size,
            "standardize": experiment_config.standardize,
            "min_train_samples": experiment_config.min_train_samples,
            "test_start": str(test_start_ts) if test_start_ts else None,
            "test_end": str(test_end_ts) if test_end_ts else None,
        },
        "column_name": column_name,
        "global_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in global_metrics.items()
            if v is not None
        },
        "statistics": {
            "n_total_rows": int(n_total),
            "n_with_predictions": int(n_with_predictions),
            "pct_with_predictions": float(pct_with_predictions),
        },
    }
    
    metadata_path = output_dir / f"ml_alpha_metadata_{model_type}_{horizon_str}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("ML alpha factor export completed successfully")
    
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export ML alpha factors from factor panels. "
            "Trains ML models via time-series cross-validation and merges predictions "
            "back into the panel as ml_alpha_{model_type}_{horizon}d columns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export with Ridge model
  python research/ml/export_ml_alpha_factor.py \\
    --factor-panel-file output/factor_panels/factor_panel_ai_tech_core_20d_1d.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge \\
    --model-param alpha=0.1

  # Random Forest with custom parameters
  python research/ml/export_ml_alpha_factor.py \\
    --factor-panel-file output/factor_panels/factor_panel_ai_tech_core_20d_1d.parquet \\
    --label-col fwd_return_20d \\
    --model-type random_forest \\
    --model-param n_estimators=200 \\
    --model-param max_depth=5 \\
    --n-splits 10

  # Custom output directory and column name
  python research/ml/export_ml_alpha_factor.py \\
    --factor-panel-file output/factor_panels/factor_panel_ai_tech_core_20d_1d.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge \\
    --output-dir output/custom_ml_alpha \\
    --column-name my_ml_alpha_factor
        """,
    )
    
    parser.add_argument(
        "--factor-panel-file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to factor panel file (Parquet or CSV) with factors and forward returns",
    )
    
    parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        metavar="COL",
        help="Name of label column (e.g., 'fwd_return_20d')",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["linear", "ridge", "lasso", "random_forest"],
        metavar="TYPE",
        help="Model type: 'linear', 'ridge', 'lasso', or 'random_forest'",
    )
    
    parser.add_argument(
        "--model-param",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Model hyperparameters (can be specified multiple times, e.g., --model-param alpha=0.1)",
    )
    
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        metavar="N",
        help="Number of time-series CV splits (default: 5)",
    )
    
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional test start date (YYYY-MM-DD, UTC)",
    )
    
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional test end date (YYYY-MM-DD, UTC)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: output/ml_alpha_factors)",
    )
    
    parser.add_argument(
        "--column-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Custom column name for ML alpha factor "
            "(default: auto-generated as ml_alpha_{model_type}_{horizon}d)"
        ),
    )
    
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    
    try:
        # Parse model parameters
        model_params = parse_model_params(args.model_param) if args.model_param else None
        
        # Run export
        output_path = export_ml_alpha_factor(
            factor_panel_file=args.factor_panel_file,
            label_col=args.label_col,
            model_type=args.model_type,
            model_params=model_params,
            n_splits=args.n_splits,
            test_start=args.test_start,
            test_end=args.test_end,
            output_dir=args.output_dir,
            column_name=args.column_name,
        )
        
        print(f"Successfully exported ML alpha factor panel to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"ML alpha factor export failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

