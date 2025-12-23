"""Model Zoo for ML Factor Validation - Automated Model Comparison Tool.

DESIGN OVERVIEW:
===============

This script provides an automated "model zoo" workflow for comparing multiple
ML models on a single factor panel. It extends the single-model validation
workflow (run_ml_factor_validation.py) to run multiple model configurations
in sequence and aggregate results for comparison.

Purpose:
--------
- Run multiple model configurations (Linear, Ridge with different alphas, Lasso, RF)
  on the same factor panel
- Aggregate metrics across all models
- Generate comparison reports (CSV + optional Markdown)
- Enable systematic model selection based on comprehensive metrics

Workflow:
---------
1. Load factor panel (Parquet/CSV)
2. Define model zoo (list of MLModelConfig instances)
3. For each model:
   a. Prepare dataset (prepare_ml_dataset)
   b. Run time-series CV (run_time_series_cv)
   c. Evaluate predictions (evaluate_ml_predictions)
   d. Collect key metrics (R², MSE, IC, IC-IR, L/S Sharpe)
4. Aggregate results into comparison table
5. Write outputs:
   - ml_model_zoo_summary.csv (structured comparison table)
   - ml_model_zoo_summary.md (optional, human-readable report)

Key Design Decisions:
---------------------
- Reuses existing ML-Core functions (prepare_ml_dataset, run_time_series_cv, evaluate_ml_predictions)
- Uses same MLExperimentConfig for all models (ensures fair comparison)
- Aggregates per-model metrics into a single DataFrame for easy comparison
- Supports custom model zoo via build_default_model_zoo() or user-provided list

Future Enhancements (Post-Skeleton):
------------------------------------
- Hyperparameter grid search integration
- Parallel execution of models (multiprocessing)
- Best model selection based on combined score (IC-IR + Sharpe)
- Visualization of model comparison (if matplotlib available)
- Integration with experiment tracking

Example Usage (Future):
-----------------------
python research/ml/model_zoo_factor_validation.py \
  --factor-panel-file output/factor_panels/core_20d_factors.parquet \
  --label-col fwd_return_20d \
  --n-splits 5 \
  --output-dir output/ml_validation/model_zoo_comparison
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    evaluate_ml_predictions,
    run_time_series_cv,
)
from src.assembled_core.qa.metrics import deflated_sharpe_ratio
from src.assembled_core.ml.explainability import (
    compute_model_feature_importance,
    summarize_feature_importance_global,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_default_model_zoo() -> list[MLModelConfig]:
    """
    Build default model zoo with common configurations.

    Returns:
        List of MLModelConfig instances representing the model zoo
    """
    models = [
        # Linear models
        MLModelConfig(name="linear", model_type="linear", params={}),
        # Ridge with different regularization strengths
        MLModelConfig(name="ridge_0_1", model_type="ridge", params={"alpha": 0.1}),
        MLModelConfig(name="ridge_1_0", model_type="ridge", params={"alpha": 1.0}),
        MLModelConfig(name="ridge_10_0", model_type="ridge", params={"alpha": 10.0}),
        # Lasso (sparse features)
        MLModelConfig(name="lasso_0_01", model_type="lasso", params={"alpha": 0.01}),
        # Random Forest with different depths
        MLModelConfig(
            name="rf_depth_3",
            model_type="random_forest",
            params={"n_estimators": 200, "max_depth": 3, "random_state": 42},
        ),
        MLModelConfig(
            name="rf_depth_5",
            model_type="random_forest",
            params={"n_estimators": 200, "max_depth": 5, "random_state": 42},
        ),
    ]

    return models


def run_model_zoo_for_panel(
    factor_panel_path: Path,
    label_col: str,
    output_dir: Path,
    experiment_cfg_kwargs: dict[str, Any] | None = None,
    model_configs: list[MLModelConfig] | None = None,
) -> pd.DataFrame:
    """
    Run model zoo on factor panel and collect metrics for all models.

    Args:
        factor_panel_path: Path to factor panel file (Parquet or CSV)
        label_col: Name of label column (e.g., "fwd_return_20d")
        output_dir: Output directory for writing results
        experiment_cfg_kwargs: Optional dict to override default MLExperimentConfig values
        model_configs: Optional list of MLModelConfig instances (default: use build_default_model_zoo())

    Returns:
        DataFrame with aggregated metrics (one row per model):
        - model_name: Name of model
        - model_type: Type of model (linear, ridge, etc.)
        - test_r2_mean: Average test R² across CV splits
        - test_mse_mean: Average test MSE
        - test_mae_mean: Average test MAE
        - ic_mean: Mean IC from evaluate_ml_predictions
        - ic_ir: IC-IR from evaluate_ml_predictions
        - rank_ic_mean: Mean Rank-IC
        - rank_ic_ir: Rank-IC-IR
        - ls_sharpe: Long/Short Sharpe ratio (if available, else NaN)
        - train_test_gap_r2: Average gap between train and test R²
        - n_splits: Number of successful CV splits
        - n_samples: Total number of predictions
    """
    # Load factor panel
    logger.info(f"Loading factor panel from {factor_panel_path}...")
    if factor_panel_path.suffix == ".parquet":
        factor_panel_df = pd.read_parquet(factor_panel_path)
    elif factor_panel_path.suffix == ".csv":
        factor_panel_df = pd.read_csv(factor_panel_path)
        if "timestamp" in factor_panel_df.columns:
            factor_panel_df["timestamp"] = pd.to_datetime(
                factor_panel_df["timestamp"], utc=True, errors="coerce"
            )
    else:
        raise ValueError(
            f"Unsupported file format: {factor_panel_path.suffix}. Use .parquet or .csv"
        )

    logger.info(
        f"Loaded {len(factor_panel_df)} rows, {len(factor_panel_df.columns)} columns"
    )

    # Build model zoo (if not provided)
    if model_configs is None:
        model_configs = build_default_model_zoo()

    logger.info(f"Running model zoo with {len(model_configs)} models...")

    # Build experiment config with defaults
    default_experiment_kwargs = {
        "label_col": label_col,
        "feature_cols": None,  # Auto-detect
        "n_splits": 5,
        "train_size": None,  # Expanding window
        "standardize": True,
        "min_train_samples": 252,  # ~1 year of daily data
    }

    # Override with user-provided kwargs
    if experiment_cfg_kwargs:
        default_experiment_kwargs.update(experiment_cfg_kwargs)

    experiment_config = MLExperimentConfig(**default_experiment_kwargs)

    # Extract horizon_days from label_col (e.g., "fwd_return_20d" -> 20)
    horizon_match = re.search(r"(\d+)d", label_col)
    horizon_days = int(horizon_match.group(1)) if horizon_match else 20
    logger.info(f"Detected horizon: {horizon_days} days from label_col '{label_col}'")

    # Run models
    results = []
    feature_importance_dfs = {}  # Collect feature importances for global summary

    for i, model_cfg in enumerate(model_configs, 1):
        logger.info(
            f"[{i}/{len(model_configs)}] Running model: {model_cfg.name} ({model_cfg.model_type})"
        )

        try:
            # Run time-series CV (this handles prepare_ml_dataset internally)
            cv_result = run_time_series_cv(
                factor_panel_df=factor_panel_df,
                experiment=experiment_config,
                model_cfg=model_cfg,
            )

            predictions_df = cv_result["predictions_df"]
            global_metrics = cv_result["global_metrics"]
            metrics_df = cv_result["metrics_df"]
            last_model = cv_result.get("last_model")
            feature_names = cv_result.get("feature_names", [])

            # Compute feature importance (optional, for cost control)
            if last_model is not None and feature_names:
                try:
                    feature_imp_df = compute_model_feature_importance(
                        last_model, feature_names
                    )
                    feature_importance_dfs[model_cfg.name] = feature_imp_df
                    logger.info(
                        f"  Feature importance computed for {model_cfg.name}: {len(feature_imp_df)} features"
                    )
                except Exception as e:
                    logger.warning(
                        f"  Failed to compute feature importance for {model_cfg.name}: {e}"
                    )
                    # Continue without feature importance for this model

            if predictions_df.empty:
                logger.warning(
                    f"Model {model_cfg.name} produced empty predictions, skipping..."
                )
                continue

            # Evaluate predictions
            eval_metrics = evaluate_ml_predictions(
                predictions_df=predictions_df,
                horizon_days=horizon_days,
            )

            # Collect metrics from global_metrics and eval_metrics
            test_r2 = global_metrics.get("r2")
            train_r2 = global_metrics.get("r2_train")

            # Extract Sharpe and compute deflated Sharpe (B4)
            ls_sharpe_raw = eval_metrics.get("ls_sharpe")
            n_samples = eval_metrics.get("n_predictions", len(predictions_df))
            n_tests = len(model_configs)  # Total number of models tested in this zoo

            # Compute deflated Sharpe if we have a valid Sharpe
            ls_sharpe_deflated = None
            if (
                ls_sharpe_raw is not None
                and not np.isnan(ls_sharpe_raw)
                and n_samples >= 2
            ):
                # n_obs: Use number of unique timestamps if available, else n_samples
                # For daily data, n_samples ≈ n_obs (one prediction per day)
                # For intraday, we'd need to adjust, but for factor panels it's usually daily
                n_obs_eff = (
                    n_samples  # Conservative: assume one observation per prediction
                )
                if "timestamp" in predictions_df.columns:
                    n_obs_eff = predictions_df["timestamp"].nunique()

                try:
                    ls_sharpe_deflated = deflated_sharpe_ratio(
                        sharpe_annual=ls_sharpe_raw,
                        n_obs=n_obs_eff,
                        n_tests=n_tests,
                        skew=0.0,  # Default: assume normal (could be computed from returns if available)
                        kurtosis=3.0,  # Default: assume normal
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to compute deflated Sharpe for {model_cfg.name}: {e}"
                    )
                    ls_sharpe_deflated = None

            result_row = {
                "model_name": model_cfg.name,
                "model_type": model_cfg.model_type,
                "test_r2_mean": test_r2,
                "test_mse_mean": global_metrics.get("mse"),
                "test_mae_mean": global_metrics.get("mae"),
                "train_r2_mean": train_r2,
                "train_test_gap_r2": (
                    (train_r2 - test_r2)
                    if (train_r2 is not None and test_r2 is not None)
                    else None
                ),
                "ic_mean": eval_metrics.get("ic_mean"),
                "ic_ir": eval_metrics.get("ic_ir"),
                "rank_ic_mean": eval_metrics.get("rank_ic_mean"),
                "rank_ic_ir": eval_metrics.get("rank_ic_ir"),
                "ls_sharpe_raw": ls_sharpe_raw,  # B4: Renamed from ls_sharpe
                "ls_sharpe_deflated": ls_sharpe_deflated,  # B4: New deflated Sharpe
                "ls_return_mean": eval_metrics.get("ls_return_mean"),
                "directional_accuracy": eval_metrics.get("directional_accuracy"),
                "n_splits": len(metrics_df) if not metrics_df.empty else 0,
                "n_samples": n_samples,
                "n_tests": n_tests,  # B4: Number of models tested in this zoo
            }

            results.append(result_row)
            r2_str = (
                f"{result_row['test_r2_mean']:.4f}"
                if result_row["test_r2_mean"] is not None
                else "N/A"
            )
            ic_ir_str = (
                f"{result_row['ic_ir']:.4f}"
                if result_row["ic_ir"] is not None
                else "N/A"
            )
            logger.info(f"  ✓ {model_cfg.name}: R²={r2_str}, IC-IR={ic_ir_str}")

        except Exception as e:
            logger.error(f"Model {model_cfg.name} failed: {e}", exc_info=True)
            # Add error row with None values
            results.append(
                {
                    "model_name": model_cfg.name,
                    "model_type": model_cfg.model_type,
                    "error": str(e)[:100],  # Truncate error message
                    **{
                        k: None
                        for k in [
                            "test_r2_mean",
                            "test_mse_mean",
                            "test_mae_mean",
                            "train_r2_mean",
                            "train_test_gap_r2",
                            "ic_mean",
                            "ic_ir",
                            "rank_ic_mean",
                            "rank_ic_ir",
                            "ls_sharpe_raw",
                            "ls_sharpe_deflated",
                            "ls_return_mean",
                            "directional_accuracy",
                            "n_splits",
                            "n_samples",
                            "n_tests",
                        ]
                    },
                }
            )
            continue

    # Convert to DataFrame
    summary_df = pd.DataFrame(results)

    # Sort by IC-IR descending (if available), else by test_r2_mean
    if "ic_ir" in summary_df.columns and summary_df["ic_ir"].notna().any():
        summary_df = summary_df.sort_values(
            "ic_ir", ascending=False, na_position="last"
        )
    elif "test_r2_mean" in summary_df.columns:
        summary_df = summary_df.sort_values(
            "test_r2_mean", ascending=False, na_position="last"
        )

    logger.info(
        f"Model zoo completed: {len([r for r in results if r.get('test_r2_mean') is not None])} successful, "
        f"{len([r for r in results if r.get('test_r2_mean') is None])} failed"
    )

    # Compute global feature importance summary (if we have feature importances)
    if feature_importance_dfs:
        try:
            logger.info("Computing global feature importance summary...")
            global_feature_importance_df = summarize_feature_importance_global(
                feature_importance_dfs
            )

            # Save global feature importance summary
            global_importance_path = (
                output_dir / "ml_model_zoo_feature_importance_summary.csv"
            )
            global_feature_importance_df.to_csv(global_importance_path, index=False)
            logger.info(
                f"Saved global feature importance summary to {global_importance_path}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute global feature importance summary: {e}",
                exc_info=True,
            )

    return summary_df


def write_model_zoo_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    write_markdown: bool = True,
) -> tuple[Path, Path | None]:
    """
    Write model zoo summary to CSV and optionally Markdown.

    Args:
        summary_df: DataFrame with aggregated model metrics (from run_model_zoo_for_panel)
        output_dir: Output directory
        write_markdown: Whether to write Markdown report (default: True)

    Returns:
        Tuple of (csv_path, md_path) where md_path may be None if write_markdown=False
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = output_dir / "ml_model_zoo_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved model zoo summary CSV to {csv_path}")

    md_path = None
    if write_markdown:
        md_path = output_dir / "ml_model_zoo_summary.md"

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Model Zoo Comparison Summary\n\n")
            f.write(f"**Total Models:** {len(summary_df)}\n\n")

            # Filter successful models (those without errors)
            if "error" in summary_df.columns:
                successful_df = summary_df[summary_df["error"].isna()]
            else:
                successful_df = summary_df

            if len(successful_df) > 0:
                f.write("## Top Models\n\n")

                # Best by IC-IR
                if (
                    "ic_ir" in successful_df.columns
                    and successful_df["ic_ir"].notna().any()
                ):
                    best_ic_ir = successful_df.loc[successful_df["ic_ir"].idxmax()]
                    f.write(
                        f"**Best IC-IR:** {best_ic_ir['model_name']} (IC-IR={best_ic_ir['ic_ir']:.4f})\n\n"
                    )

                # Best by R²
                if (
                    "test_r2_mean" in successful_df.columns
                    and successful_df["test_r2_mean"].notna().any()
                ):
                    best_r2 = successful_df.loc[successful_df["test_r2_mean"].idxmax()]
                    f.write(
                        f"**Best R²:** {best_r2['model_name']} (R²={best_r2['test_r2_mean']:.4f})\n\n"
                    )

                # Comparison table (B4: Include deflated Sharpe)
                f.write("## Model Comparison\n\n")
                f.write(
                    "| Model | Type | Test R² | IC-IR | L/S Sharpe (Raw) | L/S Sharpe (Deflated) | N Tests | N Samples |\n"
                )
                f.write(
                    "|-------|------|---------|-------|------------------|----------------------|---------|----------|\n"
                )

                for _, row in successful_df.iterrows():
                    r2_str = (
                        f"{row['test_r2_mean']:.4f}"
                        if pd.notna(row.get("test_r2_mean"))
                        else "N/A"
                    )
                    ic_ir_str = (
                        f"{row['ic_ir']:.4f}" if pd.notna(row.get("ic_ir")) else "N/A"
                    )
                    sharpe_raw_str = (
                        f"{row['ls_sharpe_raw']:.4f}"
                        if pd.notna(row.get("ls_sharpe_raw"))
                        else "N/A"
                    )
                    sharpe_deflated_str = (
                        f"{row['ls_sharpe_deflated']:.4f}"
                        if pd.notna(row.get("ls_sharpe_deflated"))
                        else "N/A"
                    )
                    n_tests_str = (
                        f"{int(row['n_tests'])}"
                        if pd.notna(row.get("n_tests"))
                        else "N/A"
                    )
                    n_samples_str = (
                        f"{int(row['n_samples'])}"
                        if pd.notna(row.get("n_samples"))
                        else "N/A"
                    )

                    f.write(
                        f"| {row['model_name']} | {row['model_type']} | {r2_str} | "
                        f"{ic_ir_str} | {sharpe_raw_str} | {sharpe_deflated_str} | {n_tests_str} | {n_samples_str} |\n"
                    )

                f.write("\n")
            else:
                f.write("**Warning:** No successful model runs.\n\n")

            # Failed models
            if "error" in summary_df.columns:
                failed_df = summary_df[summary_df["error"].notna()]
            else:
                failed_df = pd.DataFrame()
            if len(failed_df) > 0:
                f.write("## Failed Models\n\n")
                for _, row in failed_df.iterrows():
                    f.write(
                        f"- **{row['model_name']}**: {row.get('error', 'Unknown error')}\n"
                    )
                f.write("\n")

        logger.info(f"Saved model zoo summary Markdown to {md_path}")

    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run model zoo comparison on factor panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model zoo comparison
  python research/ml/model_zoo_factor_validation.py \\
    --factor-panel-file output/factor_panels/core_20d_factors.parquet \\
    --label-col fwd_return_20d
  
  # With custom CV splits
  python research/ml/model_zoo_factor_validation.py \\
    --factor-panel-file output/factor_panels/core_20d_factors.parquet \\
    --label-col fwd_return_20d \\
    --n-splits 10 \\
    --output-dir output/ml_validation/custom_zoo
        """,
    )

    parser.add_argument(
        "--factor-panel-file",
        type=Path,
        required=True,
        help="Path to factor panel file (Parquet or CSV)",
    )

    parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Label column name (e.g., 'fwd_return_20d')",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/ml_model_zoo)",
    )

    parser.add_argument(
        "--n-splits", type=int, default=None, help="Number of CV splits (default: 5)"
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Training window size in days (default: None = expanding window)",
    )

    parser.add_argument(
        "--standardize",
        type=bool,
        default=None,
        help="Whether to standardize features (default: True)",
    )

    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=None,
        help="Minimum training samples (default: 252)",
    )

    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="Test start date (YYYY-MM-DD, optional)",
    )

    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Test end date (YYYY-MM-DD, optional)",
    )

    parser.add_argument(
        "--no-markdown", action="store_true", help="Skip Markdown report generation"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for model zoo factor validation.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        args = parse_args()

        # Resolve paths
        factor_panel_path = args.factor_panel_file
        if not factor_panel_path.is_absolute():
            factor_panel_path = ROOT / factor_panel_path

        if not factor_panel_path.exists():
            logger.error(f"Factor panel file not found: {factor_panel_path}")
            return 1

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = ROOT / "output" / "ml_model_zoo"
        elif not output_dir.is_absolute():
            output_dir = ROOT / output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Model Zoo Factor Validation")
        logger.info("=" * 60)
        logger.info(f"Factor Panel: {factor_panel_path}")
        logger.info(f"Label Column: {args.label_col}")
        logger.info(f"Output Directory: {output_dir}")

        # Build experiment config kwargs from args
        experiment_cfg_kwargs: dict[str, Any] = {}
        if args.n_splits is not None:
            experiment_cfg_kwargs["n_splits"] = args.n_splits
        if args.train_size is not None:
            experiment_cfg_kwargs["train_size"] = args.train_size
        if args.standardize is not None:
            experiment_cfg_kwargs["standardize"] = args.standardize
        if args.min_train_samples is not None:
            experiment_cfg_kwargs["min_train_samples"] = args.min_train_samples
        if args.test_start:
            experiment_cfg_kwargs["test_start"] = pd.to_datetime(
                args.test_start, utc=True
            )
        if args.test_end:
            experiment_cfg_kwargs["test_end"] = pd.to_datetime(args.test_end, utc=True)

        # Run model zoo
        summary_df = run_model_zoo_for_panel(
            factor_panel_path=factor_panel_path,
            label_col=args.label_col,
            output_dir=output_dir,
            experiment_cfg_kwargs=experiment_cfg_kwargs
            if experiment_cfg_kwargs
            else None,
        )

        # Write summary
        csv_path, md_path = write_model_zoo_summary(
            summary_df=summary_df,
            output_dir=output_dir,
            write_markdown=not args.no_markdown,
        )

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Model Zoo Comparison Results")
        logger.info("=" * 60)

        if "error" in summary_df.columns:
            successful_df = summary_df[summary_df["error"].isna()]
        else:
            successful_df = summary_df

        if len(successful_df) > 0:
            logger.info(f"Successful models: {len(successful_df)}")

            # Top 3 by R²
            if (
                "test_r2_mean" in successful_df.columns
                and successful_df["test_r2_mean"].notna().any()
            ):
                top3_r2 = successful_df.nlargest(3, "test_r2_mean")
                logger.info("\nTop 3 by Test R²:")
                for i, (_, row) in enumerate(top3_r2.iterrows(), 1):
                    logger.info(
                        f"  {i}. {row['model_name']}: R²={row['test_r2_mean']:.4f}, "
                        f"IC-IR={row['ic_ir']:.4f if pd.notna(row.get('ic_ir')) else 'N/A'}"
                    )

        logger.info("")
        logger.info(f"Summary CSV: {csv_path}")
        if md_path:
            logger.info(f"Summary Markdown: {md_path}")
        logger.info("")
        logger.info("Model zoo comparison completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Model zoo comparison failed: {e}", exc_info=True)
        return 1


# Implementation Roadmap:
# =======================
#
# Phase 1 (Skeleton - Current):
# - ✅ File structure and imports
# - ✅ Function signatures
# - ✅ Design documentation
# - ✅ TODO comments for implementation
#
# Phase 2 (Core Implementation):
# - Implement build_default_model_zoo() with 5-7 common model configurations
# - Implement run_model_zoo_for_panel() with full workflow
# - Implement write_model_zoo_summary() for CSV and Markdown output
# - Implement parse_args() and main()
# - Add error handling and logging
#
# Phase 3 (Testing):
# - Test with sample factor panel
# - Verify all models run successfully
# - Check output format and correctness
# - Add to test suite (tests/test_ml_model_zoo.py)
#
# Phase 4 (Enhancements):
# - Parallel execution support (optional)
# - Best model auto-selection
# - Integration with experiment tracking
# - Visualization support

if __name__ == "__main__":
    sys.exit(main())
