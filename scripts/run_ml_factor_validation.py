"""ML Factor Validation CLI Runner.

This script provides a command-line interface for running ML validation on factor panels.
It trains ML models to predict forward returns and evaluates them using time-series cross-validation.

Usage:
    python scripts/run_ml_factor_validation.py \
      --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \
      --label-col fwd_return_20d \
      --model-type ridge \
      --n-splits 5 \
      --output-dir output/ml_validation
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.logging_config import generate_run_id, setup_logging
from src.assembled_core.ml.factor_models import (
    MLExperimentConfig,
    MLModelConfig,
    evaluate_ml_predictions,
    run_time_series_cv,
)
from src.assembled_core.qa.metrics import deflated_sharpe_ratio
from src.assembled_core.ml.explainability import (
    compute_model_feature_importance,
    compute_permutation_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_model_params(param_strings: list[str]) -> dict[str, Any]:
    """
    Parse model parameter strings in format "key=value".
    
    Args:
        param_strings: List of strings like ["alpha=0.1", "max_iter=1000"]
    
    Returns:
        Dictionary with parsed parameters (values are converted to int/float if possible)
    
    Example:
        >>> parse_model_params(["alpha=0.1", "max_iter=1000"])
        {'alpha': 0.1, 'max_iter': 1000}
    """
    params = {}
    
    for param_str in param_strings:
        if "=" not in param_str:
            logger.warning(f"Ignoring invalid parameter format: {param_str} (expected 'key=value')")
            continue
        
        key, value_str = param_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Try to convert value to int or float
        try:
            # Try int first
            value = int(value_str)
        except ValueError:
            try:
                # Try float
                value = float(value_str)
            except ValueError:
                # Keep as string
                value = value_str
        
        params[key] = value
    
    return params


def write_ml_validation_report(
    model_name: str,
    label_col: str,
    metrics: dict[str, Any],
    predictions_summary: dict[str, Any],
    output_path: Path,
    feature_importance_df: pd.DataFrame | None = None,
    permutation_importance_df: pd.DataFrame | None = None,
) -> None:
    """
    Write ML validation report as Markdown.
    
    Args:
        model_name: Model name (e.g., "ridge_20d")
        label_col: Label column name (e.g., "fwd_return_20d")
        metrics: Dictionary with evaluation metrics
        predictions_summary: Summary statistics about predictions
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# ML Factor Validation Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Label:** {label_col}\n\n")
        
        # Classical ML Metrics
        f.write("## Classical ML Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        mse = metrics.get("mse")
        mae = metrics.get("mae")
        r2 = metrics.get("r2")
        directional_accuracy = metrics.get("directional_accuracy")
        
        f.write(f"| MSE | {mse:.6f} |\n" if mse is not None else "| MSE | N/A |\n")
        f.write(f"| MAE | {mae:.6f} |\n" if mae is not None else "| MAE | N/A |\n")
        f.write(f"| R² | {r2:.4f} |\n" if r2 is not None else "| R² | N/A |\n")
        f.write(
            f"| Directional Accuracy | {directional_accuracy:.2%} |\n"
            if directional_accuracy is not None
            else "| Directional Accuracy | N/A |\n"
        )
        f.write(f"| N Samples | {metrics.get('n_samples', 0)} |\n")
        f.write("\n")
        
        # Factor-Specific Metrics
        f.write("## Factor-Specific Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        ic_mean = metrics.get("ic_mean")
        ic_ir = metrics.get("ic_ir")
        rank_ic_mean = metrics.get("rank_ic_mean")
        rank_ic_ir = metrics.get("rank_ic_ir")
        
        f.write(f"| Mean IC | {ic_mean:.4f} |\n" if ic_mean is not None else "| Mean IC | N/A |\n")
        f.write(f"| IC-IR | {ic_ir:.4f} |\n" if ic_ir is not None else "| IC-IR | N/A |\n")
        f.write(
            f"| Mean Rank-IC | {rank_ic_mean:.4f} |\n"
            if rank_ic_mean is not None
            else "| Mean Rank-IC | N/A |\n"
        )
        f.write(
            f"| Rank-IC-IR | {rank_ic_ir:.4f} |\n"
            if rank_ic_ir is not None
            else "| Rank-IC-IR | N/A |\n"
        )
        f.write(f"| N Timestamps | {metrics.get('n_timestamps', 0)} |\n")
        f.write("\n")
        
        # Portfolio Metrics
        ls_sharpe = metrics.get("ls_sharpe")
        ls_return_mean = metrics.get("ls_return_mean")
        
        if ls_sharpe is not None or ls_return_mean is not None:
            f.write("## Long/Short Portfolio Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(
                f"| L/S Return (Mean) | {ls_return_mean:.4f} |\n"
                if ls_return_mean is not None
                else "| L/S Return (Mean) | N/A |\n"
            )
            f.write(
                f"| L/S Sharpe | {ls_sharpe:.4f} |\n"
                if ls_sharpe is not None
                else "| L/S Sharpe | N/A |\n"
            )
            f.write("\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("### Classical ML Metrics\n")
        f.write("- **MSE/MAE**: Lower is better. Measures prediction error.\n")
        f.write("- **R²**: Higher is better. R² > 0 indicates model explains some variance.\n")
        f.write("- **Directional Accuracy**: Percentage of correct sign predictions. > 50% indicates predictive power.\n\n")
        
        f.write("### Factor-Specific Metrics\n")
        f.write("- **IC (Information Coefficient)**: Cross-sectional correlation between predictions and realized returns.\n")
        f.write("  - Mean IC > 0.05: Strong predictive signal\n")
        f.write("  - IC-IR > 0.5: Consistent signal over time\n")
        f.write("- **Rank-IC**: Spearman correlation (robust to outliers).\n")
        f.write("  - Higher Rank-IC than IC suggests non-linear relationships.\n\n")
        
        ls_sharpe_raw = metrics.get("ls_sharpe_raw")  # B4: Updated key
        ls_sharpe_deflated = metrics.get("ls_sharpe_deflated")  # B4: New metric
        n_tests = metrics.get("n_tests", 1)
        
        if ls_sharpe_raw is not None:
            f.write("### Portfolio Performance\n")
            f.write("- **L/S Sharpe (Raw)**: Sharpe ratio of long/short portfolio (Top vs. Bottom quintile based on predictions).\n")
            f.write(f"  - Current value: {ls_sharpe_raw:.4f} ({'Good' if ls_sharpe_raw > 1.0 else 'Moderate' if ls_sharpe_raw > 0.5 else 'Weak'} signal)\n")
            if ls_sharpe_deflated is not None:
                f.write(f"- **L/S Sharpe (Deflated)**: Deflated Sharpe ratio adjusted for multiple testing (n_tests={n_tests}).\n")
                f.write(f"  - Current value: {ls_sharpe_deflated:.4f} ({'Significant' if ls_sharpe_deflated > 0 else 'May be due to luck/multiple testing'})\n")
            f.write("\n")
        
        # Feature Importance Section
        if feature_importance_df is not None and not feature_importance_df.empty:
            f.write("## Top 10 Features (Model Coefficients / Feature Importances)\n\n")
            top_features = feature_importance_df.head(10)
            
            f.write("| Rank | Feature | Importance | Raw Value | Direction |\n")
            f.write("|------|---------|------------|-----------|----------|\n")
            for idx, row in top_features.iterrows():
                direction_str = str(row["direction"]) if pd.notna(row["direction"]) else "N/A"
                f.write(
                    f"| {idx + 1} | {row['feature']} | {row['importance']:.6f} | "
                    f"{row['raw_value']:.6f} | {direction_str} |\n"
                )
            f.write("\n")
            
            # Interpretation hints
            top_feature_names = top_features["feature"].tolist()[:5]
            
            # Check for factor groups
            momentum_features = [f for f in top_feature_names if "mom" in f.lower() or "momentum" in f.lower()]
            volatility_features = [f for f in top_feature_names if "vol" in f.lower() or "volatility" in f.lower() or "rv_" in f.lower()]
            value_features = [f for f in top_feature_names if "value" in f.lower()]
            quality_features = [f for f in top_feature_names if "quality" in f.lower()]
            
            f.write("### Feature Importance Interpretation\n\n")
            if momentum_features:
                f.write(f"- **Momentum factors** ({', '.join(momentum_features)}) are among the top features. ")
                f.write("This suggests momentum effects are important for return prediction.\n")
            if volatility_features:
                f.write(f"- **Volatility factors** ({', '.join(volatility_features)}) appear in top features. ")
                f.write("Volatility characteristics influence predictions.\n")
            if value_features:
                f.write(f"- **Value factors** ({', '.join(value_features)}) are prominent. ")
                f.write("Value-based signals contribute to the model.\n")
            if quality_features:
                f.write(f"- **Quality factors** ({', '.join(quality_features)}) are important. ")
                f.write("Quality characteristics matter for predictions.\n")
            
            if not (momentum_features or volatility_features or value_features or quality_features):
                f.write("- Top features are listed above. Review their characteristics to understand model behavior.\n")
            
            f.write("\n")
        
        # Permutation Importance Section (optional)
        if permutation_importance_df is not None and not permutation_importance_df.empty:
            f.write("## Top 10 Features (Permutation Importance)\n\n")
            f.write("*Note: Permutation importance measures how much model performance decreases when a feature is shuffled.*\n\n")
            
            top_perm_features = permutation_importance_df.head(10)
            
            f.write("| Rank | Feature | Importance (Mean) | Importance (Std) |\n")
            f.write("|------|---------|-------------------|------------------|\n")
            for idx, row in top_perm_features.iterrows():
                f.write(
                    f"| {idx + 1} | {row['feature']} | {row['importance_mean']:.6f} | "
                    f"{row['importance_std']:.6f} |\n"
                )
            f.write("\n")


def run_ml_validation(
    factor_panel_file: Path,
    label_col: str,
    model_type: str,
    model_params: dict[str, Any] | None = None,
    n_splits: int = 5,
    test_start: str | None = None,
    test_end: str | None = None,
    output_dir: Path | None = None,
) -> int:
    """
    Run ML validation on factor panel.
    
    Args:
        factor_panel_file: Path to factor panel file (Parquet or CSV)
        label_col: Name of label column (e.g., "fwd_return_20d")
        model_type: Model type ("linear", "ridge", "lasso", "random_forest")
        model_params: Model hyperparameters (dict)
        n_splits: Number of CV splits
        test_start: Optional test start date (YYYY-MM-DD)
        test_end: Optional test end date (YYYY-MM-DD)
        output_dir: Output directory (default: output/ml_validation)
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Resolve paths
    if not factor_panel_file.is_absolute():
        factor_panel_file = ROOT / factor_panel_file
    
    if not factor_panel_file.exists():
        logger.error(f"Factor panel file not found: {factor_panel_file}")
        return 1
    
    if output_dir is None:
        output_dir = ROOT / "output" / "ml_validation"
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"ML Factor Validation")
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
            logger.error(
                f"Unsupported file format: {factor_panel_file.suffix}. "
                f"Supported formats: .parquet, .csv"
            )
            return 1
        
        logger.info(f"Loaded factor panel: {len(factor_panel_df)} rows, {len(factor_panel_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load factor panel: {e}", exc_info=True)
        return 1
    
    # Validate label column
    if label_col not in factor_panel_df.columns:
        logger.error(
            f"Label column '{label_col}' not found in factor panel. "
            f"Available columns: {list(factor_panel_df.columns)}"
        )
        return 1
    
    # Check for sklearn availability
    try:
        from sklearn.linear_model import LinearRegression  # noqa: F401
    except ImportError:
        logger.error(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )
        return 1
    
    # Parse test dates
    test_start_ts = None
    test_end_ts = None
    
    if test_start:
        try:
            test_start_ts = pd.to_datetime(test_start, utc=True)
        except Exception as e:
            logger.error(f"Invalid test_start date format: {test_start}. Error: {e}")
            return 1
    
    if test_end:
        try:
            test_end_ts = pd.to_datetime(test_end, utc=True)
        except Exception as e:
            logger.error(f"Invalid test_end date format: {test_end}. Error: {e}")
            return 1
    
    # Build experiment config
    experiment = MLExperimentConfig(
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
            experiment=experiment,
            model_cfg=model_cfg,
        )
    except Exception as e:
        logger.error(f"CV failed: {e}", exc_info=True)
        return 1
    
    predictions_df = cv_result["predictions_df"]
    metrics_df = cv_result["metrics_df"]
    global_metrics_cv = cv_result["global_metrics"]
    last_model = cv_result.get("last_model")
    feature_names = cv_result.get("feature_names", [])
    
    logger.info(
        f"CV completed: {len(metrics_df)} splits, "
        f"global R² = {global_metrics_cv.get('r2', 0):.4f}"
    )
    
    # Compute feature importance (if model and feature names available)
    feature_importance_df = None
    permutation_importance_df = None
    
    if last_model is not None and feature_names:
        try:
            logger.info("Computing feature importance...")
            feature_importance_df = compute_model_feature_importance(last_model, feature_names)
            logger.info(f"Feature importance computed: {len(feature_importance_df)} features")
            
            # Save feature importance CSV
            feature_importance_path = output_dir / f"ml_feature_importance_{model_type}_{label_col.replace('fwd_return_', '')}.csv"
            feature_importance_df.to_csv(feature_importance_path, index=False)
            logger.info(f"Saved feature importance to {feature_importance_path}")
        except Exception as e:
            logger.warning(f"Failed to compute feature importance: {e}", exc_info=True)
            feature_importance_df = None
        
        # Optional: Compute permutation importance (on subsample for cost control)
        try:
            from src.assembled_core.ml.factor_models import prepare_ml_dataset
            X_full, y_full = prepare_ml_dataset(factor_panel_df, experiment)
            
            # Subsample for cost control (max 5000 rows)
            if len(X_full) > 5000:
                logger.info(f"Subsampling to 5000 rows for permutation importance (from {len(X_full)} rows)")
                sample_indices = np.random.choice(len(X_full), size=5000, replace=False)
                X_sample = X_full.iloc[sample_indices]
                y_sample = y_full.iloc[sample_indices]
            else:
                X_sample = X_full
                y_sample = y_full
            
            logger.info("Computing permutation importance (this may take a while)...")
            permutation_importance_df = compute_permutation_importance(
                last_model,
                X_sample,
                y_sample,
                n_repeats=5,  # Reduced repeats for speed
                random_state=42,
            )
            logger.info(f"Permutation importance computed: {len(permutation_importance_df)} features")
            
            # Save permutation importance CSV
            perm_importance_path = output_dir / f"ml_permutation_importance_{model_type}_{label_col.replace('fwd_return_', '')}.csv"
            permutation_importance_df.to_csv(perm_importance_path, index=False)
            logger.info(f"Saved permutation importance to {perm_importance_path}")
        except Exception as e:
            logger.warning(f"Failed to compute permutation importance: {e}", exc_info=True)
            permutation_importance_df = None
    
    # Extract horizon from label_col (e.g., "fwd_return_20d" -> 20)
    horizon_days = 20  # Default
    try:
        # Try to extract number from label_col
        import re
        match = re.search(r"(\d+)d", label_col)
        if match:
            horizon_days = int(match.group(1))
    except Exception:
        pass
    
    # Evaluate predictions (aggregated)
    logger.info("Evaluating predictions...")
    try:
        eval_metrics = evaluate_ml_predictions(
            predictions_df=predictions_df,
            horizon_days=horizon_days,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  R²: {eval_metrics.get('r2', 0):.4f}")
    logger.info(f"  IC-IR: {eval_metrics.get('ic_ir', 0):.4f}")
    ls_sharpe_raw = eval_metrics.get('ls_sharpe')
    if ls_sharpe_raw is not None:
        logger.info(f"  L/S Sharpe (Raw): {ls_sharpe_raw:.4f}")
    else:
        logger.info("  L/S Sharpe (Raw): N/A")
    
    # Log feature info
    try:
        from src.assembled_core.ml.factor_models import detect_feature_cols
        
        features = detect_feature_cols(factor_panel_df, label_col)
        logger.info(f"Used {len(features)} features: {features[:10]}{'...' if len(features) > 10 else ''}")
    except Exception:
        pass
    
    # Write outputs
    logger.info("Writing outputs...")
    
    # 1. Metrics CSV
    metrics_file = output_dir / f"ml_metrics_{model_type}_{label_col}.csv"
    
    # Combine CV metrics and evaluation metrics
    if not metrics_df.empty:
        # Add model_name and label_col to metrics_df
        metrics_df["model_name"] = model_name
        metrics_df["label_col"] = label_col
        metrics_df["horizon_days"] = horizon_days
        
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics to {metrics_file}")
    
    # 2. Predictions sample (first 10000 rows to limit size)
    predictions_sample_file = output_dir / f"ml_predictions_sample_{model_type}_{label_col}.parquet"
    
    predictions_sample = predictions_df.head(10000) if len(predictions_df) > 10000 else predictions_df
    predictions_sample.to_parquet(predictions_sample_file, index=False)
    logger.info(
        f"Saved predictions sample ({len(predictions_sample)} rows) to {predictions_sample_file}"
    )
    
    # 3. Portfolio metrics CSV (extract relevant metrics) - B4: Add deflated Sharpe
    ls_sharpe_raw = eval_metrics.get("ls_sharpe")
    n_samples = eval_metrics.get("n_predictions", len(predictions_df) if predictions_df is not None else 0)
    n_tests = 1  # Single model validation (conservative: assume this is part of a larger experiment)
    
    # Compute deflated Sharpe if we have a valid Sharpe
    ls_sharpe_deflated = None
    if ls_sharpe_raw is not None and not np.isnan(ls_sharpe_raw) and n_samples >= 2:
        # n_obs: Use number of unique timestamps if available, else n_samples
        n_obs_eff = n_samples  # Conservative: assume one observation per prediction
        if predictions_df is not None and "timestamp" in predictions_df.columns:
            n_obs_eff = predictions_df["timestamp"].nunique()
        
        try:
            ls_sharpe_deflated = deflated_sharpe_ratio(
                sharpe_annual=ls_sharpe_raw,
                n_obs=n_obs_eff,
                n_tests=n_tests,
                skew=0.0,  # Default: assume normal
                kurtosis=3.0,  # Default: assume normal
            )
        except Exception as e:
            logger.warning(f"Failed to compute deflated Sharpe: {e}")
            ls_sharpe_deflated = None
    
    portfolio_metrics = {
        "model_name": model_name,
        "label_col": label_col,
        "horizon_days": horizon_days,
        "ls_sharpe_raw": ls_sharpe_raw,  # B4: Renamed from ls_sharpe
        "ls_sharpe_deflated": ls_sharpe_deflated,  # B4: New deflated Sharpe
        "ls_return_mean": eval_metrics.get("ls_return_mean"),
        "ls_return_std": eval_metrics.get("ls_return_std"),
        "ic_mean": eval_metrics.get("ic_mean"),
        "ic_ir": eval_metrics.get("ic_ir"),
        "rank_ic_mean": eval_metrics.get("rank_ic_mean"),
        "rank_ic_ir": eval_metrics.get("rank_ic_ir"),
        "n_tests": n_tests,  # B4: Number of tests (conservative: 1 for single model)
        "n_samples": n_samples,  # B4: Number of observations
    }
    
    portfolio_metrics_file = output_dir / f"ml_portfolio_metrics_{model_type}_{label_col}.csv"
    portfolio_metrics_df = pd.DataFrame([portfolio_metrics])
    portfolio_metrics_df.to_csv(portfolio_metrics_file, index=False)
    logger.info(f"Saved portfolio metrics to {portfolio_metrics_file}")
    
    # 4. Markdown report
    report_file = output_dir / f"ml_validation_report_{model_type}_{label_col}.md"
    
    predictions_summary = {
        "n_total": len(predictions_df),
        "n_splits": len(metrics_df),
        "y_true_range": [float(predictions_df["y_true"].min()), float(predictions_df["y_true"].max())],
        "y_pred_range": [float(predictions_df["y_pred"].min()), float(predictions_df["y_pred"].max())],
    }
    
    write_ml_validation_report(
        model_name=model_name,
        label_col=label_col,
        metrics=eval_metrics,
        predictions_summary=predictions_summary,
        output_path=report_file,
        feature_importance_df=feature_importance_df,
        permutation_importance_df=permutation_importance_df,
    )
    logger.info(f"Saved validation report to {report_file}")
    
    logger.info("ML validation completed successfully")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ML validation on factor panels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with Ridge model
  python scripts/run_ml_factor_validation.py \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge

  # With custom parameters
  python scripts/run_ml_factor_validation.py \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge \\
    --model-param alpha=0.1 \\
    --model-param max_iter=1000

  # Random Forest with time filter
  python scripts/run_ml_factor_validation.py \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type random_forest \\
    --n-splits 10 \\
    --test-start 2020-01-01 \\
    --test-end 2024-12-31
        """
    )
    
    parser.add_argument(
        "--factor-panel-file",
        type=Path,
        required=True,
        help="Path to factor panel file (Parquet or CSV) with factors and forward returns"
    )
    
    parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Name of label column (e.g., 'fwd_return_20d')"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["linear", "ridge", "lasso", "random_forest"],
        help="Model type: 'linear', 'ridge', 'lasso', or 'random_forest'"
    )
    
    parser.add_argument(
        "--model-param",
        type=str,
        action="append",
        default=[],
        help="Model hyperparameter in format 'key=value' (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits (default: 5)"
    )
    
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="Test start date (YYYY-MM-DD, optional)"
    )
    
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Test end date (YYYY-MM-DD, optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/ml_validation)"
    )
    
    args = parser.parse_args()
    
    # Parse model parameters
    model_params = parse_model_params(args.model_param) if args.model_param else None
    
    return run_ml_validation(
        factor_panel_file=args.factor_panel_file,
        label_col=args.label_col,
        model_type=args.model_type,
        model_params=model_params,
        n_splits=args.n_splits,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())

