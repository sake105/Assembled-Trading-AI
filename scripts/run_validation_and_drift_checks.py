# scripts/run_validation_and_drift_checks.py
"""Run validation and drift checks on ML datasets.

This script loads the latest ML dataset and compares it with a reference dataset
to perform model validation and drift detection checks. Results are written to
a Markdown summary report.

Usage:
    python scripts/run_validation_and_drift_checks.py
    python scripts/run_validation_and_drift_checks.py --current-dataset output/ml_datasets/trend_baseline_1d.parquet
    python scripts/run_validation_and_drift_checks.py --reference-dataset output/ml_datasets/trend_baseline_1d_reference.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.qa.drift_detection import (
    detect_feature_drift,
    detect_label_drift,
)
from src.assembled_core.qa.validation import run_full_model_validation

logger = setup_logging(level="INFO")


def find_latest_ml_dataset(ml_datasets_dir: Path) -> Path | None:
    """Find the latest ML dataset file in the directory.

    Args:
        ml_datasets_dir: Directory containing ML dataset files

    Returns:
        Path to the latest dataset file (by modification time), or None if no files found
    """
    if not ml_datasets_dir.exists():
        return None

    # Find all parquet files
    dataset_files = list(ml_datasets_dir.glob("*.parquet"))

    if not dataset_files:
        return None

    # Return the most recently modified file
    return max(dataset_files, key=lambda p: p.stat().st_mtime)


def load_ml_dataset(file_path: Path) -> pd.DataFrame:
    """Load ML dataset from parquet file.

    Args:
        file_path: Path to parquet file

    Returns:
        DataFrame with ML dataset

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read or is empty
    """
    if not file_path.exists():
        raise FileNotFoundError(f"ML dataset file not found: {file_path}")

    logger.info(f"Loading ML dataset from: {file_path}")
    df = pd.read_parquet(file_path)

    if df.empty:
        raise ValueError(f"ML dataset is empty: {file_path}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def extract_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature columns from ML dataset.

    Excludes non-feature columns like 'label', 'timestamp', 'symbol', etc.

    Args:
        df: ML dataset DataFrame

    Returns:
        DataFrame with only feature columns
    """
    # Common non-feature columns to exclude
    non_feature_cols = {
        "label",
        "open_time",
        "timestamp",
        "symbol",
        "open_price",
        "close_time",
        "pnl_pct",
        "horizon_days",
    }

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    if not feature_cols:
        logger.warning("No feature columns found in dataset")
        return pd.DataFrame()

    return df[feature_cols]


def extract_labels_from_dataset(df: pd.DataFrame) -> pd.Series | None:
    """Extract labels from ML dataset.

    Args:
        df: ML dataset DataFrame

    Returns:
        Series with labels, or None if 'label' column doesn't exist
    """
    if "label" not in df.columns:
        logger.warning("No 'label' column found in dataset")
        return None

    return df["label"]


def load_performance_metrics_from_files(
    output_dir: Path, freq: str = "1d"
) -> dict[str, float | None] | None:
    """Load performance metrics from run manifest or portfolio report.

    Args:
        output_dir: Output directory to search for manifest/report files
        freq: Trading frequency ("1d" or "5min")

    Returns:
        Dictionary with performance metrics, or None if not found
    """
    import json

    # Try to load from run manifest first
    manifest_path = output_dir / f"run_manifest_{freq}.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if "qa_metrics" in manifest and manifest["qa_metrics"]:
                metrics_dict = manifest["qa_metrics"]
                logger.info(f"Loaded metrics from manifest: {manifest_path}")
                return {
                    "sharpe_ratio": metrics_dict.get("sharpe_ratio"),
                    "max_drawdown_pct": metrics_dict.get("max_drawdown_pct"),
                    "total_trades": metrics_dict.get("total_trades"),
                    "cagr": metrics_dict.get("cagr"),
                    "start_capital": metrics_dict.get("start_capital"),
                    "total_return": metrics_dict.get("total_return"),
                }
        except Exception as e:
            logger.warning(f"Failed to load metrics from manifest: {e}")

    return None


def run_validation_checks(
    model_name: str,
    current_dataset: pd.DataFrame,
    performance_metrics: dict[str, float | None] | None = None,
    deflated_sharpe: float | None = None,
    config: dict | None = None,
) -> dict:
    """Run validation checks on the current dataset.

    Args:
        model_name: Name of the model being validated
        current_dataset: Current ML dataset DataFrame
        performance_metrics: Optional performance metrics dictionary
        deflated_sharpe: Optional deflated Sharpe ratio
        config: Optional validation configuration

    Returns:
        Dictionary with validation results
    """
    logger.info("Running validation checks...")

    # Extract features for data quality validation
    feature_df = extract_features_from_dataset(current_dataset)

    # Use provided metrics or create empty dict
    if performance_metrics is None:
        performance_metrics = {}
        logger.warning("No performance metrics provided - validation will be limited")

    # Run full model validation
    validation_result = run_full_model_validation(
        model_name=model_name,
        metrics=performance_metrics,
        feature_df=feature_df if not feature_df.empty else None,
        deflated_sharpe=deflated_sharpe,
        config=config,
    )

    return {
        "model_name": validation_result.model_name,
        "is_ok": validation_result.is_ok,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
        "metadata": validation_result.metadata,
    }


def run_drift_checks(
    current_dataset: pd.DataFrame,
    reference_dataset: pd.DataFrame | None = None,
    psi_threshold: float = 0.2,
    severe_threshold: float = 0.3,
) -> dict:
    """Run drift detection checks.

    Args:
        current_dataset: Current ML dataset DataFrame
        reference_dataset: Reference dataset for comparison (if None, uses sample data)
        psi_threshold: PSI threshold for moderate drift (default: 0.2)
        severe_threshold: PSI threshold for severe drift (default: 0.3)

    Returns:
        Dictionary with drift detection results
    """
    logger.info("Running drift detection checks...")

    results = {"feature_drift": None, "label_drift": None, "performance_drift": None}

    # 1. Feature drift detection
    if reference_dataset is not None:
        current_features = extract_features_from_dataset(current_dataset)
        reference_features = extract_features_from_dataset(reference_dataset)

        if not current_features.empty and not reference_features.empty:
            logger.info("Detecting feature drift...")
            feature_drift_df = detect_feature_drift(
                base_df=reference_features,
                current_df=current_features,
                psi_threshold=psi_threshold,
                severe_threshold=severe_threshold,
            )

            # Get top features with drift (sorted by PSI, descending)
            if not feature_drift_df.empty:
                drift_features = feature_drift_df[
                    feature_drift_df["drift_flag"].isin(["MODERATE", "SEVERE"])
                ].sort_values("psi", ascending=False)

                results["feature_drift"] = {
                    "total_features_checked": len(feature_drift_df),
                    "features_with_drift": drift_features.to_dict("records")
                    if not drift_features.empty
                    else [],
                    "overall_severity": "SEVERE"
                    if (feature_drift_df["drift_flag"] == "SEVERE").any()
                    else "MODERATE"
                    if (feature_drift_df["drift_flag"] == "MODERATE").any()
                    else "NONE",
                }
            else:
                results["feature_drift"] = {
                    "total_features_checked": 0,
                    "features_with_drift": [],
                    "overall_severity": "NONE",
                }
        else:
            logger.warning("Cannot detect feature drift: empty feature DataFrames")

    # 2. Label drift detection
    current_labels = extract_labels_from_dataset(current_dataset)
    reference_labels = (
        extract_labels_from_dataset(reference_dataset)
        if reference_dataset is not None
        else None
    )

    if current_labels is not None and reference_labels is not None:
        logger.info("Detecting label drift...")
        label_drift = detect_label_drift(
            base_labels=reference_labels,
            current_labels=current_labels,
            psi_threshold=psi_threshold,
        )
        results["label_drift"] = label_drift
    else:
        logger.warning(
            "Cannot detect label drift: labels not available in one or both datasets"
        )

    # 3. Performance drift (requires equity curve - not available from ML dataset alone)
    # This would need to be loaded separately or computed from other sources
    logger.info(
        "Performance drift detection skipped (requires equity curve, not available from ML dataset)"
    )

    return results


def write_summary_report(
    output_path: Path,
    model_name: str,
    current_dataset_path: Path | None,
    reference_dataset_path: Path | None,
    validation_results: dict,
    drift_results: dict,
    timestamp: datetime,
) -> None:
    """Write Markdown summary report.

    Args:
        output_path: Path to output Markdown file
        model_name: Name of the model
        current_dataset_path: Path to current dataset (for reference)
        reference_dataset_path: Path to reference dataset (for reference)
        validation_results: Validation results dictionary
        drift_results: Drift detection results dictionary
        timestamp: Timestamp of the analysis
    """
    logger.info(f"Writing summary report to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Validation and Drift Checks Summary")
    lines.append("")
    lines.append(f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Model:** {model_name}")
    lines.append("")

    # Dataset information
    lines.append("## Dataset Information")
    lines.append("")
    if current_dataset_path:
        lines.append(f"- **Current Dataset:** `{current_dataset_path}`")
    else:
        lines.append("- **Current Dataset:** Not specified")

    if reference_dataset_path:
        lines.append(f"- **Reference Dataset:** `{reference_dataset_path}`")
    else:
        lines.append(
            "- **Reference Dataset:** Not specified (using sample/reference data)"
        )
    lines.append("")

    # Validation Results
    lines.append("## Validation Result")
    lines.append("")
    is_ok = validation_results.get("is_ok", False)
    status_text = "[PASSED]" if is_ok else "[FAILED]"
    lines.append(f"**Status:** {status_text}")
    lines.append("")

    errors = validation_results.get("errors", [])
    warnings = validation_results.get("warnings", [])

    if errors:
        lines.append("### Errors")
        lines.append("")
        for error in errors:
            lines.append(f"- [ERROR] {error}")
        lines.append("")

    if warnings:
        lines.append("### Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- [WARNING] {warning}")
        lines.append("")

    if not errors and not warnings:
        lines.append("[OK] No errors or warnings")
        lines.append("")

    # Validation metadata
    metadata = validation_results.get("metadata", {})
    if metadata:
        lines.append("### Validation Details")
        lines.append("")
        validation_details = metadata.get("validation_details", {})

        if "performance" in validation_details:
            perf_meta = validation_details["performance"].get("metadata", {})
            if perf_meta.get("sharpe_checked"):
                sharpe = perf_meta.get("sharpe_ratio")
                min_sharpe = perf_meta.get("min_sharpe_threshold", "N/A")
                lines.append(
                    f"- **Sharpe Ratio:** {sharpe:.4f if sharpe is not None else 'N/A'} (threshold: {min_sharpe})"
                )

        if "data_quality" in validation_details:
            data_meta = validation_details["data_quality"].get("metadata", {})
            total_cols = data_meta.get("total_columns", 0)
            total_rows = data_meta.get("total_rows", 0)
            lines.append(f"- **Dataset Size:** {total_rows} rows, {total_cols} columns")

        lines.append("")

    # Drift Status
    lines.append("## Drift Status")
    lines.append("")

    # Feature drift
    feature_drift = drift_results.get("feature_drift")
    if feature_drift:
        overall_severity = feature_drift.get("overall_severity", "NONE")
        severity_prefix = (
            "[SEVERE]"
            if overall_severity == "SEVERE"
            else "[MODERATE]"
            if overall_severity == "MODERATE"
            else "[OK]"
        )
        lines.append(f"**Feature Drift:** {severity_prefix} {overall_severity}")
        lines.append("")

        total_checked = feature_drift.get("total_features_checked", 0)
        features_with_drift = feature_drift.get("features_with_drift", [])
        lines.append(f"- **Features Checked:** {total_checked}")
        lines.append(f"- **Features with Drift:** {len(features_with_drift)}")
        lines.append("")

        if features_with_drift:
            lines.append("### Top Features with Drift")
            lines.append("")
            lines.append("| Feature | PSI | Severity |")
            lines.append("|---------|-----|----------|")
            for feat in features_with_drift[:10]:  # Top 10
                lines.append(
                    f"| {feat['feature']} | {feat['psi']:.4f} | {feat['drift_flag']} |"
                )
            lines.append("")
    else:
        lines.append(
            "**Feature Drift:** [WARNING] Not available (reference dataset missing)"
        )
        lines.append("")

    # Label drift
    label_drift = drift_results.get("label_drift")
    if label_drift:
        drift_detected = label_drift.get("drift_detected", False)
        drift_severity = label_drift.get("drift_severity", "NONE")
        psi = label_drift.get("psi", 0.0)

        severity_prefix = (
            "[SEVERE]"
            if drift_severity == "SEVERE"
            else "[MODERATE]"
            if drift_severity == "MODERATE"
            else "[OK]"
        )
        lines.append(
            f"**Label Drift:** {severity_prefix} {drift_severity} (PSI: {psi:.4f})"
        )
        lines.append("")

        if drift_detected:
            base_mean = label_drift.get("base_mean")
            current_mean = label_drift.get("current_mean")
            mean_shift = label_drift.get("mean_shift", 0.0)
            lines.append(
                f"- **Base Mean:** {base_mean:.4f if base_mean is not None else 'N/A'}"
            )
            lines.append(
                f"- **Current Mean:** {current_mean:.4f if current_mean is not None else 'N/A'}"
            )
            lines.append(
                f"- **Mean Shift:** {mean_shift:+.4f if mean_shift is not None else 'N/A'}"
            )
            lines.append("")
    else:
        lines.append(
            "**Label Drift:** [WARNING] Not available (labels or reference dataset missing)"
        )
        lines.append("")

    # Performance drift
    performance_drift = drift_results.get("performance_drift")
    if performance_drift:
        lines.append("**Performance Drift:** (not computed from ML dataset)")
        lines.append("")
    else:
        lines.append(
            "**Performance Drift:** [WARNING] Not available (requires equity curve)"
        )
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Report generated by run_validation_and_drift_checks.py at {timestamp.isoformat()}*"
    )

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Summary report written: {output_path}")


def run_validation_and_drift_checks(
    current_dataset_path: Path | None = None,
    reference_dataset_path: Path | None = None,
    output_path: Path | None = None,
    model_name: str = "default",
    performance_metrics: dict[str, float | None] | None = None,
    deflated_sharpe: float | None = None,
    validation_config: dict | None = None,
) -> Path:
    """Run validation and drift checks on ML datasets.

    Args:
        current_dataset_path: Path to current ML dataset (if None, finds latest)
        reference_dataset_path: Path to reference dataset (if None, uses sample data or skips comparison)
        output_path: Path to output Markdown file (if None, uses default)
        model_name: Name of the model being validated
        performance_metrics: Optional performance metrics dictionary
        deflated_sharpe: Optional deflated Sharpe ratio
        validation_config: Optional validation configuration

    Returns:
        Path to generated summary report

    Raises:
        FileNotFoundError: If dataset files cannot be found
        ValueError: If datasets are invalid
    """
    settings = get_settings()

    # Determine paths
    ml_datasets_dir = settings.output_dir / "ml_datasets"

    if current_dataset_path is None:
        logger.info("No current dataset specified, finding latest...")
        current_dataset_path = find_latest_ml_dataset(ml_datasets_dir)
        if current_dataset_path is None:
            raise FileNotFoundError(
                f"No ML dataset found in {ml_datasets_dir}. "
                "Please specify --current-dataset or create a dataset first."
            )

    current_dataset_path = Path(current_dataset_path)

    if output_path is None:
        monitoring_dir = settings.output_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        output_path = monitoring_dir / "validation_drift_summary.md"
    else:
        output_path = Path(output_path)

    # Load datasets
    logger.info("=" * 60)
    logger.info("Validation and Drift Checks")
    logger.info("=" * 60)

    current_dataset = load_ml_dataset(current_dataset_path)

    reference_dataset = None
    if reference_dataset_path:
        reference_dataset_path = Path(reference_dataset_path)
        if reference_dataset_path.exists():
            reference_dataset = load_ml_dataset(reference_dataset_path)
        else:
            logger.warning(
                f"Reference dataset not found: {reference_dataset_path}, skipping drift comparison"
            )

    # Extract model name from dataset path if not provided
    if model_name == "default":
        model_name = current_dataset_path.stem

    # Try to load performance metrics from files if not provided
    if performance_metrics is None:
        logger.info("No performance metrics provided, attempting to load from files...")
        # Extract frequency from dataset filename if possible (e.g., trend_baseline_1d.parquet)
        freq = "1d"  # Default
        if "_1d" in current_dataset_path.stem:
            freq = "1d"
        elif "_5min" in current_dataset_path.stem:
            freq = "5min"

        performance_metrics = load_performance_metrics_from_files(
            settings.output_dir, freq=freq
        )
        if performance_metrics is None:
            logger.warning(
                "Could not load performance metrics from files - validation will be limited"
            )
            performance_metrics = {}

    # Run validation checks
    validation_results = run_validation_checks(
        model_name=model_name,
        current_dataset=current_dataset,
        performance_metrics=performance_metrics,
        deflated_sharpe=deflated_sharpe,
        config=validation_config,
    )

    # Run drift checks
    drift_results = run_drift_checks(
        current_dataset=current_dataset,
        reference_dataset=reference_dataset,
        psi_threshold=validation_config.get("psi_threshold", 0.2)
        if validation_config
        else 0.2,
        severe_threshold=validation_config.get("severe_threshold", 0.3)
        if validation_config
        else 0.3,
    )

    # Write summary report
    timestamp = datetime.utcnow()
    write_summary_report(
        output_path=output_path,
        model_name=model_name,
        current_dataset_path=current_dataset_path,
        reference_dataset_path=reference_dataset_path,
        validation_results=validation_results,
        drift_results=drift_results,
        timestamp=timestamp,
    )

    logger.info("=" * 60)
    logger.info("Validation and Drift Checks Complete")
    logger.info(f"Report: {output_path}")
    logger.info("=" * 60)

    return output_path


def main() -> int:
    """Main entry point for validation and drift checks script."""
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Run validation and drift checks on ML datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Use latest dataset from {settings.output_dir / "ml_datasets"}
  python scripts/run_validation_and_drift_checks.py

  # Specify current and reference datasets
  python scripts/run_validation_and_drift_checks.py \\
      --current-dataset output/ml_datasets/trend_baseline_1d.parquet \\
      --reference-dataset output/ml_datasets/trend_baseline_1d_reference.parquet

  # Custom output path
  python scripts/run_validation_and_drift_checks.py \\
      --output output/monitoring/custom_summary.md
        """,
    )

    parser.add_argument(
        "--current-dataset",
        type=Path,
        default=None,
        help="Path to current ML dataset (default: latest in output/ml_datasets)",
    )

    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=None,
        help="Path to reference ML dataset (default: None, skips drift comparison)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Path to output Markdown file (default: {settings.output_dir / 'monitoring' / 'validation_drift_summary.md'})",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        help="Name of the model being validated (default: derived from dataset filename)",
    )

    args = parser.parse_args()

    try:
        report_path = run_validation_and_drift_checks(
            current_dataset_path=args.current_dataset,
            reference_dataset_path=args.reference_dataset,
            output_path=args.output,
            model_name=args.model_name,
        )
        print("\n[OK] Validation and drift checks completed successfully")
        print(f"Report: {report_path}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
