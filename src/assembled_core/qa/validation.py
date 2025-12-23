"""Model validation engine for performance, overfitting, and data quality checks.

This module provides structured validation functions for trading models and strategies.
It aggregates validation results from performance metrics, overfitting detection, and
data quality checks into a unified validation report.

Key features:
- Performance validation (Sharpe, Max Drawdown, Trade Count)
- Overfitting detection (Deflated Sharpe Ratio)
- Data quality validation (Missing values, feature completeness)
- Aggregated full model validation
- Structured validation results with errors and warnings

Usage:
    >>> from src.assembled_core.qa.validation import run_full_model_validation
    >>>
    >>> metrics = {
    ...     "sharpe_ratio": 1.5,
    ...     "max_drawdown_pct": -15.0,
    ...     "total_trades": 100
    ... }
    >>>
    >>> result = run_full_model_validation(
    ...     model_name="trend_baseline",
    ...     metrics=metrics,
    ...     feature_df=features_df,
    ...     deflated_sharpe=1.2
    ... )
    >>>
    >>> if result.is_ok:
    ...     print("Model validation passed")
    >>> else:
    ...     print(f"Validation failed: {result.errors}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ModelValidationResult:
    """Result of model validation checks.

    Attributes:
        model_name: Name/ID of the validated model
        is_ok: True if validation passed (no critical errors), False otherwise
        errors: List of error messages (critical issues that cause validation to fail)
        warnings: List of warning messages (non-critical issues)
        metadata: Optional dictionary with additional validation details
    """

    model_name: str
    is_ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_performance(
    metrics: dict[str, float | None],
    min_sharpe: float = 1.0,
    max_drawdown: float = 0.25,
    min_trades: int = 30,
) -> ModelValidationResult:
    """Validate model performance metrics against thresholds.

    Checks that key performance metrics meet minimum quality standards:
    - Sharpe ratio >= min_sharpe
    - Max drawdown (absolute) <= max_drawdown (as positive fraction)
    - Total trades >= min_trades

    Args:
        metrics: Dictionary with performance metrics. Expected keys:
            - sharpe_ratio: Sharpe ratio (annualized, optional)
            - max_drawdown_pct: Maximum drawdown in percent (negative value, e.g., -15.0)
            - total_trades: Total number of trades (optional)
            - max_drawdown: Maximum drawdown absolute (optional, alternative to max_drawdown_pct)
            Other keys are ignored
        min_sharpe: Minimum Sharpe ratio threshold (default: 1.0)
        max_drawdown: Maximum drawdown threshold as positive fraction (default: 0.25 = 25%)
            Note: metrics['max_drawdown_pct'] should be negative (e.g., -20.0), so we check abs() <= max_drawdown
        min_trades: Minimum number of trades required (default: 30)

    Returns:
        ModelValidationResult with is_ok=True if all checks pass, False otherwise

    Example:
        >>> metrics = {
        ...     "sharpe_ratio": 1.5,
        ...     "max_drawdown_pct": -15.0,
        ...     "total_trades": 100
        ... }
        >>> result = validate_performance(metrics, min_sharpe=1.0, max_drawdown=0.20)
        >>> assert result.is_ok is True
        >>> assert len(result.errors) == 0
    """
    errors = []
    warnings = []
    metadata = {}

    # Check Sharpe ratio
    sharpe = metrics.get("sharpe_ratio")
    if sharpe is None:
        warnings.append(
            "Sharpe ratio not available (None) - cannot validate Sharpe threshold"
        )
        metadata["sharpe_checked"] = False
    else:
        metadata["sharpe_checked"] = True
        metadata["sharpe_ratio"] = sharpe
        metadata["min_sharpe_threshold"] = min_sharpe
        if sharpe < min_sharpe:
            errors.append(
                f"Sharpe ratio {sharpe:.4f} is below minimum threshold {min_sharpe:.2f}"
            )

    # Check max drawdown (prefer max_drawdown_pct if available, else max_drawdown)
    max_dd_pct = metrics.get("max_drawdown_pct")
    max_dd_abs = metrics.get("max_drawdown")

    if max_dd_pct is not None:
        # max_drawdown_pct is negative (e.g., -15.0 for -15%)
        max_dd_fraction = abs(max_dd_pct) / 100.0  # Convert to fraction (0.15)
        metadata["max_drawdown_pct"] = max_dd_pct
        metadata["max_drawdown_fraction"] = max_dd_fraction
        if max_dd_fraction > max_drawdown:
            errors.append(
                f"Max drawdown {max_dd_pct:.2f}% ({max_dd_fraction:.2%}) exceeds threshold {max_drawdown:.2%}"
            )
    elif max_dd_abs is not None:
        # Try to compute percentage if we have start_capital
        if "start_capital" in metrics:
            start_capital = metrics.get("start_capital")
            if start_capital is not None and start_capital > 0:
                max_dd_fraction = abs(max_dd_abs) / start_capital
                metadata["max_drawdown_abs"] = max_dd_abs
                metadata["max_drawdown_fraction"] = max_dd_fraction
                if max_dd_fraction > max_drawdown:
                    errors.append(
                        f"Max drawdown {max_dd_abs:.2f} ({max_dd_fraction:.2%}) exceeds threshold {max_drawdown:.2%}"
                    )
            else:
                warnings.append(
                    "start_capital is zero - cannot validate max drawdown as fraction"
                )
        else:
            warnings.append(
                "Max drawdown (absolute) is available but start_capital is missing - "
                "cannot validate max drawdown as fraction"
            )
    else:
        warnings.append(
            "Max drawdown not available (max_drawdown_pct or max_drawdown missing) - "
            "cannot validate drawdown threshold"
        )
        metadata["drawdown_checked"] = False

    # Check trade count
    total_trades = metrics.get("total_trades")
    if total_trades is None:
        warnings.append(
            "Total trades count not available (None) - cannot validate minimum trades"
        )
        metadata["trades_checked"] = False
    else:
        metadata["trades_checked"] = True
        metadata["total_trades"] = total_trades
        metadata["min_trades_threshold"] = min_trades
        if total_trades < min_trades:
            errors.append(
                f"Total trades {total_trades} is below minimum threshold {min_trades}"
            )

    is_ok = len(errors) == 0

    return ModelValidationResult(
        model_name="performance_validation",
        is_ok=is_ok,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def validate_overfitting(
    deflated_sharpe: float | None, threshold: float = 0.5
) -> ModelValidationResult:
    """Validate that model is not overfitted using deflated Sharpe ratio.

    The deflated Sharpe ratio adjusts the Sharpe ratio for the number of trials/parameters
    to reduce the effect of multiple testing. A low deflated Sharpe suggests overfitting.

    Args:
        deflated_sharpe: Deflated Sharpe ratio (None if not computed)
        threshold: Minimum deflated Sharpe ratio threshold (default: 0.5)

    Returns:
        ModelValidationResult:
        - is_ok=True if deflated_sharpe >= threshold or deflated_sharpe is None (warning only)
        - is_ok=False if deflated_sharpe < threshold (indicates overfitting)

    Example:
        >>> # Good: Deflated Sharpe above threshold
        >>> result = validate_overfitting(deflated_sharpe=0.8, threshold=0.5)
        >>> assert result.is_ok is True
        >>>
        >>> # Overfitted: Deflated Sharpe below threshold
        >>> result = validate_overfitting(deflated_sharpe=0.3, threshold=0.5)
        >>> assert result.is_ok is False
        >>> assert "overfitting" in result.errors[0].lower()
        >>>
        >>> # Not computed: Warning only, no hard failure
        >>> result = validate_overfitting(deflated_sharpe=None, threshold=0.5)
        >>> assert result.is_ok is True  # Warning, not error
        >>> assert len(result.warnings) > 0
    """
    errors = []
    warnings = []
    metadata = {"threshold": threshold}

    if deflated_sharpe is None:
        warnings.append(
            "Deflated Sharpe ratio not available (None) - cannot validate overfitting. "
            "Consider computing deflated Sharpe for robust validation."
        )
        metadata["deflated_sharpe_checked"] = False
        is_ok = True  # Warning only, not a hard failure
    else:
        metadata["deflated_sharpe_checked"] = True
        metadata["deflated_sharpe"] = deflated_sharpe
        if deflated_sharpe < threshold:
            errors.append(
                f"Deflated Sharpe ratio {deflated_sharpe:.4f} is below threshold {threshold:.2f} "
                "- model may be overfitted"
            )
            is_ok = False
        else:
            is_ok = True

    return ModelValidationResult(
        model_name="overfitting_validation",
        is_ok=is_ok,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def validate_data_quality(
    feature_df: pd.DataFrame, max_missing_fraction: float = 0.05
) -> ModelValidationResult:
    """Validate data quality of feature DataFrame.

    Checks that no column has more than max_missing_fraction of missing values (NaNs).
    This ensures that features are sufficiently populated for reliable model performance.

    Args:
        feature_df: DataFrame with feature columns to validate
        max_missing_fraction: Maximum allowed fraction of missing values per column (default: 0.05 = 5%)

    Returns:
        ModelValidationResult:
        - is_ok=True if all columns have <= max_missing_fraction missing values
        - is_ok=False if any column exceeds the threshold

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Good: No missing values
        >>> df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> result = validate_data_quality(df)
        >>> assert result.is_ok is True
        >>>
        >>> # Bad: Column with > 5% missing
        >>> df = pd.DataFrame({
        ...     "feature1": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],
        ...     "feature2": [4, 5, 6, 7, 8, 9, 10]
        ... })
        >>> result = validate_data_quality(df, max_missing_fraction=0.05)
        >>> assert result.is_ok is False
        >>> assert "feature1" in result.errors[0]
    """
    errors = []
    warnings = []
    metadata = {
        "max_missing_fraction": max_missing_fraction,
        "total_rows": len(feature_df),
        "total_columns": len(feature_df.columns),
    }

    if feature_df.empty:
        errors.append("Feature DataFrame is empty - cannot validate data quality")
        return ModelValidationResult(
            model_name="data_quality_validation",
            is_ok=False,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    problematic_columns = []
    column_quality = {}

    for col in feature_df.columns:
        missing_count = feature_df[col].isna().sum()
        missing_fraction = missing_count / len(feature_df)

        column_quality[col] = {
            "missing_count": int(missing_count),
            "missing_fraction": float(missing_fraction),
        }

        if missing_fraction > max_missing_fraction:
            problematic_columns.append(col)
            errors.append(
                f"Column '{col}' has {missing_fraction:.2%} missing values "
                f"({missing_count}/{len(feature_df)}), exceeds threshold {max_missing_fraction:.2%}"
            )
        elif missing_fraction > 0:
            warnings.append(
                f"Column '{col}' has {missing_fraction:.2%} missing values "
                f"({missing_count}/{len(feature_df)}) - below threshold but not zero"
            )

    metadata["column_quality"] = column_quality
    metadata["problematic_columns"] = problematic_columns

    is_ok = len(errors) == 0

    return ModelValidationResult(
        model_name="data_quality_validation",
        is_ok=is_ok,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def run_full_model_validation(
    model_name: str,
    metrics: dict[str, float | None],
    feature_df: pd.DataFrame | None = None,
    deflated_sharpe: float | None = None,
    config: dict[str, Any] | None = None,
) -> ModelValidationResult:
    """Run full model validation aggregating all validation checks.

    This function runs all validation checks (performance, overfitting, data quality)
    and aggregates the results into a single validation report.

    Args:
        model_name: Name/ID of the model being validated
        metrics: Dictionary with performance metrics (see validate_performance for expected keys)
        feature_df: Optional DataFrame with features for data quality validation
        deflated_sharpe: Optional deflated Sharpe ratio for overfitting validation
        config: Optional configuration dictionary with validation parameters:
            - min_sharpe: Minimum Sharpe ratio (default: 1.0)
            - max_drawdown: Maximum drawdown as fraction (default: 0.25)
            - min_trades: Minimum number of trades (default: 30)
            - overfitting_threshold: Deflated Sharpe threshold (default: 0.5)
            - max_missing_fraction: Maximum missing fraction for data quality (default: 0.05)

    Returns:
        ModelValidationResult with aggregated errors and warnings from all checks

    Example:
        >>> import pandas as pd
        >>>
        >>> metrics = {
        ...     "sharpe_ratio": 1.5,
        ...     "max_drawdown_pct": -15.0,
        ...     "total_trades": 100
        ... }
        >>>
        >>> features = pd.DataFrame({
        ...     "feature1": [1, 2, 3],
        ...     "feature2": [4, 5, 6]
        ... })
        >>>
        >>> result = run_full_model_validation(
        ...     model_name="trend_baseline",
        ...     metrics=metrics,
        ...     feature_df=features,
        ...     deflated_sharpe=0.8
        ... )
        >>>
        >>> if result.is_ok:
        ...     print(f"Model {result.model_name} validation passed")
        >>> else:
        ...     print(f"Validation failed: {', '.join(result.errors)}")
    """
    if config is None:
        config = {}

    # Extract config parameters with defaults
    min_sharpe = config.get("min_sharpe", 1.0)
    max_drawdown = config.get("max_drawdown", 0.25)
    min_trades = config.get("min_trades", 30)
    overfitting_threshold = config.get("overfitting_threshold", 0.5)
    max_missing_fraction = config.get("max_missing_fraction", 0.05)

    # Run individual validations
    all_errors = []
    all_warnings = []
    validation_details = {}

    # 1. Performance validation
    perf_result = validate_performance(
        metrics=metrics,
        min_sharpe=min_sharpe,
        max_drawdown=max_drawdown,
        min_trades=min_trades,
    )
    all_errors.extend(perf_result.errors)
    all_warnings.extend(perf_result.warnings)
    validation_details["performance"] = {
        "is_ok": perf_result.is_ok,
        "metadata": perf_result.metadata,
    }

    # 2. Overfitting validation
    overfitting_result = validate_overfitting(
        deflated_sharpe=deflated_sharpe, threshold=overfitting_threshold
    )
    all_errors.extend(overfitting_result.errors)
    all_warnings.extend(overfitting_result.warnings)
    validation_details["overfitting"] = {
        "is_ok": overfitting_result.is_ok,
        "metadata": overfitting_result.metadata,
    }

    # 3. Data quality validation (if feature_df provided)
    if feature_df is not None:
        data_quality_result = validate_data_quality(
            feature_df=feature_df, max_missing_fraction=max_missing_fraction
        )
        all_errors.extend(data_quality_result.errors)
        all_warnings.extend(data_quality_result.warnings)
        validation_details["data_quality"] = {
            "is_ok": data_quality_result.is_ok,
            "metadata": data_quality_result.metadata,
        }
    else:
        all_warnings.append(
            "Feature DataFrame not provided - skipping data quality validation"
        )
        validation_details["data_quality"] = {
            "is_ok": None,
            "metadata": {"skipped": True, "reason": "feature_df not provided"},
        }

    # Aggregate result: is_ok = True only if all checks passed
    is_ok = len(all_errors) == 0

    # Add config to metadata for traceability
    metadata = {
        "validation_config": config,
        "validation_details": validation_details,
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
    }

    return ModelValidationResult(
        model_name=model_name,
        is_ok=is_ok,
        errors=all_errors,
        warnings=all_warnings,
        metadata=metadata,
    )
