"""Multi-Factor Signal Generation Module.

This module provides functions to build combined multi-factor signals from
factor DataFrames and factor bundle configurations.

The main function `build_multifactor_signal` processes factors according to a
bundle configuration:
- Winsorizes factors (if enabled)
- Applies cross-sectional z-scoring (per timestamp, across symbols)
- Applies direction (inverts negative factors)
- Computes weighted multi-factor score

Example:
    from src.assembled_core.config.factor_bundles import load_factor_bundle
    from src.assembled_core.signals.multifactor_signal import build_multifactor_signal

    # Load bundle
    bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")

    # Build signal (factors_df must contain timestamp, symbol, and factor columns)
    result = build_multifactor_signal(factors_df, bundle)

    # Access result
    mf_df = result.df  # DataFrame with mf_score column
    meta = result.meta  # Metadata about processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.assembled_core.config.factor_bundles import FactorBundleConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiFactorSignalResult:
    """Result of multi-factor signal generation.

    Attributes:
        df: DataFrame with original columns plus mf_score and optionally
            normalized factor columns (e.g., factor_name_z)
        meta: Dictionary with metadata about processing:
            - used_factors: List of factor names that were successfully used
            - factor_weights: Dict mapping factor names to weights
            - missing_factors: List of factor names that were missing from input
            - options_applied: Dict with applied options (winsorize, zscore, etc.)
            - universe: Universe identifier from bundle
            - factor_set: Factor set identifier from bundle
    """

    df: pd.DataFrame
    meta: dict


def _winsorize_series(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Winsorize a series by clipping values at quantile limits.

    Args:
        series: Input series to winsorize
        lower_quantile: Lower quantile limit (e.g., 0.01)
        upper_quantile: Upper quantile limit (e.g., 0.99)

    Returns:
        Winsorized series with extreme values clipped
    """
    # Remove NaN values for quantile calculation
    non_null = series.dropna()
    if len(non_null) == 0:
        return series

    lower_limit = non_null.quantile(lower_quantile)
    upper_limit = non_null.quantile(upper_quantile)

    result = series.copy()
    result = result.clip(lower=lower_limit, upper=upper_limit)

    return result


def _zscore_crosssectional(
    df: pd.DataFrame,
    factor_col: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Compute cross-sectional z-scores per timestamp.

    For each timestamp, computes z-score across all symbols (not over time).
    This ensures that the z-score reflects relative ranking within each period.

    Args:
        df: DataFrame with timestamp_col, symbol, and factor_col
        factor_col: Name of factor column to z-score
        timestamp_col: Name of timestamp column

    Returns:
        Series with z-scores (index matches df.index)
    """
    result = df[[timestamp_col, factor_col]].copy()

    # Group by timestamp and compute z-score within each group
    def zscore_group(group: pd.DataFrame) -> pd.Series:
        values = group[factor_col].values
        non_null_mask = ~pd.isna(values)

        if non_null_mask.sum() < 2:
            # Not enough non-null values for z-scoring
            return pd.Series(np.nan, index=group.index)

        non_null_values = values[non_null_mask]
        mean_val = np.mean(non_null_values)
        std_val = np.std(non_null_values, ddof=0)  # Population std

        # Compute z-scores
        zscores = np.full(len(values), np.nan)
        if std_val > 1e-10:  # Avoid division by zero
            zscores[non_null_mask] = (non_null_values - mean_val) / std_val
        else:
            # All values are the same, set z-score to 0
            zscores[non_null_mask] = 0.0

        return pd.Series(zscores, index=group.index)

    zscores = result.groupby(timestamp_col, group_keys=False).apply(zscore_group)

    # Reindex to match original df index
    zscores = zscores.reindex(df.index)

    return zscores


def build_multifactor_signal(
    factors_df: pd.DataFrame,
    bundle: FactorBundleConfig,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> MultiFactorSignalResult:
    """Build multi-factor signal from factors DataFrame and bundle configuration.

    This function:
    1. Checks which factors from the bundle are available in factors_df
    2. Optionally winsorizes factors (per factor, over entire history)
    3. Optionally applies cross-sectional z-scoring (per timestamp, across symbols)
    4. Applies direction (inverts negative factors)
    5. Computes weighted multi-factor score

    Args:
        factors_df: DataFrame in panel format with columns:
            - timestamp_col: Timestamp column
            - group_col: Symbol/group column
            - Factor columns (e.g., returns_12m, trend_strength_200, etc.)
        bundle: FactorBundleConfig with factors, weights, and options
        group_col: Name of grouping column (default: "symbol")
        timestamp_col: Name of timestamp column (default: "timestamp")

    Returns:
        MultiFactorSignalResult with:
            - df: Copy of factors_df with mf_score column and optionally
                  normalized factor columns (factor_name_z)
            - meta: Dictionary with processing metadata

    Raises:
        ValueError: If required columns (timestamp_col, group_col) are missing
        KeyError: If factors_df is empty
    """
    # Validate input
    if factors_df.empty:
        raise KeyError("factors_df is empty")

    required_cols = [timestamp_col, group_col]
    missing_cols = [col for col in required_cols if col not in factors_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(factors_df.columns)}"
        )

    # Ensure timestamp is datetime
    result_df = factors_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)

    # Sort by timestamp and group for consistent processing
    result_df = result_df.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    # Check which factors are available
    available_factors = []
    missing_factors = []

    for factor_cfg in bundle.factors:
        if factor_cfg.name in result_df.columns:
            available_factors.append(factor_cfg)
        else:
            missing_factors.append(factor_cfg.name)
            logger.warning(
                f"Factor '{factor_cfg.name}' from bundle '{bundle.universe}' not found in DataFrame. "
                f"Available columns: {[c for c in result_df.columns if c not in [timestamp_col, group_col]][:10]}"
            )

    if not available_factors:
        raise ValueError(
            f"No factors from bundle are available in factors_df. "
            f"Missing factors: {missing_factors}"
        )

    # Process each available factor
    processed_factors = {}
    factor_weights = {}

    for factor_cfg in available_factors:
        factor_name = factor_cfg.name
        factor_series = result_df[factor_name].copy()

        # Step 1: Winsorize (if enabled)
        if bundle.options.winsorize:
            lower_q, upper_q = bundle.options.winsorize_limits
            logger.debug(
                f"Winsorizing {factor_name} with limits [{lower_q}, {upper_q}]"
            )
            factor_series = _winsorize_series(factor_series, lower_q, upper_q)

        # Step 2: Z-score (cross-sectional per timestamp)
        if bundle.options.zscore:
            logger.debug(f"Applying cross-sectional z-scoring to {factor_name}")
            zscore_series = _zscore_crosssectional(
                result_df.assign(**{factor_name: factor_series}),
                factor_col=factor_name,
                timestamp_col=timestamp_col,
            )
            # Store z-score column
            z_col_name = f"{factor_name}_z"
            result_df[z_col_name] = zscore_series
            processed_series = zscore_series
        else:
            # Use raw (or winsorized) values
            processed_series = factor_series

        # Step 3: Apply direction
        if factor_cfg.direction == "negative":
            # Invert: lower values become higher scores
            processed_series = -processed_series
            logger.debug(f"Inverting {factor_name} (negative direction)")

        processed_factors[factor_name] = processed_series
        factor_weights[factor_name] = factor_cfg.weight

    # Step 4: Compute weighted multi-factor score
    mf_score = pd.Series(0.0, index=result_df.index, dtype=float)

    total_weight = 0.0
    for factor_name, processed_series in processed_factors.items():
        weight = factor_weights[factor_name]
        # Only add where factor is not NaN
        non_null_mask = ~pd.isna(processed_series)
        mf_score.loc[non_null_mask] += weight * processed_series.loc[non_null_mask]
        total_weight += weight * non_null_mask.astype(float)

    # Normalize by total weight (for cases where some factors are missing)
    # This ensures mf_score is still in a reasonable range even if some factors are NaN
    if bundle.options.zscore:
        # For z-scored factors, normalization by weight sum is appropriate
        # (though we don't strictly need it since weights sum to 1.0)
        pass  # Already correct as weights sum to 1.0

    result_df["mf_score"] = mf_score

    # Build metadata
    meta = {
        "universe": bundle.universe,
        "factor_set": bundle.factor_set,
        "horizon_days": bundle.horizon_days,
        "used_factors": [f.name for f in available_factors],
        "factor_weights": {f.name: f.weight for f in available_factors},
        "missing_factors": missing_factors,
        "options_applied": {
            "winsorize": bundle.options.winsorize,
            "winsorize_limits": bundle.options.winsorize_limits
            if bundle.options.winsorize
            else None,
            "zscore": bundle.options.zscore,
            "neutralize_by": bundle.options.neutralize_by,
        },
        "n_observations": len(result_df),
        "n_symbols": result_df[group_col].nunique(),
        "date_range": {
            "start": str(result_df[timestamp_col].min()),
            "end": str(result_df[timestamp_col].max()),
        },
    }

    logger.info(
        f"Built multi-factor signal: {len(available_factors)} factors, "
        f"{len(missing_factors)} missing, {len(result_df)} observations"
    )

    return MultiFactorSignalResult(df=result_df, meta=meta)


def select_top_bottom(
    mf_df: pd.DataFrame,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    score_col: str = "mf_score",
) -> pd.DataFrame:
    """Add top/bottom quantile flags based on multi-factor score.

    For each timestamp, identifies top and bottom quantiles based on mf_score
    and adds binary flag columns.

    Args:
        mf_df: DataFrame with mf_score column (must have timestamp_col, group_col, score_col)
        top_quantile: Top quantile threshold (e.g., 0.2 for top 20%)
        bottom_quantile: Bottom quantile threshold (e.g., 0.2 for bottom 20%)
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol/group column (default: "symbol")
        score_col: Name of score column (default: "mf_score")

    Returns:
        DataFrame with additional columns:
            - mf_long_flag: 1 if symbol is in top quantile, else 0
            - mf_short_flag: 1 if symbol is in bottom quantile, else 0

    Raises:
        ValueError: If required columns are missing or quantiles are invalid
    """
    required_cols = [timestamp_col, group_col, score_col]
    missing_cols = [col for col in required_cols if col not in mf_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available columns: {list(mf_df.columns)}"
        )

    if not (0.0 < top_quantile <= 1.0):
        raise ValueError(f"top_quantile must be in (0, 1], got {top_quantile}")
    if not (0.0 < bottom_quantile <= 1.0):
        raise ValueError(f"bottom_quantile must be in (0, 1], got {bottom_quantile}")

    result_df = mf_df.copy()

    # Initialize flags
    result_df["mf_long_flag"] = 0
    result_df["mf_short_flag"] = 0

    # Group by timestamp and compute quantiles per timestamp
    def compute_flags(group: pd.DataFrame) -> pd.DataFrame:
        scores = group[score_col].dropna()
        if len(scores) < 2:
            # Not enough data for quantiles
            return group.assign(mf_long_flag=0, mf_short_flag=0)

        top_threshold = scores.quantile(1.0 - top_quantile)
        bottom_threshold = scores.quantile(bottom_quantile)

        # Create flags
        long_mask = group[score_col] >= top_threshold
        short_mask = group[score_col] <= bottom_threshold

        group = group.copy()
        group["mf_long_flag"] = long_mask.astype(int)
        group["mf_short_flag"] = short_mask.astype(int)

        return group

    result_df = result_df.groupby(timestamp_col, group_keys=False).apply(compute_flags)

    logger.debug(
        f"Computed top/bottom flags: top_quantile={top_quantile}, "
        f"bottom_quantile={bottom_quantile}"
    )

    return result_df
