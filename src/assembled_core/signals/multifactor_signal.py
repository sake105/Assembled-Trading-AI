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
    bundle = load_factor_bundle("configs/factor_bundles/macro_world_etfs_core_bundle.yaml")

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
    # Track per-row effective weight sum for renormalization when factors are NaN
    mf_score = pd.Series(0.0, index=result_df.index, dtype=float)
    effective_weight_sum = pd.Series(0.0, index=result_df.index, dtype=float)

    for factor_name, processed_series in processed_factors.items():
        weight = factor_weights[factor_name]
        # Only add where factor is not NaN
        non_null_mask = ~pd.isna(processed_series)
        mf_score.loc[non_null_mask] += weight * processed_series.loc[non_null_mask]
        effective_weight_sum.loc[non_null_mask] += weight

    # Renormalize: when some factors are NaN for a row, scale the score
    # so that available factors' weights effectively sum to the original total.
    # This prevents rows with fewer valid factors from having systematically
    # lower scores simply due to missing data.
    total_configured_weight = sum(factor_weights.values())
    if total_configured_weight > 0:
        renorm = total_configured_weight / effective_weight_sum.replace(0.0, np.nan)
        mf_score = mf_score * renorm
        # Rows where ALL factors were NaN remain NaN (not 0.0)
        mf_score.loc[effective_weight_sum == 0.0] = np.nan

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
            "winsorize_limits": (
                bundle.options.winsorize_limits if bundle.options.winsorize else None
            ),
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

    # Vectorized flag computation using transform — avoids one DataFrame copy per
    # timestamp group that groupby.apply() would create.
    grouped_scores = result_df.groupby(timestamp_col, sort=False)[score_col]

    def _top_threshold(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 2:
            return pd.Series(float("nan"), index=s.index)
        return pd.Series(valid.quantile(1.0 - top_quantile), index=s.index)

    def _bottom_threshold(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 2:
            return pd.Series(float("nan"), index=s.index)
        return pd.Series(valid.quantile(bottom_quantile), index=s.index)

    top_thresh = grouped_scores.transform(_top_threshold)
    bottom_thresh = grouped_scores.transform(_bottom_threshold)

    # Where threshold is NaN (< 2 valid scores), flags stay 0
    result_df["mf_long_flag"] = (
        (result_df[score_col] >= top_thresh) & top_thresh.notna()
    ).astype(int)
    result_df["mf_short_flag"] = (
        (result_df[score_col] <= bottom_thresh) & bottom_thresh.notna()
    ).astype(int)

    logger.debug(
        f"Computed top/bottom flags: top_quantile={top_quantile}, "
        f"bottom_quantile={bottom_quantile}"
    )

    return result_df


# ── IC-weighted adaptive factor combination (Plan 1.1) ────────────────


def compute_ic_weights(
    factor_df: pd.DataFrame,
    forward_returns_col: str,
    factor_cols: list[str],
    *,
    ic_window: int = 60,
    ic_halflife: int = 20,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    min_ic_obs: int = 5,
) -> pd.DataFrame:
    """Compute time-varying IC-based weights for each factor.

    For each timestamp, computes the rolling Spearman rank-IC of each factor
    against forward returns, smooths with EWMA, and normalises positive ICs
    into portfolio weights.

    This implements the adaptive factor weighting scheme described in
    Plan §1.1: factors with recent predictive power receive higher weight;
    factors with negative IC are zeroed out automatically.

    Args:
        factor_df: Panel DataFrame with timestamp, symbol, factors, and
            forward returns.
        forward_returns_col: Column with forward returns for IC.
        factor_cols: List of factor column names to weight.
        ic_window: Rolling window (trading days) for IC estimation.
        ic_halflife: EWMA half-life for IC smoothing (reduces noise).
        timestamp_col: Timestamp column.
        group_col: Symbol column.
        min_ic_obs: Minimum cross-sectional observations for valid IC.

    Returns:
        DataFrame indexed by timestamp with columns ``ic_{factor}``,
        ``ic_smooth_{factor}``, ``weight_{factor}``, and
        ``aggregate_ic`` (sum of positive smoothed ICs — confidence
        indicator for overall exposure scaling).
    """
    from scipy.stats import spearmanr

    timestamps = sorted(factor_df[timestamp_col].unique())
    if len(timestamps) < ic_window:
        return pd.DataFrame()

    # Step 1: compute raw cross-sectional IC per timestamp per factor
    raw_ics: dict[str, list[float]] = {f: [] for f in factor_cols}
    ts_list: list = []

    for ts in timestamps:
        slice_df = factor_df[factor_df[timestamp_col] == ts]
        ts_list.append(ts)
        for fcol in factor_cols:
            valid = slice_df[[fcol, forward_returns_col]].dropna()
            if len(valid) < min_ic_obs:
                raw_ics[fcol].append(np.nan)
            else:
                corr, _ = spearmanr(valid[fcol], valid[forward_returns_col])
                raw_ics[fcol].append(corr)

    ic_df = pd.DataFrame({"timestamp": ts_list})
    for fcol in factor_cols:
        ic_df[f"ic_{fcol}"] = raw_ics[fcol]

    # Step 2: rolling mean IC + EWMA smoothing
    for fcol in factor_cols:
        raw_col = f"ic_{fcol}"
        rolling_ic = ic_df[raw_col].rolling(ic_window, min_periods=max(10, ic_window // 3)).mean()
        ic_df[f"ic_smooth_{fcol}"] = rolling_ic.ewm(halflife=ic_halflife, min_periods=5).mean()

    # Step 3: normalise positive smoothed ICs into weights
    smooth_cols = [f"ic_smooth_{f}" for f in factor_cols]

    for i in range(len(ic_df)):
        values = ic_df.loc[i, smooth_cols].values.astype(float)
        positive = np.where(values > 0, values, 0.0)
        total = positive.sum()
        for j, fcol in enumerate(factor_cols):
            ic_df.loc[i, f"weight_{fcol}"] = positive[j] / total if total > 0 else 0.0

    # Aggregate IC: confidence indicator
    ic_df["aggregate_ic"] = ic_df[smooth_cols].clip(lower=0).sum(axis=1)

    ic_df = ic_df.set_index("timestamp")
    return ic_df


# ── Sector / group neutralization (Plan 1.3) ──────────────────────────


def neutralize_by_group(
    df: pd.DataFrame,
    factor_col: str,
    group_col: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Cross-sectionally neutralize a factor within groups (e.g., sectors).

    For each timestamp and group, computes:
        ``z_neutral = (x - mean(x|group)) / std(x|group)``

    This removes sector/industry bias from factor scores so that only
    intra-group (pure stock) alpha remains.  Without neutralization,
    a "value" signal is often dominated by sector rotation rather than
    stock selection.

    Args:
        df: Panel DataFrame.
        factor_col: Factor column to neutralize.
        group_col: Column defining groups (e.g., "sector", "industry").
        timestamp_col: Timestamp column.

    Returns:
        Series of neutralized z-scores (same index as ``df``).
    """
    def _neutralize_group(sub: pd.DataFrame) -> pd.Series:
        vals = sub[factor_col]
        valid = vals.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=sub.index)
        mean = valid.mean()
        std = valid.std(ddof=0)
        if std < 1e-10:
            return pd.Series(0.0, index=sub.index)
        return (vals - mean) / std

    neutralized = df.groupby([timestamp_col, group_col], group_keys=False).apply(
        _neutralize_group
    )
    return neutralized.reindex(df.index)


def build_adaptive_multifactor_signal(
    factors_df: pd.DataFrame,
    bundle: FactorBundleConfig,
    forward_returns_col: str | None = None,
    ic_weights_df: pd.DataFrame | None = None,
    *,
    neutralize_col: str | None = None,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> MultiFactorSignalResult:
    """Build multi-factor signal with IC-adaptive weights and optional neutralization.

    Extends ``build_multifactor_signal`` with:
    - IC-weighted factor combination (if ``ic_weights_df`` provided or
      ``forward_returns_col`` given for on-the-fly IC computation)
    - Sector/group neutralization (if ``neutralize_col`` provided)

    Falls back to static bundle weights when IC data is unavailable.

    Args:
        factors_df: Panel DataFrame with timestamp, symbol, factor columns.
        bundle: Factor bundle configuration.
        forward_returns_col: Forward return column for IC calculation.
            If provided and ``ic_weights_df`` is None, ICs will be
            computed on-the-fly.
        ic_weights_df: Pre-computed IC weights (from ``compute_ic_weights``).
        neutralize_col: Column for group neutralization (e.g., "sector").
        group_col: Symbol column.
        timestamp_col: Timestamp column.

    Returns:
        MultiFactorSignalResult with adaptive mf_score.
    """
    result_df = factors_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)
    result_df = result_df.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    # Determine available factors
    available_factors = [f for f in bundle.factors if f.name in result_df.columns]
    missing_factors = [f.name for f in bundle.factors if f.name not in result_df.columns]

    if not available_factors:
        raise ValueError(f"No factors from bundle available. Missing: {missing_factors}")

    factor_names = [f.name for f in available_factors]

    # Optional: compute IC weights on the fly
    if ic_weights_df is None and forward_returns_col and forward_returns_col in result_df.columns:
        ic_weights_df = compute_ic_weights(
            result_df, forward_returns_col, factor_names,
            timestamp_col=timestamp_col, group_col=group_col,
        )

    # Neutralize factors if requested
    if neutralize_col and neutralize_col in result_df.columns:
        for f in available_factors:
            result_df[f.name] = neutralize_by_group(
                result_df, f.name, neutralize_col, timestamp_col,
            )

    # Process factors: winsorize + zscore (same as build_multifactor_signal)
    processed: dict[str, pd.Series] = {}
    for f in available_factors:
        series = result_df[f.name].copy()
        if bundle.options.winsorize:
            low, high = bundle.options.winsorize_limits
            series = _winsorize_series(series, low, high)
        if bundle.options.zscore:
            z = _zscore_crosssectional(
                result_df.assign(**{f.name: series}), f.name, timestamp_col,
            )
            result_df[f"{f.name}_z"] = z
            series = z
        if f.direction == "negative":
            series = -series
        processed[f.name] = series

    # Compute weighted score — either IC-adaptive or static
    mf_score = pd.Series(0.0, index=result_df.index, dtype=float)
    effective_weight_sum = pd.Series(0.0, index=result_df.index, dtype=float)

    use_ic = ic_weights_df is not None and not ic_weights_df.empty

    for fname, pseries in processed.items():
        non_null = ~pd.isna(pseries)
        if use_ic:
            # Map IC weight per timestamp
            weight_col = f"weight_{fname}"
            if weight_col in ic_weights_df.columns:
                ts_weights = result_df[timestamp_col].map(
                    ic_weights_df[weight_col]
                ).fillna(0.0)
            else:
                # Fallback: static weight
                cfg = next((f for f in available_factors if f.name == fname), None)
                ts_weights = pd.Series(cfg.weight if cfg else 0.0, index=result_df.index)
        else:
            cfg = next((f for f in available_factors if f.name == fname), None)
            ts_weights = pd.Series(cfg.weight if cfg else 0.0, index=result_df.index)

        mf_score.loc[non_null] += ts_weights.loc[non_null] * pseries.loc[non_null]
        effective_weight_sum.loc[non_null] += ts_weights.loc[non_null]

    # Renormalize
    total_w = effective_weight_sum.replace(0, np.nan)
    mf_score = mf_score / total_w  # normalize to unit weights
    mf_score.loc[effective_weight_sum == 0.0] = np.nan

    result_df["mf_score"] = mf_score

    # Aggregate IC as confidence
    if use_ic:
        result_df["mf_aggregate_ic"] = result_df[timestamp_col].map(
            ic_weights_df["aggregate_ic"]
        )

    meta = {
        "universe": bundle.universe,
        "factor_set": bundle.factor_set,
        "used_factors": factor_names,
        "missing_factors": missing_factors,
        "ic_weighted": use_ic,
        "neutralized_by": neutralize_col,
        "n_observations": len(result_df),
    }

    return MultiFactorSignalResult(df=result_df, meta=meta)


# ── Regime-conditional factor weights (Plan 1.2) ──────────────────────

REGIME_FACTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {
        "momentum": 0.30, "quality": 0.20, "value": 0.10,
        "low_vol": 0.05, "growth": 0.20, "size": 0.10,
        "mean_reversion": 0.00, "safe_haven": 0.00, "carry": 0.05,
    },
    "bear": {
        "momentum": -0.10, "quality": 0.35, "value": 0.10,
        "low_vol": 0.25, "growth": 0.00, "size": 0.00,
        "mean_reversion": 0.10, "safe_haven": 0.10, "carry": 0.00,
    },
    "crisis": {
        "momentum": -0.10, "quality": 0.15, "value": 0.00,
        "low_vol": 0.30, "growth": 0.00, "size": 0.00,
        "mean_reversion": 0.00, "safe_haven": 0.40, "carry": 0.00,
    },
    "recovery": {
        "momentum": -0.05, "quality": 0.10, "value": 0.35,
        "low_vol": 0.05, "growth": 0.15, "size": 0.20,
        "mean_reversion": 0.10, "safe_haven": 0.00, "carry": 0.05,
    },
    "sideways": {
        "momentum": 0.10, "quality": 0.15, "value": 0.15,
        "low_vol": 0.10, "growth": 0.10, "size": 0.05,
        "mean_reversion": 0.25, "safe_haven": 0.00, "carry": 0.10,
    },
}


def compute_regime_blended_weights(
    regime_probabilities: dict[str, float],
    factor_categories: dict[str, str] | None = None,
    custom_regime_weights: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Blend factor weights across regimes using HMM probabilities.

    When HMM gives P(bull)=0.6, P(bear)=0.4, weights become:
    ``w = 0.6 * bull_weights + 0.4 * bear_weights``

    Args:
        regime_probabilities: Regime → probability (sums to ~1).
        factor_categories: Optional factor_name → category mapping.
        custom_regime_weights: Override default regime-factor matrix.

    Returns:
        Dict mapping category (or factor name) → blended weight.
    """
    rw = custom_regime_weights or REGIME_FACTOR_WEIGHTS

    total_p = sum(regime_probabilities.values())
    if total_p < 1e-10:
        return {}

    blended: dict[str, float] = {}
    for regime, prob in regime_probabilities.items():
        norm_prob = prob / total_p
        weights = rw.get(regime, rw.get("sideways", {}))
        for category, w in weights.items():
            blended[category] = blended.get(category, 0.0) + norm_prob * w

    if factor_categories:
        return {fn: round(blended.get(cat, 0.0), 6) for fn, cat in factor_categories.items()}

    return {k: round(v, 6) for k, v in blended.items()}


def extract_regime_posteriors(
    regime_state_df: "pd.DataFrame",
    timestamp: "pd.Timestamp | str | None" = None,
) -> dict[str, float]:
    """Extract HMM posterior probabilities from a regime state DataFrame.

    Instead of using a discrete regime_label, this returns the continuous
    probability distribution across regimes, enabling smooth blending
    via ``compute_regime_blended_weights``.

    Args:
        regime_state_df: DataFrame from ``build_regime_state`` or
            ``build_regime_state_hmm`` containing ``hmm_*_prob`` columns.
        timestamp: If provided, extract posteriors for this specific timestamp.
            If None, uses the last row (most recent).

    Returns:
        Dict mapping regime_name -> probability (e.g. {"bull": 0.6, "bear": 0.3, "sideways": 0.1}).
        Falls back to uniform if no HMM probabilities are found.
    """
    if regime_state_df is None or regime_state_df.empty:
        return {}

    # Select the row
    if timestamp is not None:
        ts_col = "timestamp" if "timestamp" in regime_state_df.columns else regime_state_df.columns[0]
        row_mask = regime_state_df[ts_col] == timestamp
        if row_mask.any():
            row = regime_state_df.loc[row_mask].iloc[-1]
        else:
            row = regime_state_df.iloc[-1]
    else:
        row = regime_state_df.iloc[-1]

    # Extract all hmm_*_prob columns
    proba_cols = [c for c in regime_state_df.columns if c.startswith("hmm_") and c.endswith("_prob")]
    if not proba_cols:
        # Fallback: use discrete label with confidence
        label = row.get("regime_label", "sideways")
        conf = float(row.get("regime_confidence", 0.8))
        return {str(label): conf, "sideways": 1.0 - conf} if label != "sideways" else {"sideways": 1.0}

    posteriors: dict[str, float] = {}
    for col in proba_cols:
        # hmm_bull_prob -> bull
        regime_name = col.replace("hmm_", "").replace("_prob", "")
        val = float(row.get(col, 0.0))
        if not (val != val):  # not NaN
            posteriors[regime_name] = val

    # Normalize to sum to 1.0
    total = sum(posteriors.values())
    if total > 1e-10:
        posteriors = {k: v / total for k, v in posteriors.items()}

    return posteriors


# ---------------------------------------------------------------------------
# Signal Hysteresis and Flip Cooldown (Plan 1.6)
# ---------------------------------------------------------------------------


def apply_signal_hysteresis(
    signal_series: pd.Series,
    threshold: float = 0.0,
    hysteresis_pct: float = 0.15,
    cooldown_bars: int = 3,
) -> pd.Series:
    """Apply hysteresis band and flip cooldown to a signal series.

    A signal must exceed threshold by hysteresis_pct to flip direction.
    After a flip, wait cooldown_bars before allowing another flip.

    Args:
        signal_series: Raw signal values (positive=long, negative=short).
        threshold: Base threshold for direction change.
        hysteresis_pct: Percent above threshold to trigger flip.
        cooldown_bars: Minimum bars between direction flips.

    Returns:
        Filtered signal series with hysteresis applied.
    """
    result = signal_series.copy()
    current_dir = 0  # 0=flat, 1=long, -1=short
    bars_since_flip = cooldown_bars  # start ready

    for i in range(len(result)):
        val = float(result.iloc[i])
        bars_since_flip += 1

        if bars_since_flip < cooldown_bars:
            # In cooldown — keep current direction
            if current_dir == 0:
                result.iloc[i] = 0.0
            continue

        # Check for direction change with hysteresis
        upper = threshold + abs(threshold) * hysteresis_pct + hysteresis_pct * 0.01
        lower = -threshold - abs(threshold) * hysteresis_pct - hysteresis_pct * 0.01

        if current_dir <= 0 and val > upper:
            current_dir = 1
            bars_since_flip = 0
        elif current_dir >= 0 and val < lower:
            current_dir = -1
            bars_since_flip = 0
        elif current_dir != 0 and abs(val) < abs(threshold) * 0.5:
            current_dir = 0
            bars_since_flip = 0

    return result


# ---------------------------------------------------------------------------
# Meta-Model Confidence Filter (Phase 2 integration)
# ---------------------------------------------------------------------------


def apply_meta_model_filter(
    signals_df: pd.DataFrame,
    model_path: str = "models/meta/meta_model_latest.joblib",
    confidence_threshold: float = 0.55,
    scale_by_confidence: bool = True,
    score_col: str = "mf_score",
) -> pd.DataFrame:
    """Filter and scale signals using meta-model confidence predictions.

    Loads a trained MetaModel from disk, predicts confidence_score for each
    signal row, then:
    1. Drops signals below confidence_threshold
    2. Optionally scales mf_score by confidence_score

    Args:
        signals_df: DataFrame with factor columns and mf_score.
        model_path: Path to saved MetaModel joblib.
        confidence_threshold: Minimum confidence to keep a signal (default 0.55).
        scale_by_confidence: If True, mf_score *= confidence_score.
        score_col: Name of the score column to scale.

    Returns:
        Filtered (and optionally scaled) DataFrame.
    """
    import pathlib

    model_file = pathlib.Path(model_path)
    if not model_file.exists():
        logger.debug(
            "[META-FILTER] Model not found at %s — passing through",
            model_path,
        )
        return signals_df

    if signals_df.empty:
        return signals_df

    try:
        from src.assembled_core.signals.meta_model import load_meta_model

        meta_model = load_meta_model(model_file)
        confidence = meta_model.predict_proba(signals_df)

        signals_df = signals_df.copy()
        signals_df["confidence_score"] = confidence.values

        n_before = len(signals_df)
        signals_df = signals_df[signals_df["confidence_score"] >= confidence_threshold]
        n_after = len(signals_df)

        if scale_by_confidence and score_col in signals_df.columns:
            signals_df[score_col] = (
                signals_df[score_col] * signals_df["confidence_score"]
            )

        logger.info(
            "[META-FILTER] %d/%d passed (thr=%.2f, scaled=%s)",
            n_after, n_before, confidence_threshold, scale_by_confidence,
        )
        return signals_df

    except Exception as exc:
        logger.warning(
            "[META-FILTER] Failed: %s — passing through", exc,
        )
        return signals_df
