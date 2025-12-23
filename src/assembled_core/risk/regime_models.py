"""Regime Detection and State Classification Module.

This module implements Phase D1 from the Advanced Analytics & Factor Labs roadmap.
It provides functions to detect market regimes (bull, bear, sideways, crisis, reflation)
based on macro factors, market breadth, volatility, and trend indicators.

Example:
    from src.assembled_core.risk.regime_models import build_regime_state, RegimeStateConfig

    # Build regime state from factors
    regime_state = build_regime_state(
        prices=prices_df,
        macro_factors=macro_factors_df,
        breadth_df=breadth_df,
        vol_df=vol_df,
    )

    # Evaluate factor IC by regime
    ic_by_regime = evaluate_factor_ic_by_regime(ic_df, regime_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeStateConfig:
    """Configuration for regime state detection.

    Attributes:
        trend_ma_windows: Moving average windows for trend detection (default: [50, 200])
        vol_window: Window for realized volatility calculation (default: 20)
        vov_window: Window for volatility of volatility calculation (default: 60)
        breadth_ma_window: Moving average window for market breadth (default: 50)
        combine_macro_and_market: Whether to combine macro and market factors (default: True)
    """

    trend_ma_windows: tuple[int, ...] = (50, 200)
    vol_window: int = 20
    vov_window: int = 60
    breadth_ma_window: int = 50
    combine_macro_and_market: bool = True


def build_regime_state(
    prices: pd.DataFrame,
    macro_factors: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    vol_df: pd.DataFrame | None = None,
    config: RegimeStateConfig | None = None,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Build regime state DataFrame from various input factors.

    Combines macro factors, market breadth, volatility, and trend indicators
    to produce daily regime labels and sub-scores.

    Args:
        prices: DataFrame with price data (panel format)
            Required columns: timestamp_col, group_col, close
        macro_factors: Optional DataFrame with macro regime factors (panel format)
                      Must have timestamp_col and columns:
                      - macro_growth_regime, macro_inflation_regime, macro_risk_aversion_proxy
        breadth_df: Optional DataFrame with market breadth indicators (time-series format)
                    Must have timestamp_col and columns:
                    - fraction_above_ma_50 (or fraction_above_ma_{ma_window}),
                    - risk_on_off_score (optional)
        vol_df: Optional DataFrame with volatility factors (panel format)
                Must have timestamp_col, group_col, and columns:
                - rv_20 (or rv_{vol_window}), vov_20_60 (optional)
        config: Optional RegimeStateConfig for configuration (default: RegimeStateConfig())
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol column (default: "symbol")

    Returns:
        DataFrame with columns:
            - timestamp: Timestamp (UTC)
            - regime_label: Regime label (str: "bull", "bear", "sideways", "crisis", "reflation", "neutral")
            - regime_trend_score: Trend score (-1.0 to +1.0)
            - regime_macro_score: Macro score (-1.0 to +1.0)
            - regime_risk_score: Risk score (-1.0 to +1.0)
            - regime_confidence: Confidence score (0.0 to 1.0)

        One row per timestamp, sorted by timestamp.

    Raises:
        ValueError: If required columns are missing or inputs are invalid
        KeyError: If timestamp_col not found in inputs
    """
    if config is None:
        config = RegimeStateConfig()

    # Validate inputs
    required_price_cols = [timestamp_col, group_col, "close"]
    missing_cols = [col for col in required_price_cols if col not in prices.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in prices: {', '.join(missing_cols)}. "
            f"Available: {list(prices.columns)}"
        )

    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    # Ensure timestamp is datetime
    prices = prices.copy()
    if not pd.api.types.is_datetime64_any_dtype(prices[timestamp_col]):
        prices[timestamp_col] = pd.to_datetime(prices[timestamp_col], utc=True)

    # Get all unique timestamps
    all_timestamps = sorted(prices[timestamp_col].unique())

    # Prepare data structures for aggregation
    regime_data = []

    for timestamp in all_timestamps:
        # Initialize scores
        trend_score = np.nan
        macro_score = np.nan
        risk_score = np.nan
        confidence = 0.0

        # 1. Compute Trend Score (from Market Breadth and Trend Strength)
        trend_scores = []

        # Try to get trend strength from prices if available (from ta_factors_core)
        prices_ts = prices[prices[timestamp_col] == timestamp]
        if not prices_ts.empty:
            # Check for trend_strength columns
            for trend_col in ["trend_strength_200", "trend_strength_50"]:
                if trend_col in prices_ts.columns:
                    trend_values = prices_ts[trend_col].dropna()
                    if not trend_values.empty:
                        # Use median trend strength
                        median_trend = trend_values.median()
                        # Normalize: assume trend_strength is already roughly in [-1, 1] range
                        # Clip to be safe
                        trend_scores.append(np.clip(median_trend, -1.0, 1.0))
                        confidence += 0.2
                        break

        # Use Market Breadth as additional trend indicator
        if breadth_df is not None:
            breadth_data = breadth_df[breadth_df[timestamp_col] == timestamp]
            if not breadth_data.empty:
                # Use fraction_above_ma_50 (or configurable window)
                breadth_col = f"fraction_above_ma_{config.breadth_ma_window}"
                if breadth_col not in breadth_data.columns:
                    # Try alternative column names
                    for alt_col in ["fraction_above_ma_50", "fraction_above_ma_200"]:
                        if alt_col in breadth_data.columns:
                            breadth_col = alt_col
                            break

                if breadth_col in breadth_data.columns:
                    fraction_above = breadth_data[breadth_col].iloc[0]
                    if not pd.isna(fraction_above):
                        # Trend score: fraction_above_ma normalized to [-1, 1]
                        # 0.5 (50%) = 0.0, 1.0 (100%) = +1.0, 0.0 (0%) = -1.0
                        breadth_trend_score = 2.0 * (fraction_above - 0.5)
                        trend_scores.append(breadth_trend_score)
                        confidence += 0.3

        # Combine trend scores (average if multiple available)
        if trend_scores:
            trend_score = np.mean(trend_scores)
        else:
            trend_score = np.nan

        # 2. Compute Macro Score (from Macro Factors)
        if macro_factors is not None:
            macro_data = macro_factors[macro_factors[timestamp_col] == timestamp]
            if not macro_data.empty:
                # Get first row (macro factors should be same for all symbols on a given date)
                row = macro_data.iloc[0]

                growth_regime = row.get("macro_growth_regime", np.nan)
                inflation_regime = row.get("macro_inflation_regime", np.nan)

                if not pd.isna(growth_regime) or not pd.isna(inflation_regime):
                    # Macro score: combination of growth and inflation
                    # Growth: +1 = expansion, -1 = recession
                    # Inflation: +1 = high inflation, -1 = low/deflation
                    # Combined: (growth + inflation) / 2, normalized
                    growth_val = growth_regime if not pd.isna(growth_regime) else 0.0
                    inflation_val = (
                        inflation_regime if not pd.isna(inflation_regime) else 0.0
                    )

                    # Simple combination: average of growth and inflation (weighted)
                    # Growth is more important (0.7 weight), inflation less (0.3 weight)
                    macro_score = 0.7 * growth_val + 0.3 * inflation_val

                    # Clip to [-1, 1]
                    macro_score = np.clip(macro_score, -1.0, 1.0)
                    confidence += 0.3

        # 3. Compute Risk Score (from Volatility and Risk-On/Off)
        if vol_df is not None:
            vol_data = vol_df[vol_df[timestamp_col] == timestamp]
            if not vol_data.empty:
                # Aggregate volatility across symbols (use median to be robust)
                rv_col = f"rv_{config.vol_window}"
                if rv_col not in vol_data.columns:
                    # Try alternative
                    if "rv_20" in vol_data.columns:
                        rv_col = "rv_20"

                if rv_col in vol_data.columns:
                    rv_values = vol_data[rv_col].dropna()
                    if not rv_values.empty:
                        # Use median realized volatility
                        median_rv = rv_values.median()

                        # Normalize RV to score: high RV = negative score (risk-off)
                        # We need to use percentiles or a threshold approach
                        # For now, use a simple heuristic:
                        # RV > 0.3 (30% annualized) = -1.0 (crisis)
                        # RV < 0.15 (15% annualized) = +1.0 (risk-on)
                        # Linear interpolation between
                        if median_rv > 0.5:
                            risk_score_from_vol = -1.0
                        elif median_rv < 0.1:
                            risk_score_from_vol = 1.0
                        else:
                            # Linear interpolation: (0.1, 1.0) to (0.5, -1.0)
                            risk_score_from_vol = 1.0 - 2.0 * (median_rv - 0.1) / (
                                0.5 - 0.1
                            )

                        risk_score = risk_score_from_vol
                        confidence += 0.2

        # Add risk_on_off_score from breadth_df if available
        if breadth_df is not None:
            breadth_data = breadth_df[breadth_df[timestamp_col] == timestamp]
            if not breadth_data.empty and "risk_on_off_score" in breadth_data.columns:
                risk_on_off = breadth_data["risk_on_off_score"].iloc[0]
                if not pd.isna(risk_on_off):
                    # risk_on_off_score: -1 = risk-off, +1 = risk-on
                    # Combine with volatility-based risk score
                    if not pd.isna(risk_score):
                        # Average of volatility-based and risk_on_off_score
                        risk_score = 0.6 * risk_score + 0.4 * risk_on_off
                    else:
                        risk_score = risk_on_off
                    confidence += 0.2

        # 4. Determine Regime Label from Scores
        regime_label = "neutral"

        # Fill missing scores with 0.0 for label determination
        trend_score_val = trend_score if not pd.isna(trend_score) else 0.0
        macro_score_val = macro_score if not pd.isna(macro_score) else 0.0
        risk_score_val = risk_score if not pd.isna(risk_score) else 0.0

        # Regime classification logic (based on design document)
        if risk_score_val < -0.8:
            # Extreme volatility/risk-off = crisis
            regime_label = "crisis"
        elif trend_score_val < -0.5 and risk_score_val < 0.0:
            # Negative trend + risk-off = bear
            regime_label = "bear"
        elif trend_score_val > 0.5 and risk_score_val > 0.0:
            # Positive trend + risk-on = bull
            regime_label = "bull"
        elif macro_score_val > 0.3 and macro_factors is not None:
            # Check for reflation: positive macro + high inflation
            macro_data = macro_factors[macro_factors[timestamp_col] == timestamp]
            if not macro_data.empty:
                inflation_regime = macro_data.iloc[0].get("macro_inflation_regime", 0.0)
                if not pd.isna(inflation_regime) and inflation_regime > 0:
                    regime_label = "reflation"
        elif abs(trend_score_val) < 0.3 and abs(risk_score_val) < 0.3:
            # Moderate scores = sideways
            regime_label = "sideways"
        else:
            # Default: neutral
            regime_label = "neutral"

        # Normalize confidence to [0, 1]
        confidence = min(confidence, 1.0)

        regime_data.append(
            {
                timestamp_col: timestamp,
                "regime_label": regime_label,
                "regime_trend_score": trend_score if not pd.isna(trend_score) else 0.0,
                "regime_macro_score": macro_score if not pd.isna(macro_score) else 0.0,
                "regime_risk_score": risk_score if not pd.isna(risk_score) else 0.0,
                "regime_confidence": confidence,
            }
        )

    regime_df = pd.DataFrame(regime_data)

    if regime_df.empty:
        logger.warning("No regime state data computed. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                timestamp_col,
                "regime_label",
                "regime_trend_score",
                "regime_macro_score",
                "regime_risk_score",
                "regime_confidence",
            ]
        )

    # Sort by timestamp
    regime_df = regime_df.sort_values(timestamp_col).reset_index(drop=True)

    logger.info(
        f"Computed regime state for {len(regime_df)} timestamps. "
        f"Regime distribution: {regime_df['regime_label'].value_counts().to_dict()}"
    )

    return regime_df


def compute_regime_transition_stats(
    regime_state_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """Compute statistics about regime transitions.

    Analyzes how often regimes transition to other regimes and typical
    duration of each regime.

    Args:
        regime_state_df: DataFrame from build_regime_state()
                        Must have timestamp_col and regime_col
        timestamp_col: Name of timestamp column (default: "timestamp")
        regime_col: Name of regime label column (default: "regime_label")

    Returns:
        DataFrame with regime transition statistics:
            - from_regime: Source regime label
            - to_regime: Target regime label
            - count: Number of transitions
            - avg_duration_days: Average duration of source regime (before transition)
            - transition_probability: Probability of transitioning from source to target

        Also includes summary statistics:
            - regime: Regime label
            - avg_duration_days: Average duration of this regime
            - total_count: Total number of times this regime occurred

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = [timestamp_col, regime_col]
    missing_cols = [col for col in required_cols if col not in regime_state_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available: {list(regime_state_df.columns)}"
        )

    if regime_state_df.empty:
        logger.warning("regime_state_df is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # Ensure timestamp is datetime
    df = regime_state_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Compute transitions
    transitions = []
    regime_durations = {regime: [] for regime in df[regime_col].unique()}

    current_regime = None
    regime_start_idx = None

    for i, row in df.iterrows():
        regime = row[regime_col]

        if current_regime is None:
            current_regime = regime
            regime_start_idx = i
        elif regime != current_regime:
            # Transition occurred
            # Calculate duration of previous regime
            if regime_start_idx is not None and i > regime_start_idx:
                duration = (
                    df.iloc[i][timestamp_col] - df.iloc[regime_start_idx][timestamp_col]
                ).days
                if duration > 0:
                    regime_durations[current_regime].append(duration)

                transitions.append(
                    {
                        "from_regime": current_regime,
                        "to_regime": regime,
                    }
                )

            current_regime = regime
            regime_start_idx = i

    # Handle last regime duration
    if regime_start_idx is not None:
        last_idx = len(df) - 1
        if last_idx > regime_start_idx:
            duration = (
                df.iloc[last_idx][timestamp_col]
                - df.iloc[regime_start_idx][timestamp_col]
            ).days
            if duration > 0:
                regime_durations[current_regime].append(duration)

    # Build transition statistics
    if not transitions:
        logger.warning("No regime transitions found. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "from_regime",
                "to_regime",
                "count",
                "avg_duration_days",
                "transition_probability",
            ]
        )

    transitions_df = pd.DataFrame(transitions)

    # Count transitions
    transition_counts = (
        transitions_df.groupby(["from_regime", "to_regime"])
        .size()
        .reset_index(name="count")
    )

    # Compute average durations
    avg_durations = {
        regime: np.mean(durations) if durations else 0.0
        for regime, durations in regime_durations.items()
    }

    # Compute transition probabilities
    transition_probs = []
    for _, row in transition_counts.iterrows():
        from_regime = row["from_regime"]
        count = row["count"]

        # Total transitions from this regime
        total_from = transition_counts[transition_counts["from_regime"] == from_regime][
            "count"
        ].sum()
        prob = count / total_from if total_from > 0 else 0.0

        transition_probs.append(
            {
                "from_regime": from_regime,
                "to_regime": row["to_regime"],
                "count": count,
                "avg_duration_days": avg_durations.get(from_regime, 0.0),
                "transition_probability": prob,
            }
        )

    result_df = pd.DataFrame(transition_probs)

    if not result_df.empty:
        result_df = result_df.sort_values(
            ["from_regime", "transition_probability"], ascending=[True, False]
        )

    logger.info(
        f"Computed transition stats: {len(result_df)} transition pairs, "
        f"{result_df['from_regime'].nunique()} unique source regimes"
    )

    return result_df


def evaluate_factor_ic_by_regime(
    ic_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
    ic_col_suffix: str = "_ic",
) -> pd.DataFrame:
    """Evaluate factor effectiveness (IC) by regime.

    Computes IC statistics (mean IC, IC-IR, hit ratio) separately for each regime.
    This allows identification of factors that work well in specific regimes.

    Args:
        ic_df: DataFrame with IC time-series (from compute_ic or compute_rank_ic)
               Must have timestamp_col (as index or column) and IC columns (e.g., "returns_12m_ic")
        regime_state_df: DataFrame from build_regime_state()
                         Must have timestamp_col and regime_col
        timestamp_col: Name of timestamp column (default: "timestamp")
        regime_col: Name of regime label column (default: "regime_label")
        ic_col_suffix: Suffix for IC columns (default: "_ic")
                       IC columns are identified by this suffix

    Returns:
        DataFrame with IC statistics per factor and regime:
            - factor: Factor name
            - regime: Regime label
            - mean_ic: Mean IC in this regime
            - std_ic: Std of IC in this regime
            - ic_ir: IC-IR (mean_ic / std_ic) in this regime
            - hit_ratio: Percentage of periods with positive IC in this regime
            - n_observations: Number of observations (timestamps) in this regime

    Raises:
        ValueError: If required columns are missing
    """
    # Validate inputs
    if ic_df.empty or regime_state_df.empty:
        logger.warning("Empty input DataFrames. Returning empty result.")
        return pd.DataFrame(
            columns=[
                "factor",
                "regime",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "n_observations",
            ]
        )

    # Handle timestamp index in ic_df
    ic_df = ic_df.copy()
    if timestamp_col not in ic_df.columns:
        if ic_df.index.name == timestamp_col or isinstance(
            ic_df.index, pd.DatetimeIndex
        ):
            ic_df = ic_df.reset_index()
        else:
            raise ValueError(
                f"timestamp_col '{timestamp_col}' not found in ic_df columns or index. "
                f"Available columns: {list(ic_df.columns)}"
            )

    # Ensure timestamp is datetime in both DataFrames
    if not pd.api.types.is_datetime64_any_dtype(ic_df[timestamp_col]):
        ic_df[timestamp_col] = pd.to_datetime(ic_df[timestamp_col], utc=True)

    regime_df = regime_state_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(regime_df[timestamp_col]):
        regime_df[timestamp_col] = pd.to_datetime(regime_df[timestamp_col], utc=True)

    # Merge IC and regime data
    merged = pd.merge(
        ic_df,
        regime_df[[timestamp_col, regime_col]],
        on=timestamp_col,
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping timestamps between ic_df and regime_state_df.")
        return pd.DataFrame(
            columns=[
                "factor",
                "regime",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "n_observations",
            ]
        )

    # Identify IC columns (those ending with ic_col_suffix)
    ic_cols = [col for col in merged.columns if col.endswith(ic_col_suffix)]

    if not ic_cols:
        logger.warning(
            f"No IC columns found with suffix '{ic_col_suffix}'. Available columns: {list(merged.columns)}"
        )
        return pd.DataFrame(
            columns=[
                "factor",
                "regime",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "hit_ratio",
                "n_observations",
            ]
        )

    # Compute statistics per factor and regime
    results = []

    for ic_col in ic_cols:
        factor_name = ic_col.replace(ic_col_suffix, "")

        for regime in merged[regime_col].unique():
            regime_data = merged[merged[regime_col] == regime][ic_col].dropna()

            if len(regime_data) == 0:
                continue

            mean_ic = regime_data.mean()
            std_ic = regime_data.std(ddof=0)  # Population std
            ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
            hit_ratio = (regime_data > 0).sum() / len(regime_data)

            results.append(
                {
                    "factor": factor_name,
                    "regime": regime,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "ic_ir": ic_ir,
                    "hit_ratio": hit_ratio,
                    "n_observations": len(regime_data),
                }
            )

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(["factor", "regime"]).reset_index(drop=True)

    logger.info(
        f"Computed IC-by-regime stats: {len(result_df)} factor-regime pairs, "
        f"{result_df['factor'].nunique()} factors, {result_df['regime'].nunique()} regimes"
    )

    return result_df
