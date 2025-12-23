"""Multi-Factor Long/Short Strategy Module.

This module provides a multi-factor long/short strategy that:
1. Computes factors from prices
2. Builds multi-factor scores using factor bundles
3. Selects top/bottom quantiles for long/short positions
4. Rebalances at specified frequencies

Example:
    from src.assembled_core.strategies.multifactor_long_short import (
        MultiFactorStrategyConfig,
        generate_multifactor_long_short_signals,
        compute_multifactor_long_short_positions,
    )

    config = MultiFactorStrategyConfig(
        bundle_path="config/factor_bundles/macro_world_etfs_core_bundle.yaml",
        top_quantile=0.2,
        bottom_quantile=0.2,
        rebalance_freq="M",
    )

    # Generate signals
    signals = generate_multifactor_long_short_signals(prices_df, factors_df, config)

    # Compute positions
    positions = compute_multifactor_long_short_positions(signals, capital, config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.assembled_core.config.factor_bundles import load_factor_bundle
from src.assembled_core.risk.regime_models import RegimeStateConfig, build_regime_state
from src.assembled_core.signals.multifactor_signal import (
    build_multifactor_signal,
    select_top_bottom,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiFactorStrategyConfig:
    """Configuration for multi-factor long/short strategy.

    Attributes:
        bundle_path: Path to factor bundle YAML file
        top_quantile: Top quantile threshold for long positions (e.g., 0.2 for top 20%)
        bottom_quantile: Bottom quantile threshold for short positions (e.g., 0.2 for bottom 20%)
        rebalance_freq: Rebalancing frequency string (e.g., "M" for monthly, "W" for weekly, "D" for daily)
                       Must be a valid pandas frequency alias
        max_gross_exposure: Maximum gross exposure (long + short) as fraction of capital (default: 1.0)
                           Used as default if regime overlay is disabled or regime not in risk_map
        max_leverage: Maximum leverage (default: 1.0, meaning long + short <= capital)
        transaction_cost_bps: Transaction cost in basis points per roundtrip (default: 5.0)
        use_regime_overlay: Whether to use regime-based risk overlay (default: False)
        regime_config: Optional RegimeStateConfig for regime detection (default: None, uses default config)
        regime_risk_map: Optional dictionary mapping regime labels to risk parameters (default: None)
                        Example: {
                            "bull": {"max_gross_exposure": 1.2, "target_net_exposure": 0.6},
                            "neutral": {"max_gross_exposure": 1.0, "target_net_exposure": 0.2},
                            "bear": {"max_gross_exposure": 0.6, "target_net_exposure": 0.0},
                            "crisis": {"max_gross_exposure": 0.3, "target_net_exposure": 0.0},
                        }
    """

    bundle_path: str
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    rebalance_freq: str = "M"
    max_gross_exposure: float = 1.0
    max_leverage: float = 1.0
    transaction_cost_bps: float = 5.0
    use_regime_overlay: bool = False
    regime_config: RegimeStateConfig | None = None
    regime_risk_map: dict[str, dict[str, float]] | None = None


def _is_rebalance_date(
    timestamp: pd.Timestamp,
    rebalance_freq: str,
    timestamp_col: str = "timestamp",
) -> bool:
    """Check if a timestamp is a rebalancing date.

    Args:
        timestamp: Timestamp to check
        rebalance_freq: Rebalancing frequency (e.g., "M", "W", "D")
        timestamp_col: Name of timestamp column (unused, kept for API consistency)

    Returns:
        True if timestamp is a rebalancing date, False otherwise
    """
    # For monthly rebalancing, check if it's the first day of the month
    if rebalance_freq.upper() == "M":
        return timestamp.day == 1

    # For weekly rebalancing, check if it's Monday
    if rebalance_freq.upper() == "W":
        return timestamp.weekday() == 0  # Monday

    # For daily rebalancing, always return True
    if rebalance_freq.upper() == "D":
        return True

    # For custom frequencies, try pandas date offset
    try:
        # This is a simplified check - in practice, you'd want to track
        # the last rebalance date and check if enough time has passed
        # For now, we assume daily rebalancing if freq is not recognized
        logger.warning(
            f"Unknown rebalance_freq '{rebalance_freq}', defaulting to daily"
        )
        return True
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning(
            f"Could not parse rebalance_freq '{rebalance_freq}': {e}, defaulting to daily"
        )
        return True


def generate_multifactor_long_short_signals(
    prices: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    config: MultiFactorStrategyConfig | None = None,
) -> pd.DataFrame:
    """Generate multi-factor long/short signals.

    This function:
    1. Loads the factor bundle
    2. If factors are not provided, computes them from prices
    3. Builds multi-factor scores
    4. Selects top/bottom quantiles
    5. Filters to rebalancing dates

    Args:
        prices: DataFrame with columns: timestamp, symbol, close, ... (and optionally factor columns)
        factors: Optional pre-computed factors DataFrame. If None, factors will be computed from prices.
                 Must have same structure as prices but with additional factor columns.
        config: MultiFactorStrategyConfig with bundle path and strategy parameters.
                If None, uses default config with first available bundle.

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG" for top quantile, "SHORT" for bottom quantile, "FLAT" otherwise
        score: Multi-factor score (mf_score from bundle)
        regime: Regime label (if use_regime_overlay=True)
        Only includes rows for rebalancing dates

        Also returns regime_state_df as attribute (if use_regime_overlay=True),
        accessible via signals_df.attrs["regime_state_df"]

    Raises:
        FileNotFoundError: If bundle file not found
        ValueError: If required columns are missing or config is invalid
    """
    if config is None:
        # Use default config with first available bundle
        from src.assembled_core.config.factor_bundles import (
            list_available_factor_bundles,
        )

        bundles = list_available_factor_bundles()
        if not bundles:
            raise ValueError(
                "No factor bundles found. Please provide config.bundle_path"
            )

        config = MultiFactorStrategyConfig(bundle_path=str(bundles[0]))
        logger.info(f"Using default bundle: {config.bundle_path}")

    # Load bundle
    bundle = load_factor_bundle(config.bundle_path)
    logger.info(
        f"Loaded bundle: {bundle.universe}, factor_set={bundle.factor_set}, "
        f"{len(bundle.factors)} factors"
    )

    # Use provided factors or compute from prices
    if factors is None:
        from scripts.run_factor_analysis import compute_factors

        logger.info(
            f"Computing factors from prices (factor_set: {bundle.factor_set})..."
        )
        factors_df = compute_factors(
            prices,
            factor_set=bundle.factor_set,
            output_dir=None,  # Use default altdata directory
        )
    else:
        factors_df = factors.copy()
        logger.info(
            f"Using provided factors DataFrame with {len(factors_df.columns)} columns"
        )

    # Build multi-factor signal
    logger.info("Building multi-factor scores...")
    mf_result = build_multifactor_signal(
        factors_df,
        bundle=bundle,
        group_col="symbol",
        timestamp_col="timestamp",
    )

    mf_df = mf_result.df

    # Log metadata
    logger.info(
        f"Multi-factor signal built: {len(mf_result.meta['used_factors'])} factors used, "
        f"{len(mf_result.meta['missing_factors'])} missing"
    )
    logger.info(f"Used factors: {', '.join(mf_result.meta['used_factors'])}")

    # Select top/bottom quantiles
    logger.info(
        f"Selecting top {config.top_quantile:.1%} / bottom {config.bottom_quantile:.1%} quantiles..."
    )
    mf_df = select_top_bottom(
        mf_df,
        top_quantile=config.top_quantile,
        bottom_quantile=config.bottom_quantile,
        timestamp_col="timestamp",
        group_col="symbol",
        score_col="mf_score",
    )

    # Filter to rebalancing dates
    logger.info(f"Filtering to rebalancing dates (freq: {config.rebalance_freq})...")
    rebalance_mask = mf_df["timestamp"].apply(
        lambda ts: _is_rebalance_date(ts, config.rebalance_freq)
    )
    mf_df_rebalance = mf_df[rebalance_mask].copy()

    logger.info(
        f"Rebalancing dates: {mf_df_rebalance['timestamp'].nunique()} unique dates, "
        f"{len(mf_df_rebalance)} total rows"
    )

    # Build regime state if regime overlay is enabled
    regime_state_df = None
    if config.use_regime_overlay:
        logger.info("Regime overlay enabled. Building regime state...")

        try:
            # Try to extract macro factors, breadth, and vol from factors_df
            # (they might already be computed if factor_set includes them)
            macro_factors_df = None
            breadth_df = None
            vol_df = None

            # Check if macro factors are in factors_df
            macro_cols = [
                "macro_growth_regime",
                "macro_inflation_regime",
                "macro_risk_aversion_proxy",
            ]
            if all(col in factors_df.columns for col in macro_cols):
                macro_factors_df = factors_df[
                    ["timestamp", "symbol"] + macro_cols
                ].copy()

            # Check if volatility factors are in factors_df
            vol_cols = ["rv_20", "vov_20_60"]
            if any(col in factors_df.columns for col in vol_cols):
                available_vol_cols = [
                    col for col in vol_cols if col in factors_df.columns
                ]
                vol_df = factors_df[["timestamp", "symbol"] + available_vol_cols].copy()

            # Compute market breadth (if not already in factors_df)
            # We need to compute it from prices
            try:
                from src.assembled_core.features.market_breadth import (
                    compute_market_breadth_ma,
                )

                breadth_df = compute_market_breadth_ma(
                    prices,
                    ma_window=config.regime_config.breadth_ma_window
                    if config.regime_config
                    else 50,
                )

                # Also try to get risk_on_off_score
                try:
                    from src.assembled_core.features.market_breadth import (
                        compute_risk_on_off_indicators,
                    )

                    risk_indicators = compute_risk_on_off_indicators(prices)
                    if (
                        not risk_indicators.empty
                        and "risk_on_off_score" in risk_indicators.columns
                    ):
                        breadth_df = breadth_df.merge(
                            risk_indicators[["timestamp", "risk_on_off_score"]],
                            on="timestamp",
                            how="left",
                        )
                except Exception as e:
                    logger.debug(f"Could not compute risk_on_off_score: {e}")
            except Exception as e:
                logger.warning(
                    f"Could not compute market breadth for regime detection: {e}"
                )

            # Build regime state
            regime_state_df = build_regime_state(
                prices=prices,
                macro_factors=macro_factors_df,
                breadth_df=breadth_df,
                vol_df=vol_df,
                config=config.regime_config,
            )

            logger.info(
                f"Regime state built: {len(regime_state_df)} timestamps, "
                f"regime distribution: {regime_state_df['regime_label'].value_counts().to_dict()}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to build regime state: {e}. Continuing without regime overlay."
            )
            regime_state_df = None

    # Build signals DataFrame
    signals = []
    for timestamp in mf_df_rebalance["timestamp"].unique():
        timestamp_df = mf_df_rebalance[mf_df_rebalance["timestamp"] == timestamp]

        # Get regime label for this timestamp if regime overlay is enabled
        regime_label = None
        if regime_state_df is not None:
            regime_for_ts = regime_state_df[regime_state_df["timestamp"] == timestamp]
            if not regime_for_ts.empty:
                regime_label = regime_for_ts["regime_label"].iloc[0]

        # Long signals (top quantile)
        long_symbols = timestamp_df[timestamp_df["mf_long_flag"] == 1]
        for _, row in long_symbols.iterrows():
            signal_dict = {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": "LONG",
                "score": row["mf_score"],
            }
            if regime_label is not None:
                signal_dict["regime"] = regime_label
            signals.append(signal_dict)

        # Short signals (bottom quantile)
        short_symbols = timestamp_df[timestamp_df["mf_short_flag"] == 1]
        for _, row in short_symbols.iterrows():
            signal_dict = {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": "SHORT",
                "score": row["mf_score"],
            }
            if regime_label is not None:
                signal_dict["regime"] = regime_label
            signals.append(signal_dict)

    signals_df = pd.DataFrame(signals)

    # Store regime_state_df as attribute for position sizing function
    if regime_state_df is not None:
        signals_df.attrs["regime_state_df"] = regime_state_df

    if not signals_df.empty:
        signals_df = signals_df.sort_values(["timestamp", "symbol"]).reset_index(
            drop=True
        )
        logger.info(
            f"Generated signals: {len(signals_df)} total, "
            f"{len(signals_df[signals_df['direction'] == 'LONG'])} long, "
            f"{len(signals_df[signals_df['direction'] == 'SHORT'])} short"
        )

        # Log per-rebalance statistics
        for timestamp in signals_df["timestamp"].unique():
            ts_signals = signals_df[signals_df["timestamp"] == timestamp]
            n_long = len(ts_signals[ts_signals["direction"] == "LONG"])
            n_short = len(ts_signals[ts_signals["direction"] == "SHORT"])
            logger.debug(
                f"  {timestamp.strftime('%Y-%m-%d')}: {n_long} long, {n_short} short"
            )
    else:
        logger.warning("No signals generated (no symbols in top/bottom quantiles)")

    return signals_df


def compute_multifactor_long_short_positions(
    signals: pd.DataFrame,
    capital: float,
    config: MultiFactorStrategyConfig,
    regime_state_df: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
) -> pd.DataFrame:
    """Compute target positions from multi-factor long/short signals.

    This function:
    1. Splits signals into long and short
    2. Equal-weights positions within each side
    3. Applies regime-based or fixed exposure constraints
    4. Applies max_gross_exposure and max_leverage constraints

    Args:
        signals: DataFrame with columns: timestamp, symbol, direction, score
                 direction must be "LONG" or "SHORT"
                 Optionally contains "regime" column if regime overlay is enabled
        capital: Total capital available
        config: MultiFactorStrategyConfig with strategy parameters
        regime_state_df: Optional DataFrame with regime state (from build_regime_state)
                        Used if config.use_regime_overlay=True
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol column (default: "symbol")

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
        target_weight: Target weight (can be positive for long, negative for short)
        target_qty: Target quantity (positive for long, negative for short)

    Raises:
        ValueError: If signals DataFrame is empty or missing required columns
    """
    if signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    required_cols = [timestamp_col, group_col, "direction"]
    missing_cols = [col for col in required_cols if col not in signals.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Available: {list(signals.columns)}"
        )

    # Compute positions for the latest timestamp in signals
    # Note: This function is called per timestamp during backtest,
    # so signals should already be filtered to the current timestamp
    latest_timestamp = signals[timestamp_col].max()
    latest_signals = signals[signals[timestamp_col] == latest_timestamp].copy()

    if latest_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    # Get regime_state_df from signals.attrs if available
    if (
        regime_state_df is None
        and hasattr(signals, "attrs")
        and "regime_state_df" in signals.attrs
    ):
        regime_state_df = signals.attrs["regime_state_df"]

    # Determine risk parameters (regime-based or fixed)
    max_gross_exp = config.max_gross_exposure
    target_net_exp = 0.0  # Default: market neutral

    current_regime = None
    if config.use_regime_overlay and regime_state_df is not None:
        # Get regime for current timestamp
        regime_for_ts = regime_state_df[
            regime_state_df[timestamp_col] == latest_timestamp
        ]
        if not regime_for_ts.empty:
            current_regime = regime_for_ts["regime_label"].iloc[0]

            # Get risk parameters from regime_risk_map
            if config.regime_risk_map and current_regime in config.regime_risk_map:
                risk_params = config.regime_risk_map[current_regime]
                max_gross_exp = risk_params.get(
                    "max_gross_exposure", config.max_gross_exposure
                )
                target_net_exp = risk_params.get("target_net_exposure", 0.0)

                logger.info(
                    f"Regime '{current_regime}' at {latest_timestamp.strftime('%Y-%m-%d')}: "
                    f"max_gross_exposure={max_gross_exp:.2%}, target_net_exposure={target_net_exp:.2%}"
                )
            else:
                logger.debug(
                    f"Regime '{current_regime}' not found in regime_risk_map. "
                    f"Using default parameters."
                )

    # Split long and short
    long_signals = latest_signals[latest_signals["direction"] == "LONG"].copy()
    short_signals = latest_signals[latest_signals["direction"] == "SHORT"].copy()

    positions = []

    # Calculate weights based on regime parameters
    # Formula:
    # Long side = (max_gross_exposure / 2 + target_net_exposure / 2)
    # Short side = (max_gross_exposure / 2 - target_net_exposure / 2)
    # This ensures:
    # - Gross exposure = long + short = max_gross_exposure
    # - Net exposure = long - short = target_net_exposure

    # Long positions: equal-weighted within long side
    if not long_signals.empty:
        n_long = len(long_signals)
        # Long side allocation
        long_side_allocation = (max_gross_exp / 2.0) + (target_net_exp / 2.0)
        long_weight_per_symbol = long_side_allocation / n_long

        for _, row in long_signals.iterrows():
            positions.append(
                {
                    "symbol": row[group_col],
                    "target_weight": long_weight_per_symbol,
                    "target_qty": long_weight_per_symbol * capital,
                }
            )

    # Short positions: equal-weighted within short side (negative weights)
    if not short_signals.empty:
        n_short = len(short_signals)
        # Short side allocation (negative)
        short_side_allocation = (max_gross_exp / 2.0) - (target_net_exp / 2.0)
        short_weight_per_symbol = -short_side_allocation / n_short

        for _, row in short_signals.iterrows():
            positions.append(
                {
                    "symbol": row[group_col],
                    "target_weight": short_weight_per_symbol,
                    "target_qty": short_weight_per_symbol
                    * capital,  # Negative quantity for short
                }
            )

    positions_df = pd.DataFrame(positions)

    if not positions_df.empty:
        positions_df = positions_df.sort_values("symbol").reset_index(drop=True)

        # Log position summary
        total_long_weight = positions_df[positions_df["target_weight"] > 0][
            "target_weight"
        ].sum()
        total_short_weight = abs(
            positions_df[positions_df["target_weight"] < 0]["target_weight"].sum()
        )
        net_weight = positions_df["target_weight"].sum()
        gross_weight = total_long_weight + total_short_weight

        logger.info(
            f"Computed positions: {len(positions_df)} total, "
            f"{len(positions_df[positions_df['target_weight'] > 0])} long, "
            f"{len(positions_df[positions_df['target_weight'] < 0])} short"
        )
        if current_regime is not None:
            logger.info(
                f"  Regime: {current_regime}, "
                f"Gross exposure: {gross_weight:.2%}, Net exposure: {net_weight:.2%}, "
                f"Long: {total_long_weight:.2%}, Short: {total_short_weight:.2%}"
            )
        else:
            logger.info(
                f"  Gross exposure: {gross_weight:.2%}, Net exposure: {net_weight:.2%}, "
                f"Long: {total_long_weight:.2%}, Short: {total_short_weight:.2%}"
            )

    return positions_df
