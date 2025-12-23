"""Strategy adapters for Paper-Track.

This module provides thin adapter functions that map PaperTrackConfig
to existing strategy implementations (trend_baseline, multifactor_long_short)
without duplizierende Finanzlogik.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import pandas as pd

from src.assembled_core.strategies import (
    MultiFactorStrategyConfig,
    compute_multifactor_long_short_positions,
    generate_multifactor_long_short_signals,
)
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)
from src.assembled_core.signals.rules_trend import (
    generate_trend_signals_from_prices,
)

logger = logging.getLogger(__name__)


def generate_signals_and_targets_for_day(
    config: Any,
    state_before: Any,
    prices_full: pd.DataFrame,
    prices_filtered: pd.DataFrame,
    prices_with_features: pd.DataFrame,
    as_of: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate signals and target positions for a given day via strategy adapter.

    Args:
        config: PaperTrackConfig-like object (expects strategy_type, strategy_params)
        state_before: PaperTrackState-like object (expects equity attribute)
        prices_full: Full price panel up to as_of (used for multi-factor strategies)
        prices_filtered: One row per symbol (last <= as_of), used for pricing
        prices_with_features: Feature-enriched prices (for trend strategies)
        as_of: Run date (pd.Timestamp, UTC)

    Returns:
        Tuple of (signals_df, target_positions_df)
    """
    strategy_type = getattr(config, "strategy_type", None)

    if strategy_type == "trend_baseline":
        # Trend baseline uses existing TA feature pipeline + EMA rules
        params = getattr(config, "strategy_params", {}) or {}
        ma_fast = params.get("ma_fast")
        ma_slow = params.get("ma_slow")

        signals = generate_trend_signals_from_prices(
            prices_with_features,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
        )

        target_positions = compute_target_positions_from_trend_signals(
            signals,
            total_capital=getattr(state_before, "equity"),
            top_n=params.get("top_n"),
            min_score=params.get("min_score", 0.0),
        )

        return signals, target_positions

    if strategy_type == "multifactor_long_short":
        params = getattr(config, "strategy_params", {}) or {}
        bundle_path = params.get("bundle_path")
        if not bundle_path:
            raise ValueError(
                "strategy_type 'multifactor_long_short' requires "
                "strategy_params.bundle_path to be set"
            )

        mf_config = MultiFactorStrategyConfig(
            bundle_path=str(bundle_path),
            top_quantile=float(params.get("top_quantile", 0.2)),
            bottom_quantile=float(params.get("bottom_quantile", 0.2)),
            rebalance_freq=str(params.get("rebalance_freq", "M")),
            max_gross_exposure=float(params.get("max_gross_exposure", 1.0)),
            max_leverage=float(params.get("max_leverage", 1.0)),
            transaction_cost_bps=float(params.get("transaction_cost_bps", 5.0)),
            use_regime_overlay=bool(params.get("use_regime_overlay", False)),
            # regime_config / regime_risk_map können später ergänzt werden
        )

        # Use full price history for factor computation and signal generation
        signals_all = generate_multifactor_long_short_signals(
            prices_full,
            factors=None,
            config=mf_config,
        )

        if signals_all.empty:
            logger.info(
                "Multi-factor adapter: no signals generated for full history "
                f"(as_of={as_of.date()})"
            )
            return signals_all, pd.DataFrame(
                columns=["symbol", "target_weight", "target_qty"]
            )

        signals_all = signals_all.copy()
        timestamps = pd.to_datetime(signals_all["timestamp"], utc=True, errors="coerce")
        mask = timestamps.dt.normalize() == as_of.normalize()
        signals = signals_all[mask].copy()

        if signals.empty:
            # Kein Rebalance-Tag → keine Positionsänderung
            logger.info(
                "Multi-factor adapter: no rebalance on %s (no signals for this date)",
                as_of.date(),
            )
            return signals, pd.DataFrame(
                columns=["symbol", "target_weight", "target_qty"]
            )

        target_positions = compute_multifactor_long_short_positions(
            signals=signals,
            capital=float(getattr(state_before, "equity")),
            config=mf_config,
        )

        return signals, target_positions

    raise ValueError(f"Unsupported strategy_type in adapter: {strategy_type}")


