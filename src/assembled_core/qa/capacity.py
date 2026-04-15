"""Strategy Capacity Estimation (M16.5).

Estimates the maximum AUM at which a strategy's alpha remains positive
after accounting for market impact costs via the square-root law.

Usage:
    from src.assembled_core.qa.capacity import estimate_strategy_capacity
    cap = estimate_strategy_capacity(trades_df, adv_df)
    print(f"Max AUM: ${cap.max_aum_usd:,.0f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapacityEstimate:
    """Result of strategy capacity estimation."""

    max_aum_usd: float
    breakeven_aum_usd: float
    alpha_gross_bps: float
    alpha_net_at_target_bps: float
    participation_at_target_pct: float
    target_aum_usd: float
    verdict: str  # "ok", "warning", "exceeded"


def estimate_strategy_capacity(
    trades_df: pd.DataFrame,
    adv_df: pd.DataFrame | None = None,
    *,
    alpha_gross_bps: float = 1500.0,
    target_aum_usd: float = 10_000_000.0,
    avg_adv_usd: float = 500_000_000.0,
    max_participation_pct: float = 1.0,
    impact_coeff: float = 0.1,
    n_positions: int = 20,
) -> CapacityEstimate:
    """Estimate strategy capacity using square-root market impact model.

    Model: alpha_net(AUM) = alpha_gross - impact_coeff * sqrt(AUM / (n * avg_ADV))
    Breakeven: AUM where alpha_net = 0.

    Args:
        trades_df: DataFrame with trade history (columns: symbol, notional, date).
        adv_df: DataFrame with ADV per symbol (columns: symbol, adv_usd).
            If None, uses avg_adv_usd for all symbols.
        alpha_gross_bps: Gross alpha in basis points (before impact).
        target_aum_usd: Target AUM to evaluate capacity at.
        avg_adv_usd: Default ADV per symbol if adv_df not provided.
        max_participation_pct: Max % of ADV per trade (default: 1%).
        impact_coeff: Market impact coefficient (default: 0.1).
        n_positions: Average number of positions.

    Returns:
        CapacityEstimate with breakeven AUM and net alpha at target.
    """
    # Compute average ADV from provided data or use default
    if adv_df is not None and not adv_df.empty and "adv_usd" in adv_df.columns:
        mean_adv = float(adv_df["adv_usd"].median())
    else:
        mean_adv = avg_adv_usd

    if mean_adv <= 0:
        mean_adv = avg_adv_usd

    alpha_gross = alpha_gross_bps / 10_000.0  # Convert bps to decimal

    # Square-root impact model: cost = k * sqrt(trade_size / ADV)
    # With AUM spread across n positions: trade_per_position = AUM / n
    # participation = (AUM / n) / ADV
    # impact_per_trade = k * sqrt(participation) * 10000 bps

    def _alpha_net_bps(aum: float) -> float:
        participation = (aum / n_positions) / mean_adv
        impact_bps = impact_coeff * np.sqrt(participation) * 10_000
        return alpha_gross_bps - impact_bps

    # Breakeven: find AUM where alpha_net = 0
    # alpha_gross = k * sqrt(AUM / (n * ADV)) * 10000
    # => AUM_breakeven = n * ADV * (alpha_gross / (k * 10000))^2
    breakeven = n_positions * mean_adv * (alpha_gross / impact_coeff) ** 2

    # Net alpha at target AUM
    alpha_net_target = _alpha_net_bps(target_aum_usd)
    participation_target = (target_aum_usd / n_positions) / mean_adv * 100

    # Max AUM at max_participation constraint
    max_aum_participation = n_positions * mean_adv * (max_participation_pct / 100)

    # Effective max = min(breakeven, participation limit)
    max_aum = min(breakeven, max_aum_participation)

    if target_aum_usd > breakeven:
        verdict = "exceeded"
    elif target_aum_usd > breakeven * 0.7:
        verdict = "warning"
    else:
        verdict = "ok"

    logger.info(
        "[Capacity] Gross=%.0f bps, Net@$%.0fM=%.0f bps, Breakeven=$%.0fM, verdict=%s",
        alpha_gross_bps, target_aum_usd / 1e6,
        alpha_net_target, breakeven / 1e6, verdict,
    )

    return CapacityEstimate(
        max_aum_usd=round(max_aum, 0),
        breakeven_aum_usd=round(breakeven, 0),
        alpha_gross_bps=alpha_gross_bps,
        alpha_net_at_target_bps=round(alpha_net_target, 1),
        participation_at_target_pct=round(participation_target, 4),
        target_aum_usd=target_aum_usd,
        verdict=verdict,
    )
