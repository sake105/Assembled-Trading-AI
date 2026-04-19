"""Regime-Conditional Transaction Cost Model (M39 Task 39.2).

Adjusts transaction cost estimates based on market regime:
- Crisis: spreads widen 2-5x, market impact increases, fill rates drop
- Stressed: moderate cost increase
- Normal: baseline costs
- Calm: potentially lower costs due to tight spreads

Reference:
    Almgren et al. (2005) "Optimal execution with nonlinear impact functions"
    Cont & Kukanov (2017) "Optimal order placement in limit order markets"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeCostConfig:
    """Regime-conditional cost model configuration."""
    # Base costs (normal regime)
    base_spread_bps: float = 5.0
    base_impact_bps: float = 3.0
    base_commission_bps: float = 1.0

    # Regime multipliers
    crisis_spread_mult: float = 4.0
    crisis_impact_mult: float = 3.0
    stressed_spread_mult: float = 2.0
    stressed_impact_mult: float = 1.8
    calm_spread_mult: float = 0.7
    calm_impact_mult: float = 0.8

    # Fill rate by regime
    normal_fill_rate: float = 0.95
    stressed_fill_rate: float = 0.80
    crisis_fill_rate: float = 0.60

    # Urgency scaling
    urgency_cost_mult: float = 2.0  # Max multiplier for urgent trades


@dataclass
class RegimeCostEstimate:
    """Transaction cost estimate for a trade."""
    spread_cost_bps: float
    impact_cost_bps: float
    commission_bps: float
    total_cost_bps: float
    expected_fill_rate: float
    regime: str
    regime_multiplier: float
    urgency_factor: float


def estimate_regime_costs(
    trade_value: float,
    adv: float,
    regime: str = "normal",
    urgency: float = 0.5,
    vix_level: float | None = None,
    config: RegimeCostConfig | None = None,
) -> RegimeCostEstimate:
    """Estimate transaction costs conditional on market regime.

    Args:
        trade_value: Absolute trade notional value.
        adv: Average daily volume in dollars.
        regime: Market regime ("crisis", "stressed", "normal", "calm").
        urgency: Trade urgency [0=patient, 1=immediate].
        vix_level: Optional VIX for continuous regime adjustment.
        config: Cost model configuration.

    Returns:
        RegimeCostEstimate.
    """
    cfg = config or RegimeCostConfig()

    # Participation rate
    participation = trade_value / max(adv, 1) if adv > 0 else 1.0

    # Regime-specific multipliers
    regime_map = {
        "crisis": (cfg.crisis_spread_mult, cfg.crisis_impact_mult, cfg.crisis_fill_rate),
        "stressed": (cfg.stressed_spread_mult, cfg.stressed_impact_mult, cfg.stressed_fill_rate),
        "normal": (1.0, 1.0, cfg.normal_fill_rate),
        "calm": (cfg.calm_spread_mult, cfg.calm_impact_mult, min(0.99, cfg.normal_fill_rate + 0.03)),
    }

    spread_mult, impact_mult, fill_rate = regime_map.get(regime, (1.0, 1.0, 0.95))

    # Continuous VIX adjustment (override discrete regime if VIX available)
    if vix_level is not None:
        vix_ratio = vix_level / 20.0  # Normalize to "normal" VIX
        spread_mult = max(spread_mult, np.clip(vix_ratio, 0.5, 5.0))
        impact_mult = max(impact_mult, np.clip(vix_ratio ** 0.7, 0.5, 3.5))
        fill_rate = min(fill_rate, np.clip(1.0 - (vix_ratio - 1) * 0.15, 0.5, 0.99))

    # Urgency factor
    urgency_factor = 1.0 + urgency * (cfg.urgency_cost_mult - 1.0)

    # Cost components
    spread_cost = cfg.base_spread_bps * spread_mult * urgency_factor
    impact_cost = cfg.base_impact_bps * impact_mult * np.sqrt(participation) * urgency_factor
    commission = cfg.base_commission_bps

    total = spread_cost + impact_cost + commission
    regime_mult = total / (cfg.base_spread_bps + cfg.base_impact_bps + cfg.base_commission_bps)

    logger.debug(
        "[RegimeCost] %s regime: spread=%.1f, impact=%.1f, total=%.1f bps (%.1fx), fill=%.0f%%",
        regime, spread_cost, impact_cost, total, regime_mult, fill_rate * 100,
    )

    return RegimeCostEstimate(
        spread_cost_bps=round(spread_cost, 2),
        impact_cost_bps=round(impact_cost, 2),
        commission_bps=round(commission, 2),
        total_cost_bps=round(total, 2),
        expected_fill_rate=round(fill_rate, 4),
        regime=regime,
        regime_multiplier=round(regime_mult, 2),
        urgency_factor=round(urgency_factor, 2),
    )


def compute_crisis_cost_multiplier(
    vix_level: float,
    bid_ask_spread: float | None = None,
    historical_spread: float | None = None,
) -> float:
    """Compute crisis cost multiplier from market observables.

    Args:
        vix_level: Current VIX.
        bid_ask_spread: Current bid-ask spread.
        historical_spread: Historical median bid-ask spread.

    Returns:
        Cost multiplier (1.0 = normal, >1 = elevated costs).
    """
    # VIX component
    vix_mult = np.clip(vix_level / 20.0, 0.5, 5.0)

    # Spread component (if available)
    spread_mult = 1.0
    if bid_ask_spread is not None and historical_spread is not None and historical_spread > 0:
        spread_mult = np.clip(bid_ask_spread / historical_spread, 0.5, 5.0)

    # Combined: geometric mean
    multiplier = float(np.sqrt(vix_mult * spread_mult))
    return round(np.clip(multiplier, 0.5, 5.0), 2)


def adjust_sizing_for_regime(
    target_weights: pd.Series,
    regime: str,
    regime_cost_config: RegimeCostConfig | None = None,
    max_turnover: float = 0.20,
) -> pd.Series:
    """Reduce position sizing when regime costs are elevated.

    In crisis: reduce turnover to avoid excessive transaction costs.
    Implements cost-aware trade throttling.

    Args:
        target_weights: Target portfolio weights.
        regime: Current market regime.
        regime_cost_config: Cost configuration.
        max_turnover: Maximum allowed daily turnover.

    Returns:
        Adjusted target weights (closer to current if costs high).
    """
    cfg = regime_cost_config or RegimeCostConfig()  # noqa: F841

    # Regime-specific turnover limits
    turnover_limits = {
        "crisis": max_turnover * 0.25,    # 75% reduction in crisis
        "stressed": max_turnover * 0.50,  # 50% reduction
        "normal": max_turnover,
        "calm": max_turnover * 1.1,       # Slightly more active in calm
    }

    allowed_turnover = turnover_limits.get(regime, max_turnover)
    total_turnover = target_weights.abs().sum()

    if total_turnover > allowed_turnover and total_turnover > 0:
        scale = allowed_turnover / total_turnover
        adjusted = target_weights * scale
        logger.info("[RegimeCost] Turnover throttled: %.1f%% → %.1f%% (%s regime)",
                    total_turnover * 100, allowed_turnover * 100, regime)
        return adjusted

    return target_weights


__all__ = [
    "RegimeCostConfig",
    "RegimeCostEstimate",
    "estimate_regime_costs",
    "compute_crisis_cost_multiplier",
    "adjust_sizing_for_regime",
]
