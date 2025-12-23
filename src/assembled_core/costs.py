"""Cost model configuration for portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Cost model parameters for portfolio simulation.

    Attributes:
        commission_bps: Commission in basis points (1 bps = 0.01%)
        spread_w: Spread weight (multiplier for bid/ask spread)
        impact_w: Market impact weight (multiplier for price impact)
    """

    commission_bps: float
    spread_w: float
    impact_w: float


def get_default_cost_model() -> CostModel:
    """Get default cost model parameters.

    Default values are based on best grid result from cost sensitivity analysis:
    - commission_bps = 0.0 (no commission)
    - spread_w = 0.25 (25% of spread)
    - impact_w = 0.5 (50% of impact)

    Returns:
        CostModel instance with default parameters
    """
    return CostModel(commission_bps=0.0, spread_w=0.25, impact_w=0.5)
