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

    This is the **single source of truth** for cost defaults used by both
    the backtest engine and the paper-trading engine.  Any change here
    propagates to both paths automatically.

    Values:
    - commission_bps = 1.0 (realistic for US equities, ~$0.001/share equiv.)
    - spread_w = 0.25 (25% of estimated bid/ask spread)
    - impact_w = 0.5 (50% of estimated market impact)

    Returns:
        CostModel instance with default parameters
    """
    return CostModel(commission_bps=1.0, spread_w=0.25, impact_w=0.5)
