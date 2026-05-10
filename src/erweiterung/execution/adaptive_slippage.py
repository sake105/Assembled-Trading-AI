"""Adaptive Slippage-Modell — empirisch kalibriert.

Theorie
-------
Slippage hat mehrere Komponenten:
1. **Half-Spread**: Bid-Ask-Spread / 2.
2. **Linear Impact**: a × (order_size / ADV).
3. **Volatility-Impact**: b × σ × √(order_size / ADV).
4. **Time-of-Day-Effekt**: höhere Slippage bei Open/Close.

Ein robustes Modell:
    slippage_bps = half_spread_bps + a × (Q/ADV) + b × σ × √(Q/ADV)

mit a, b empirisch geschätzt aus historischen Fills.
"""

from __future__ import annotations

import numpy as np


def slippage_bps(
    order_size_shares: float,
    avg_daily_volume: float,
    half_spread_bps: float = 1.0,
    daily_volatility: float = 0.02,
    a_linear: float = 100.0,
    b_sqrt: float = 50.0,
    time_of_day_multiplier: float = 1.0,
) -> float:
    """Slippage in basis points.

    Args:
        order_size_shares: |Q|.
        avg_daily_volume: ADV (shares).
        half_spread_bps: typical bid-ask half-spread.
        daily_volatility: σ.
        a_linear: linear impact coefficient (bps per 100% of ADV).
        b_sqrt: sqrt impact coefficient.
        time_of_day_multiplier: 1.0 default; >1 for open/close.

    Returns:
        Slippage in bps.
    """
    if avg_daily_volume <= 0:
        return half_spread_bps + 100.0 * time_of_day_multiplier
    participation = abs(order_size_shares) / avg_daily_volume
    linear = a_linear * participation
    sqrt_term = b_sqrt * daily_volatility * 100 * np.sqrt(participation)
    return float(time_of_day_multiplier * (half_spread_bps + linear + sqrt_term))


def execution_price(
    target_price: float,
    side: int,
    slippage_bps_value: float,
) -> float:
    """Apply slippage.  Buy: price * (1 + bps/10000); Sell: * (1 - bps/10000)."""
    sign = 1 if side > 0 else -1
    return float(target_price * (1 + sign * slippage_bps_value / 10000))


__all__ = ["slippage_bps", "execution_price"]
