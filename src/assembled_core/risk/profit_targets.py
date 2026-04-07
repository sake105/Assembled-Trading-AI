"""Tiered profit targets and partial exit logic (M16 E2).

Implements staged profit locking:
  +10% gain → sell 25% of position
  +20% gain → sell additional 25% (50% total out)
  +35% gain → sell additional 25% (75% total out)
  Remainder → managed by trailing stops

For short positions the logic is mirrored (negative returns = profit).

Usage in backtest loop:
    pt_reductions = check_profit_targets(current_positions, config)
    for sym, factor in pt_reductions.items():
        new_weights[sym] = new_weights.get(sym, 0.0) * factor
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


@dataclass
class ProfitTargetConfig:
    """Configuration for tiered profit targets.

    tiers: list of (gain_threshold, sell_fraction) tuples.
      gain_threshold: unrealized gain at which to trigger partial exit.
      sell_fraction: fraction of ORIGINAL position to sell at this tier.
    Tiers should be ordered by ascending threshold.
    """
    tiers: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.10, 0.25),   # +10%: sell 25% of position
        (0.20, 0.25),   # +20%: sell another 25% (50% total)
        (0.35, 0.25),   # +35%: sell another 25% (75% total)
    ])
    apply_to_shorts: bool = True   # mirror logic for short positions


@dataclass
class PositionRecord:
    """Tracks entry price and cumulative exits for one position."""
    symbol: str
    entry_price: float
    is_long: bool = True
    tiers_triggered: set[int] = field(default_factory=set)

    def unrealized_pnl(self, current_price: float) -> float:
        """Fractional unrealized P&L (positive = gain)."""
        if self.entry_price <= 0:
            return 0.0
        pnl = (current_price / self.entry_price) - 1.0
        return pnl if self.is_long else -pnl


def check_profit_targets(
    positions: dict[str, PositionRecord],
    current_prices: dict[str, float],
    config: ProfitTargetConfig | None = None,
) -> dict[str, float]:
    """Check each position against profit tiers.

    Args:
        positions: Dict of symbol → PositionRecord (tracks entry and state).
        current_prices: Dict of symbol → current close price.
        config: Profit target configuration.

    Returns:
        Dict of symbol → weight_scale_factor (1.0 = no change, 0.75 = sell 25%, etc.)
        Only symbols with triggered tiers appear in the result.
    """
    config = config or ProfitTargetConfig()
    reductions: dict[str, float] = {}

    for sym, pos in positions.items():
        price = current_prices.get(sym)
        if price is None or price <= 0:
            continue
        if not pos.is_long and not config.apply_to_shorts:
            continue

        pnl = pos.unrealized_pnl(price)

        # Find newly triggered tiers
        newly_triggered_fraction = 0.0
        for i, (threshold, sell_frac) in enumerate(config.tiers):
            if i in pos.tiers_triggered:
                continue
            if pnl >= threshold:
                pos.tiers_triggered.add(i)
                newly_triggered_fraction += sell_frac
                _log.info(
                    "PROFIT TARGET tier %d triggered for %s: pnl=+%.1f%%, selling %.0f%%",
                    i + 1, sym, pnl * 100, sell_frac * 100,
                )

        if newly_triggered_fraction > 0:
            # Scale factor: keep (1 - sell_fraction) of current weight
            reductions[sym] = max(0.0, 1.0 - newly_triggered_fraction)

    return reductions


def build_position_records(
    current_weights: dict[str, float],
    entry_prices: dict[str, float],
    existing_records: dict[str, PositionRecord] | None = None,
) -> dict[str, PositionRecord]:
    """Build or update position records for the current holdings.

    Preserves existing records (with their tier state) for known positions.
    Creates new records for new entries.

    Args:
        current_weights: Current portfolio weights.
        entry_prices: Entry prices per symbol (current price for new positions).
        existing_records: Previously tracked PositionRecord objects.

    Returns:
        Updated dict of symbol → PositionRecord.
    """
    existing = existing_records or {}
    result: dict[str, PositionRecord] = {}

    for sym, weight in current_weights.items():
        if abs(weight) < 1e-6:
            continue
        is_long = weight > 0

        if sym in existing:
            rec = existing[sym]
            # If direction changed, reset the record
            if rec.is_long != is_long:
                rec = PositionRecord(
                    symbol=sym,
                    entry_price=entry_prices.get(sym, 0.0),
                    is_long=is_long,
                )
            result[sym] = rec
        else:
            result[sym] = PositionRecord(
                symbol=sym,
                entry_price=entry_prices.get(sym, 0.0),
                is_long=is_long,
            )

    return result


__all__ = [
    "ProfitTargetConfig",
    "PositionRecord",
    "check_profit_targets",
    "build_position_records",
]
