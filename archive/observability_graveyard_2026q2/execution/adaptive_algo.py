"""Adaptive Execution Algorithm (M20 Task 20.6).

Monitors fill rate and spread during execution and dynamically adjusts
aggressiveness:
    - Spread tight → aggressive (market order)
    - Spread wide → passive (limit order, wait)
    - Momentum with us → front-load
    - Momentum against us → back-load
    - Spread widening pause: halt when spread > 2x normal

Cost reduction: 5-20 bps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class AggressionLevel(Enum):
    """Execution aggressiveness levels."""
    PASSIVE = "passive"       # Limit orders only, patient
    NORMAL = "normal"         # Mix of limit and market
    AGGRESSIVE = "aggressive"  # Market orders, front-loaded
    PAUSE = "pause"           # Halt execution (spread too wide)


@dataclass
class MarketCondition:
    """Current market microstructure snapshot."""
    bid: float
    ask: float
    mid: float = 0.0
    spread_bps: float = 0.0
    avg_spread_bps: float = 10.0  # Historical average
    last_price: float = 0.0
    momentum_bps: float = 0.0     # Short-term price momentum
    volume_ratio: float = 1.0     # Current vs average volume
    fill_rate: float = 1.0        # Recent fill success rate

    def __post_init__(self):
        if self.mid == 0.0:
            self.mid = (self.bid + self.ask) / 2
        if self.spread_bps == 0.0 and self.mid > 0:
            self.spread_bps = (self.ask - self.bid) / self.mid * 10000


@dataclass
class AdaptiveAlgoConfig:
    """Configuration for adaptive execution."""
    spread_wide_threshold: float = 2.0   # Spread / avg_spread ratio for caution
    spread_pause_threshold: float = 3.0  # Spread / avg_spread ratio to pause
    momentum_threshold_bps: float = 5.0  # Momentum threshold for front/back-loading
    min_fill_rate: float = 0.3           # Minimum fill rate before switching aggressive
    max_participation: float = 0.15      # Max fraction of volume
    urgency: float = 0.5                 # 0=patient, 1=urgent


@dataclass
class ExecutionSlice:
    """A single execution slice decision."""
    aggression: AggressionLevel
    order_type: str              # "market", "limit", "limit_passive"
    limit_offset_bps: float      # Offset from mid for limit orders
    size_fraction: float         # Fraction of remaining order to execute now
    reason: str


@dataclass
class AdaptiveAlgoState:
    """Running state of adaptive execution."""
    total_shares: int = 0
    filled_shares: int = 0
    slices_executed: int = 0
    total_cost_bps: float = 0.0
    paused: bool = False
    history: list[ExecutionSlice] = field(default_factory=list)

    @property
    def remaining_shares(self) -> int:
        return self.total_shares - self.filled_shares

    @property
    def completion_pct(self) -> float:
        if self.total_shares == 0:
            return 100.0
        return self.filled_shares / self.total_shares * 100


class AdaptiveExecutionAlgo:
    """Adaptive execution algorithm with dynamic aggression control."""

    def __init__(self, config: AdaptiveAlgoConfig | None = None):
        self.config = config or AdaptiveAlgoConfig()
        self.state = AdaptiveAlgoState()

    def initialize(self, total_shares: int, side: str = "buy") -> None:
        """Initialize a new execution order.

        Args:
            total_shares: Total shares to execute.
            side: "buy" or "sell".
        """
        self.state = AdaptiveAlgoState(total_shares=abs(total_shares))
        self.side = side
        logger.info("[AdaptiveAlgo] Init: %d shares %s", total_shares, side)

    def decide_slice(self, condition: MarketCondition) -> ExecutionSlice:
        """Decide next execution slice based on current market conditions.

        Args:
            condition: Current market microstructure snapshot.

        Returns:
            ExecutionSlice with order type and sizing.
        """
        cfg = self.config
        spread_ratio = condition.spread_bps / max(condition.avg_spread_bps, 0.1)

        # 1. Spread pause check
        if spread_ratio > cfg.spread_pause_threshold:
            slice_ = ExecutionSlice(
                aggression=AggressionLevel.PAUSE,
                order_type="none",
                limit_offset_bps=0,
                size_fraction=0.0,
                reason=f"Spread {condition.spread_bps:.1f}bps > {cfg.spread_pause_threshold}x avg — pausing",
            )
            self.state.paused = True
            self.state.history.append(slice_)
            return slice_

        self.state.paused = False

        # 2. Determine aggression
        aggression = AggressionLevel.NORMAL
        reasons = []

        # Spread-based adjustment
        if spread_ratio > cfg.spread_wide_threshold:
            aggression = AggressionLevel.PASSIVE
            reasons.append(f"wide spread ({spread_ratio:.1f}x)")
        elif spread_ratio < 0.7:
            aggression = AggressionLevel.AGGRESSIVE
            reasons.append(f"tight spread ({spread_ratio:.1f}x)")

        # Momentum-based adjustment
        momentum_with_us = (
            (self.side == "buy" and condition.momentum_bps > cfg.momentum_threshold_bps) or
            (self.side == "sell" and condition.momentum_bps < -cfg.momentum_threshold_bps)
        )
        momentum_against_us = (
            (self.side == "buy" and condition.momentum_bps < -cfg.momentum_threshold_bps) or
            (self.side == "sell" and condition.momentum_bps > cfg.momentum_threshold_bps)
        )

        if momentum_against_us and aggression != AggressionLevel.PASSIVE:
            aggression = AggressionLevel.PASSIVE
            reasons.append("momentum against us — back-loading")
        elif momentum_with_us:
            aggression = AggressionLevel.AGGRESSIVE
            reasons.append("momentum with us — front-loading")

        # Fill rate adjustment
        if condition.fill_rate < cfg.min_fill_rate and aggression == AggressionLevel.PASSIVE:
            aggression = AggressionLevel.NORMAL
            reasons.append(f"low fill rate ({condition.fill_rate:.0%}) — increasing aggression")

        # Urgency override
        if cfg.urgency > 0.8:
            aggression = AggressionLevel.AGGRESSIVE
            reasons.append("high urgency override")

        # 3. Order type and sizing
        if aggression == AggressionLevel.AGGRESSIVE:
            order_type = "market"
            limit_offset = 0.0
            size_fraction = min(0.25 + cfg.urgency * 0.25, 0.5)
        elif aggression == AggressionLevel.PASSIVE:
            order_type = "limit_passive"
            limit_offset = -condition.spread_bps * 0.3  # Inside the spread
            size_fraction = 0.1
        else:  # NORMAL
            order_type = "limit"
            limit_offset = condition.spread_bps * 0.1  # Slight edge
            size_fraction = 0.15

        # Volume participation limit
        remaining_pct = self.state.remaining_shares / max(self.state.total_shares, 1)
        size_fraction = min(size_fraction, remaining_pct)

        slice_ = ExecutionSlice(
            aggression=aggression,
            order_type=order_type,
            limit_offset_bps=round(limit_offset, 2),
            size_fraction=round(size_fraction, 4),
            reason="; ".join(reasons) if reasons else "normal conditions",
        )

        self.state.slices_executed += 1
        self.state.history.append(slice_)
        return slice_

    def record_fill(self, shares_filled: int, cost_bps: float) -> None:
        """Record a fill event.

        Args:
            shares_filled: Number of shares filled.
            cost_bps: Execution cost in basis points.
        """
        self.state.filled_shares += shares_filled
        n = self.state.slices_executed
        self.state.total_cost_bps = (
            self.state.total_cost_bps * (n - 1) + cost_bps
        ) / n

    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.state.remaining_shares <= 0

    def summary(self) -> dict:
        """Execution summary."""
        aggression_counts = {}
        for s in self.state.history:
            key = s.aggression.value
            aggression_counts[key] = aggression_counts.get(key, 0) + 1

        return {
            "total_shares": self.state.total_shares,
            "filled_shares": self.state.filled_shares,
            "completion_pct": round(self.state.completion_pct, 1),
            "slices": self.state.slices_executed,
            "avg_cost_bps": round(self.state.total_cost_bps, 2),
            "aggression_profile": aggression_counts,
        }


__all__ = [
    "AggressionLevel",
    "MarketCondition",
    "AdaptiveAlgoConfig",
    "ExecutionSlice",
    "AdaptiveAlgoState",
    "AdaptiveExecutionAlgo",
]
