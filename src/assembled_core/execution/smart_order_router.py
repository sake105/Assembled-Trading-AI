"""Smart Order Router (M20.3) — Institutional-Grade Multi-Venue Routing.

Replaces the original 3-venue stub with an ADV-aware, regime-aware SOR
that supports order splitting across venues, participation rate limits,
and cost-optimized routing.

Key features:
  - Multi-venue routing with configurable venue characteristics
  - ADV-based participation rate limits (default 5% per venue per interval)
  - Regime-aware spread/fill adjustments (wider spreads in crisis)
  - Order splitting for large orders across multiple venues
  - Cost scoring that balances spread, fill probability, and latency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Venue configuration
# ---------------------------------------------------------------------------


@dataclass
class VenueConfig:
    """Configuration for a simulated execution venue.

    Attributes:
        name: Venue identifier.
        spread_bps: Typical bid-ask spread in basis points.
        fill_probability: Base probability of fill (0-1).
        latency_ms: Simulated latency in milliseconds.
        max_participation_pct: Max fraction of ADV allowed per interval.
        rebate_bps: Maker rebate in basis points (positive = rebate).
        fee_bps: Taker fee in basis points.
        dark_pool: Whether this is a dark pool (no pre-trade transparency).
        min_order_size: Minimum order size (shares) for this venue.
    """

    name: str
    spread_bps: float = 5.0
    fill_probability: float = 0.95
    latency_ms: float = 1.0
    max_participation_pct: float = 0.05
    rebate_bps: float = 0.0
    fee_bps: float = 0.3
    dark_pool: bool = False
    min_order_size: float = 1.0


DEFAULT_VENUES = [
    VenueConfig(
        "NYSE",
        spread_bps=3.0,
        fill_probability=0.95,
        latency_ms=0.5,
        fee_bps=0.30,
        rebate_bps=0.20,
    ),
    VenueConfig(
        "NASDAQ",
        spread_bps=3.5,
        fill_probability=0.93,
        latency_ms=0.3,
        fee_bps=0.30,
        rebate_bps=0.25,
    ),
    VenueConfig(
        "ARCA",
        spread_bps=4.0,
        fill_probability=0.90,
        latency_ms=0.8,
        fee_bps=0.25,
        rebate_bps=0.15,
    ),
    VenueConfig(
        "IEX",
        spread_bps=3.0,
        fill_probability=0.85,
        latency_ms=1.5,
        fee_bps=0.09,
        rebate_bps=0.0,
        min_order_size=100.0,
    ),
    VenueConfig(
        "dark_pool_1",
        spread_bps=0.5,
        fill_probability=0.35,
        latency_ms=5.0,
        fee_bps=0.10,
        rebate_bps=0.0,
        dark_pool=True,
        max_participation_pct=0.02,
    ),
    VenueConfig(
        "dark_pool_2",
        spread_bps=1.0,
        fill_probability=0.45,
        latency_ms=3.0,
        fee_bps=0.15,
        rebate_bps=0.0,
        dark_pool=True,
        max_participation_pct=0.03,
    ),
]


# Regime spread multipliers (widen spreads in adverse markets)
REGIME_SPREAD_MULT: dict[str, float] = {
    "bull": 1.0,
    "sideways": 1.2,
    "bear": 1.8,
    "crisis": 3.0,
}

# Regime fill probability adjustments (harder fills in crisis)
REGIME_FILL_MULT: dict[str, float] = {
    "bull": 1.0,
    "sideways": 0.95,
    "bear": 0.80,
    "crisis": 0.60,
}


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """Result of smart order routing decision.

    Attributes:
        allocations: List of venue allocations with quantities.
        total_expected_cost_bps: Weighted average expected cost.
        total_expected_fill_pct: Expected fill percentage across venues.
        regime: Market regime used for adjustments.
    """

    allocations: list[VenueAllocation]
    total_expected_cost_bps: float
    total_expected_fill_pct: float
    regime: str = "bull"


@dataclass
class VenueAllocation:
    """Allocation of shares to a specific venue.

    Attributes:
        venue: Venue name.
        quantity: Shares allocated to this venue.
        expected_spread_bps: Expected spread cost for this venue.
        expected_fill_prob: Expected fill probability.
        expected_cost_bps: Expected total cost (spread + fees - rebates).
        is_dark: Whether this is a dark pool allocation.
        participation_pct: Fraction of ADV this allocation represents.
    """

    venue: str
    quantity: float
    expected_spread_bps: float
    expected_fill_prob: float
    expected_cost_bps: float
    is_dark: bool = False
    participation_pct: float = 0.0


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------


def route_order(
    order_size: float,
    signal_urgency: float = 0.5,
    venues: list[VenueConfig] | None = None,
    seed: int | None = None,
    *,
    adv: float = 1_000_000.0,
    regime: str = "bull",
    price: float = 100.0,
    allow_dark_pools: bool = True,
    max_venues: int = 3,
) -> RoutingResult:
    """Route an order across optimal venues with cost minimization.

    Supports backward compatibility with the original route_order() signature
    while adding ADV-awareness, regime adjustments, and multi-venue splitting.

    Args:
        order_size: Order size in shares (positive).
        signal_urgency: 0-1 urgency (1 = fill immediately, 0 = minimize cost).
        venues: Available venues (defaults to DEFAULT_VENUES).
        seed: Random seed for fill simulation.
        adv: Average daily volume for participation rate calculation.
        regime: Current market regime for spread/fill adjustments.
        price: Current price per share (for notional calculations).
        allow_dark_pools: Whether to include dark pools in routing.
        max_venues: Maximum number of venues to split across.

    Returns:
        RoutingResult with venue allocations and cost estimates.
    """
    venues = venues or DEFAULT_VENUES

    if not allow_dark_pools:
        venues = [v for v in venues if not v.dark_pool]

    if not venues:
        venues = DEFAULT_VENUES[:1]

    order_size = abs(order_size)
    spread_mult = REGIME_SPREAD_MULT.get(regime, 1.0)
    fill_mult = REGIME_FILL_MULT.get(regime, 1.0)

    # Score each venue
    scored_venues: list[tuple[float, VenueConfig, float, float, float]] = []
    for v in venues:
        # ADV participation check
        max_shares = adv * v.max_participation_pct
        allocatable = min(order_size, max_shares)
        if allocatable < v.min_order_size:
            continue

        # Regime-adjusted metrics
        adj_spread = v.spread_bps * spread_mult
        adj_fill = min(v.fill_probability * fill_mult, 0.99)

        # Net cost: spread + fee - rebate
        net_cost_bps = adj_spread + v.fee_bps - v.rebate_bps

        # Composite score: higher is better
        # Urgency weights fill probability; patience weights cost savings
        urgency_w = signal_urgency
        cost_w = 1.0 - signal_urgency

        # Dark pool bonus for low urgency (better prices)
        dark_bonus = 0.0
        if v.dark_pool and signal_urgency < 0.5:
            dark_bonus = 2.0

        # Latency penalty scales with urgency
        latency_penalty = urgency_w * v.latency_ms * 0.1

        score = (
            urgency_w * adj_fill * 100
            - cost_w * net_cost_bps
            + dark_bonus
            - latency_penalty
        )

        scored_venues.append((score, v, adj_spread, adj_fill, allocatable))

    if not scored_venues:
        # Fallback: use first venue with full allocation
        v0 = (venues or DEFAULT_VENUES)[0]
        return RoutingResult(
            allocations=[
                VenueAllocation(
                    venue=v0.name,
                    quantity=order_size,
                    expected_spread_bps=v0.spread_bps * spread_mult,
                    expected_fill_prob=v0.fill_probability * fill_mult,
                    expected_cost_bps=v0.spread_bps * spread_mult
                    + v0.fee_bps
                    - v0.rebate_bps,
                )
            ],
            total_expected_cost_bps=v0.spread_bps * spread_mult
            + v0.fee_bps
            - v0.rebate_bps,
            total_expected_fill_pct=v0.fill_probability * fill_mult * 100,
            regime=regime,
        )

    # Sort by score descending
    scored_venues.sort(key=lambda x: -x[0])

    # Allocate across top venues
    allocations: list[VenueAllocation] = []
    remaining = order_size
    used_venues = 0

    for score, v, adj_spread, adj_fill, max_alloc in scored_venues:
        if remaining <= 0 or used_venues >= max_venues:
            break

        qty = min(remaining, max_alloc)
        if qty < v.min_order_size:
            continue

        net_cost = adj_spread + v.fee_bps - v.rebate_bps
        participation = qty / max(adv, 1.0)

        allocations.append(
            VenueAllocation(
                venue=v.name,
                quantity=round(qty, 2),
                expected_spread_bps=round(adj_spread, 2),
                expected_fill_prob=round(adj_fill, 3),
                expected_cost_bps=round(net_cost, 2),
                is_dark=v.dark_pool,
                participation_pct=round(participation * 100, 3),
            )
        )

        remaining -= qty
        used_venues += 1

    # If remaining shares, allocate to best venue (relaxing participation limit)
    if remaining > 0 and allocations:
        allocations[0].quantity = round(allocations[0].quantity + remaining, 2)
        allocations[0].participation_pct = round(
            allocations[0].quantity / max(adv, 1.0) * 100, 3
        )

    # Compute weighted averages
    total_qty = sum(a.quantity for a in allocations)
    if total_qty > 0:
        total_cost = (
            sum(a.expected_cost_bps * a.quantity for a in allocations) / total_qty
        )
        total_fill = (
            sum(a.expected_fill_prob * a.quantity for a in allocations)
            / total_qty
            * 100
        )
    else:
        total_cost = 0.0
        total_fill = 0.0

    result = RoutingResult(
        allocations=allocations,
        total_expected_cost_bps=round(total_cost, 2),
        total_expected_fill_pct=round(total_fill, 1),
        regime=regime,
    )

    logger.info(
        "[SOR] %.0f shares -> %d venues, cost=%.1f bps, fill=%.1f%%, regime=%s",
        order_size,
        len(allocations),
        total_cost,
        total_fill,
        regime,
    )

    return result


def simulate_fills(
    routing: RoutingResult,
    seed: int | None = None,
) -> dict:
    """Simulate fills based on routing result.

    Args:
        routing: RoutingResult from route_order.
        seed: Random seed.

    Returns:
        Dict with filled_qty, unfilled_qty, fill_pct, total_cost_bps, venue_fills.
    """
    rng = np.random.RandomState(seed)
    venue_fills = []
    total_filled = 0.0
    total_cost = 0.0

    for alloc in routing.allocations:
        filled = rng.random() < alloc.expected_fill_prob
        fill_qty = alloc.quantity if filled else 0.0
        cost = fill_qty * alloc.expected_cost_bps if filled else 0.0

        venue_fills.append(
            {
                "venue": alloc.venue,
                "ordered": alloc.quantity,
                "filled": fill_qty,
                "cost_bps": alloc.expected_cost_bps if filled else 0.0,
            }
        )
        total_filled += fill_qty
        total_cost += cost

    total_ordered = sum(a.quantity for a in routing.allocations)
    avg_cost = total_cost / total_filled if total_filled > 0 else 0.0

    return {
        "filled_qty": total_filled,
        "unfilled_qty": total_ordered - total_filled,
        "fill_pct": (
            round(total_filled / total_ordered * 100, 1) if total_ordered > 0 else 0.0
        ),
        "total_cost_bps": round(avg_cost, 2),
        "venue_fills": venue_fills,
    }


__all__ = [
    "VenueConfig",
    "VenueAllocation",
    "RoutingResult",
    "DEFAULT_VENUES",
    "REGIME_SPREAD_MULT",
    "REGIME_FILL_MULT",
    "route_order",
    "simulate_fills",
]
