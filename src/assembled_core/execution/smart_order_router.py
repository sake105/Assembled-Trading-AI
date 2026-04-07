"""Smart Order Router Simulation (Plan 6.9).

Simulates routing across 3 venues:
- Primary: wider spread, reliable fill
- Dark Pool: tighter spread, uncertain fill
- ATS: moderate spread, moderate fill probability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VenueConfig:
    """Configuration for a simulated venue."""
    name: str
    spread_bps: float
    fill_probability: float
    latency_ms: float = 0.0


DEFAULT_VENUES = [
    VenueConfig("primary", spread_bps=5.0, fill_probability=0.95, latency_ms=1.0),
    VenueConfig("dark_pool", spread_bps=1.0, fill_probability=0.40, latency_ms=5.0),
    VenueConfig("ats", spread_bps=3.0, fill_probability=0.70, latency_ms=2.0),
]


def route_order(
    order_size: float,
    signal_urgency: float = 0.5,
    venues: list[VenueConfig] | None = None,
    seed: int | None = None,
) -> dict:
    """Route an order to the best venue.

    High urgency → primary (reliable fill).
    Low urgency → dark pool (better price).

    Args:
        order_size: Order notional value.
        signal_urgency: 0-1 urgency (1 = fill immediately).
        venues: Available venues.
        seed: Random seed for fill simulation.

    Returns:
        Dict with venue, filled, fill_cost_bps, latency_ms.
    """
    venues = venues or DEFAULT_VENUES
    rng = np.random.RandomState(seed)

    # Score venues: urgency-weighted fill_probability vs spread
    scores = []
    for v in venues:
        score = signal_urgency * v.fill_probability * 100 - (1 - signal_urgency) * v.spread_bps
        scores.append((score, v))

    scores.sort(key=lambda x: -x[0])
    best_venue = scores[0][1]

    filled = rng.random() < best_venue.fill_probability

    return {
        "venue": best_venue.name,
        "filled": filled,
        "fill_cost_bps": best_venue.spread_bps if filled else 0.0,
        "latency_ms": best_venue.latency_ms,
    }


__all__ = ["VenueConfig", "DEFAULT_VENUES", "route_order"]
