"""Multi-channel shock propagation model (Plan 4.1).

Extends the base shock propagation with 3 distinct transmission channels
that have different speeds, dampening, and decay characteristics:

1. **FINANCIAL** (dampening=0.70, lag=0-2d): Capital flows, currencies,
   credit spreads. Fastest channel.
2. **TRADE** (dampening=0.80, lag=5-30d): Supply chains, import/export,
   commodities. Medium speed.
3. **SENTIMENT** (dampening=0.90, lag=0d): News flow, investor sentiment,
   risk-off. Instantaneous but short-lived.

Each edge type has a primary channel:
- ``TRADE_DEPENDENT`` → TRADE channel
- ``LENDS_TO`` → FINANCIAL channel
- ``MEDIA_AMPLIFIES`` → SENTIMENT channel

Total impact at time t:
    ``impact(t) = sum_channels(channel_impact × decay(t - channel_lag))``

Decay model:
    ``exponential_decay(t) = magnitude × exp(-ln(2) × t / half_life)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class PropagationChannel(str, Enum):
    """Transmission channels for shock propagation."""

    FINANCIAL = "financial"
    TRADE = "trade"
    SENTIMENT = "sentiment"


@dataclass
class ChannelConfig:
    """Configuration for a propagation channel."""

    dampening: float  # fraction of shock absorbed per hop (0-1)
    lag_days_min: int  # minimum lag before impact starts
    lag_days_max: int  # maximum lag (peak impact)
    half_life_days: float  # decay half-life after peak
    weight: float = 1.0  # relative importance


# Default channel configurations
CHANNEL_CONFIGS: dict[PropagationChannel, ChannelConfig] = {
    PropagationChannel.FINANCIAL: ChannelConfig(
        dampening=0.70, lag_days_min=0, lag_days_max=2, half_life_days=5, weight=0.40,
    ),
    PropagationChannel.TRADE: ChannelConfig(
        dampening=0.80, lag_days_min=5, lag_days_max=30, half_life_days=20, weight=0.35,
    ),
    PropagationChannel.SENTIMENT: ChannelConfig(
        dampening=0.90, lag_days_min=0, lag_days_max=0, half_life_days=3, weight=0.25,
    ),
}

# Edge type → primary channel mapping
EDGE_CHANNEL_MAP: dict[str, PropagationChannel] = {
    "TRADE_DEPENDENT": PropagationChannel.TRADE,
    "IMPORTS_FROM": PropagationChannel.TRADE,
    "EXPORTS_TO": PropagationChannel.TRADE,
    "SUPPLY_CHAIN": PropagationChannel.TRADE,
    "LENDS_TO": PropagationChannel.FINANCIAL,
    "BORROWS_FROM": PropagationChannel.FINANCIAL,
    "INVESTS_IN": PropagationChannel.FINANCIAL,
    "CURRENCY_PEG": PropagationChannel.FINANCIAL,
    "MEDIA_AMPLIFIES": PropagationChannel.SENTIMENT,
    "NEIGHBOR_OF": PropagationChannel.SENTIMENT,
    "ALLIANCE_WITH": PropagationChannel.SENTIMENT,
    "SANCTIONS": PropagationChannel.FINANCIAL,
    "RIVAL_OF": PropagationChannel.SENTIMENT,
}


@dataclass
class ChannelImpact:
    """Impact via a single channel."""

    channel: PropagationChannel
    magnitude: float  # peak impact magnitude
    lag_days: int  # days until peak
    half_life_days: float  # decay after peak
    current_impact: float  # impact at current time


@dataclass
class MultiChannelShockResult:
    """Result of multi-channel shock propagation."""

    target_node: str
    total_impact: float
    channel_impacts: list[ChannelImpact] = field(default_factory=list)
    expected_peak_day: int = 0
    dominant_channel: str = ""


def exponential_decay(
    magnitude: float,
    days_since_peak: float,
    half_life: float,
) -> float:
    """Compute exponential decay of shock impact.

    Args:
        magnitude: Peak impact magnitude.
        days_since_peak: Days since the impact peaked.
        half_life: Decay half-life in days.

    Returns:
        Current impact value.
    """
    if days_since_peak < 0 or half_life <= 0:
        return magnitude
    return magnitude * np.exp(-np.log(2) * days_since_peak / half_life)


def compute_channel_impact(
    initial_magnitude: float,
    channel: PropagationChannel,
    days_since_event: int = 0,
    n_hops: int = 1,
    config: ChannelConfig | None = None,
) -> ChannelImpact:
    """Compute impact through a specific channel.

    Args:
        initial_magnitude: Shock magnitude at source (0-1 scale).
        channel: Propagation channel.
        days_since_event: Days since the triggering event.
        n_hops: Number of hops in the dependency graph.
        config: Channel configuration (uses defaults if None).

    Returns:
        ChannelImpact with current impact value.
    """
    if config is None:
        config = CHANNEL_CONFIGS[channel]

    # Dampening per hop
    magnitude_after_hops = initial_magnitude * (1 - config.dampening) ** n_hops

    # Lag: impact ramps up from lag_min to lag_max
    lag_center = (config.lag_days_min + config.lag_days_max) / 2

    # Before lag: no impact yet (or partial for sentiment)
    if config.lag_days_min == config.lag_days_max:
        # Instantaneous channel (e.g., sentiment): peak at day 0, then decay
        if days_since_event == 0:
            current = magnitude_after_hops
        else:
            current = exponential_decay(
                magnitude_after_hops, days_since_event, config.half_life_days,
            )
    elif days_since_event < config.lag_days_min:
        current = 0.0
    elif days_since_event <= config.lag_days_max:
        # Ramp-up phase
        progress = (days_since_event - config.lag_days_min) / max(
            config.lag_days_max - config.lag_days_min, 1,
        )
        current = magnitude_after_hops * progress
    else:
        # Decay phase
        days_past_peak = days_since_event - config.lag_days_max
        current = exponential_decay(
            magnitude_after_hops, days_past_peak, config.half_life_days,
        )

    return ChannelImpact(
        channel=channel,
        magnitude=round(magnitude_after_hops, 6),
        lag_days=int(lag_center),
        half_life_days=config.half_life_days,
        current_impact=round(current, 6),
    )


def propagate_multichannel(
    initial_magnitude: float,
    edge_types: list[str],
    days_since_event: int = 0,
    n_hops: int = 1,
) -> MultiChannelShockResult:
    """Propagate shock through multiple channels based on edge types.

    Args:
        initial_magnitude: Shock magnitude at source.
        edge_types: List of edge types traversed.
        days_since_event: Days since triggering event.
        n_hops: Number of hops in dependency graph.

    Returns:
        MultiChannelShockResult with per-channel and total impact.
    """
    # Determine active channels from edge types
    active_channels: set[PropagationChannel] = set()
    for et in edge_types:
        channel = EDGE_CHANNEL_MAP.get(et, PropagationChannel.SENTIMENT)
        active_channels.add(channel)

    # If no specific channels, use all three
    if not active_channels:
        active_channels = {PropagationChannel.FINANCIAL, PropagationChannel.TRADE, PropagationChannel.SENTIMENT}

    channel_impacts: list[ChannelImpact] = []
    total_impact = 0.0

    for channel in active_channels:
        config = CHANNEL_CONFIGS[channel]
        impact = compute_channel_impact(
            initial_magnitude, channel, days_since_event, n_hops, config,
        )
        channel_impacts.append(impact)
        total_impact += impact.current_impact * config.weight

    # Determine dominant channel
    dominant = ""
    if channel_impacts:
        dominant = max(channel_impacts, key=lambda ci: ci.current_impact).channel.value

    # Expected peak day
    peak_days = [ci.lag_days for ci in channel_impacts if ci.magnitude > 0]
    expected_peak = int(np.mean(peak_days)) if peak_days else 0

    return MultiChannelShockResult(
        target_node="",
        total_impact=round(total_impact, 6),
        channel_impacts=channel_impacts,
        expected_peak_day=expected_peak,
        dominant_channel=dominant,
    )


def compute_impact_timeline(
    initial_magnitude: float,
    edge_types: list[str],
    n_hops: int = 1,
    horizon_days: int = 60,
) -> dict[int, float]:
    """Compute impact over time for multi-channel propagation.

    Args:
        initial_magnitude: Shock magnitude.
        edge_types: Edge types traversed.
        n_hops: Graph hops.
        horizon_days: Number of days to project.

    Returns:
        Dict mapping day → total impact value.
    """
    timeline = {}
    for day in range(horizon_days + 1):
        result = propagate_multichannel(
            initial_magnitude, edge_types, day, n_hops,
        )
        timeline[day] = result.total_impact
    return timeline


__all__ = [
    "CHANNEL_CONFIGS",
    "ChannelConfig",
    "ChannelImpact",
    "MultiChannelShockResult",
    "PropagationChannel",
    "compute_channel_impact",
    "compute_impact_timeline",
    "exponential_decay",
    "propagate_multichannel",
]
