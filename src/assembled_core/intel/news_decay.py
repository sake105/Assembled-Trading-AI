"""News-impact decay curves.

A single news event's market impact is not constant — it decays over time.
This module provides a small library of decay functions keyed by event type,
plus a helper that returns the remaining fraction of impact at time `t` after
the original event.

Two decay families:

* **exponential**: `impact(t) = exp(-t / half_life * ln(2))`
  Appropriate for events whose effect fades smoothly (sanctions, central-bank
  decisions, earnings surprises).

* **linear**: `impact(t) = max(0, 1 - t / lifetime)`
  Appropriate for events with a hard shelf-life (intraday shocks that mean-
  revert by close).

Defaults can be overridden per event_type. No external dependencies.

Usage:
    d = NewsDecay()
    frac = d.impact_remaining("sanctions", minutes_since=120)
    # 0.63 (for default 3h half-life)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


DecayKind = Literal["exponential", "linear"]


@dataclass(frozen=True)
class DecayProfile:
    kind: DecayKind
    # For exponential: half-life in minutes. For linear: total lifetime in minutes.
    parameter_min: float


# Per-event-type defaults. Numbers are heuristic; the calibrator (F9) is the
# intended pathway for learning data-driven values over time.
_DEFAULTS: dict[str, DecayProfile] = {
    # Geopolitics
    "war_escalation":     DecayProfile("exponential", 8 * 60),
    "military_strike":    DecayProfile("exponential", 6 * 60),
    "sanctions":          DecayProfile("exponential", 3 * 60),
    "diplomatic":         DecayProfile("exponential", 4 * 60),
    "political_crisis":   DecayProfile("exponential", 6 * 60),
    # Markets / macro
    "rate_surprise":      DecayProfile("exponential", 2 * 60),
    "earnings":           DecayProfile("exponential", 90),
    "guidance_change":    DecayProfile("exponential", 90),
    "credit_downgrade":   DecayProfile("exponential", 6 * 60),
    # Intraday-only
    "market_stress":      DecayProfile("linear", 120),
    "liquidity_event":    DecayProfile("linear", 60),
    # Energy / shipping
    "energy_disruption":  DecayProfile("exponential", 4 * 60),
    "shipping_disruption":DecayProfile("exponential", 4 * 60),
    # Other
    "cyber_attack":       DecayProfile("exponential", 3 * 60),
    "natural_disaster":   DecayProfile("exponential", 12 * 60),
    "default":            DecayProfile("exponential", 2 * 60),
}


class NewsDecay:
    """Decay-profile registry + remaining-impact calculator."""

    def __init__(self, overrides: dict[str, DecayProfile] | None = None) -> None:
        self._profiles: dict[str, DecayProfile] = dict(_DEFAULTS)
        if overrides:
            for key, prof in overrides.items():
                self._profiles[key.lower()] = prof

    def profile(self, event_type: str) -> DecayProfile:
        return self._profiles.get(event_type.lower(), self._profiles["default"])

    def impact_remaining(
        self,
        event_type: str,
        minutes_since: float,
    ) -> float:
        """Return a value in [0, 1]. 1.0 = full impact, 0.0 = fully decayed."""
        if minutes_since <= 0:
            return 1.0
        prof = self.profile(event_type)
        if prof.kind == "linear":
            life = max(1.0, prof.parameter_min)
            return max(0.0, 1.0 - (minutes_since / life))
        # exponential
        half = max(1.0, prof.parameter_min)
        return math.exp(-minutes_since * math.log(2.0) / half)

    def scale_bps(
        self,
        event_type: str,
        original_bps: float,
        minutes_since: float,
    ) -> float:
        """Scale a bps impact by the remaining fraction at `minutes_since`."""
        return original_bps * self.impact_remaining(event_type, minutes_since)


__all__ = ["NewsDecay", "DecayProfile"]
