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
    # Gap tuning (2026-04-20): geopolitics events persist for days, not hours.
    # Prior values underestimated half-lives — a sanctions announcement is
    # still moving energy ETFs the following week. Reference: observed
    # repricing windows for 2022-2024 Russia/Iran packages.
    # Geopolitics
    "war_escalation":     DecayProfile("exponential", 72 * 60),   # was 8h → 3 days
    "military_strike":    DecayProfile("exponential", 24 * 60),   # was 6h → 1 day
    "sanctions":          DecayProfile("exponential", 72 * 60),   # was 3h → 3 days
    "diplomatic":         DecayProfile("exponential", 24 * 60),   # was 4h → 1 day
    "political_crisis":   DecayProfile("exponential", 48 * 60),   # was 6h → 2 days
    # Markets / macro — earnings decay slower than a single session.
    "rate_surprise":      DecayProfile("exponential", 4 * 60),
    "earnings":           DecayProfile("exponential", 5 * 60),    # was 90m → 5h
    "guidance_change":    DecayProfile("exponential", 5 * 60),
    "credit_downgrade":   DecayProfile("exponential", 24 * 60),
    # Intraday-only
    "market_stress":      DecayProfile("linear", 240),
    "liquidity_event":    DecayProfile("linear", 120),
    # Energy / shipping
    "energy_disruption":  DecayProfile("exponential", 12 * 60),
    "shipping_disruption":DecayProfile("exponential", 12 * 60),
    # Capital-structure / labour / sell-side (new event types)
    "buyback":            DecayProfile("exponential", 48 * 60),
    "ipo":                DecayProfile("exponential", 24 * 60),
    "layoffs":            DecayProfile("exponential", 24 * 60),
    "analyst_rating":     DecayProfile("exponential", 4 * 60),
    # Other
    "cyber_attack":       DecayProfile("exponential", 12 * 60),
    "natural_disaster":   DecayProfile("exponential", 24 * 60),
    "default":            DecayProfile("exponential", 4 * 60),
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
