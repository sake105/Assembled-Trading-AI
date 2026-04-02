"""Static source registry with tier and trust weight information."""

from __future__ import annotations

from .models import SourceTier

# ---------------------------------------------------------------------------
# Static registry: source_id -> (tier, description)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, SourceTier] = {
    "OFAC": SourceTier.T0,
    "UN_SANCTIONS": SourceTier.T0,
    "AP_MEDIA": SourceTier.T1,
    "REUTERS": SourceTier.T1,
    "GDELT": SourceTier.T2,
    "ACLED": SourceTier.T2,
    "WORLD_BANK": SourceTier.T2,
    "IMF": SourceTier.T2,
    "EIA": SourceTier.T2,
    "NEWSAPI": SourceTier.T3,
}

_TRUST_WEIGHTS: dict[SourceTier, float] = {
    SourceTier.T0: 1.0,
    SourceTier.T1: 0.9,
    SourceTier.T2: 0.7,
    SourceTier.T3: 0.4,
}


def get_source_tier(source_id: str) -> SourceTier:
    """Return the tier for a known source. Defaults to T3 for unknown sources."""
    return _REGISTRY.get(source_id, SourceTier.T3)


def get_trust_weight(source_id: str) -> float:
    """Return the trust weight for a source (T0=1.0, T1=0.9, T2=0.7, T3=0.4)."""
    tier = get_source_tier(source_id)
    return _TRUST_WEIGHTS[tier]


def list_sources(tier: SourceTier | None = None) -> list[str]:
    """Return all source IDs, optionally filtered by tier."""
    if tier is None:
        return list(_REGISTRY.keys())
    return [sid for sid, t in _REGISTRY.items() if t == tier]


def get_all_tiers() -> dict[str, SourceTier]:
    """Return a copy of the full source-to-tier mapping."""
    return dict(_REGISTRY)
