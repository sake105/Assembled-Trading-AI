"""Nation resource/vulnerability profiles for geopolitical risk modeling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .models import NationProfile, ShockType

logger = logging.getLogger(__name__)

# Default path — three parents up from this file reaches the repo root.
DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "nation_profiles.yaml"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_nation_profiles(
    path: str | Path | None = None,
) -> dict[str, NationProfile]:
    """Load nation profiles from YAML.

    Parameters
    ----------
    path:
        Explicit path to the YAML file.  Falls back to
        ``configs/nation_profiles.yaml`` relative to the repo root.

    Returns
    -------
    dict[str, NationProfile]
        Mapping *nation_id* -> populated :class:`NationProfile`.
    """
    path = Path(path) if path else DEFAULT_PROFILES_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    profiles: dict[str, NationProfile] = {}
    for nation_id, raw in data.items():
        profiles[nation_id] = NationProfile(
            nation_id=nation_id,
            name=raw.get("name", nation_id),
            imports=raw.get("imports", {}),
            exports=raw.get("exports", {}),
            transit_dependencies=raw.get("transit_dependencies", {}),
            fiscal=raw.get("fiscal", {}),
            military=raw.get("military", {}),
            tech_sovereignty=raw.get("tech_sovereignty", {}),
            vulnerabilities=raw.get("vulnerabilities", {}),
        )
    logger.info("[NationProfiles] Loaded %d profiles", len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Shock → vulnerability mapping
# ---------------------------------------------------------------------------

# Maps human-readable shock categories to the YAML keys that contribute to
# a nation's exposure.  Keys are looked up across *imports*,
# *vulnerabilities*, *transit_dependencies* and *fiscal* dicts.
_SHOCK_VULNERABILITY_MAP: dict[str, list[str]] = {
    "oil_supply_risk": ["oil_pct", "energy_import_shock"],
    "rare_earth_supply_risk": ["rare_earths_pct", "rare_earth_disruption"],
    "semiconductor_supply_risk": ["semiconductors_pct", "china_decoupling_impact"],
    "food_supply_risk": ["food_pct", "food_security_risk"],
    "lng_supply_risk": ["lng_pct", "energy_import_shock"],
    "shipping_lane_disruption": [
        "malacca_blockade_impact",
        "hormuz_blockade_impact",
        "suez_disruption_impact",
    ],
    "currency_crisis": ["dollarization_pct", "reserve_drain_risk"],
    "sanctions_exposure": ["sanctions_resilience"],
    "hegemonic_decoupling": ["china_decoupling_impact", "us_decoupling_impact"],
    "fiscal_shock": ["debt_to_gdp", "deficit_pct"],
    # Additional mappings aligned with ShockType enum values
    "energy_price_spike": ["oil_pct", "lng_pct", "energy_import_shock"],
    "shipping_cost_risk": [
        "malacca_blockade_impact",
        "hormuz_blockade_impact",
        "suez_disruption_impact",
    ],
    "chip_shortage": ["semiconductors_pct"],
    "lithium_supply_risk": ["lithium_pct"],
    "uranium_supply_risk": ["uranium_pct"],
    "copper_supply_risk": ["copper_pct"],
    "climate_disruption": ["food_security_risk", "water_scarcity_risk"],
    "nuclear_escalation_risk": [],  # binary — not profile-derivable
    "reserve_depletion": ["reserve_drain_risk"],
    "capital_flight": ["dollarization_pct", "reserve_drain_risk"],
    "inflation_spike": ["dollarization_pct"],
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_vulnerability_score(
    profile: NationProfile,
    shock_type: ShockType | str,
) -> float:
    """Compute how vulnerable a nation is to a specific shock type.

    Returns a float in ``[0.0, 1.0]`` where 0 means *not vulnerable* and
    1 means *extremely vulnerable*.

    The score is the arithmetic mean of all matching keys found in the
    profile's *imports*, *vulnerabilities*, *transit_dependencies* and
    *fiscal* dicts.  Values are clamped to ``[0, 1]``.
    """
    shock_key = shock_type.value if isinstance(shock_type, ShockType) else str(shock_type)
    keys = _SHOCK_VULNERABILITY_MAP.get(shock_key, [])
    if not keys:
        return 0.0

    scores: list[float] = []
    for key in keys:
        for source in [
            profile.imports,
            profile.vulnerabilities,
            profile.transit_dependencies,
            profile.fiscal,
        ]:
            if key in source:
                val = float(source[key])
                scores.append(min(abs(val), 1.0))
                break  # use first match per key

    return sum(scores) / len(scores) if scores else 0.0


def compute_bilateral_dependency(
    profile_a: NationProfile,
    profile_b: NationProfile,
) -> dict[str, float]:
    """Compute mutual dependency between two nations.

    This is a *rough heuristic* based on average vulnerability scores.
    Returns a dict with ``a_depends_on_b``, ``b_depends_on_a`` and
    ``asymmetry``.
    """
    a_vuln = (
        sum(profile_a.vulnerabilities.values())
        / max(len(profile_a.vulnerabilities), 1)
    )
    b_vuln = (
        sum(profile_b.vulnerabilities.values())
        / max(len(profile_b.vulnerabilities), 1)
    )
    return {
        "a_depends_on_b": min(a_vuln, 1.0),
        "b_depends_on_a": min(b_vuln, 1.0),
        "asymmetry": abs(a_vuln - b_vuln),
    }


def rank_nations_by_exposure(
    profiles: dict[str, NationProfile],
    shock_type: ShockType | str,
) -> list[tuple[str, float]]:
    """Rank all nations by vulnerability to *shock_type* (descending)."""
    ranked: list[tuple[str, float]] = []
    for nation_id, profile in profiles.items():
        score = compute_vulnerability_score(profile, shock_type)
        ranked.append((nation_id, score))
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def compute_sanctions_resilience(profile: NationProfile) -> float:
    """Estimate how well a nation can withstand comprehensive sanctions.

    Returns a float in ``[0.0, 1.0]`` where 0 means *extremely fragile*
    and 1 means *highly resilient*.

    The estimate is a weighted combination of:

    * **Forex reserves** relative to 12 months of import cover (40 %).
    * **Import diversity** — 1 minus the highest single-commodity import
      share (30 %).
    * **Tech self-sufficiency** — average of *tech_sovereignty* values
      (30 %).
    """
    reserve_score = min(
        profile.fiscal.get("reserve_months_import", 0) / 12, 1.0
    )
    import_diversity = 1.0 - max(profile.imports.values(), default=0)
    tech_self = (
        sum(profile.tech_sovereignty.values())
        / max(len(profile.tech_sovereignty), 1)
    )
    return reserve_score * 0.4 + import_diversity * 0.3 + tech_self * 0.3
