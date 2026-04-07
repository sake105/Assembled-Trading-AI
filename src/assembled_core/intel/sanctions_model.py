"""Sanctions package modeling for geopolitical risk analysis.

Models historical and hypothetical sanctions packages, cascade effects
on third-party nations, and beneficiary identification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import SanctionPackage

if TYPE_CHECKING:
    from .nation_profiles import NationProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Historical Sanctions Database
# ---------------------------------------------------------------------------

HISTORICAL_SANCTIONS: dict[str, SanctionPackage] = {
    "RUSSIA_2022_FULL": SanctionPackage(
        package_id="RUSSIA_2022_FULL",
        issuer="MULTI",  # US+EU+UK+Canada+Japan+Australia
        target_nation="RUSSIA",
        target_entities=["SBERBANK", "VTB", "GAZPROM", "ROSNEFT",
                          "CENTRAL_BANK_RUSSIA", "RUSSIAN_DIRECT_INVESTMENT_FUND"],
        domains=["finance", "energy", "tech", "military", "luxury", "aviation"],
        severity=5,
        secondary_sanctions=True,
        swift_exclusion=True,
        affected_sectors=["ENERGY", "FINANCE", "TECH", "DEFENSE", "MINING",
                           "AGRICULTURE", "AVIATION", "SHIPPING"],
        estimated_gdp_impact_pct=0.04,  # IMF: ~2.1% GDP drop 2022, ~4% long-run drag with adaptation
        evasion_difficulty=0.40,  # Russia has workarounds via China/India/Turkey
    ),
    "IRAN_2018_MAX_PRESSURE": SanctionPackage(
        package_id="IRAN_2018_MAX_PRESSURE",
        issuer="US",
        target_nation="IRAN",
        target_entities=["NIOC", "CENTRAL_BANK_IRAN", "IRISL", "IRGC",
                          "BANK_SADERAT", "BANK_MELLI"],
        domains=["energy", "finance", "shipping", "military"],
        severity=5,
        secondary_sanctions=True,  # Secondary sanctions on buyers of Iranian oil
        swift_exclusion=True,
        affected_sectors=["ENERGY", "FINANCE", "SHIPPING", "DEFENSE"],
        estimated_gdp_impact_pct=0.20,
        evasion_difficulty=0.35,  # Iran uses ghost fleet, crypto, barter
    ),
    "CHINA_TECH_2022": SanctionPackage(
        package_id="CHINA_TECH_2022",
        issuer="US",
        target_nation="CHINA",
        target_entities=["HUAWEI", "SMIC", "YANGTZE_MEMORY", "HIKVISION",
                          "DAHUA", "DJI", "NUCTECH"],
        domains=["tech", "semiconductors", "surveillance", "ai"],
        severity=3,
        secondary_sanctions=False,
        swift_exclusion=False,
        affected_sectors=["SEMIS", "TECH", "TELECOM", "DEFENSE"],
        estimated_gdp_impact_pct=0.02,  # Limited direct GDP, but tech sector large
        evasion_difficulty=0.60,  # Difficult: cutting-edge chips hard to substitute
    ),
    "NORTH_KOREA_COMPREHENSIVE": SanctionPackage(
        package_id="NORTH_KOREA_COMPREHENSIVE",
        issuer="UN",
        target_nation="NORTH_KOREA",
        target_entities=["KOREAN_WORKERS_PARTY", "MILITARY", "EXPORT_AGENCIES"],
        domains=["energy", "finance", "military", "trade"],
        severity=5,
        secondary_sanctions=True,
        swift_exclusion=True,
        affected_sectors=["ENERGY", "MINING", "DEFENSE"],
        estimated_gdp_impact_pct=0.25,
        evasion_difficulty=0.20,  # North Korea still finds evasion routes via China
    ),
    "VENEZUELA_2019": SanctionPackage(
        package_id="VENEZUELA_2019",
        issuer="US",
        target_nation="VENEZUELA",
        target_entities=["PDVSA", "CENTRAL_BANK_VENEZUELA", "MADURO_GOVERNMENT"],
        domains=["energy", "finance", "gold"],
        severity=4,
        secondary_sanctions=True,
        swift_exclusion=False,
        affected_sectors=["ENERGY", "FINANCE", "MINING"],
        estimated_gdp_impact_pct=0.08,
        evasion_difficulty=0.30,
    ),
    "CUBA_EMBARGO": SanctionPackage(
        package_id="CUBA_EMBARGO",
        issuer="US",
        target_nation="CUBA",
        target_entities=["CUBAN_GOVERNMENT", "GAESA"],
        domains=["trade", "finance", "tourism"],
        severity=4,
        secondary_sanctions=False,
        swift_exclusion=False,
        affected_sectors=["FINANCE", "CONSUMER", "AGRICULTURE"],
        estimated_gdp_impact_pct=0.05,
        evasion_difficulty=0.50,
    ),
    "BELARUS_2021": SanctionPackage(
        package_id="BELARUS_2021",
        issuer="MULTI",  # US+EU
        target_nation="BELARUS",
        target_entities=["LUKASHENKO_REGIME", "BELARUSKALI", "BELNEFTEKHIM"],
        domains=["finance", "energy", "potash", "aviation"],
        severity=3,
        secondary_sanctions=False,
        swift_exclusion=False,
        affected_sectors=["AGRICULTURE", "ENERGY", "FINANCE", "AVIATION"],
        estimated_gdp_impact_pct=0.07,
        evasion_difficulty=0.45,
    ),
}

# Third-party nation exposure to major sanctions packages
# Format: {package_id: {nation: exposure_score 0-1}}
_THIRD_PARTY_EXPOSURE: dict[str, dict[str, float]] = {
    "RUSSIA_2022_FULL": {
        "CHINA": 0.30,    # Major buyer of Russian energy + evasion route
        "INDIA": 0.25,    # Buying discounted Russian oil
        "TURKEY": 0.35,   # Transit hub, secondary sanctions risk
        "UAE": 0.25,      # Financial hub for Russians
        "BRAZIL": 0.10,
        "INDONESIA": 0.10,
    },
    "IRAN_2018_MAX_PRESSURE": {
        "CHINA": 0.40,    # Largest buyer of Iranian oil
        "INDIA": 0.30,    # Major oil importer from Iran (before 2018 waiver)
        "TURKEY": 0.25,
        "SOUTH_KOREA": 0.15,  # Had waivers, now reduced
        "JAPAN": 0.10,
    },
    "CHINA_TECH_2022": {
        "TAIWAN": 0.60,   # TSMC is key, competing demands
        "SOUTH_KOREA": 0.45,  # Samsung/SK Hynix compliance required
        "JAPAN": 0.35,    # ASML, semiconductor equipment
        "NETHERLANDS": 0.30,  # ASML
        "GERMANY": 0.20,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sanction_package(package_id: str) -> SanctionPackage | None:
    """Return a sanction package by ID."""
    return HISTORICAL_SANCTIONS.get(package_id)


def compute_sanction_cascade(
    package: SanctionPackage,
    nation_profiles: "dict[str, NationProfile] | None" = None,
) -> dict[str, float]:
    """Compute cascade impact of a sanction package on third-party nations.

    Returns {nation_id: exposure_score 0-1}.
    Higher score = more exposed to secondary sanctions risk.
    """
    known = _THIRD_PARTY_EXPOSURE.get(package.package_id, {})
    result = dict(known)

    # If secondary sanctions enabled, amplify exposure
    if package.secondary_sanctions:
        result = {nation: score * 1.3 for nation, score in result.items()}

    # If SWIFT exclusion, financial exposure for all trading partners
    if package.swift_exclusion:
        # Nations that heavily trade with target face payment disruption
        if nation_profiles:
            for nation_id, profile in nation_profiles.items():
                if nation_id not in result:
                    # Check if they import from the sanctioned nation
                    trade_dep = profile.imports.get(
                        f"{package.target_nation.lower()}_trade_pct", 0.0
                    )
                    if trade_dep > 0.05:
                        result[nation_id] = trade_dep * 2.0

    # Cap at 1.0
    result = {k: min(v, 1.0) for k, v in result.items()}

    logger.debug(
        "[Sanctions] %s cascade: %d nations exposed, secondary=%s",
        package.package_id, len(result), package.secondary_sanctions
    )
    return result


def identify_sanction_beneficiaries(
    package: SanctionPackage,
) -> list[tuple[str, float]]:
    """Identify which nations/sectors benefit from a sanction package.

    Returns [(nation_or_sector, benefit_score)] sorted descending.
    Saudi profits when Iran is sanctioned (oil market share).
    """
    beneficiaries: dict[str, float] = {}

    # Energy sanctions → competing producers benefit
    if "energy" in package.domains:
        if package.target_nation in ("IRAN", "RUSSIA", "VENEZUELA"):
            beneficiaries["SAUDI_ARABIA"] = 0.70
            beneficiaries["UAE"] = 0.60
            beneficiaries["US"] = 0.55  # US LNG/oil
            beneficiaries["QATAR"] = 0.50
            beneficiaries["NIGERIA"] = 0.35
            beneficiaries["NORWAY"] = 0.30
            beneficiaries["ENERGY"] = 0.65  # Sector

    # Tech sanctions on China → competing chip producers benefit
    if "tech" in package.domains or "semiconductors" in package.domains:
        if package.target_nation == "CHINA":
            beneficiaries["TAIWAN"] = 0.50  # TSMC gets more orders
            beneficiaries["SOUTH_KOREA"] = 0.45
            beneficiaries["JAPAN"] = 0.35
            beneficiaries["US"] = 0.40  # Intel, Qualcomm
            beneficiaries["SEMIS"] = 0.55

    # Finance sanctions → neighboring financial hubs benefit
    if "finance" in package.domains or package.swift_exclusion:
        if package.target_nation == "RUSSIA":
            beneficiaries["UAE"] = 0.40  # Dubai as alternative hub
            beneficiaries["TURKEY"] = 0.30
            beneficiaries["CHINA"] = 0.25  # CIPS as SWIFT alternative
        if package.target_nation == "IRAN":
            beneficiaries["TURKEY"] = 0.35
            beneficiaries["UAE"] = 0.30

    # Defense sector always benefits from major sanctions (arms demand up)
    if package.severity >= 4:
        beneficiaries["DEFENSE"] = 0.40
        beneficiaries["US"] = max(beneficiaries.get("US", 0.0), 0.30)

    return sorted(beneficiaries.items(), key=lambda x: x[1], reverse=True)


def estimate_evasion_routes(package: SanctionPackage) -> list[dict[str, Any]]:
    """Estimate potential evasion paths for a sanctions target.

    Returns list of route dicts with feasibility score.
    """
    routes = []

    if package.evasion_difficulty < 0.7:
        routes.append({
            "method": "FRIENDLY_NATION_TRANSIT",
            "via": ["CHINA", "INDIA", "TURKEY", "UAE"][: 2],
            "feasibility": 1.0 - package.evasion_difficulty,
            "domains": [d for d in package.domains if d in ("energy", "trade")],
        })

    if "finance" in package.domains and package.evasion_difficulty < 0.6:
        routes.append({
            "method": "ALTERNATIVE_PAYMENT_SYSTEM",
            "via": ["CIPS", "CRYPTO", "BARTER"],
            "feasibility": (1.0 - package.evasion_difficulty) * 0.6,
            "domains": ["finance"],
        })

    if "energy" in package.domains and package.evasion_difficulty < 0.5:
        routes.append({
            "method": "GHOST_FLEET_SHIPPING",
            "via": ["FLAG_OF_CONVENIENCE"],
            "feasibility": (1.0 - package.evasion_difficulty) * 0.5,
            "domains": ["energy"],
        })

    return sorted(routes, key=lambda x: x["feasibility"], reverse=True)


def compute_secondary_sanction_risk(
    nation: str,
    package: SanctionPackage,
) -> float:
    """Estimate secondary sanction risk for a third-party nation.

    Returns 0.0 (no risk) to 1.0 (high risk of secondary sanctions).
    """
    if not package.secondary_sanctions:
        return 0.0
    cascade = _THIRD_PARTY_EXPOSURE.get(package.package_id, {})
    base = cascade.get(nation, 0.0)
    # Severity amplification
    return min(base * (package.severity / 3.0), 1.0)


def simulate_new_sanction_package(
    target_nation: str,
    domains: list[str],
    severity: int = 3,
    secondary_sanctions: bool = False,
    swift_exclusion: bool = False,
) -> dict[str, Any]:
    """Model a hypothetical new sanction package (what-if analysis).

    Returns estimated impacts without creating a persistent package.
    """
    hypo = SanctionPackage(
        package_id=f"HYPO_{target_nation}_{severity}",
        issuer="MULTI",
        target_nation=target_nation,
        domains=domains,
        severity=severity,
        secondary_sanctions=secondary_sanctions,
        swift_exclusion=swift_exclusion,
        evasion_difficulty=0.3 + severity * 0.1,
        estimated_gdp_impact_pct=severity * 0.03,
    )

    beneficiaries = identify_sanction_beneficiaries(hypo)
    evasion = estimate_evasion_routes(hypo)
    cascade = compute_sanction_cascade(hypo)

    return {
        "target": target_nation,
        "severity": severity,
        "estimated_gdp_impact_pct": hypo.estimated_gdp_impact_pct,
        "key_beneficiaries": beneficiaries[:5],
        "evasion_routes": evasion[:3],
        "third_party_exposure": dict(sorted(cascade.items(), key=lambda x: x[1], reverse=True)[:8]),
        "secondary_sanctions_risk": secondary_sanctions,
        "swift_exclusion": swift_exclusion,
    }
