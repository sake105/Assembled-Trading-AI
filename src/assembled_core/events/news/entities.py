from __future__ import annotations

import re
from typing import Dict, List, Set

# Alias (lowercase) -> ISO2 country code
COUNTRY_ALIASES: Dict[str, str] = {
    # North America
    "united states": "US",
    "united states of america": "US",
    "u.s.": "US",
    "u.s": "US",
    "us": "US",
    "usa": "US",
    "u.s.a.": "US",
    "america": "US",
    "canada": "CA",
    "mexico": "MX",
    # Europe
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "french republic": "FR",
    "italy": "IT",
    "spain": "ES",
    "united kingdom": "GB",
    "u.k.": "GB",
    "uk": "GB",
    "britain": "GB",
    "great britain": "GB",
    "england": "GB",
    "netherlands": "NL",
    "holland": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "poland": "PL",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "czech republic": "CZ",
    "greece": "GR",
    "portugal": "PT",
    "ireland": "IE",
    # Eastern Europe / Eurasia
    "russia": "RU",
    "russian federation": "RU",
    "ukraine": "UA",
    "belarus": "BY",
    "turkey": "TR",
    # Middle East
    "israel": "IL",
    "palestine": "PS",
    "iran": "IR",
    "iraq": "IQ",
    "saudi arabia": "SA",
    "kingdom of saudi arabia": "SA",
    "united arab emirates": "AE",
    "uae": "AE",
    "qatar": "QA",
    "kuwait": "KW",
    "oman": "OM",
    "bahrain": "BH",
    # Asia
    "china": "CN",
    "people's republic of china": "CN",
    "prc": "CN",
    "india": "IN",
    "japan": "JP",
    "south korea": "KR",
    "republic of korea": "KR",
    "north korea": "KP",
    "democratic people's republic of korea": "KP",
    "taiwan": "TW",
    "hong kong": "HK",
    "singapore": "SG",
    "indonesia": "ID",
    "vietnam": "VN",
    "thailand": "TH",
    "malaysia": "MY",
    "philippines": "PH",
    # Africa
    "south africa": "ZA",
    "nigeria": "NG",
    "egypt": "EG",
    "kenya": "KE",
    "ethiopia": "ET",
    # Americas (rest)
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "peru": "PE",
    # Oceania
    "australia": "AU",
    "new zealand": "NZ",
}


HIGH_IMPACT_ENTITIES: List[str] = [
    "NATO",
    "European Union",
    "EU",
    "OPEC",
    "UN",
    "United Nations",
    "Red Sea",
    "Suez Canal",
    "Strait of Hormuz",
    "Hormuz Strait",
    "Taiwan Strait",
    "South China Sea",
    "East China Sea",
    "Panama Canal",
    "Gulf of Aden",
    "Black Sea",
    "Baltic Sea",
]


def _canonical_country_code(value: str) -> str | None:
    """Map various country representations to ISO2 code."""
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    lower = v.lower()

    # Direct alias match (names, common abbreviations)
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]

    # ISO2 code
    if len(v) == 2 and v.isalpha():
        return v.upper()

    # Selected ISO3 / common short forms
    iso3_map = {
        "deu": "DE",
        "ger": "DE",
        "usa": "US",
        "gbr": "GB",
        "uk": "GB",
        "chn": "CN",
        "rus": "RU",
        "ukr": "UA",
        "fra": "FR",
        "ita": "IT",
        "esp": "ES",
    }
    if lower in iso3_map:
        return iso3_map[lower]

    return None


def extract_countries(text: str) -> List[str]:
    """Extract country ISO2 codes from free text using alias lookup."""
    if not text:
        return []

    found: Set[str] = set()
    for alias, code in COUNTRY_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(code)

    # Deterministic ordering: alphabetic by ISO2 code
    return sorted(found)


def extract_entities(text: str) -> List[str]:
    """Extract high-impact entities and (optionally) country codes from text."""
    if not text:
        return []

    entities: List[str] = []

    # High-impact geo / political entities
    for name in HIGH_IMPACT_ENTITIES:
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            entities.append(name)

    # Optionally include country codes as entities (kept simple for v1)
    country_codes = extract_countries(text)
    for cc in country_codes:
        entities.append(cc)

    return entities


__all__ = [
    "COUNTRY_ALIASES",
    "HIGH_IMPACT_ENTITIES",
    "extract_countries",
    "extract_entities",
    "_canonical_country_code",
]
