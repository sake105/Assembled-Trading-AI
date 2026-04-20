"""Lightweight news-to-ticker entity mapper.

Maps company names and common aliases found in news headlines to equity tickers.
Used by RSSFetcher when no external EntityLinker is available.
"""

from __future__ import annotations

import re

# Company name → ticker (curated, high-confidence only)
_ENTITY_MAP: dict[str, str] = {
    # Big Tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "tsmc": "TSM",
    "samsung": "005930.KS",
    # Finance
    "jpmorgan": "JPM",
    "j.p. morgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "citigroup": "C",
    "wells fargo": "WFC",
    "blackrock": "BLK",
    "visa": "V",
    "mastercard": "MA",
    # Energy
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "shell": "SHEL",
    "bp": "BP",
    "totalenergies": "TTE",
    "conocophillips": "COP",
    "halliburton": "HAL",
    "schlumberger": "SLB",
    "saudi aramco": "2222.SR",
    # Defense
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "boeing": "BA",
    "northrop grumman": "NOC",
    "general dynamics": "GD",
    "l3harris": "LHX",
    "bae systems": "BAESY",
    "rheinmetall": "RHM.DE",
    # Pharma
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "moderna": "MRNA",
    "astrazeneca": "AZN",
    "novartis": "NVS",
    "eli lilly": "LLY",
    # Autos
    "general motors": "GM",
    "ford": "F",
    "volkswagen": "VWAGY",
    "toyota": "TM",
    "ferrari": "RACE",
    # Commodities / Mining
    "barrick gold": "GOLD",
    "newmont": "NEM",
    "freeport-mcmoran": "FCX",
    "rio tinto": "RIO",
    "bhp": "BHP",
    # Shipping / Logistics
    "fedex": "FDX",
    "ups": "UPS",
    "maersk": "AMKBY",
    "cosco": "CICOY",
    # China
    "alibaba": "BABA",
    "tencent": "TCEHY",
    "baidu": "BIDU",
    "jd.com": "JD",
    "pinduoduo": "PDD",
    "nio": "NIO",
    "byd": "BYDDF",
    # Indices / macro proxies (not real tickers but useful labels)
    "s&p 500": "SPY",
    "nasdaq": "QQQ",
    "dow jones": "DIA",
    "vix": "VIX",
}

# Pre-compiled patterns: sorted by length descending (match longest first)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE), ticker)
    for name, ticker in sorted(_ENTITY_MAP.items(), key=lambda x: -len(x[0]))
]


def extract_tickers_from_title(title: str) -> list[str]:
    """Extract equity tickers mentioned in a news headline.

    Returns a deduplicated list of ticker strings, preserving first-match order.
    """
    if not title:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pattern, ticker in _PATTERNS:
        if pattern.search(title):
            if ticker not in seen:
                found.append(ticker)
                seen.add(ticker)
    return found


class SimpleEntityLinker:
    """Drop-in EntityLinker backed by the curated name→ticker map.

    Compatible with the `entity_linker.link(entity)` interface expected
    by RSSFetcher.
    """

    def link(self, entity: str) -> str | None:
        if not entity:
            return None
        lower = entity.lower().strip()
        # Exact map lookup
        if lower in _ENTITY_MAP:
            return _ENTITY_MAP[lower]
        # Partial prefix match (entity is a substring of a known name)
        for name, ticker in _ENTITY_MAP.items():
            if lower in name or name in lower:
                return ticker
        return None
