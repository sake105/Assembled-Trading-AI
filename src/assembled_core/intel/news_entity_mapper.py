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
    # Extended Tech
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "shopify": "SHOP",
    "uber": "UBER",
    "lyft": "LYFT",
    "snap": "SNAP",
    "snapchat": "SNAP",
    "spotify": "SPOT",
    "roblox": "RBLX",
    "palantir": "PLTR",
    "datadog": "DDOG",
    "cloudflare": "NET",
    "zscaler": "ZS",
    "okta": "OKTA",
    "palo alto": "PANW",
    "palo alto networks": "PANW",
    "crowdstrike": "CRWD",
    "sentinelone": "S",
    "twitter": "X",
    "x corp": "X",
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
    # Extended Finance
    "american express": "AXP",
    "amex": "AXP",
    "capital one": "COF",
    "us bancorp": "USB",
    "us bank": "USB",
    "pnc": "PNC",
    "bnp paribas": "BNPQY",
    "deutsche bank": "DB",
    "ubs": "UBS",
    "credit suisse": "UBS",
    "barclays": "BCS",
    "hsbc": "HSBC",
    "standard chartered": "SCBFF",
    "societe generale": "SCGLY",
    "ing": "ING",
    "allianz": "ALIZY",
    "aig": "AIG",
    "metlife": "MET",
    "prudential": "PRU",
    "berkshire hathaway": "BRK-B",
    "berkshire": "BRK-B",
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
    # Extended Energy
    "pioneer natural resources": "PXD",
    "pioneer natural": "PXD",
    "devon energy": "DVN",
    "eog resources": "EOG",
    "coterra": "CTRA",
    "williams companies": "WMB",
    "williams cos": "WMB",
    "kinder morgan": "KMI",
    "enbridge": "ENB",
    "baker hughes": "BKR",
    "valero": "VLO",
    "marathon petroleum": "MPC",
    "phillips 66": "PSX",
    "petrobras": "PBR",
    "ypf": "YPF",
    "repsol": "REPYY",
    # Defense
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "boeing": "BA",
    "northrop grumman": "NOC",
    "general dynamics": "GD",
    "l3harris": "LHX",
    "bae systems": "BAESY",
    "rheinmetall": "RHM.DE",
    # Extended Defense
    "heico": "HEI",
    "curtiss-wright": "CW",
    "kratos": "KTOS",
    "axon": "AXON",
    "teledyne": "TDY",
    "leidos": "LDOS",
    "saic": "SAIC",
    "booz allen": "BAH",
    "booz allen hamilton": "BAH",
    "caci": "CACI",
    "mantech": "MAN",
    # Pharma
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "moderna": "MRNA",
    "astrazeneca": "AZN",
    "novartis": "NVS",
    "eli lilly": "LLY",
    # Extended Pharma/Healthcare
    "abbvie": "ABBV",
    "bristol myers": "BMY",
    "bristol-myers squibb": "BMY",
    "merck": "MRK",
    "gilead": "GILD",
    "biogen": "BIIB",
    "regeneron": "REGN",
    "vertex": "VRTX",
    "amgen": "AMGN",
    "danaher": "DHR",
    "thermo fisher": "TMO",
    "illumina": "ILMN",
    "edwards lifesciences": "EW",
    # Autos
    "general motors": "GM",
    "ford": "F",
    "volkswagen": "VWAGY",
    "toyota": "TM",
    "ferrari": "RACE",
    # Extended Industrials
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "honeywell": "HON",
    "3m": "MMM",
    "emerson": "EMR",
    "eaton": "ETN",
    "parker hannifin": "PH",
    "cummins": "CMI",
    "textron": "TXT",
    "paccar": "PCAR",
    "oshkosh": "OSK",
    "general electric": "GE",
    # Commodities / Mining
    "barrick gold": "GOLD",
    "newmont": "NEM",
    "freeport-mcmoran": "FCX",
    "rio tinto": "RIO",
    "bhp": "BHP",
    # Extended Mining/Materials
    "glencore": "GLNCY",
    "anglo american": "AAUKF",
    "alcoa": "AA",
    "nucor": "NUE",
    "us steel": "X",
    "cleveland-cliffs": "CLF",
    "mosaic": "MOS",
    "cf industries": "CF",
    "nutrien": "NTR",
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
    # Crypto / Fintech
    "coinbase": "COIN",
    "block": "SQ",
    "square": "SQ",
    "paypal": "PYPL",
    "microstrategy": "MSTR",
    "marathon digital": "MARA",
    "riot platforms": "RIOT",
    "riot blockchain": "RIOT",
    # Consumer / Retail
    "nike": "NKE",
    "adidas": "ADDYY",
    "lvmh": "LVMUY",
    "hermes": "HESAY",
    "hermès": "HESAY",
    "starbucks": "SBUX",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "yum brands": "YUM",
    "dollar general": "DG",
    "dollar tree": "DLTR",
    "home depot": "HD",
    "lowe's": "LOW",
    "lowes": "LOW",
    "tjx": "TJX",
    "tjx companies": "TJX",
    "ross stores": "ROST",
    "walmart": "WMT",
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


# ---------------------------------------------------------------------------
# Sector / Country asset helpers (mirrors news_classifier.py mappings)
# ---------------------------------------------------------------------------

SECTOR_TO_ETFS: dict[str, list[str]] = {
    "energy": ["XLE", "USO", "XOM", "CVX", "OIH"],
    "defense": ["XAR", "ITA", "LMT", "RTX", "NOC"],
    "financials": ["XLF", "KRE", "GS", "JPM", "BAC"],
    "tech": ["QQQ", "XLK", "SMH", "NVDA", "INTC"],
    "healthcare": ["XLV", "IBB", "JNJ", "PFE", "MRNA"],
    "industrials": ["XLI", "FXI", "GE", "BA", "CAT"],
    "materials": ["XLB", "GLD", "COPX", "FCX", "NEM"],
    "consumer": ["XLY", "XLP", "AMZN", "WMT", "TGT"],
    "utilities": ["XLU", "NEE", "DUK"],
}

COUNTRY_TO_ASSETS: dict[str, list[str]] = {
    "RU": ["RSXJ", "OIL", "GAZP", "EURUSD"],
    "CN": ["FXI", "KWEB", "MCHI", "CNYUSD"],
    "IR": ["USO", "XLE", "OIH"],
    "SA": ["USO", "XLE", "ARAMCO"],
    "US": ["SPY", "QQQ", "DXY"],
    "DE": ["EWG", "DAX"],
    "JP": ["EWJ", "JPYUSD"],
    "IL": ["EIS", "XAR"],
    "UA": ["XLE", "WEAT"],
    "KP": ["XAR", "ITA"],
    "TR": ["TUR", "TRYUSD"],
    "IN": ["INDA", "INRUSD"],
    "BR": ["EWZ", "BRLUSD"],
    "GB": ["EWU", "GBPUSD"],
    "FR": ["EWQ", "EURUSD"],
    "EU": ["EZU", "EURUSD"],
}


def get_sector_etfs(sector: str) -> list[str]:
    """Return ETFs/tickers for the given sector. Returns [] if unknown."""
    return SECTOR_TO_ETFS.get(sector.lower(), [])


def get_country_assets(iso2: str) -> list[str]:
    """Return assets associated with the given ISO-2 country code. Returns [] if unknown."""
    return COUNTRY_TO_ASSETS.get(iso2.upper(), [])
