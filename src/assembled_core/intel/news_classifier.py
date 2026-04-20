"""Rule-based news event classifier.

Assigns multi-label event types, severity score, market direction,
time horizon, affected sectors, and affected asset tickers to a news headline.

Zero external dependencies — pure Python, deterministic, fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Event-type keyword maps
# ---------------------------------------------------------------------------

_EVENT_KEYWORDS: dict[str, frozenset[str]] = {
    "sanctions": frozenset({
        "sanction", "embargo", "ban", "restrict", "freeze assets",
        "export control", "asset freeze", "blocked", "blacklisted",
    }),
    "military_strike": frozenset({
        "strike", "attack", "bomb", "missile", "explosion", "airstrike",
        "shelling", "rocket", "mortar", "drone strike", "air raid",
    }),
    "war_escalation": frozenset({
        "war", "escalat", "invasion", "offensive", "frontline", "troops advance",
        "troops deploy", "ground offensive", "territorial", "occupied",
    }),
    "energy_disruption": frozenset({
        "pipeline", "oil supply", "gas supply", "lng", "shortage",
        "energy crisis", "supply cut", "energy supply", "fuel supply",
        "refinery attack", "oil field",
    }),
    "central_bank": frozenset({
        "fed", "ecb", "boe", "boj", "rate hike", "rate cut",
        "interest rate", "monetary policy", "inflation", "fomc",
        "central bank", "quantitative easing", "tapering", "basis points",
    }),
    "trade_policy": frozenset({
        "tariff", "trade war", "import ban", "export ban", "wto",
        "customs", "trade deal", "trade agreement", "import duty",
        "trade barrier", "quota", "trade restriction",
    }),
    "political_crisis": frozenset({
        "coup", "resign", "protest", "riot", "election fraud",
        "constitutional crisis", "government collapse", "martial law",
        "impeach", "overthrow", "junta",
    }),
    "earnings": frozenset({
        "earnings", "revenue", "profit", "loss", "quarterly", "eps",
        "guidance", "beat", "miss", "results", "income", "fiscal year",
        "q1", "q2", "q3", "q4", "annual results",
    }),
    "ma_activity": frozenset({
        "merger", "acquisition", "takeover", "deal", "buyout",
        "offer", "bid", "acquir", "merge", "purchase agreement",
        "strategic acquisition",
    }),
    "regulatory": frozenset({
        "fda", "sec", "ftc", "doj", "antitrust", "approval", "fine",
        "investigation", "penalty", "probe", "lawsuit", "regulator",
        "compliance", "violation", "enforcement",
    }),
    "cyber_attack": frozenset({
        "cyber", "hack", "ransomware", "breach", "malware", "zero-day",
        "phishing", "data breach", "cybersecurity", "intrusion",
        "compromised", "stolen data",
    }),
    "natural_disaster": frozenset({
        "earthquake", "hurricane", "typhoon", "flood", "wildfire",
        "storm", "drought", "tsunami", "volcanic", "cyclone",
        "tornado", "natural disaster",
    }),
    "diplomatic": frozenset({
        "treaty", "summit", "talks", "ceasefire", "peace deal",
        "agreement", "expelled", "diplomatic", "ambassador", "embassy",
        "bilateral", "multilateral", "negotiation",
    }),
    "market_stress": frozenset({
        "crash", "rout", "plunge", "sell-off", "circuit breaker",
        "default", "downgrade", "bankruptcy", "selloff", "meltdown",
        "market rout", "capitulation", "bear market", "collapse",
    }),
}

# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

_SEVERITY_KEYWORDS: list[tuple[frozenset[str], float]] = [
    (frozenset({"nuclear", "catastrophic", "ww3", "world war", "armageddon"}), 10.0),
    (frozenset({"war", "invasion", "collapse"}), 8.0),
    (frozenset({"coup", "default", "banking crisis", "financial crisis"}), 7.5),
    (frozenset({"sanctions", "attack", "explosion", "strike", "bomb", "missile"}), 6.5),
    (frozenset({"crisis", "recession", "downgrade", "bankruptcy", "meltdown"}), 5.5),
    (frozenset({"tariff", "protest", "resignation", "escalat", "shutdown"}), 4.0),
    (frozenset({"election", "merger", "acquisition", "regulatory", "probe"}), 3.0),
    (frozenset({"rate decision", "trade", "earnings", "guidance", "approval"}), 2.5),
]

_TIER_SEVERITY_MULTIPLIER: dict[str, float] = {
    "T0": 1.0,
    "T1": 0.9,
    "T2": 0.7,
    "T3": 0.5,
}

# ---------------------------------------------------------------------------
# Market direction keywords
# ---------------------------------------------------------------------------

_BEARISH_WORDS = frozenset({
    "war", "sanction", "attack", "crash", "default", "recession", "crisis",
    "bomb", "invasion", "collapse", "downgrade", "shortage", "rout", "plunge",
    "sell-off", "selloff", "bankrupt", "missile", "explosion", "coup",
    "meltdown", "bear market", "downfall", "spiral", "airstrike", "strike",
    "shelling", "offensive", "escalation", "conflict", "embargo",
})

_BULLISH_WORDS = frozenset({
    "deal", "agreement", "stimulus", "recovery", "merger", "approval",
    "ceasefire", "peace", "rate cut", "earnings beat", "acquisition",
    "bullish", "surge", "rally", "record high", "buy", "upgrade",
    "beat expectations", "positive", "growth", "rebound",
})

# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------

_INTRADAY_WORDS = frozenset({
    "breaking", "flash", "urgent", "explosion", "missile", "attack",
    "explosion", "shoot", "bombing", "immediate",
})

_SHORT_WORDS = frozenset({
    "sanctions", "coup", "election result", "earnings", "bankruptcy",
    "vote", "referendum", "result",
})

_MEDIUM_WORDS = frozenset({
    "trade policy", "tariff", "regulatory", "approval", "merger",
    "acquisition", "rate decision", "rate hike", "rate cut", "m&a",
})

_LONG_WORDS = frozenset({
    "war resolution", "treaty", "demographic", "structural",
    "constitutional", "peace agreement", "long-term",
})

# ---------------------------------------------------------------------------
# Sector keyword maps
# ---------------------------------------------------------------------------

_SECTOR_KEYWORDS: dict[str, frozenset[str]] = {
    "energy": frozenset({
        "oil", "gas", "lng", "pipeline", "opec", "crude", "refin",
        "energy", "fuel", "coal", "brent", "wti", "petrol",
    }),
    "defense": frozenset({
        "military", "nato", "weapon", "missile", "defense", "army",
        "navy", "air force", "procurement", "arms", "soldier", "troops",
        "pentagon", "war", "combat", "aircraft carrier",
    }),
    "financials": frozenset({
        "bank", "fed", "ecb", "interest rate", "credit", "loan",
        "debt", "bond", "yield", "currency", "forex", "inflation",
        "monetary", "rate hike", "rate cut", "liquidity",
    }),
    "tech": frozenset({
        "semiconductor", "chip", "ai", "tech", "software", "cyber",
        "hack", "silicon", "nvidia", "intel", "processor", "cloud",
        "data center", "quantum",
    }),
    "healthcare": frozenset({
        "fda", "drug", "vaccine", "clinical", "pharma", "cancer",
        "biotech", "approval", "trial", "therapeutics", "genomic",
    }),
    "industrials": frozenset({
        "supply chain", "shipping", "logistics", "trade", "tariff",
        "manufacturing", "freight", "factory", "production", "export",
    }),
    "materials": frozenset({
        "gold", "copper", "mining", "rare earth", "lithium", "commodity",
        "aluminum", "zinc", "nickel", "iron ore", "steel", "cobalt",
    }),
    "consumer": frozenset({
        "retail", "consumer confidence", "spending", "amazon", "walmart",
        "consumer", "shopping", "e-commerce",
    }),
    "utilities": frozenset({
        "power grid", "electricity", "nuclear power plant", "energy transition",
        "renewable", "solar", "wind power", "grid",
    }),
}

# ---------------------------------------------------------------------------
# Sector → ETF / asset mappings
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
    "FR": ["EWQ", "CADUSD"],
    "EU": ["EZU", "EURUSD"],
}

_COMMODITY_KEYWORDS: dict[str, list[str]] = {
    "gold": ["GLD", "GC=F", "GOLD"],
    "oil": ["USO", "CL=F", "XLE"],
    "natural gas": ["UNG", "NG=F"],
    "wheat": ["WEAT", "ZW=F"],
    "copper": ["COPX", "HG=F", "FCX"],
    "silver": ["SLV", "SI=F"],
    "bitcoin": ["GBTC", "BTC-USD"],
    "lithium": ["LIT", "LTHM"],
    "uranium": ["URA", "CCJ"],
}


# ---------------------------------------------------------------------------
# Main dataclass + classify function
# ---------------------------------------------------------------------------


@dataclass
class NewsClassification:
    event_types: list[str] = field(default_factory=list)
    severity: float = 0.0  # 0–10
    market_direction: str = "neutral"  # "bearish"/"bullish"/"neutral"/"mixed"
    time_horizon: str = "short"  # "intraday"/"short"/"medium"/"long"
    affected_sectors: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0–1


def classify_news_event(
    title: str,
    geo_tags: list[str] | None = None,
    source_tier: str = "T2",
    tickers: list[str] | None = None,
) -> NewsClassification:
    """Classify a news headline into multi-label event types and market signals.

    Args:
        title: News headline text.
        geo_tags: Optional list of ISO-2 country codes from geo extraction.
        source_tier: Source tier string ("T0"/"T1"/"T2"/"T3").
        tickers: Optional pre-extracted ticker symbols.

    Returns:
        NewsClassification dataclass with all signal fields populated.
    """
    if not title:
        return NewsClassification(
            event_types=[],
            severity=0.0,
            market_direction="neutral",
            time_horizon="short",
            affected_sectors=[],
            affected_assets=[],
            confidence=0.0,
        )

    title_lower = title.lower()
    geo_tags = geo_tags or []
    tickers = tickers or []

    # --- Event types (multi-label) ---
    event_types: list[str] = []
    for etype, keywords in _EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                event_types.append(etype)
                break

    # --- Severity ---
    raw_severity = 1.0
    for keywords, score in _SEVERITY_KEYWORDS:
        for kw in keywords:
            if kw in title_lower:
                raw_severity = max(raw_severity, score)
                break
    tier_mult = _TIER_SEVERITY_MULTIPLIER.get(source_tier, 0.7)
    severity = round(min(raw_severity * tier_mult, 10.0), 2)

    # --- Market direction ---
    has_bearish = any(w in title_lower for w in _BEARISH_WORDS)
    has_bullish = any(w in title_lower for w in _BULLISH_WORDS)
    if has_bearish and has_bullish:
        market_direction = "mixed"
    elif has_bearish:
        market_direction = "bearish"
    elif has_bullish:
        market_direction = "bullish"
    else:
        market_direction = "neutral"

    # --- Time horizon ---
    time_horizon = "short"  # default
    if any(w in title_lower for w in _INTRADAY_WORDS):
        time_horizon = "intraday"
    elif any(w in title_lower for w in _LONG_WORDS):
        time_horizon = "long"
    elif any(w in title_lower for w in _MEDIUM_WORDS):
        time_horizon = "medium"
    # Intraday overrides if military event type present
    if "military_strike" in event_types or "war_escalation" in event_types:
        if time_horizon not in ("long",):
            time_horizon = "intraday"

    # --- Affected sectors ---
    affected_sectors: list[str] = []
    for sector, keywords in _SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                affected_sectors.append(sector)
                break
    # Event-type → implicit sector hints
    _EVENT_SECTOR_HINTS: dict[str, list[str]] = {
        "sanctions": ["financials"],
        "military_strike": ["defense"],
        "war_escalation": ["defense", "energy"],
        "energy_disruption": ["energy"],
        "central_bank": ["financials"],
        "trade_policy": ["industrials"],
        "cyber_attack": ["tech"],
        "natural_disaster": ["utilities", "industrials"],
        "ma_activity": [],
        "regulatory": [],
        "earnings": [],
        "diplomatic": [],
        "market_stress": ["financials"],
        "political_crisis": [],
    }
    for etype in event_types:
        for implicit_sector in _EVENT_SECTOR_HINTS.get(etype, []):
            if implicit_sector not in affected_sectors:
                affected_sectors.append(implicit_sector)
    # Deduplicate
    seen_sectors: set[str] = set()
    deduped_sectors: list[str] = []
    for s in affected_sectors:
        if s not in seen_sectors:
            deduped_sectors.append(s)
            seen_sectors.add(s)
    affected_sectors = deduped_sectors

    # --- Affected assets ---
    affected_assets: list[str] = list(tickers)
    seen_assets: set[str] = set(tickers)

    # From sectors
    for sector in affected_sectors:
        for asset in SECTOR_TO_ETFS.get(sector, []):
            if asset not in seen_assets:
                affected_assets.append(asset)
                seen_assets.add(asset)

    # From geo_tags
    for iso2 in geo_tags:
        for asset in COUNTRY_TO_ASSETS.get(iso2.upper(), []):
            if asset not in seen_assets:
                affected_assets.append(asset)
                seen_assets.add(asset)

    # From commodity keywords
    for commodity, assets in _COMMODITY_KEYWORDS.items():
        if commodity in title_lower:
            for asset in assets:
                if asset not in seen_assets:
                    affected_assets.append(asset)
                    seen_assets.add(asset)

    # --- Confidence (0-1) ---
    # Based on: number of event types matched, severity, source tier
    base_confidence = min(0.4 + 0.05 * len(event_types) + severity / 50.0, 0.95)
    tier_conf_mult = {"T0": 1.0, "T1": 0.95, "T2": 0.8, "T3": 0.6}.get(source_tier, 0.8)
    confidence = round(base_confidence * tier_conf_mult, 3)

    return NewsClassification(
        event_types=event_types,
        severity=severity,
        market_direction=market_direction,
        time_horizon=time_horizon,
        affected_sectors=affected_sectors,
        affected_assets=affected_assets,
        confidence=confidence,
    )


def classify_batch(
    titles: list[str],
    geo_tags_list: list[list[str]] | None = None,
    source_tier: str = "T2",
) -> list[NewsClassification]:
    """Classify a batch of headlines. geo_tags_list aligns with titles by index."""
    if geo_tags_list is None:
        geo_tags_list = [[] for _ in titles]
    return [
        classify_news_event(title, geo_tags=geo_tags, source_tier=source_tier)
        for title, geo_tags in zip(titles, geo_tags_list)
    ]


# ---------------------------------------------------------------------------
# Source bias tagging (Point 40)
# ---------------------------------------------------------------------------

# Source ID → known bias attributes
# Bias is not a judgement of quality — it reflects systematic slant that should
# be accounted for when aggregating signals across sources.
_SOURCE_BIAS: dict[str, dict[str, str]] = {
    # State / government media
    "xinhua": {"geo_bias": "CN", "editorial_bias": "pro_government"},
    "cgtn": {"geo_bias": "CN", "editorial_bias": "pro_government"},
    "rt": {"geo_bias": "RU", "editorial_bias": "pro_government"},
    "tass": {"geo_bias": "RU", "editorial_bias": "pro_government"},
    "sputnik": {"geo_bias": "RU", "editorial_bias": "pro_government"},
    "press_tv": {"geo_bias": "IR", "editorial_bias": "pro_government"},
    "al_jazeera": {"geo_bias": "QA", "editorial_bias": "neutral"},
    "aljazeera": {"geo_bias": "QA", "editorial_bias": "neutral"},
    "aljaz": {"geo_bias": "QA", "editorial_bias": "neutral"},
    "presstv": {"geo_bias": "IR", "editorial_bias": "pro_government"},
    # Western mainstream
    "reuters": {"geo_bias": "GB", "editorial_bias": "neutral"},
    "ap": {"geo_bias": "US", "editorial_bias": "neutral"},
    "apnews": {"geo_bias": "US", "editorial_bias": "neutral"},
    "bbc": {"geo_bias": "GB", "editorial_bias": "center"},
    "bbc_world": {"geo_bias": "GB", "editorial_bias": "center"},
    "cnn": {"geo_bias": "US", "editorial_bias": "center_left"},
    "fox": {"geo_bias": "US", "editorial_bias": "right"},
    "nyt": {"geo_bias": "US", "editorial_bias": "center_left"},
    "wsj": {"geo_bias": "US", "editorial_bias": "center_right"},
    "ft": {"geo_bias": "GB", "editorial_bias": "center"},
    "guardian": {"geo_bias": "GB", "editorial_bias": "left"},
    "axios": {"geo_bias": "US", "editorial_bias": "center"},
    "politico": {"geo_bias": "US", "editorial_bias": "center"},
    "bloomberg": {"geo_bias": "US", "editorial_bias": "center"},
    "wapo": {"geo_bias": "US", "editorial_bias": "center_left"},
    "npr": {"geo_bias": "US", "editorial_bias": "center_left"},
    # Financial
    "cnbc": {"geo_bias": "US", "editorial_bias": "center"},
    "marketwatch": {"geo_bias": "US", "editorial_bias": "center"},
    "seeking_alpha": {"geo_bias": "US", "editorial_bias": "crowdsourced"},
    # Regional / specialty
    "haaretz": {"geo_bias": "IL", "editorial_bias": "left"},
    "kyiv_independent": {"geo_bias": "UA", "editorial_bias": "pro_ukrainian"},
    "kyivindependent": {"geo_bias": "UA", "editorial_bias": "pro_ukrainian"},
    "nikkei": {"geo_bias": "JP", "editorial_bias": "center"},
    "nikkei_asia": {"geo_bias": "JP", "editorial_bias": "center"},
    "spiegel": {"geo_bias": "DE", "editorial_bias": "center_left"},
    "euractiv": {"geo_bias": "EU", "editorial_bias": "pro_eu"},
    "dw": {"geo_bias": "DE", "editorial_bias": "center"},
    "france24": {"geo_bias": "FR", "editorial_bias": "center"},
    "rfi": {"geo_bias": "FR", "editorial_bias": "center"},
    "voa": {"geo_bias": "US", "editorial_bias": "pro_western"},
    "sky_news": {"geo_bias": "GB", "editorial_bias": "center_right"},
}

# T3 (state media) source IDs that carry discount warnings
_STATE_MEDIA_SOURCE_IDS = frozenset({
    "xinhua", "cgtn", "rt", "tass", "sputnik", "press_tv", "presstv",
})


def get_source_bias(source_id: str) -> dict[str, str]:
    """Return known bias attributes for a source_id.

    Args:
        source_id: Feed/source identifier (matched case-insensitively).

    Returns:
        Dict with 'geo_bias' and 'editorial_bias' keys, or empty dict if unknown.
    """
    return _SOURCE_BIAS.get(source_id.lower().strip(), {})


def is_state_media(source_id: str) -> bool:
    """Return True if the source is known state-controlled media."""
    return source_id.lower().strip() in _STATE_MEDIA_SOURCE_IDS


def apply_source_bias_discount(confidence: float, source_id: str) -> float:
    """Apply a confidence discount for known state media sources.

    State media (RT, Xinhua, TASS, etc.) gets a 30% confidence penalty
    because their editorial slant is systematic. Other biased sources
    get a smaller 10% discount.

    Args:
        confidence: Base confidence score [0, 1].
        source_id: Feed/source identifier.

    Returns:
        Adjusted confidence score [0, 1].
    """
    if is_state_media(source_id):
        return round(confidence * 0.70, 4)
    bias = get_source_bias(source_id)
    if bias.get("editorial_bias") in ("pro_government", "pro_western", "pro_ukrainian"):
        return round(confidence * 0.90, 4)
    return confidence


__all__ = [
    "NewsClassification",
    "SECTOR_TO_ETFS",
    "COUNTRY_TO_ASSETS",
    "classify_news_event",
    "classify_batch",
    "get_source_bias",
    "is_state_media",
    "apply_source_bias_discount",
]
