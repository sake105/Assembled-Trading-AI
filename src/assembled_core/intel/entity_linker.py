"""Entity → Ticker Linker (X2).

Maps free-text entity names (companies, countries, sectors) to tradeable symbols.
Uses exact match, alias lookup, and optional fuzzy match (rapidfuzz if installed).

Usage:
    linker = EntityLinker.from_security_master("configs/security_master.csv")
    matches = linker.link("Apple")          # → ["AAPL"]
    matches = linker.link("United States")  # → [] (geo → no direct ticker)
    geo = linker.geo_to_etf("US")          # → ["SPY", "QQQ"]
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process as rfprocess
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.debug("[SKIP] rapidfuzz not installed — fuzzy matching disabled in EntityLinker")

# ---------------------------------------------------------------------------
# Built-in alias table (company name → ticker)
# ---------------------------------------------------------------------------

_BUILTIN_ALIASES: dict[str, list[str]] = {
    # US mega-caps
    "apple": ["AAPL"],
    "amazon": ["AMZN"],
    "microsoft": ["MSFT"],
    "alphabet": ["GOOGL"],
    "google": ["GOOGL"],
    "meta": ["META"],
    "facebook": ["META"],
    "nvidia": ["NVDA"],
    "tesla": ["TSLA"],
    "berkshire": ["BRK.B"],
    "jpmorgan": ["JPM"],
    "johnson": ["JNJ"],
    "visa": ["V"],
    "mastercard": ["MA"],
    "chevron": ["CVX"],
    "exxon": ["XOM"],
    "walmart": ["WMT"],
    "costco": ["COST"],
    "abbvie": ["ABBV"],
    "broadcom": ["AVGO"],
    "salesforce": ["CRM"],
    "adobe": ["ADBE"],
    # ETF sector proxies
    "technology": ["QQQ", "XLK"],
    "energy": ["XLE", "USO"],
    "healthcare": ["XLV"],
    "financials": ["XLF"],
    "consumer": ["XLY", "XLP"],
    "utilities": ["XLU"],
    "materials": ["XLB"],
    "industrials": ["XLI"],
    "real estate": ["VNQ", "IYR"],
    "communication": ["XLC"],
    # Geo → broad ETF proxies
    "us": ["SPY", "QQQ"],
    "usa": ["SPY", "QQQ"],
    "united states": ["SPY", "QQQ"],
    "europe": ["VGK", "EZU"],
    "china": ["FXI", "MCHI"],
    "japan": ["EWJ"],
    "emerging markets": ["EEM", "VWO"],
    "global": ["VT", "ACWI"],
    # Commodity
    "oil": ["USO", "XLE"],
    "gold": ["GLD"],
    "silver": ["SLV"],
    "natural gas": ["UNG"],
}

# ---------------------------------------------------------------------------
# Geo region → ETF mapping (separate from equity mapping)
# ---------------------------------------------------------------------------

_GEO_ETF_MAP: dict[str, list[str]] = {
    "US": ["SPY", "QQQ"],
    "EU": ["VGK", "EZU"],
    "CN": ["FXI", "MCHI"],
    "JP": ["EWJ"],
    "EM": ["EEM", "VWO"],
    "GLOBAL": ["VT", "ACWI"],
    "ENERGY": ["XLE", "USO"],
    "OIL": ["USO", "XLE"],
    "GOLD": ["GLD"],
}


class EntityLinker:
    """Maps entity names to tradeable symbols.

    Construction:
        EntityLinker.from_security_master(path) — load symbols from CSV
        EntityLinker()                           — alias-only mode
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        extra_aliases: dict[str, list[str]] | None = None,
        fuzzy_threshold: int = 85,
    ) -> None:
        self._symbols: set[str] = set(symbols or [])
        # Build lookup: normalized name → [tickers]
        self._alias_map: dict[str, list[str]] = dict(_BUILTIN_ALIASES)
        if extra_aliases:
            for k, v in extra_aliases.items():
                self._alias_map[k.lower().strip()] = list(v)
        # Add exact ticker match
        for sym in self._symbols:
            self._alias_map[sym.lower()] = [sym]
        self._fuzzy_threshold = fuzzy_threshold
        logger.debug(
            "[OK] EntityLinker initialized: %d symbols, %d aliases, fuzzy=%s",
            len(self._symbols), len(self._alias_map), _RAPIDFUZZ_AVAILABLE,
        )

    @classmethod
    def from_security_master(
        cls,
        csv_path: str | Path,
        *,
        extra_aliases: dict[str, list[str]] | None = None,
        fuzzy_threshold: int = 85,
    ) -> "EntityLinker":
        """Load symbol universe from a CSV with a 'symbol' column."""
        p = Path(csv_path)
        symbols: list[str] = []
        if p.exists():
            with open(p, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    sym = (row.get("symbol") or "").strip()
                    if sym:
                        symbols.append(sym)
        else:
            logger.warning("[WARN] EntityLinker: security_master not found at %s", p)
        return cls(symbols=symbols, extra_aliases=extra_aliases, fuzzy_threshold=fuzzy_threshold)

    def link(self, entity_name: str, *, max_results: int = 5) -> list[str]:
        """Map an entity name to a list of tickers.

        Lookup order:
        1. Exact ticker match (case-insensitive)
        2. Alias table lookup
        3. Fuzzy match (if rapidfuzz installed and threshold met)

        Returns empty list if no match above threshold.
        """
        if not entity_name or not isinstance(entity_name, str):
            return []
        key = entity_name.lower().strip()

        # 1. Exact / alias
        if key in self._alias_map:
            return self._alias_map[key][:max_results]

        # 2. Partial alias scan (entity_name contains alias key or vice-versa)
        for alias_key, tickers in self._alias_map.items():
            if alias_key in key or key in alias_key:
                if len(alias_key) >= 4:  # avoid short false-positives
                    return tickers[:max_results]

        # 3. Fuzzy (optional)
        if _RAPIDFUZZ_AVAILABLE and self._alias_map:
            choices = list(self._alias_map.keys())
            result = rfprocess.extractOne(key, choices, scorer=fuzz.token_sort_ratio)
            if result is not None:
                match_key, score, _ = result
                if score >= self._fuzzy_threshold:
                    logger.debug(
                        "[OK] EntityLinker fuzzy: '%s' → '%s' (score=%d)",
                        entity_name, match_key, score,
                    )
                    return self._alias_map[match_key][:max_results]

        return []

    def geo_to_etf(self, geo_code: str) -> list[str]:
        """Map a two-letter geo code (e.g. 'US', 'CN') to ETF proxies."""
        return list(_GEO_ETF_MAP.get(geo_code.upper(), []))

    def link_many(
        self, entity_names: list[str], *, max_per_entity: int = 3
    ) -> dict[str, list[str]]:
        """Batch link. Returns {entity_name: [tickers]}."""
        return {name: self.link(name, max_results=max_per_entity) for name in entity_names}

    def add_alias(self, name: str, tickers: list[str]) -> None:
        """Dynamically add an alias (runtime-only, not persisted)."""
        self._alias_map[name.lower().strip()] = list(tickers)

    @property
    def symbol_count(self) -> int:
        return len(self._symbols)

    @property
    def alias_count(self) -> int:
        return len(self._alias_map)
