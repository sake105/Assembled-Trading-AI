"""ETF Universe Loader for M10 — Universe Upgrade.

Loads and filters the ETF universe from configs/universe_etf_v1.yaml.
Provides symbol lists, asset class groupings, and correlation-cluster-aware filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default universe config path
# parents[0] = data/, parents[1] = assembled_core/, parents[2] = src/, parents[3] = repo root
_DEFAULT_UNIVERSE_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "universe_etf_v1.yaml"
)

_ETF_UNIVERSE_CACHE: dict[str, dict[str, Any]] = {}


def load_etf_universe(path: str | Path | None = None) -> dict[str, Any]:
    """Load the ETF universe config.

    Args:
        path: Path to universe YAML. Defaults to configs/universe_etf_v1.yaml.

    Returns:
        Parsed universe dict with 'etfs' key containing asset class groups.
    """
    import yaml

    resolved = Path(path) if path else _DEFAULT_UNIVERSE_PATH
    cache_key = str(resolved.resolve())
    if cache_key in _ETF_UNIVERSE_CACHE:
        return _ETF_UNIVERSE_CACHE[cache_key]
    if not resolved.exists():
        raise FileNotFoundError(f"ETF universe config not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _ETF_UNIVERSE_CACHE[cache_key] = data
    return data


def get_all_symbols(universe: dict[str, Any]) -> list[str]:
    """Return all ETF symbols from the universe config.

    Args:
        universe: Loaded universe dict from load_etf_universe().

    Returns:
        Sorted list of symbol strings.
    """
    symbols = []
    for group in universe.get("etfs", {}).values():
        for entry in group:
            sym = entry.get("symbol")
            if sym:
                symbols.append(sym)
    return sorted(set(symbols))


def get_symbols_by_asset_class(
    universe: dict[str, Any],
    asset_class: str,
) -> list[str]:
    """Return symbols filtered by asset_class field.

    Args:
        universe: Loaded universe dict.
        asset_class: e.g. "equity", "fixed_income", "commodity", "volatility".

    Returns:
        Sorted list of matching symbols.
    """
    symbols = []
    for group in universe.get("etfs", {}).values():
        for entry in group:
            if entry.get("asset_class") == asset_class:
                sym = entry.get("symbol")
                if sym:
                    symbols.append(sym)
    return sorted(set(symbols))


def get_symbols_by_group(
    universe: dict[str, Any],
    group_name: str,
) -> list[str]:
    """Return symbols from a named group (e.g. 'equity_broad', 'fixed_income').

    Args:
        universe: Loaded universe dict.
        group_name: Top-level group key under 'etfs'.

    Returns:
        List of symbols in that group.
    """
    group = universe.get("etfs", {}).get(group_name, [])
    return [e["symbol"] for e in group if "symbol" in e]


def get_defensive_symbols(universe: dict[str, Any]) -> list[str]:
    """Return defensive / safe-haven symbols: fixed_income + gold + volatility.

    Used by Crisis Alpha baskets for flight-to-safety positioning.
    """
    defensive = []
    for group in universe.get("etfs", {}).values():
        for entry in group:
            ac = entry.get("asset_class", "")
            sub = entry.get("sub_type", "")
            if ac in ("fixed_income", "volatility") or sub == "gold":
                sym = entry.get("symbol")
                if sym:
                    defensive.append(sym)
    return sorted(set(defensive))


# ---------------------------------------------------------------------------
# Inverse ETF Map — long-only proxies for short exposure
# ---------------------------------------------------------------------------

#: Maps a long ETF symbol to its inverse/short ETF counterpart.
#: Used for market-neutral construction without short-selling directly.
INVERSE_ETF_MAP: dict[str, str] = {
    # Broad equity
    "SPY": "SH",    # ProShares Short S&P500
    "QQQ": "PSQ",   # ProShares Short QQQ
    "IWM": "RWM",   # ProShares Short Russell 2000
    "DIA": "DOG",   # ProShares Short Dow30
    # Sector ETFs
    "XLK": "REW",   # ProShares UltraShort Technology (2×, use carefully)
    "XLF": "SKF",   # ProShares UltraShort Financials (2×, use carefully)
    "XLE": "DDG",   # ProShares Short Oil & Gas
    "XLV": "RXD",   # ProShares UltraShort Health Care (2×)
    "XLI": "SIJ",   # ProShares UltraShort Industrials (2×)
    "XLY": "SCC",   # ProShares UltraShort Consumer Disc. (2×)
    "XLP": "SZK",   # ProShares UltraShort Consumer Staples (2×)
    "XLU": "SDP",   # ProShares UltraShort Utilities (2×)
    "XLB": "SMN",   # ProShares UltraShort Basic Materials (2×)
    "XLRE": "REK",  # ProShares Short Real Estate
    # Fixed income
    "TLT": "TBF",   # ProShares Short 20+ Year Treasury
    "IEF": "TBX",   # ProShares Short 7-10 Year Treasury
    "HYG": "SJB",   # ProShares Short High Yield
    # International
    "EFA": "EFZ",   # ProShares Short MSCI EAFE
    "EEM": "EEV",   # ProShares UltraShort MSCI Emerging Mkts (2×)
}


def get_inverse_etf(symbol: str) -> str | None:
    """Return the inverse ETF symbol for a given long ETF symbol.

    Args:
        symbol: Long ETF symbol (e.g. "SPY").

    Returns:
        Inverse ETF symbol (e.g. "SH") or None if not mapped.
    """
    return INVERSE_ETF_MAP.get(symbol.upper())


def get_inverse_etf_map() -> dict[str, str]:
    """Return the full inverse ETF mapping dict."""
    return dict(INVERSE_ETF_MAP)


def build_symbol_metadata(universe: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a symbol -> metadata dict for all ETFs.

    Returns:
        Dict mapping symbol -> {name, asset_class, sub_type, group}.
    """
    result: dict[str, dict[str, str]] = {}
    for group_name, group in universe.get("etfs", {}).items():
        for entry in group:
            sym = entry.get("symbol")
            if not sym:
                continue
            result[sym] = {
                "name": entry.get("name", ""),
                "asset_class": entry.get("asset_class", ""),
                "sub_type": entry.get("sub_type", ""),
                "group": group_name,
            }
    return result
