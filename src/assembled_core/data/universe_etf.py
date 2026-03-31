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


def load_etf_universe(path: str | Path | None = None) -> dict[str, Any]:
    """Load the ETF universe config.

    Args:
        path: Path to universe YAML. Defaults to configs/universe_etf_v1.yaml.

    Returns:
        Parsed universe dict with 'etfs' key containing asset class groups.
    """
    import yaml

    resolved = Path(path) if path else _DEFAULT_UNIVERSE_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"ETF universe config not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

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
