"""Cost model configuration for portfolio simulation.

Two layers:

* :class:`CostModel` — backtest-engine legacy defaults. Unchanged since
  the institutional hardening sprints; kept stable because downstream
  backtests depend on its exact numbers for 1e-9 regression.
* :func:`get_tier_for_symbol` / :func:`get_tier_costs` — per-symbol cost
  tiers driven by ``config/cost_tiers.yaml``. These are what the paper
  engine's Almgren-Chriss fill simulator consumes so that a mega-cap
  like AAPL and a micro-cap like an obscure biotech pay realistically
  different costs instead of both getting a hardcoded 5 bps half-spread.

The tier YAML is loaded once at first call and cached in a module-level
dict. Reload is available via :func:`_reset_tier_cache` (primarily for
tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CostModel:
    """Backtest-side cost model parameters.

    Attributes:
        commission_bps: Commission in basis points (1 bps = 0.01%)
        spread_w: Spread weight (multiplier for bid/ask spread)
        impact_w: Market impact weight (multiplier for price impact)
    """

    commission_bps: float
    spread_w: float
    impact_w: float


def get_default_cost_model() -> CostModel:
    """Backtest-engine default cost model.

    Used by legacy backtest paths that have not yet migrated to per-symbol
    tier lookup. Kept stable for 1e-9 regression compatibility.
    """
    return CostModel(commission_bps=1.0, spread_w=0.25, impact_w=0.5)


# ---------------------------------------------------------------------------
# Per-symbol cost tiers (E0.2)
# ---------------------------------------------------------------------------

_TIER_CACHE: dict[str, Any] | None = None
_TIER_YAML_PATH = Path("config/cost_tiers.yaml")

_FALLBACK_TIERS: dict[str, Any] = {
    "tiers": {
        "mega_cap": {
            "adv_min_usd": 100_000_000,
            "commission_bps": 0.2,
            "half_spread_bps": 1.0,
            "slippage_bps": 1.5,
        },
        "large_cap": {
            "adv_min_usd": 20_000_000,
            "commission_bps": 0.5,
            "half_spread_bps": 2.0,
            "slippage_bps": 2.5,
        },
        "mid_cap": {
            "adv_min_usd": 5_000_000,
            "commission_bps": 0.8,
            "half_spread_bps": 3.5,
            "slippage_bps": 5.0,
        },
        "small_cap": {
            "adv_min_usd": 1_000_000,
            "commission_bps": 1.0,
            "half_spread_bps": 5.0,
            "slippage_bps": 8.0,
        },
        "micro_cap": {
            "adv_min_usd": 0,
            "commission_bps": 1.5,
            "half_spread_bps": 8.0,
            "slippage_bps": 12.0,
        },
    },
    "default_tier": "mid_cap",
    "adv_window": 20,
}


def _load_tier_config() -> dict[str, Any]:
    global _TIER_CACHE
    if _TIER_CACHE is not None:
        return _TIER_CACHE

    try:
        import yaml  # type: ignore

        if _TIER_YAML_PATH.exists():
            data = yaml.safe_load(_TIER_YAML_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict) and data.get("tiers"):
                _TIER_CACHE = data
                return _TIER_CACHE
    except Exception:
        # yaml missing or file malformed — fall back silently.
        pass

    _TIER_CACHE = _FALLBACK_TIERS
    return _TIER_CACHE


def _reset_tier_cache() -> None:
    """Test hook: drop the cached tier config so the next call reloads it."""
    global _TIER_CACHE
    _TIER_CACHE = None


def get_tier_for_symbol(symbol: str, adv_usd: float | None) -> str:
    """Classify a symbol by its 20-day ADV in USD.

    If ``adv_usd`` is ``None`` or non-positive, the default tier from the
    YAML is returned. Otherwise the highest-liquidity tier whose
    ``adv_min_usd`` floor is satisfied is selected.

    ``symbol`` is currently unused but kept in the signature so per-symbol
    overrides (e.g. HTB list) can be added without a breaking API change.
    """
    cfg = _load_tier_config()
    tiers: dict[str, dict[str, Any]] = cfg["tiers"]
    default_tier = str(cfg.get("default_tier", "mid_cap"))

    if adv_usd is None or adv_usd <= 0:
        return default_tier

    best: tuple[float, str] | None = None
    for name, spec in tiers.items():
        floor = float(spec.get("adv_min_usd", 0))
        if adv_usd >= floor:
            if best is None or floor > best[0]:
                best = (floor, name)
    return best[1] if best is not None else default_tier


def get_tier_costs(tier_name: str) -> dict[str, float]:
    """Return commission_bps, half_spread_bps, slippage_bps for a tier.

    Unknown tier names fall back to ``default_tier`` from the YAML.
    """
    cfg = _load_tier_config()
    tiers: dict[str, dict[str, Any]] = cfg["tiers"]
    default_tier = str(cfg.get("default_tier", "mid_cap"))
    spec = tiers.get(tier_name) or tiers.get(default_tier) or {}
    return {
        "commission_bps": float(spec.get("commission_bps", 1.0)),
        "half_spread_bps": float(spec.get("half_spread_bps", 5.0)),
        "slippage_bps": float(spec.get("slippage_bps", 5.0)),
    }


def get_tier_costs_for_symbol(
    symbol: str, adv_usd: float | None
) -> tuple[str, dict[str, float]]:
    """Convenience: classify a symbol and return (tier_name, cost_dict)."""
    tier = get_tier_for_symbol(symbol, adv_usd)
    return tier, get_tier_costs(tier)
