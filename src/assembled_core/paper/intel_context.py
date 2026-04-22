"""Part B wiring: populate TradingContext intel attrs from artifacts.

The trading_cycle intel signal_layer + bayesian_confidence paths read optional
ctx attributes (intel_active_shocks, intel_sector_impacts, signal_historical_scores).
Paper_runner previously never set these, so the flipped flags silent-skipped.

This helper wires two conservative paths:

1. ``ctx.intel_active_shocks`` — built from news trigger topic_ids via a
   curated topic_id → shock_type map. Drives the shock beneficiary path in
   trading_cycle (guarded by policy.intel.signal_layer.allow_short=false +
   INVERSE_ETF_BLACKLIST in intel_signal_adapter.py).

2. ``ctx.signal_historical_scores`` — left as None on purpose. The Bayesian
   confidence module falls back to current cross-section for prior estimation
   when historical_scores is None. Wiring a true history requires persisting
   per-run signal scores; tracked as follow-up.

Not wired (remains X2 PARK — see docs/intel/T4.5-signal_layer-investigation.md):
- intel_sector_impacts
- intel_supply_vulnerability
- intel_sanctions_beneficiary
- intel_chokepoint_exposure
- intel_confidence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# news trigger topic_id → list of ShockType keys used by SHOCK_BENEFICIARY_MAP
# in intel_signal_adapter. Curated, high-confidence only.
TOPIC_TO_SHOCKS: dict[str, list[str]] = {
    "geopolitical_conflict": ["defense_demand_surge", "global_risk_off"],
    "sanctions_trade": ["global_risk_off", "inflation_spike"],
    "shipping_disruption": ["shipping_cost_risk", "oil_supply_risk"],
    "taiwan_strait": ["semiconductor_supply_risk", "defense_demand_surge", "global_risk_off"],
    "energy_crisis": ["oil_supply_risk", "energy_price_spike"],
    "market_crash": ["global_risk_off"],
    "central_bank": ["rate_shock"],
    "nuclear_risk": ["nuclear_escalation_risk", "global_risk_off"],
}

# Minimum trigger severity to count as an active shock (1 = WATCH, 2 = ACTIVE)
MIN_SHOCK_SEVERITY = 2


def active_shocks_from_triggers(
    items: list[dict[str, Any]],
    *,
    min_severity: int = MIN_SHOCK_SEVERITY,
) -> list[str]:
    """Extract active shock types from a news triggers list.

    Args:
        items: Raw list of trigger dicts (triggers_latest.json -> items).
        min_severity: Skip triggers below this severity.

    Returns:
        De-duplicated list of shock types in SHOCK_BENEFICIARY_MAP form.
    """
    if not items:
        return []

    shocks: set[str] = set()
    for t in items:
        try:
            sev = int(t.get("severity", 0))
        except (TypeError, ValueError):
            continue
        if sev < min_severity:
            continue
        topic_id = str(t.get("topic_id", "")).strip().lower()
        if not topic_id:
            continue
        mapped = TOPIC_TO_SHOCKS.get(topic_id)
        if mapped:
            shocks.update(mapped)

    return sorted(shocks)


def populate_ctx_from_artifacts(
    ctx: Any,
    root: Path,
    *,
    news_triggers_path: str | None = None,
) -> None:
    """Populate ctx intel attributes from on-disk artifacts.

    Silently degrades (no-op) when artifacts are missing or unparseable —
    the downstream code uses ``getattr(ctx, ..., None)`` with defensive
    defaults, so an empty ctx is the safe fallback.
    """
    # Default path unless caller overrides
    triggers_path = (
        Path(news_triggers_path)
        if news_triggers_path
        else root / "output" / "intel" / "news" / "triggers_latest.json"
    )

    active_shocks: list[str] = []
    try:
        if triggers_path.exists():
            data = json.loads(triggers_path.read_text(encoding="utf-8"))
            items = data.get("items", []) if isinstance(data, dict) else []
            active_shocks = active_shocks_from_triggers(items)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "[INTEL-CTX] failed to load news triggers from %s: %s",
            triggers_path,
            exc,
        )

    # Only set when non-empty so downstream `if active_shocks:` gates stay clean
    if active_shocks:
        ctx.intel_active_shocks = active_shocks
        log.info(
            "[INTEL-CTX] populated intel_active_shocks (%d): %s",
            len(active_shocks),
            active_shocks,
        )
    else:
        log.debug("[INTEL-CTX] no active shocks from triggers (empty or low severity)")

    # Leave historical_scores as None — Bayesian fallback handles this.
    # Explicitly leaving the attribute unset keeps semantics unchanged.


__all__ = [
    "TOPIC_TO_SHOCKS",
    "MIN_SHOCK_SEVERITY",
    "active_shocks_from_triggers",
    "populate_ctx_from_artifacts",
]
