"""BENCH-1/BENCH-2: Intel simulation harness for policy A/B — deterministic news_geo + disclosures + geo_spikes."""

from __future__ import annotations

from typing import Any

from src.assembled_core.intel.disclosures_triggers_loader import DisclosuresTriggerSnapshot


def apply_intel_sim(
    ctx: Any,
    day_index: int,
    cfg: dict[str, Any],
) -> None:
    """Inject deterministic intel into ctx for paper/experiment runs. Call before run_trading_cycle.

    - stress_ok True -> news_geo ACTIVE (geo_score=2); else WATCH (geo_score=1).
    - If geo_spikes.enabled and day_index % every_n_days == 0: override to geo_score=spike_score (default 3).
    - Every disclosures_confirm_every_n_days: disclosures_triggers snapshot with max_severity=1; else None.
    - market_stress.stress_ok set from same rule; intel_health_flags cleared.
    - Sets ctx.intel_sim_applied = True so trading_cycle skips real intel loading.
    """
    mode = (cfg.get("mode") or "stress_based").strip().lower()
    n_days = int(cfg.get("disclosures_confirm_every_n_days") or 5)
    if n_days <= 0:
        n_days = 5

    # Geo spikes (BENCH-2): optional score=3 on schedule
    spikes = cfg.get("geo_spikes") or {}
    spikes_enabled = bool(spikes.get("enabled", False))
    spike_n = int(spikes.get("every_n_days", 7) or 7)
    spike_score = int(spikes.get("geo_score", 3) or 3)
    spike_conf = float(spikes.get("geo_confidence", 0.85) or 0.85)

    # Base stress_ok
    if mode == "stress_based":
        stress_ok = day_index % 2 == 0
    else:
        stress_ok = True

    # Geo score/conf: spike day overrides base
    if spikes_enabled and spike_n > 0 and (day_index % spike_n == 0):
        geo_score = spike_score
        geo_conf = spike_conf
        state_hint = "ACTIVE"
    else:
        if stress_ok:
            geo_score = 2
            geo_conf = 0.8
            state_hint = "ACTIVE"
        else:
            geo_score = 1
            geo_conf = 0.7
            state_hint = "WATCH"

    ctx.news_geo = {
        "geo_score": geo_score,
        "geo_confidence": geo_conf,
        "state_hint": state_hint,
        "top_triggers": [],
    }

    if day_index % n_days == 0:
        ctx.disclosures_triggers = DisclosuresTriggerSnapshot(
            generated_utc="intel_sim",
            triggers=[],
            summary={"max_severity": 1, "count_sev1plus": 1, "count_sev2plus": 0},
        )
    else:
        ctx.disclosures_triggers = None

    ctx.market_stress = {"stress_ok": stress_ok}
    ctx.intel_health_flags = {}
    ctx.intel_sim_applied = True


__all__ = ["apply_intel_sim"]
