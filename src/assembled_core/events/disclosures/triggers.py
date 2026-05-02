"""Disclosure trigger scoring v1: events -> triggers with severity, confidence, TTL, decay."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List

from .evidence import summarize_evidence
from .models import DisclosureEvent


def _parse_utc(s: str) -> datetime | None:
    """Parse ISO UTC string to datetime. Returns None on failure."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _age_hours(published_utc: str, now_utc: str) -> float:
    """Hours between published_utc and now_utc. Positive if published in the past."""
    pub = _parse_utc(published_utc)
    now = _parse_utc(now_utc)
    if pub is None or now is None:
        return 0.0
    delta = now - pub
    return max(0.0, delta.total_seconds() / 3600.0)


def score_disclosure_triggers(
    events: List[DisclosureEvent],
    source_meta: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    now_utc: str,
) -> List[Dict[str, Any]]:
    """Score disclosure events into trigger dicts (severity 0..3, confidence 0..1, TTL, decay)."""
    if not events:
        return []

    severity_cfg = cfg.get("severity") or {}
    base_by_action = severity_cfg.get("base_by_action") or {}
    max_sev = int(severity_cfg.get("max") or 3)
    confidence_cfg = cfg.get("confidence") or {}
    tier_a_alone = float(confidence_cfg.get("tierA_alone", 0.85))
    tier_b_two = float(confidence_cfg.get("tierB_two_domains", 0.70))
    otherwise = float(confidence_cfg.get("otherwise", 0.40))
    gating = cfg.get("gating") or {}
    require_evidence_ok = bool(gating.get("require_evidence_ok", True))
    ttl_cfg = cfg.get("ttl") or {}
    default_ttl = int(ttl_cfg.get("default_hours") or 168)
    by_action_ttl = ttl_cfg.get("by_action") or {}
    decay_cfg = cfg.get("decay") or {}
    half_life = float(decay_cfg.get("half_life_hours", 72))
    min_conf_floor = float(decay_cfg.get("min_confidence_floor", 0.25))
    sev_floor = int(decay_cfg.get("severity_floor") or 0)

    triggers: List[Dict[str, Any]] = []

    for ev in events:
        # Single-event evidence
        evidence = summarize_evidence([ev], source_meta)
        evidence_ok = evidence.get("evidence_ok", False)

        action_type = ev.action_type or ""
        base_sev = int(base_by_action.get(action_type, 1))
        base_sev = min(max_sev, max(0, base_sev))

        # Confidence by tier (single event: A => tierA_alone, B => otherwise)
        if evidence.get("tierA_count", 0) >= 1:
            conf = tier_a_alone
        elif evidence.get("tierB_independent_domains", 0) >= 2:
            conf = tier_b_two
        else:
            conf = otherwise

        # Gating: require_evidence_ok and not evidence_ok -> severity=0, confidence=otherwise
        if require_evidence_ok and not evidence_ok:
            base_sev = 0
            conf = otherwise

        ttl_hours = int(by_action_ttl.get(action_type) or default_ttl)
        age_h = _age_hours(ev.published_utc or now_utc, now_utc)

        if age_h >= ttl_hours:
            # Expired: severity=0, confidence decay
            sev = 0
            conf = max(min_conf_floor, conf * 0.5)
            decay_factor = 0.5
        else:
            decay_factor = 0.5 ** (age_h / half_life) if half_life > 0 else 1.0
            sev = max(sev_floor, int(round(base_sev * decay_factor)))
            sev = min(max_sev, sev)
            conf = max(min_conf_floor, conf * decay_factor)

        trigger_id = (
            "dtr_"
            + hashlib.sha256(
                (ev.event_id + "|" + action_type).encode("utf-8")
            ).hexdigest()[:12]
        )

        trigger: Dict[str, Any] = {
            "trigger_id": trigger_id,
            "schema_version": "disclosures.trigger.v1",
            "generated_utc": now_utc,
            "event_id": ev.event_id,
            "source_id": ev.source_id,
            "action_type": action_type,
            "person_or_entity": ev.person_or_entity or "",
            "ticker": ev.ticker,
            "severity": sev,
            "confidence": round(conf, 4),
            "evidence_ok": evidence_ok,
            "evidence": {
                "tierA_count": evidence.get("tierA_count", 0),
                "tierB_count": evidence.get("tierB_count", 0),
                "tierB_independent_domains": evidence.get(
                    "tierB_independent_domains", 0
                ),
            },
            "ttl_hours": ttl_hours,
            "decay": {
                "age_hours": round(age_h, 2),
                "factor": round(decay_factor, 4),
                "half_life_hours": half_life,
            },
        }
        triggers.append(trigger)

    triggers.sort(
        key=lambda t: (
            -t["severity"],
            -t["confidence"],
            t.get("action_type", ""),
            t.get("event_id", ""),
        )
    )
    return triggers


def apply_qc_caps(
    triggers: List[Dict[str, Any]],
    health_status: str,
    qc_gates: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Cap trigger severity by health status (DEGRADED => degraded_max_severity, ERROR => error_max_severity)."""
    status = (health_status or "").strip().upper()
    max_sev_degraded = int(qc_gates.get("degraded_max_severity") or 1)
    max_sev_error = int(qc_gates.get("error_max_severity") or 0)
    if status == "ERROR":
        cap = max_sev_error
    elif status == "DEGRADED":
        cap = max_sev_degraded
    else:
        return list(triggers)
    out = []
    for t in triggers:
        t = dict(t)
        t["severity"] = min(t.get("severity", 0), cap)
        out.append(t)
    return out
