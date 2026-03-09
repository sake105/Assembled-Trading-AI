"""Risk state machine: WATCH / ACTIVE / COOLDOWN / PAUSE.

Persisted state in output/state/risk_state.json (atomic write).
Transitions are deterministic from ctx.news_geo + policy with hysteresis/cooldown.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal

RiskState = Literal["WATCH", "ACTIVE", "COOLDOWN", "PAUSE"]

VERSION = "risk_state.v1"


@dataclass
class RiskStateRecord:
    """Persisted risk state record."""

    state: RiskState
    since_utc: str
    last_transition_utc: str
    reason: str
    geo_score: int
    geo_confidence: float
    version: str = VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hours_since(iso_utc: str, now_utc: str) -> float:
    """Return hours between iso_utc and now_utc (positive if now_utc later)."""
    try:
        t0 = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
        delta = t1 - t0
        return delta.total_seconds() / 3600.0
    except Exception:
        return 0.0


def _default_record(now_utc: str) -> RiskStateRecord:
    return RiskStateRecord(
        state="WATCH",
        since_utc=now_utc,
        last_transition_utc=now_utc,
        reason="default",
        geo_score=0,
        geo_confidence=0.0,
        version=VERSION,
    )


def atomic_write_json_with_retry(
    path: Path,
    data: Dict[str, Any],
    retries: int = 5,
    backoff_ms: int = 50,
) -> None:
    """Write JSON to path atomically (tmp + replace). Retry on PermissionError with exponential backoff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_path), str(path))
            return
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff_ms * (2**attempt) / 1000.0)
    if last_err is not None:
        raise last_err


def load_risk_state(path: str | Path) -> RiskStateRecord:
    """Load risk state from JSON file. Tolerant: missing/corrupt -> WATCH now."""
    p = Path(path)
    if not p.exists():
        return _default_record(_now_utc_str())
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _default_record(_now_utc_str())
    if not isinstance(data, dict):
        return _default_record(_now_utc_str())
    state = data.get("state", "WATCH")
    if state not in ("WATCH", "ACTIVE", "COOLDOWN", "PAUSE"):
        state = "WATCH"
    now = _now_utc_str()
    return RiskStateRecord(
        state=state,
        since_utc=str(data.get("since_utc", now)),
        last_transition_utc=str(data.get("last_transition_utc", now)),
        reason=str(data.get("reason", "loaded")),
        geo_score=int(data.get("geo_score", 0)),
        geo_confidence=float(data.get("geo_confidence", 0.0)),
        version=str(data.get("version", VERSION)),
    )


def save_risk_state(
    record: RiskStateRecord,
    path: str | Path,
    policy: Dict[str, Any] | None = None,
) -> None:
    """Write risk state to JSON file atomically (tmp + replace). Optional lock+retry from policy."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = record.to_dict()
    rsm = (policy or {}).get("risk_state_machine") or {}
    persistence = rsm.get("persistence") or {}
    lock_cfg = persistence.get("lock") or {}
    use_lock = lock_cfg.get("enabled", False)
    retries = int(lock_cfg.get("retries", 5) or 5)
    backoff_ms = int(lock_cfg.get("backoff_ms", 50) or 50)

    if use_lock and retries > 0:
        atomic_write_json_with_retry(p, data, retries=retries, backoff_ms=backoff_ms)
        return
    # No lock: single write (original behavior)
    tmp_path = p.parent / (p.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp_path), str(p))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle import TradingContext


def _effective_geo(ctx: "TradingContext", policy: Dict[str, Any]) -> tuple[int, float]:
    """Return (geo_score, geo_confidence). Intel degraded or missing -> (0, 0)."""
    rsm = (policy or {}).get("risk_state_machine") or {}
    conf_floor = float((rsm.get("hysteresis") or {}).get("confidence_floor", 0.60) or 0.60)

    intel_flags = getattr(ctx, "intel_health_flags", None) or {}
    if intel_flags.get("intel_geo_score") == "DEGRADED" or intel_flags.get("intel_news_triggers") == "DEGRADED":
        return 0, 0.0

    news_geo = getattr(ctx, "news_geo", None)
    if not news_geo:
        return 0, 0.0

    score = int(news_geo.get("geo_score", 0))
    conf = float(news_geo.get("geo_confidence", 0.0))
    if conf < conf_floor:
        return 0, conf

    return score, conf


def compute_next_state(
    ctx: "TradingContext",
    policy: Dict[str, Any],
    now_utc: str,
    prev: RiskStateRecord,
) -> RiskStateRecord:
    """Compute next risk state from context, policy and previous record. Deterministic."""
    rsm = (policy or {}).get("risk_state_machine") or {}
    if not rsm.get("enabled", True):
        return prev

    hyst = rsm.get("hysteresis") or {}
    activate_score = int(hyst.get("activate_score", 2))
    deactivate_score = int(hyst.get("deactivate_score", 1))
    pause_score = int(hyst.get("pause_score", 3))
    confidence_floor = float(hyst.get("confidence_floor", 0.60) or 0.60)
    require_market_stress_confirm = bool(hyst.get("require_market_stress_confirm", False))
    require_disclosures_confirm = bool(hyst.get("require_disclosures_confirm", False))
    disclosures_min_severity = int(hyst.get("disclosures_min_severity", 1))

    if require_market_stress_confirm:
        market_stress = getattr(ctx, "market_stress", None) or {}
        stress_ok = bool(market_stress.get("stress_ok", False))
    else:
        stress_ok = True

    score, conf = _effective_geo(ctx, policy)
    if conf < confidence_floor:
        score = 0
    geo_score_effective = score

    gate_cfg = hyst.get("disclosures_confirm_gate") or {}
    gate_enabled = bool(gate_cfg.get("enabled", False))
    gate_min_geo = int(gate_cfg.get("min_geo_score", 3))
    gate_apply_states = set(gate_cfg.get("on_states", ["WATCH", "COOLDOWN"]))
    require_confirm_now = (
        require_disclosures_confirm
        or (gate_enabled and prev.state in gate_apply_states and geo_score_effective >= gate_min_geo)
    )

    # Disclosures confirm gate for WATCH/COOLDOWN -> ACTIVE (when require_confirm_now)
    disclosures_confirmed = True
    if require_confirm_now:
        intel_flags_rsm = getattr(ctx, "intel_health_flags", None) or {}
        if intel_flags_rsm.get("intel_disclosures_triggers") == "DEGRADED":
            disclosures_confirmed = False
        else:
            disc_triggers = getattr(ctx, "disclosures_triggers", None)
            if disc_triggers is None:
                disclosures_confirmed = False
            else:
                max_sev = 0
                if hasattr(disc_triggers, "summary") and isinstance(getattr(disc_triggers, "summary"), dict):
                    max_sev = int((disc_triggers.summary or {}).get("max_severity", 0))
                if max_sev == 0 and hasattr(disc_triggers, "triggers"):
                    for t in (getattr(disc_triggers, "triggers") or []):
                        if isinstance(t, dict):
                            max_sev = max(max_sev, int(t.get("severity", 0)))
                disclosures_confirmed = max_sev >= disclosures_min_severity

    cooldown_cfg = rsm.get("cooldown") or {}
    cooldown_hours = float(cooldown_cfg.get("hours", 24) or 24)

    pause_cfg = rsm.get("pause") or {}
    pause_enabled = bool(pause_cfg.get("enabled", False))
    pause_require_conf = float(pause_cfg.get("require_confidence", 0.80) or 0.80)
    pause_hours = float(pause_cfg.get("hours", 12) or 12)

    # score, conf, geo_score_effective already set above
    if pause_enabled and score >= pause_score and conf >= pause_require_conf:
        if prev.state == "PAUSE":
            hours_since = _hours_since(prev.last_transition_utc, now_utc)
            if hours_since < pause_hours:
                return RiskStateRecord(
                    state="PAUSE",
                    since_utc=prev.since_utc,
                    last_transition_utc=prev.last_transition_utc,
                    reason="pause_duration",
                    geo_score=score,
                    geo_confidence=conf,
                    version=VERSION,
                )
        return RiskStateRecord(
            state="PAUSE",
            since_utc=now_utc,
            last_transition_utc=now_utc,
            reason="pause_score",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )

    # PAUSE -> COOLDOWN when min duration elapsed
    if prev.state == "PAUSE":
        hours_since = _hours_since(prev.last_transition_utc, now_utc)
        if hours_since < pause_hours:
            return RiskStateRecord(
                state="PAUSE",
                since_utc=prev.since_utc,
                last_transition_utc=prev.last_transition_utc,
                reason="pause_duration",
                geo_score=score,
                geo_confidence=conf,
                version=VERSION,
            )
        return RiskStateRecord(
            state="COOLDOWN",
            since_utc=now_utc,
            last_transition_utc=now_utc,
            reason="pause_to_cooldown",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )

    if prev.state == "WATCH":
        if score >= activate_score and stress_ok and (not require_confirm_now or disclosures_confirmed):
            return RiskStateRecord(
                state="ACTIVE",
                since_utc=now_utc,
                last_transition_utc=now_utc,
                reason="activate_score",
                geo_score=score,
                geo_confidence=conf,
                version=VERSION,
            )
        if score >= activate_score and stress_ok and require_confirm_now and not disclosures_confirmed:
            return RiskStateRecord(
                state="WATCH",
                since_utc=prev.since_utc,
                last_transition_utc=prev.last_transition_utc,
                reason="disclosures_confirm",
                geo_score=score,
                geo_confidence=conf,
                version=VERSION,
            )
        if score >= activate_score and not stress_ok:
            return RiskStateRecord(
                state="WATCH",
                since_utc=prev.since_utc,
                last_transition_utc=prev.last_transition_utc,
                reason="stress_confirm",
                geo_score=score,
                geo_confidence=conf,
                version=VERSION,
            )
        return RiskStateRecord(
            state="WATCH",
            since_utc=prev.since_utc,
            last_transition_utc=prev.last_transition_utc,
            reason="watch_hold",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )

    if prev.state == "ACTIVE":
        if score <= deactivate_score:
            return RiskStateRecord(
                state="COOLDOWN",
                since_utc=now_utc,
                last_transition_utc=now_utc,
                reason="deactivate_score",
                geo_score=score,
                geo_confidence=conf,
                version=VERSION,
            )
        return RiskStateRecord(
            state="ACTIVE",
            since_utc=prev.since_utc,
            last_transition_utc=prev.last_transition_utc,
            reason="active_hold",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )

    # prev.state == "COOLDOWN"
    hours_since = _hours_since(prev.last_transition_utc, now_utc)
    # Sim/backtest: if now_utc is before last_transition (e.g. wall-clock state vs sim time), treat cooldown as expired
    if hours_since < 0:
        hours_since = cooldown_hours + 1.0
    if hours_since < cooldown_hours:
        return RiskStateRecord(
            state="COOLDOWN",
            since_utc=prev.since_utc,
            last_transition_utc=prev.last_transition_utc,
            reason="cooldown_timer",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )
    if score >= activate_score and stress_ok and (not require_confirm_now or disclosures_confirmed):
        return RiskStateRecord(
            state="ACTIVE",
            since_utc=now_utc,
            last_transition_utc=now_utc,
            reason="cooldown_to_active",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )
    if score >= activate_score and stress_ok and require_confirm_now and not disclosures_confirmed:
        return RiskStateRecord(
            state="WATCH",
            since_utc=now_utc,
            last_transition_utc=now_utc,
            reason="disclosures_confirm",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )
    if score >= activate_score and not stress_ok:
        return RiskStateRecord(
            state="WATCH",
            since_utc=now_utc,
            last_transition_utc=now_utc,
            reason="stress_confirm",
            geo_score=score,
            geo_confidence=conf,
            version=VERSION,
        )
    return RiskStateRecord(
        state="WATCH",
        since_utc=now_utc,
        last_transition_utc=now_utc,
        reason="cooldown_to_watch",
        geo_score=score,
        geo_confidence=conf,
        version=VERSION,
    )


__all__ = [
    "RiskState",
    "RiskStateRecord",
    "atomic_write_json_with_retry",
    "load_risk_state",
    "save_risk_state",
    "compute_next_state",
]
