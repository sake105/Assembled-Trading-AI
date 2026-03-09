"""OPS-3: Alerts/Anomalies v1 — deterministic alerts from run_kpis, reasons, diff artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ALERT_KINDS = (
    "STATE_CHANGE",
    "UNDERINVESTED",
    "TURNOVER_SPIKE",
    "TURNOVER_GATE",
    "HIGH_TRIGGERS",
    "NO_PREV",
    "QC_DEGRADED",
    "RECONCILE_FAIL",
)


def _severity_value(level: str, severity_map: dict[str, int]) -> int:
    return severity_map.get(level, 0)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _state_from_risk(risk: Any) -> str | None:
    if risk is None:
        return None
    if isinstance(risk, dict):
        return risk.get("state")
    return getattr(risk, "state", None)


def _make_alert_id(kind: str, message: str, generated_utc: str) -> str:
    raw = f"{kind}|{message}|{generated_utc}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def make_reconcile_fail_alert(generated_utc: str) -> dict[str, Any]:
    """Build a CRITICAL RECONCILE_FAIL alert (OPS-5)."""
    msg = "Reconcile invariants failed; see reconcile_latest.json."
    return {
        "alert_id": _make_alert_id("RECONCILE_FAIL", msg, generated_utc),
        "level": "critical",
        "kind": "RECONCILE_FAIL",
        "message": msg,
        "details": {},
    }


def compute_alerts(
    run_kpis: dict[str, Any],
    reasons: dict[str, Any],
    diff: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute alerts from run_kpis, reasons, and diff artifacts (v1 rules). Deterministic ordering."""
    alerts: list[dict[str, Any]] = []
    alerts_cfg = cfg.get("alerts") or {}
    if not alerts_cfg.get("enabled", True):
        return alerts

    thresholds = alerts_cfg.get("thresholds") or {}
    final_multiplier_drop = _safe_float(thresholds.get("final_multiplier_drop")) or 0.20
    abs_delta_weight_sum_thr = _safe_float(thresholds.get("abs_delta_weight_sum")) or 0.20
    turnover_scale_factor_below = _safe_float(thresholds.get("turnover_scale_factor_below")) or 0.70
    max_severity_ge = int(thresholds.get("max_severity_ge", 2))

    severity_map = alerts_cfg.get("severity_map") or {"info": 0, "warn": 1, "critical": 2}
    generated_utc = run_kpis.get("generated_utc") or diff.get("generated_utc") or ""

    # 1) NO_PREV
    notes = diff.get("notes") or []
    if "no_prev_run_found" in notes:
        msg = "No previous run found; diff baseline unavailable."
        alerts.append({
            "alert_id": _make_alert_id("NO_PREV", msg, generated_utc),
            "level": "info",
            "kind": "NO_PREV",
            "message": msg,
            "details": {"notes": notes},
        })

    # 2) QC_DEGRADED
    qc_flags = reasons.get("qc_flags")
    if qc_flags is not None and (
        (isinstance(qc_flags, list) and len(qc_flags) > 0)
        or (isinstance(qc_flags, dict) and len(qc_flags) > 0)
    ):
        msg = "QC/intel flags present; data quality may be degraded."
        alerts.append({
            "alert_id": _make_alert_id("QC_DEGRADED", msg, generated_utc),
            "level": "warn",
            "kind": "QC_DEGRADED",
            "message": msg,
            "details": {"qc_flags": qc_flags},
        })

    # 3) STATE_CHANGE
    delta_risk = diff.get("delta_risk_state")
    if delta_risk and isinstance(delta_risk, dict):
        prev_risk = delta_risk.get("prev")
        curr_risk = delta_risk.get("curr")
        prev_state = _state_from_risk(prev_risk)
        curr_state = _state_from_risk(curr_risk)
        if prev_state is not None and curr_state is not None and prev_state != curr_state:
            level = "critical" if curr_state == "PAUSE" else "warn"
            msg = f"Risk state changed from {prev_state} to {curr_state}."
            alerts.append({
                "alert_id": _make_alert_id("STATE_CHANGE", msg, generated_utc),
                "level": level,
                "kind": "STATE_CHANGE",
                "message": msg,
                "details": {"prev_state": prev_state, "curr_state": curr_state},
            })

    # 4) UNDERINVESTED
    delta_mult = diff.get("delta_multipliers") or {}
    fem = delta_mult.get("final_exposure_multiplier")
    if fem and isinstance(fem, dict):
        delta = _safe_float(fem.get("delta"))
        if delta is not None and delta <= -final_multiplier_drop:
            msg = f"Final exposure multiplier dropped by {abs(delta):.2f} (threshold {final_multiplier_drop})."
            alerts.append({
                "alert_id": _make_alert_id("UNDERINVESTED", msg, generated_utc),
                "level": "warn",
                "kind": "UNDERINVESTED",
                "message": msg,
                "details": {"delta": delta, "threshold": final_multiplier_drop},
            })

    # 5) TURNOVER_SPIKE
    summary = diff.get("summary") or {}
    abs_delta_sum = _safe_float(summary.get("abs_delta_weight_sum"))
    if abs_delta_sum is not None and abs_delta_sum >= abs_delta_weight_sum_thr:
        msg = f"Turnover proxy abs_delta_weight_sum={abs_delta_sum:.2f} >= {abs_delta_weight_sum_thr}."
        alerts.append({
            "alert_id": _make_alert_id("TURNOVER_SPIKE", msg, generated_utc),
            "level": "warn",
            "kind": "TURNOVER_SPIKE",
            "message": msg,
            "details": {"abs_delta_weight_sum": abs_delta_sum, "threshold": abs_delta_weight_sum_thr},
        })

    # 6) TURNOVER_GATE
    turnover_budget = run_kpis.get("turnover_budget") or {}
    scale_factor = _safe_float(turnover_budget.get("scale_factor"))
    behavior = turnover_budget.get("behavior")
    if scale_factor is not None and scale_factor < turnover_scale_factor_below:
        level = "critical" if behavior == "block" else "warn"
        msg = f"Turnover scale factor {scale_factor:.2f} below {turnover_scale_factor_below}."
        if behavior == "block":
            msg = "Turnover gate blocking; scale factor below threshold."
        alerts.append({
            "alert_id": _make_alert_id("TURNOVER_GATE", msg, generated_utc),
            "level": level,
            "kind": "TURNOVER_GATE",
            "message": msg,
            "details": {"scale_factor": scale_factor, "threshold": turnover_scale_factor_below, "behavior": behavior},
        })
    elif behavior == "block":
        msg = "Turnover gate behavior is block."
        alerts.append({
            "alert_id": _make_alert_id("TURNOVER_GATE", msg, generated_utc),
            "level": "critical",
            "kind": "TURNOVER_GATE",
            "message": msg,
            "details": {"behavior": behavior},
        })

    # 7) HIGH_TRIGGERS
    triggers_summary = run_kpis.get("triggers_summary") or {}
    max_sev = triggers_summary.get("max_severity")
    if max_sev is not None:
        try:
            ms = int(max_sev)
        except (TypeError, ValueError):
            ms = 0
        if ms >= max_severity_ge:
            level = "warn" if ms >= 2 else "info"
            msg = f"Triggers max_severity={ms} >= {max_severity_ge}."
            alerts.append({
                "alert_id": _make_alert_id("HIGH_TRIGGERS", msg, generated_utc),
                "level": level,
                "kind": "HIGH_TRIGGERS",
                "message": msg,
                "details": {"max_severity": ms, "threshold": max_severity_ge},
            })

    # Deterministic sort: level severity desc, kind, alert_id
    def sort_key(a: dict[str, Any]) -> tuple[int, str, str]:
        return (
            -_severity_value(a["level"], severity_map),
            a["kind"],
            a["alert_id"],
        )

    alerts.sort(key=sort_key)
    return alerts


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically via tmp file + rename."""
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=True)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(path)
    return path


def write_alerts_artifact(
    output_dir: str | Path,
    alerts: list[dict[str, Any]],
    generated_utc: str,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Write alerts_latest.json (schema run.alerts.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "alerts_latest.json"
    payload: dict[str, Any] = {
        "schema_version": "run.alerts.v1",
        "generated_utc": generated_utc,
        "count": len(alerts),
        "items": alerts,
    }
    return _atomic_write_json(path, payload)
