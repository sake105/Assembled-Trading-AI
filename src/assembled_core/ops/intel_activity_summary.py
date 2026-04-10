"""OPS-13: Intel activity summary across experiment run days."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "run.intel_activity.v1"


def _safe_int(x: Any) -> int:
    try:
        return int(x) if x is not None else 0
    except (TypeError, ValueError):
        return 0


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _status_counts(
    status_list: list[str],
) -> tuple[int, int, int]:
    """Return (days_ok, days_degraded, days_error)."""
    ok = sum(1 for s in status_list if (s or "").upper() == "OK")
    degraded = sum(1 for s in status_list if (s or "").upper() == "DEGRADED")
    error = sum(1 for s in status_list if (s or "").upper() == "ERROR")
    return ok, degraded, error


def _max_severity_from_run_kpis(kpis: Dict[str, Any]) -> int | None:
    """Extract max trigger severity from run_kpis (triggers_summary or top_triggers). Fallback when per-run summaries missing."""
    max_sev: int | None = None
    summary = kpis.get("triggers_summary") or {}
    if isinstance(summary, dict):
        m = summary.get("max_severity")
        if m is not None:
            try:
                max_sev = int(m)
            except (TypeError, ValueError) as exc:
                logger.warning("[IntelActivitySummary] failed to parse max_severity: %s", exc)
    for t in kpis.get("top_triggers") or []:
        if not isinstance(t, dict):
            continue
        s = t.get("severity")
        if s is not None:
            try:
                si = int(s)
                if max_sev is None or si > max_sev:
                    max_sev = si
            except (TypeError, ValueError) as exc:
                logger.warning("[IntelActivitySummary] failed to parse trigger severity: %s", exc)
    return max_sev


def _news_has_triggers(kpis: Dict[str, Any]) -> bool:
    """True if run_kpis indicates any news triggers. Prefer news_triggers_summary (OPS-14), else triggers_summary/top_triggers."""
    per_run = kpis.get("news_triggers_summary")
    if isinstance(per_run, dict) and _safe_int(per_run.get("count")) > 0:
        return True
    return _has_triggers_run_kpis(kpis)


def _news_max_severity(kpis: Dict[str, Any]) -> int | None:
    """Max news trigger severity for this run. Prefer news_triggers_summary (OPS-14), else triggers_summary/top_triggers."""
    per_run = kpis.get("news_triggers_summary")
    if isinstance(per_run, dict):
        m = per_run.get("max_severity")
        if m is not None:
            try:
                return int(m)
            except (TypeError, ValueError) as exc:
                logger.warning("[IntelActivitySummary] failed to parse news max_severity: %s", exc)
    return _max_severity_from_run_kpis(kpis)


def _disclosures_has_triggers(kpis: Dict[str, Any]) -> bool:
    """True if run_kpis indicates any disclosures triggers (OPS-14 per-run summary)."""
    per_run = kpis.get("disclosures_triggers_summary")
    if isinstance(per_run, dict) and _safe_int(per_run.get("count")) > 0:
        return True
    return False


def _disclosures_max_severity(kpis: Dict[str, Any]) -> int | None:
    """Max disclosures trigger severity for this run (OPS-14 per-run summary)."""
    per_run = kpis.get("disclosures_triggers_summary")
    if isinstance(per_run, dict):
        m = per_run.get("max_severity")
        if m is not None:
            try:
                return int(m)
            except (TypeError, ValueError) as exc:
                logger.warning("[IntelActivitySummary] failed to parse disclosures max_severity: %s", exc)
    return None


def _has_triggers_run_kpis(kpis: Dict[str, Any]) -> bool:
    """True if run_kpis indicates any triggers (news layer). Fallback when per-run summary missing."""
    top = kpis.get("top_triggers") or []
    if top:
        return True
    summary = kpis.get("triggers_summary") or {}
    if isinstance(summary, dict):
        n = summary.get("n_triggers") or summary.get("count")
        if n is not None and _safe_int(n) > 0:
            return True
    return False


def build_intel_activity_summary(
    runs_root: Path,
    *,
    intel_output_root: Path | None = None,
) -> Dict[str, Any]:
    """Build intel activity summary from run_kpis.json in each date subdir under runs_root.

    Optionally pass intel_output_root (e.g. repo_root / "output" / "intel") to read
    triggers_latest.json once for current snapshot; per-day trigger stats come from run_kpis.
    """
    runs_root = Path(runs_root)
    date_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    n_days = len(date_dirs)

    news_statuses: list[str] = []
    discl_statuses: list[str] = []
    news_days_with_triggers = 0
    news_max_severity_seen: int | None = None
    news_geo_scores: list[float | None] = []
    discl_days_with_triggers = 0
    discl_max_severity_seen: int | None = None
    risk_state_counts: Dict[str, int] = {
        "WATCH": 0,
        "ACTIVE": 0,
        "COOLDOWN": 0,
        "PAUSE": 0,
    }
    # NEWS-DEBUG-1: funnel aggregates (when run_kpis contains news_debug_funnel)
    total_candidate_triggers = 0
    total_triggers = 0
    total_triggers_sev1plus = 0
    total_triggers_evidence_blocked = 0
    total_triggers_qc_capped = 0
    max_clusters_count_seen: int = 0

    for day_dir in date_dirs:
        kpis_path = day_dir / "run_kpis.json"
        if not kpis_path.exists():
            continue
        try:
            kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(kpis, dict):
            continue

        # Intel orchestration status
        io = kpis.get("intel_orchestration") or {}
        if isinstance(io, dict):
            news_statuses.append(
                (io.get("news") or {}).get("status")
                if isinstance(io.get("news"), dict)
                else None
            )
            d = io.get("disclosures")
            discl_statuses.append(
                (d or {}).get("status") if isinstance(d, dict) else None
            )

        # Risk state
        rs = kpis.get("risk_state")
        if rs is not None and isinstance(rs, dict):
            state = (rs.get("state") or "").strip().upper() or "WATCH"
            risk_state_counts[state] = risk_state_counts.get(state, 0) + 1

        # News geo_score
        ng = kpis.get("news_geo")
        if ng is not None and isinstance(ng, dict):
            news_geo_scores.append(_safe_float(ng.get("geo_score")))
        else:
            news_geo_scores.append(None)

        # News triggers from run_kpis (OPS-14: prefer news_triggers_summary)
        if _news_has_triggers(kpis):
            news_days_with_triggers += 1
        m = _news_max_severity(kpis)
        if m is not None:
            if news_max_severity_seen is None or m > news_max_severity_seen:
                news_max_severity_seen = m

        # Disclosures triggers from run_kpis (OPS-14: per-run disclosures_triggers_summary)
        if _disclosures_has_triggers(kpis):
            discl_days_with_triggers += 1
        dm = _disclosures_max_severity(kpis)
        if dm is not None:
            if discl_max_severity_seen is None or dm > discl_max_severity_seen:
                discl_max_severity_seen = dm

        # NEWS-DEBUG-1: aggregate funnel metrics from news_debug_funnel
        funnel = kpis.get("news_debug_funnel")
        if isinstance(funnel, dict):
            total_candidate_triggers += _safe_int(
                funnel.get("candidate_triggers_count")
            )
            total_triggers += _safe_int(funnel.get("triggers_count"))
            total_triggers_sev1plus += _safe_int(
                funnel.get("triggers_severity_ge_1_count")
            )
            total_triggers_evidence_blocked += _safe_int(
                funnel.get("triggers_evidence_blocked_count")
            )
            total_triggers_qc_capped += _safe_int(
                funnel.get("triggers_qc_capped_count")
            )
            cc = _safe_int(funnel.get("clusters_count"))
            if cc > max_clusters_count_seen:
                max_clusters_count_seen = cc

    # Optional: read global triggers files once for snapshot (e.g. last run state)
    if intel_output_root is not None:
        intel_root = Path(intel_output_root)
        for kind in ["news", "disclosures"]:
            path = intel_root / kind / "triggers_latest.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    items = data.get("triggers") or data.get("items") or []
                    if (
                        isinstance(items, list)
                        and items
                        and kind == "news"
                        and news_max_severity_seen is None
                    ):
                        for t in items:
                            if isinstance(t, dict):
                                s = t.get("severity")
                                if s is not None:
                                    try:
                                        si = int(s)
                                        if (
                                            news_max_severity_seen is None
                                            or si > news_max_severity_seen
                                        ):
                                            news_max_severity_seen = si
                                    except (TypeError, ValueError) as exc:
                                        logger.warning("[IntelActivitySummary] bad news trigger severity value: %s", exc)
                    if (
                        kind == "disclosures"
                        and discl_max_severity_seen is None
                        and items
                    ):
                        for t in items:
                            if isinstance(t, dict):
                                s = t.get("severity")
                                if s is not None:
                                    try:
                                        si = int(s)
                                        if (
                                            discl_max_severity_seen is None
                                            or si > discl_max_severity_seen
                                        ):
                                            discl_max_severity_seen = si
                                    except (TypeError, ValueError) as exc:
                                        logger.warning("[IntelActivitySummary] bad disclosures trigger severity value: %s", exc)
                except Exception as exc:
                    logger.warning("[IntelActivitySummary] failed to parse intel artifact for day: %s", exc)

    days_ok_n, days_degraded_n, days_error_n = _status_counts(news_statuses)
    days_ok_d, days_degraded_d, days_error_d = _status_counts(discl_statuses)

    days_geo_ge_1 = sum(1 for g in news_geo_scores if g is not None and g >= 1)
    days_geo_ge_2 = sum(1 for g in news_geo_scores if g is not None and g >= 2)
    days_geo_ge_3 = sum(1 for g in news_geo_scores if g is not None and g >= 3)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_days": n_days,
        "news": {
            "days_ok": days_ok_n,
            "days_degraded": days_degraded_n,
            "days_error": days_error_n,
            "days_with_triggers": news_days_with_triggers,
            "max_trigger_severity_seen": news_max_severity_seen,
            "days_geo_score_ge_1": days_geo_ge_1,
            "days_geo_score_ge_2": days_geo_ge_2,
            "days_geo_score_ge_3": days_geo_ge_3,
            "news_funnel": {
                "total_candidate_triggers": total_candidate_triggers,
                "total_triggers": total_triggers,
                "total_triggers_sev1plus": total_triggers_sev1plus,
                "total_triggers_evidence_blocked": total_triggers_evidence_blocked,
                "total_triggers_qc_capped": total_triggers_qc_capped,
                "max_clusters_count_seen": max_clusters_count_seen,
            },
        },
        "disclosures": {
            "days_ok": days_ok_d,
            "days_degraded": days_degraded_d,
            "days_error": days_error_d,
            "days_with_triggers": discl_days_with_triggers,
            "max_trigger_severity_seen": discl_max_severity_seen,
        },
        "risk_state": dict(risk_state_counts),
    }


__all__ = ["build_intel_activity_summary", "SCHEMA_VERSION"]
