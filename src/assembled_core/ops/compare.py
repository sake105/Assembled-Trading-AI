"""OPS-7: Compare two paper experiment summaries (A/B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


SCHEMA_VERSION = "paper.compare.v1"


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _extract_key_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a paper summary for compare output."""
    alerts_by_level = data.get("alerts_count_by_level") or {}
    risk_state_pct = data.get("risk_state_pct") or {}
    reason_counts = data.get("risk_state_reason_counts") or {}
    return {
        "total_return": _safe_float(data.get("total_return")),
        "max_drawdown": _safe_float(data.get("max_drawdown")),
        "avg_final_multiplier": _safe_float(data.get("avg_final_multiplier")),
        "alerts_warn": int(alerts_by_level.get("warn", 0)),
        "alerts_critical": int(alerts_by_level.get("critical", 0)),
        "risk_state_transitions": int(data.get("risk_state_transitions", 0)),
        "active_pct": _safe_float(risk_state_pct.get("ACTIVE")),
        "disclosures_confirm_blocks": int(reason_counts.get("disclosures_confirm", 0)),
    }


def compare_summaries(path_a: str | Path, path_b: str | Path) -> Dict[str, Any]:
    """Compare two paper summary JSON files. Returns paper.compare.v1 dict with a, b, delta."""
    pa = Path(path_a)
    pb = Path(path_b)
    if not pa.exists():
        raise FileNotFoundError(f"Summary A not found: {pa}")
    if not pb.exists():
        raise FileNotFoundError(f"Summary B not found: {pb}")
    data_a = json.loads(pa.read_text(encoding="utf-8"))
    data_b = json.loads(pb.read_text(encoding="utf-8"))
    if not isinstance(data_a, dict):
        data_a = {}
    if not isinstance(data_b, dict):
        data_b = {}

    ka = _extract_key_metrics(data_a)
    kb = _extract_key_metrics(data_b)

    def _delta(va: Any, vb: Any) -> Any:
        if va is None and vb is None:
            return None
        fa = _safe_float(va)
        fb = _safe_float(vb)
        if fa is not None and fb is not None:
            return round(fb - fa, 6)
        if fb is not None:
            return float(fb)
        if fa is not None:
            return -float(fa)
        return None

    delta = {
        "total_return": _delta(ka.get("total_return"), kb.get("total_return")),
        "max_drawdown": _delta(ka.get("max_drawdown"), kb.get("max_drawdown")),
        "avg_final_multiplier": _delta(
            ka.get("avg_final_multiplier"), kb.get("avg_final_multiplier")
        ),
        "alerts_warn": (kb.get("alerts_warn") or 0) - (ka.get("alerts_warn") or 0),
        "alerts_critical": (kb.get("alerts_critical") or 0)
        - (ka.get("alerts_critical") or 0),
        "risk_state_transitions": (kb.get("risk_state_transitions") or 0)
        - (ka.get("risk_state_transitions") or 0),
        "active_pct": _delta(ka.get("active_pct"), kb.get("active_pct")),
        "disclosures_confirm_blocks": (kb.get("disclosures_confirm_blocks") or 0)
        - (ka.get("disclosures_confirm_blocks") or 0),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "a": ka,
        "b": kb,
        "delta": delta,
    }


__all__ = ["compare_summaries", "SCHEMA_VERSION"]
