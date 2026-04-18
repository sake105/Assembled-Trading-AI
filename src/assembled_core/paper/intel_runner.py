"""Minimal real-intel orchestration for paper track runner.

Runs NEWS (and optionally DISCLOSURES) pipelines before each trading day,
loads trigger summaries, and derives a compact news_geo state for meta output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def run_real_intel_once(
    *,
    output_dir: Path,
    run_news: bool = True,
    run_disclosures: bool = False,
    sources_path: str = "configs/news/sources.yaml",
    news_path: str = "configs/news/news.yaml",
    cadence: str = "hourly",
) -> Dict[str, Any]:
    """Run NEWS and/or DISCLOSURES pipelines once, defensively.

    Returns:
        Dict with per-pipeline status:
        {"news": {"ran": bool, "status": str}, "disclosures": {"ran": bool, "status": str}}
    """
    result: Dict[str, Any] = {
        "news": {"ran": False, "status": "SKIPPED"},
        "disclosures": {"ran": False, "status": "SKIPPED"},
    }

    if run_news:
        try:
            from src.assembled_core.events.news import run_news_pipeline

            news_out = output_dir / "intel" / "news"
            pipeline_result = run_news_pipeline(
                sources_path=sources_path,
                news_path=news_path,
                cadence=cadence,
                output_dir=news_out,
            )
            health = pipeline_result.get("health")
            status = getattr(health, "status", "ERROR") if health else "ERROR"
            result["news"] = {"ran": True, "status": status}
            logger.info(f"NEWS pipeline completed: status={status}")
        except Exception as exc:
            logger.warning(f"NEWS pipeline failed (non-fatal): {exc}")
            result["news"] = {"ran": True, "status": "ERROR"}

    if run_disclosures:
        result["disclosures"] = {"ran": False, "status": "SKIPPED"}

    return result


def load_intel_summaries(output_dir: Path) -> Dict[str, Any]:
    """Load trigger summaries from pipeline artifacts.

    Returns:
        {"news_triggers_summary": {...}, "disclosures_triggers_summary": {...}}
    """
    summaries: Dict[str, Any] = {
        "news_triggers_summary": _empty_trigger_summary(),
        "disclosures_triggers_summary": _empty_trigger_summary(),
    }

    news_triggers_path = output_dir / "intel" / "news" / "triggers_latest.json"
    if news_triggers_path.exists():
        try:
            data = json.loads(news_triggers_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            summaries["news_triggers_summary"] = _summarize_triggers(items)
        except Exception as exc:
            logger.warning(f"Failed to load NEWS triggers: {exc}")

    return summaries


def _empty_trigger_summary() -> Dict[str, Any]:
    return {"count": 0, "max_severity": 0, "sev1plus": 0, "sev2plus": 0}


def _summarize_triggers(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return _empty_trigger_summary()
    severities = [int(t.get("severity", 0)) for t in items]
    return {
        "count": len(items),
        "max_severity": max(severities),
        "sev1plus": sum(1 for s in severities if s >= 1),
        "sev2plus": sum(1 for s in severities if s >= 2),
    }


def compute_news_geo(output_dir: Path) -> Dict[str, Any]:
    """Derive a compact news_geo state from NEWS trigger snapshot.

    Returns:
        {
            "geo_score": int (0-3),
            "geo_confidence": float (0-1),
            "state_hint": "WATCH" | "ACTIVE",
            "top_triggers": [up to 5 compact trigger dicts],
        }
    """
    news_triggers_path = output_dir / "intel" / "news" / "triggers_latest.json"
    if not news_triggers_path.exists():
        return _empty_news_geo()

    try:
        data = json.loads(news_triggers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Previously a bare ``except Exception: return _empty_news_geo()``
        # silently downgraded a corrupt/unreadable trigger file to a clean
        # "no geo risk" state (geo_score=0, state_hint=WATCH). That looks
        # identical to a truly quiet day and disables the downstream risk
        # overlay exactly when the IO path is broken.
        logger.warning(
            "[PAPER-INTEL] Failed to parse news triggers from %s; returning"
            " empty geo state. Error: %s",
            news_triggers_path,
            exc,
        )
        return _empty_news_geo()

    items = data.get("items", [])
    if not items:
        return _empty_news_geo()

    max_sev = max(int(t.get("severity", 0)) for t in items)
    top_sev_triggers = [t for t in items if int(t.get("severity", 0)) == max_sev]
    geo_confidence = max(float(t.get("confidence", 0.0)) for t in top_sev_triggers)

    top_triggers = [
        {
            "trigger_id": t.get("trigger_id", ""),
            "topic_id": t.get("topic_id", ""),
            "severity": int(t.get("severity", 0)),
            "confidence": float(t.get("confidence", 0.0)),
            "sample_title": str(t.get("sample_title", ""))[:80],
        }
        for t in items[:5]
    ]

    return {
        "geo_score": max_sev,
        "geo_confidence": round(geo_confidence, 3),
        "state_hint": "ACTIVE" if max_sev >= 2 else "WATCH",
        "top_triggers": top_triggers,
    }


def _empty_news_geo() -> Dict[str, Any]:
    return {
        "geo_score": 0,
        "geo_confidence": 0.0,
        "state_hint": "WATCH",
        "top_triggers": [],
    }


def build_intel_summary(
    *,
    intel_orchestration: Dict[str, Any],
    news_triggers_summary: Dict[str, Any],
    disclosures_triggers_summary: Dict[str, Any],
    news_geo: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the intel_summary artifact dict."""
    return {
        "schema_version": "paper.intel_summary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "intel_orchestration": intel_orchestration,
        "news_triggers_summary": news_triggers_summary,
        "disclosures_triggers_summary": disclosures_triggers_summary,
        "news_geo": news_geo,
    }
