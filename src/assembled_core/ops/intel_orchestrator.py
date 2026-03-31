"""OPS-11: Intel orchestrator — run real NEWS + DISCLOSURES pipelines before trading cycle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)

PipelineStatus = str  # "OK" | "DEGRADED" | "ERROR" | "SKIPPED"


def run_intel_pipelines(
    app_cfg: Dict[str, Any], root: Path | None = None
) -> Dict[str, Any]:
    """Run NEWS and/or DISCLOSURES pipelines when paper_runner.intel.mode == "real".

    Returns:
        {
            "news": {"ran": bool, "status": "OK"|"DEGRADED"|"ERROR"|"SKIPPED"},
            "disclosures": {"ran": bool, "status": "OK"|"DEGRADED"|"ERROR"|"SKIPPED"},
        }
    """
    out: Dict[str, Any] = {
        "news": {"ran": False, "status": "SKIPPED"},
        "disclosures": {"ran": False, "status": "SKIPPED"},
    }
    paper_cfg = app_cfg.get("paper_runner") or {}
    intel_cfg = paper_cfg.get("intel") or {}
    mode = (intel_cfg.get("mode") or "sim").strip().lower()
    if mode != "real":
        return out

    base = Path(root) if root is not None else Path(".").resolve()
    run_news = bool(intel_cfg.get("run_news_pipeline", False))
    run_disclosures = bool(intel_cfg.get("run_disclosures_pipeline", False))

    if run_news:
        news_cfg = intel_cfg.get("news") or {}
        sources_path = news_cfg.get("sources_path") or "configs/news/sources.yaml"
        config_path = news_cfg.get("config_path") or "configs/news/news.yaml"
        cadence = news_cfg.get("cadence") or "hourly"
        output_dir = news_cfg.get("output_dir") or "output/intel/news"
        if not Path(sources_path).is_absolute():
            sources_path = str(base / sources_path)
        if not Path(config_path).is_absolute():
            config_path = str(base / config_path)
        if not Path(output_dir).is_absolute():
            output_dir = str(base / output_dir)
        try:
            from src.assembled_core.events.news import run_news_pipeline

            result = run_news_pipeline(
                sources_path=sources_path,
                news_path=config_path,
                cadence=cadence,
                output_dir=output_dir,
            )
            health = result.get("health")
            status = getattr(health, "status", "ERROR")
            if status not in ("OK", "DEGRADED", "ERROR"):
                status = "DEGRADED"
            out["news"] = {"ran": True, "status": status}
        except Exception as e:
            log.exception("NEWS pipeline failed: %s", e)
            out["news"] = {"ran": True, "status": "ERROR"}

    if run_disclosures:
        disc_cfg = intel_cfg.get("disclosures") or {}
        sources_path = (
            disc_cfg.get("sources_path") or "configs/disclosures/sources.yaml"
        )
        config_path = (
            disc_cfg.get("config_path") or "configs/disclosures/disclosures.yaml"
        )
        cadence = disc_cfg.get("cadence") or "daily"
        output_dir = disc_cfg.get("output_dir") or "output/intel/disclosures"
        if not Path(sources_path).is_absolute():
            sources_path = str(base / sources_path)
        if not Path(config_path).is_absolute():
            config_path = str(base / config_path)
        if not Path(output_dir).is_absolute():
            output_dir = str(base / output_dir)
        try:
            from src.assembled_core.events.disclosures import run_disclosures_pipeline

            result = run_disclosures_pipeline(
                sources_path=sources_path,
                disclosures_path=config_path,
                cadence=cadence,
                output_dir=output_dir,
            )
            health = result.get("health")
            status = getattr(health, "status", "ERROR")
            if status not in ("OK", "DEGRADED", "ERROR"):
                status = "DEGRADED"
            out["disclosures"] = {"ran": True, "status": status}
        except Exception as e:
            log.exception("Disclosures pipeline failed: %s", e)
            out["disclosures"] = {"ran": True, "status": "ERROR"}

    return out


__all__ = ["run_intel_pipelines"]
