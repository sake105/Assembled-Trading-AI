"""APScheduler-based job scheduler for EOD pipeline and cron jobs.

From 12_FREE_INFRASTRUKTUR.md §12.7.
Replaces the need for Airflow/Prefect/Dagster for solo quant setup.

Usage:
    from src.assembled_core.ops.scheduler import build_scheduler, start_scheduler
    scheduler = build_scheduler()
    start_scheduler(scheduler)

Install: pip install apscheduler==3.10.4
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _try_apscheduler():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        return AsyncIOScheduler
    except ImportError:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            return BackgroundScheduler
        except ImportError:
            logger.warning("apscheduler not installed — pip install apscheduler==3.10.4")
            return None


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

_JOB_REGISTRY: dict[str, dict[str, Any]] = {
    "eod_pipeline": {
        "trigger": "cron",
        "day_of_week": "mon-fri",
        "hour": 16,
        "minute": 15,
        "timezone": "America/New_York",
        "id": "eod_pipeline",
        "description": "EOD pipeline: fetch prices, compute features, generate signals",
    },
    "news_poll": {
        "trigger": "cron",
        "minute": "*/5",
        "hour": "9-16",
        "day_of_week": "mon-fri",
        "timezone": "America/New_York",
        "id": "news_poll",
        "description": "Poll RSS feeds and news sources every 5 minutes during market hours",
    },
    "weekly_hmm_retrain": {
        "trigger": "cron",
        "day_of_week": "sun",
        "hour": 18,
        "minute": 0,
        "timezone": "America/New_York",
        "id": "weekly_hmm_retrain",
        "description": "Weekly HMM regime model retraining",
    },
    "macro_features_refresh": {
        "trigger": "cron",
        "day_of_week": "mon-fri",
        "hour": 7,
        "minute": 0,
        "timezone": "America/New_York",
        "id": "macro_features_refresh",
        "description": "Refresh FRED macro features before market open",
    },
    "disclosures_poll": {
        "trigger": "cron",
        "day_of_week": "mon-fri",
        "hour": "9,12,16",
        "minute": 0,
        "timezone": "America/New_York",
        "id": "disclosures_poll",
        "description": "Poll SEC EDGAR for new 8-K and Form-4 filings 3x/day",
    },
    "drift_monitor": {
        "trigger": "cron",
        "day_of_week": "mon-fri",
        "hour": 17,
        "minute": 30,
        "timezone": "America/New_York",
        "id": "drift_monitor",
        "description": "Check feature drift and model performance after market close",
    },
}


def build_scheduler(jobs: dict[str, Callable] | None = None) -> Any:
    """Build and configure APScheduler instance.

    Args:
        jobs: Mapping of job_id → callable. Keys must match _JOB_REGISTRY.
              Jobs not in dict are registered as no-ops (for testing).

    Returns:
        Configured scheduler instance (AsyncIOScheduler or BackgroundScheduler).
        Returns None if apscheduler not installed.
    """
    Scheduler = _try_apscheduler()
    if Scheduler is None:
        return None

    scheduler = Scheduler()
    jobs = jobs or {}

    for job_id, config in _JOB_REGISTRY.items():
        func = jobs.get(job_id, lambda: logger.debug("Noop job: %s", job_id))
        trigger = config["trigger"]
        kwargs = {k: v for k, v in config.items()
                  if k not in ("trigger", "id", "description")}
        try:
            scheduler.add_job(
                func,
                trigger=trigger,
                id=job_id,
                replace_existing=True,
                **kwargs,
            )
            logger.debug("Registered job: %s — %s", job_id, config.get("description", ""))
        except Exception as exc:
            logger.error("[SCHEDULER] Failed to register job %s — this job will NOT run: %s", job_id, exc)

    return scheduler


def start_scheduler(scheduler: Any) -> None:
    """Start the scheduler. Logs a warning if not initialized."""
    if scheduler is None:
        logger.warning("Scheduler not initialized — apscheduler may not be installed")
        return

    try:
        scheduler.start()
        logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    except Exception as exc:
        logger.error("Scheduler start failed: %s", exc)


def shutdown_scheduler(scheduler: Any, wait: bool = True) -> None:
    """Gracefully shut down the scheduler."""
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=wait)
        logger.info("Scheduler shut down")
    except Exception as exc:
        logger.warning("Scheduler shutdown error: %s", exc)


def add_one_shot_job(
    scheduler: Any,
    func: Callable,
    run_date: str,
    job_id: str | None = None,
) -> None:
    """Schedule a one-shot job at a specific datetime string (ISO format)."""
    if scheduler is None:
        logger.warning("Scheduler not initialized")
        return

    try:
        scheduler.add_job(
            func,
            trigger="date",
            run_date=run_date,
            id=job_id or f"oneshot_{id(func)}",
            replace_existing=True,
        )
    except Exception as exc:
        logger.warning("One-shot job scheduling failed: %s", exc)


def list_jobs(scheduler: Any) -> list[dict[str, str]]:
    """Return list of registered jobs with their next run time."""
    if scheduler is None:
        return []

    try:
        return [
            {
                "id": job.id,
                "next_run": str(job.next_run_time),
                "trigger": str(job.trigger),
            }
            for job in scheduler.get_jobs()
        ]
    except Exception:
        return []


__all__ = [
    "build_scheduler",
    "start_scheduler",
    "shutdown_scheduler",
    "add_one_shot_job",
    "list_jobs",
    "_JOB_REGISTRY",
]
