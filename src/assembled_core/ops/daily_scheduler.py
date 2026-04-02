"""
daily_scheduler.py — Autonomous daily operations orchestrator.

Ties together ingest, post-trade analysis, reconcile, and health-check
workers into a coherent autonomous daily cycle. Uses stdlib only — no
external scheduling libraries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result of a single worker execution."""

    worker_name: str
    status: str  # "ok" | "skip" | "error"
    duration_s: float
    error_msg: Optional[str] = None


def _ingest_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Placeholder ingest worker — no live feed configured."""
    t0 = time.monotonic()
    logger.info("[SKIP] ingest: skip (no live feed configured)")
    return WorkerResult(
        worker_name="ingest_worker",
        status="skip",
        duration_s=time.monotonic() - t0,
    )


def _post_trade_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Post-trade analysis worker — checks availability and skips if no data."""
    t0 = time.monotonic()
    try:
        import importlib

        importlib.import_module("assembled_core.qa.post_trade_analyzer")
        logger.info("[SKIP] post_trade: available (skip: no data)")
        return WorkerResult(
            worker_name="post_trade_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )
    except ImportError:
        logger.info("[SKIP] post_trade: skip (not available)")
        return WorkerResult(
            worker_name="post_trade_worker",
            status="skip",
            duration_s=time.monotonic() - t0,
        )


def _reconcile_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Placeholder ledger reconciliation worker."""
    t0 = time.monotonic()
    logger.info("[SKIP] reconcile: skip (no ledger configured)")
    return WorkerResult(
        worker_name="reconcile_worker",
        status="skip",
        duration_s=time.monotonic() - t0,
    )


def _health_check_worker(date_str: str, output_dir: str, dry_run: bool) -> WorkerResult:
    """Health check — verifies output directory exists and is writable."""
    t0 = time.monotonic()
    path = Path(output_dir)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        # Try writing a temp file to verify write access
        test_file = path / ".health_check_probe"
        test_file.write_text("ok")
        test_file.unlink()
        logger.info("[OK] health_check: output_dir writable")
        return WorkerResult(
            worker_name="health_check_worker",
            status="ok",
            duration_s=time.monotonic() - t0,
        )
    except Exception as exc:
        msg = str(exc)
        logger.warning("[ERROR] health_check: %s", msg)
        return WorkerResult(
            worker_name="health_check_worker",
            status="error",
            duration_s=time.monotonic() - t0,
            error_msg=msg,
        )


# Default worker registry (callables that accept date_str, output_dir, dry_run)
_DEFAULT_WORKERS: List[Callable] = [
    _ingest_worker,
    _post_trade_worker,
    _reconcile_worker,
    _health_check_worker,
]


class DailyScheduler:
    """Orchestrates a sequence of daily operational workers."""

    def __init__(self, workers: Optional[List[Callable]] = None) -> None:
        self.workers: List[Callable] = (
            workers if workers is not None else list(_DEFAULT_WORKERS)
        )

    def run_daily_cycle(
        self,
        date_str: str,
        output_dir: str,
        dry_run: bool = False,
    ) -> List[WorkerResult]:
        """Run all workers in sequence and return their results."""
        results: List[WorkerResult] = []
        logger.info("[START] daily_cycle date=%s dry_run=%s", date_str, dry_run)
        for worker_fn in self.workers:
            name = getattr(worker_fn, "__name__", repr(worker_fn))
            try:
                result = worker_fn(date_str, output_dir, dry_run)
                results.append(result)
                logger.info(
                    "[%s] %s duration=%.3fs",
                    result.status.upper(),
                    name,
                    result.duration_s,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("[ERROR] %s caught unhandled exception: %s", name, msg)
                results.append(
                    WorkerResult(
                        worker_name=name,
                        status="error",
                        duration_s=0.0,
                        error_msg=msg,
                    )
                )
        logger.info("[OK] daily_cycle complete workers=%d", len(results))
        return results


# Module-level convenience function backed by a default scheduler instance
def run_daily_cycle(
    date_str: str,
    output_dir: str,
    dry_run: bool = False,
) -> List[WorkerResult]:
    """Run the default daily worker cycle and return results."""
    scheduler = DailyScheduler()
    return scheduler.run_daily_cycle(date_str, output_dir, dry_run)


def build_cycle_summary(results: List[WorkerResult]) -> dict:
    """Build a summary dict from a list of WorkerResult objects."""
    ok = sum(1 for r in results if r.status == "ok")
    skip = sum(1 for r in results if r.status == "skip")
    error = sum(1 for r in results if r.status == "error")
    return {
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total": len(results),
        "ok": ok,
        "skip": skip,
        "error": error,
        "workers": [
            {
                "worker_name": r.worker_name,
                "status": r.status,
                "duration_s": r.duration_s,
                "error_msg": r.error_msg,
            }
            for r in results
        ],
    }


def schedule_loop(
    interval_hours: float,
    output_dir: str,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
) -> None:
    """Run the daily cycle repeatedly at interval_hours cadence.

    Args:
        interval_hours: Hours to sleep between cycles.
        output_dir: Output directory passed to each cycle.
        dry_run: Passed through to each cycle.
        max_iterations: Stop after this many iterations (None = run forever).
                        Useful for testing.
    """
    scheduler = DailyScheduler()
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        results = scheduler.run_daily_cycle(date_str, output_dir, dry_run)
        summary = build_cycle_summary(results)
        logger.info(
            "[OK] schedule_loop iteration=%d summary=%s",
            iteration + 1,
            summary,
        )
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        time.sleep(interval_hours * 3600)
