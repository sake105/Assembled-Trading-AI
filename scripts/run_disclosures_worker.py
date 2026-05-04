"""DISCLOSURES v1 worker entry point — M2.

Runs the disclosures pipeline (fetch → normalize → dedupe → score → emit)
for a given cadence and writes all artifacts to the output directory.

Supports a simple file lock so that concurrent invocations do not overlap.
A second invocation while the first is still running will log a warning and
exit cleanly with code 0.

Usage examples:

    python scripts/run_disclosures_worker.py
    python scripts/run_disclosures_worker.py --cadence daily
    python scripts/run_disclosures_worker.py --output-dir output/intel/disclosures
    python scripts/run_disclosures_worker.py --sources configs/disclosures/sources.yaml \\
        --disclosures configs/disclosures/disclosures.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.events.disclosures.pipeline import run_disclosures_pipeline

logger = logging.getLogger("disclosures_worker")

_POLICY_PATH = ROOT / "configs" / "policy.yaml"


def _is_kill_switch_active(policy_path: Path = _POLICY_PATH) -> bool:
    """Return True if intel.kill_switch.enabled is set in policy.yaml."""
    try:
        with open(policy_path, "r", encoding="utf-8") as fh:
            policy = yaml.safe_load(fh)
        return bool(
            (policy or {}).get("intel", {}).get("kill_switch", {}).get("enabled", False)
        )
    except Exception as exc:
        logger.warning("[WARN] Could not read kill_switch from policy: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Simple cross-platform file lock (exclusive-create pattern)
# ---------------------------------------------------------------------------


class _WorkerLock:
    """Minimal file-based lock: exclusive O_CREAT|O_EXCL on the lockfile."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True on success, False if already held."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "_WorkerLock":
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the DISCLOSURES v1 pipeline worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cadence",
        choices=["hourly", "daily"],
        default="hourly",
        help="Pipeline cadence.",
    )
    p.add_argument(
        "--sources",
        default="configs/disclosures/sources.yaml",
        help="Path to disclosures sources registry YAML.",
    )
    p.add_argument(
        "--disclosures",
        default="configs/disclosures/disclosures.yaml",
        help="Path to disclosures pipeline parameters YAML.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for artifacts. Defaults to output/intel/disclosures.",
    )
    p.add_argument(
        "--no-lock",
        action="store_true",
        default=False,
        help="Disable file locking (useful for testing).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    args = _parse_args()

    if _is_kill_switch_active():
        logger.info("[SKIP] kill_switch_active — disclosures_worker halted by policy.")
        return 0

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output") / "intel" / "disclosures"
    )
    lock_path = output_dir / "cache" / ".disclosures_worker.lock"

    lock = _WorkerLock(lock_path)

    if not args.no_lock:
        if not lock.acquire():
            logger.warning(
                "[SKIP] disclosures_worker already running (lockfile: %s). Exiting.",
                lock_path,
            )
            return 0

    t0 = time.monotonic()
    exit_code = 0

    try:
        logger.info(
            "[START] disclosures_worker cadence=%s output_dir=%s",
            args.cadence,
            output_dir,
        )

        result = run_disclosures_pipeline(
            sources_path=args.sources,
            disclosures_path=args.disclosures,
            cadence=args.cadence,
            output_dir=output_dir,
        )

        events = result.get("events") or []
        health = result.get("health")
        health_status = getattr(health, "status", "UNKNOWN") if health else "UNKNOWN"
        trigger_count = 0
        max_sev = 0
        if health is not None:
            trig = getattr(health, "metrics", {}).get("triggers", {})
            trigger_count = int(trig.get("trigger_count", 0))
            max_sev = int(trig.get("max_severity", 0))

        elapsed = time.monotonic() - t0
        logger.info(
            "[OK] disclosures_worker done in %.1fs | events=%d health=%s triggers=%d max_sev=%d",
            elapsed,
            len(events),
            health_status,
            trigger_count,
            max_sev,
        )

        if health_status == "ERROR":
            logger.warning(
                "[WARN] Health status is ERROR — triggers may be suppressed."
            )
        elif health_status == "DEGRADED":
            logger.warning(
                "[WARN] Health status is DEGRADED — trigger severity capped at 1."
            )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "[ERROR] disclosures_worker failed after %.1fs: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        exit_code = 1
    finally:
        if not args.no_lock:
            lock.release()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
