"""Crisis-Alpha v1 worker — M5.

Runs one crisis-alpha evaluation cycle:
    1. Loads news trigger snapshot (triggers_latest.json) for geo signal.
    2. Builds CrisisAlphaContext.
    3. Calls run_crisis_alpha_pipeline().
    4. Logs results with structured [OK]/[WARN]/[SKIP] prefixes.
    5. Writes a JSON manifest for audit purposes.

This worker does NOT submit orders automatically.  When ACTIVE, it logs
the target_weights; when deactivation or exits are triggered, it logs the
flatten intent.  Order generation and submission require a separate manual
or automated step.

Usage:
    python scripts/run_crisis_alpha_worker.py
    python scripts/run_crisis_alpha_worker.py --triggers output/intel/news/triggers_latest.json
    python scripts/run_crisis_alpha_worker.py --dry-run --geo-score 2.5 --geo-sources 3
    python scripts/run_crisis_alpha_worker.py --reset-pause --reason "Manual geo-risk clear"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.pipeline import run_crisis_alpha_pipeline

logger = logging.getLogger("crisis_alpha_worker")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crisis-Alpha v1 worker — evaluate geo risk and manage crisis sub-portfolio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--triggers",
        default="output/intel/news/triggers_latest.json",
        help="Path to news triggers_latest.json for geo signal.",
    )
    p.add_argument(
        "--config",
        default="configs/crisis_alpha/crisis_alpha.yaml",
        help="Path to crisis_alpha.yaml config.",
    )
    p.add_argument(
        "--state-path",
        default="output/ops/crisis_alpha_state.json",
        help="Path to persist crisis state.",
    )
    p.add_argument(
        "--output-dir",
        default="output/ops",
        help="Directory for the run manifest.",
    )
    # Manual overrides for testing / debugging
    p.add_argument(
        "--geo-score",
        type=float,
        default=None,
        help="Override geo_score (skips trigger file parsing). For testing only.",
    )
    p.add_argument(
        "--geo-sources",
        type=int,
        default=None,
        help="Override geo_sources count. For testing only.",
    )
    p.add_argument(
        "--market-stress-ok",
        action="store_true",
        default=False,
        help="Assert market stress is confirmed (testing override).",
    )
    p.add_argument(
        "--health-ok",
        action="store_true",
        default=True,
        help="Assert health is OK (default True; use --no-health-ok to set False).",
    )
    p.add_argument(
        "--no-health-ok",
        dest="health_ok",
        action="store_false",
        help="Assert health is NOT OK.",
    )
    p.add_argument(
        "--daily-pnl",
        type=float,
        default=0.0,
        help="Today's crisis sub-portfolio PnL (negative = loss).",
    )
    p.add_argument(
        "--reset-pause",
        action="store_true",
        default=False,
        help="Manually reset PAUSE state to WATCH.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Evaluate without persisting state changes.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trigger file loader
# ---------------------------------------------------------------------------


def _load_triggers(triggers_path: str) -> tuple[list[dict], float, int, bool]:
    """Load triggers_latest.json and extract geo signal.

    Returns:
        (trigger_items, geo_score, geo_sources, social_only)
    """
    path = Path(triggers_path)
    if not path.exists():
        logger.warning("[WARN] triggers file not found: %s — using geo_score=0.0", path)
        return [], 0.0, 0, False

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("[WARN] could not parse triggers file %s: %s", path, exc)
        return [], 0.0, 0, False

    items = data.get("items", [])
    if not items:
        return [], 0.0, 0, False

    # Geo score: max severity across geo-relevant triggers
    geo_items = [
        t
        for t in items
        if str(t.get("topic", "")).lower()
        in ("geopolitical", "military", "sanctions", "trade_war")
    ]
    if not geo_items:
        return items, 0.0, 0, False

    geo_score = max(float(t.get("severity", 0)) for t in geo_items)

    # Count distinct sources
    source_set = {str(t.get("source", "")) for t in geo_items if t.get("source")}
    geo_sources = len(source_set)

    # Social-only: True if ALL sources are tagged as social
    social_only = (
        all(str(t.get("source_tier", "")).lower() == "social" for t in geo_items)
        if geo_items
        else False
    )

    return items, geo_score, geo_sources, social_only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / f"crisis_alpha_manifest_{ts_str}.json"

    t0 = time.monotonic()
    logger.info(
        "[START] crisis_alpha_worker dry_run=%s reset_pause=%s",
        args.dry_run,
        args.reset_pause,
    )

    exit_code = 0

    try:
        # Load config
        policy: dict = {}
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
        else:
            logger.warning("[WARN] config not found: %s — using defaults", config_path)

        # Build geo signal from trigger file (or CLI overrides)
        if args.geo_score is not None:
            # CLI override mode (testing / debugging)
            trigger_items: list[dict] = []
            geo_score = args.geo_score
            geo_sources = args.geo_sources if args.geo_sources is not None else 0
            social_only = False
            logger.info(
                "[INFO] CLI override: geo_score=%.2f sources=%d", geo_score, geo_sources
            )
        else:
            trigger_items, geo_score, geo_sources, social_only = _load_triggers(
                args.triggers
            )
            logger.info(
                "[INFO] triggers loaded: geo_score=%.2f sources=%d social_only=%s items=%d",
                geo_score,
                geo_sources,
                social_only,
                len(trigger_items),
            )

        # Build CrisisAlphaContext
        daily_loss_limit = float(
            (policy.get("crisis_alpha") or {}).get("daily_loss", {}).get("limit", 0.02)
        )
        ctx = CrisisAlphaContext(
            timestamp_utc=now_utc,
            geo_score=geo_score,
            geo_sources=geo_sources,
            social_only=social_only,
            market_stress_ok=args.market_stress_ok,
            health_ok=args.health_ok,
            daily_pnl=args.daily_pnl,
            daily_loss_limit=daily_loss_limit,
            news_trigger_items=trigger_items,
        )

        # Run pipeline
        result = run_crisis_alpha_pipeline(
            ctx,
            policy,
            state_path=Path(args.state_path),
            reset_pause=args.reset_pause,
            dry_run=args.dry_run,
        )

        # Log summary
        elapsed = time.monotonic() - t0
        state = result["state"]
        prev_state = result["previous_state"]
        target_weights = result["target_weights"]
        should_flatten = result["should_flatten_all"]
        errors = result["errors"]

        if errors:
            for err in errors:
                logger.error("[ERROR] %s", err)

        if state == "ACTIVE" and target_weights:
            logger.info(
                "[OK] crisis_alpha_worker done in %.2fs | %s→%s | " "targets: %s",
                elapsed,
                prev_state,
                state,
                {s: f"{w:.4f}" for s, w in target_weights.items()},
            )
            logger.warning(
                "[WARN] CRISIS ACTIVE — target weights logged above. "
                "Manual review required before order submission."
            )
        elif should_flatten:
            logger.warning(
                "[WARN] crisis_alpha_worker done in %.2fs | %s→%s | " "FLATTEN ALL: %s",
                elapsed,
                prev_state,
                state,
                result["flatten_reason"],
            )
        else:
            log_fn = logger.warning if state == "PAUSE" else logger.info
            log_fn(
                "[OK] crisis_alpha_worker done in %.2fs | %s→%s | " "gates_ok=%s",
                elapsed,
                prev_state,
                state,
                result["gates_ok"],
            )

        # Write manifest
        manifest = {
            "timestamp_utc": now_utc.isoformat(),
            "state": state,
            "previous_state": prev_state,
            "geo_score": geo_score,
            "geo_sources": geo_sources,
            "social_only": social_only,
            "health_ok": args.health_ok,
            "market_stress_ok": args.market_stress_ok,
            "gates_ok": result["gates_ok"],
            "gate_reasons": result["gate_reasons"],
            "target_weights": result["target_weights"],
            "should_flatten_all": should_flatten,
            "flatten_reason": result["flatten_reason"],
            "positions_to_exit_count": len(result["positions_to_exit"]),
            "dry_run": args.dry_run,
            "errors": errors,
            "elapsed_s": round(elapsed, 3),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        logger.info("[OK] manifest written to %s", manifest_path)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "[ERROR] crisis_alpha_worker failed after %.2fs: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
