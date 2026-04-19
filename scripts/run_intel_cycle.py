#!/usr/bin/env python3
"""Intel cycle runner — fetches GDELT every 15 minutes, updates crisis state.

Usage:
    python scripts/run_intel_cycle.py              # run once
    python scripts/run_intel_cycle.py --loop       # run every 15 minutes
    python scripts/run_intel_cycle.py --loop --interval 900  # custom interval (seconds)
    python scripts/run_intel_cycle.py --dry-run    # fetch but don't write state

Output artifacts (written to data/intel/):
    triggers_latest.json    — current geo triggers (schema: news.triggers.v1)
    crisis_state.json       — current crisis mode + risk posture
    dependency_signal.json  — current beneficiaries/losers
    intel_health.json       — component freshness status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running as script from repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assembled_core.intel import (  # noqa: E402
    crisis_alpha_worker,
    geo_trigger,
    shock_propagation,
)
from src.assembled_core.intel.dependency_graph import load_graph  # noqa: E402
from src.assembled_core.intel.health_monitor import HealthMonitor  # noqa: E402
from src.assembled_core.intel.market_confirmation import compute_market_confirmation  # noqa: E402
from src.assembled_core.intel.models import CrisisMode, CrisisState  # noqa: E402
from src.assembled_core.intel.news_cluster import ClusterManager  # noqa: E402
from src.assembled_core.intel.news_dedupe import NewsDedupeIndex  # noqa: E402
from src.assembled_core.intel.news_ingest import GdeltFetcher  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = "data/intel"
_POLICY_PATH = _REPO_ROOT / "configs" / "policy.yaml"


def _is_kill_switch_active(policy_path: Path = _POLICY_PATH) -> bool:
    """Return True if intel.kill_switch.enabled is set in policy.yaml."""
    try:
        with open(policy_path, "r", encoding="utf-8") as fh:
            policy = yaml.safe_load(fh)
        return bool((policy or {}).get("intel", {}).get("kill_switch", {}).get("enabled", False))
    except Exception as exc:
        logger.warning("[WARN] Could not read kill_switch from policy: %s", exc)
        return False
_DEFAULT_STATE_DIR = "data/intel/state"
_DEFAULT_GRAPH_PATH = "configs/dependency_graph.yaml"
_DEFAULT_INTERVAL = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_init_crisis_state(output_dir: Path) -> CrisisState:
    """Load persisted crisis state or return a fresh NORMAL state."""
    crisis_path = output_dir / "crisis_state.json"
    if crisis_path.exists():
        try:
            with open(crisis_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return CrisisState.model_validate(data)
        except Exception as exc:
            logger.warning("[WARN] Could not load crisis_state.json: %s — using fresh state", exc)

    return CrisisState(
        mode=CrisisMode.NORMAL,
        geo_score=0,
        active_triggers=[],
        dependency_signal_id=None,
        entered_at=datetime.now(tz=timezone.utc),
        risk_posture={},
        basket_overrides={},
        audit_trail=[],
    )


def _build_triggers_artifact(
    triggers: list,
    generated_utc: datetime,
) -> dict:
    """Build the triggers_latest.json artifact (schema: news.triggers.v1)."""
    trigger_list = []
    for t in triggers:
        trigger_list.append({
            "trigger_id": t.trigger_id,
            "trigger_type": t.trigger_type.value,
            "severity": t.trigger_score,
            "confidence": round(t.confidence, 4),
            "ttl_minutes": t.ttl_minutes,
        })

    sev1_plus = sum(1 for t in triggers if t.trigger_score >= 1)
    sev2_plus = sum(1 for t in triggers if t.trigger_score >= 2)
    max_sev = max((t.trigger_score for t in triggers), default=0)

    return {
        "schema_version": "news.triggers.v1",
        "generated_utc": generated_utc.isoformat(),
        "triggers": trigger_list,
        "summary": {
            "max_severity": max_sev,
            "watch_count_sev1plus": sev1_plus,
            "active_count_sev2plus": sev2_plus,
        },
    }


def _write_artifact(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        logger.debug("[SKIP] dry-run: would write %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Single cycle
# ---------------------------------------------------------------------------


def run_single_cycle(config: dict) -> dict:
    """
    Execute one complete intel cycle.

    Steps:
    1. Fetch new GDELT events
    2. Deduplicate
    3. Cluster into EvidenceClusters
    4. Score clusters → GeoTriggers
    5. Aggregate geo_score
    6. Propagate shocks → DependencySignal (if geo_score >= 1)
    7. Update CrisisState
    8. Update HealthMonitor
    9. Write output artifacts
    10. Return summary dict
    """
    cycle_start = time.monotonic()
    now = datetime.now(tz=timezone.utc)
    dry_run: bool = config["dry_run"]
    output_dir: Path = config["output_dir"]

    # --- Component instances (or from config cache) ---
    fetcher: GdeltFetcher = config["fetcher"]
    dedupe: NewsDedupeIndex = config["dedupe"]
    cluster_mgr: ClusterManager = config["cluster_mgr"]
    health: HealthMonitor = config["health"]
    dep_graph = config["dep_graph"]
    prev_state: CrisisState = config["crisis_state"]

    logger.info("[START] Intel cycle")

    # Step 1: Fetch
    events, is_new_batch = fetcher.fetch_new_events()
    raw_count = len(events)

    # Step 2: Deduplicate
    new_events = dedupe.filter_new(events)
    new_count = len(new_events)

    if not dry_run:
        dedupe.save()

    gdelt_status = "OK" if is_new_batch or raw_count > 0 else "STALE"
    health.update("gdelt", gdelt_status, now=now)

    logger.info(
        "[OK] GDELT: fetched %d events (%d new after dedupe)",
        raw_count,
        new_count,
    )

    # Step 3: Cluster
    active_clusters = cluster_mgr.update_clusters(new_events, now=now)

    cluster_summary = ", ".join(
        f"{cl.trigger_type.value} x{len(cl.supporting_events)}"
        for cl in active_clusters
    ) or "none"
    logger.info("[OK] Clusters: %d active (%s)", len(active_clusters), cluster_summary)

    # Step 3.5: D10 — FinBERT sentiment enrichment (optional)
    finbert_enabled = config.get("finbert_enabled", False)
    if finbert_enabled and active_clusters:
        try:
            from src.assembled_core.ml.nlp_sentiment import score_texts_finbert
            for cluster in active_clusters:
                texts = [ev.headline for ev in getattr(cluster, "events", []) if getattr(ev, "headline", None)]
                if texts:
                    sentiment_scores = score_texts_finbert(texts[:10])  # cap at 10 texts per cluster
                    if sentiment_scores:
                        avg_sentiment = sum(s.get("score", 0) * (1 if s.get("label") == "positive" else -1)
                                           for s in sentiment_scores) / len(sentiment_scores)
                        # Attach to cluster if it supports it
                        if hasattr(cluster, "sentiment_score"):
                            cluster.sentiment_score = float(avg_sentiment)
            logger.info("[OK] FinBERT sentiment enrichment applied to %d clusters", len(active_clusters))
        except Exception as e:
            logger.debug("[SKIP] FinBERT enrichment skipped: %s", e)

    # Step 4: Score clusters → GeoTriggers
    geo_triggers = []
    all_events_for_scoring = list(new_events)  # events in this cycle for score_cluster
    for cluster in active_clusters:
        trigger = geo_trigger.score_cluster(cluster, all_events_for_scoring)
        geo_triggers.append(trigger)

    # Step 5: Aggregate
    agg = geo_trigger.aggregate_triggers(geo_triggers)
    geo_score: int = agg["geo_score"]

    logger.info(
        "[OK] Triggers: geo_score=%d, %s mode",
        geo_score,
        _score_to_label(geo_score),
    )

    # Step 6: Shock propagation (only if geo_score >= 1 and we have triggers)
    dep_signal = None
    if geo_score >= 1 and geo_triggers:
        # Use the highest-scoring trigger for propagation
        top_trigger = max(geo_triggers, key=lambda t: t.trigger_score)
        shocks = shock_propagation.map_trigger_to_shocks(top_trigger)
        if shocks and dep_graph is not None:
            # B7: Pass regime and magnitude to enhanced propagation
            crisis_mode = getattr(config.get("crisis_state"), "mode", "NORMAL")
            prop_regime = "crisis" if crisis_mode == "CRISIS" else "sideways"
            prop_magnitude = 1.0 + (top_trigger.trigger_score - 1) * 0.3  # score 1→1.0, 3→1.6
            transmissions = shock_propagation.propagate(
                shocks, dep_graph,
                trigger_id=top_trigger.trigger_id,
                magnitude=prop_magnitude,
                regime=prop_regime,
            )
            if transmissions:
                dep_signal = shock_propagation.to_dependency_signal(
                    transmissions,
                    trigger_id=top_trigger.trigger_id,
                    trigger_score=top_trigger.trigger_score,
                    now=now,
                )

    # Step 7: Update crisis state — compute real market confirmation
    market_confirm = compute_market_confirmation(lookback_days=5, cache=config.get("_mc_cache"))
    new_state = crisis_alpha_worker.update_crisis_state(
        prev_state=prev_state,
        geo_score=geo_score,
        active_triggers=geo_triggers,
        dependency_signal=dep_signal,
        market_confirm=market_confirm,
        now=now,
    )
    # Store updated state back into config for next cycle
    config["crisis_state"] = new_state

    # Step 8: Write artifacts
    triggers_artifact = _build_triggers_artifact(geo_triggers, now)
    _write_artifact(output_dir / "triggers_latest.json", triggers_artifact, dry_run)

    crisis_artifact = json.loads(new_state.model_dump_json())
    _write_artifact(output_dir / "crisis_state.json", crisis_artifact, dry_run)

    if dep_signal is not None:
        dep_artifact = json.loads(dep_signal.model_dump_json())
        _write_artifact(output_dir / "dependency_signal.json", dep_artifact, dry_run)

    health_artifact = health.snapshot(now=now)
    _write_artifact(output_dir / "intel_health.json", health_artifact, dry_run)

    if not dry_run:
        logger.info("[OK] Artifacts written to %s", output_dir)

    cycle_elapsed = time.monotonic() - cycle_start
    logger.info(
        "[OK] Cycle complete in %.1fs.",
        cycle_elapsed,
    )

    return {
        "raw_events": raw_count,
        "new_events": new_count,
        "active_clusters": len(active_clusters),
        "geo_score": geo_score,
        "crisis_mode": new_state.mode.value,
        "elapsed_s": round(cycle_elapsed, 2),
    }


def _score_to_label(geo_score: int) -> str:
    if geo_score >= 3:
        return "ACTIVE"
    if geo_score >= 2:
        return "WATCH"
    if geo_score >= 1:
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Load dependency graph
    graph_path = Path(args.graph_path)
    dep_graph = None
    if graph_path.exists():
        try:
            dep_graph = load_graph(graph_path)
            logger.info("[OK] Dependency graph loaded from %s", graph_path)
        except Exception as exc:
            logger.warning("[WARN] Could not load dependency graph: %s", exc)
    else:
        logger.warning("[WARN] Dependency graph not found at %s — propagation disabled", graph_path)

    fetcher = GdeltFetcher(state_dir / "gdelt_state.json")
    dedupe = NewsDedupeIndex(
        persist_path=state_dir / "dedupe_index.json",
        max_size=10_000,
    )
    cluster_mgr = ClusterManager(cluster_ttl_minutes=360)
    health = HealthMonitor()
    health.register("gdelt", stale_threshold_minutes=30)

    prev_state = _load_or_init_crisis_state(output_dir)

    return {
        "dry_run": args.dry_run,
        "output_dir": output_dir,
        "state_dir": state_dir,
        "fetcher": fetcher,
        "dedupe": dedupe,
        "cluster_mgr": cluster_mgr,
        "health": health,
        "dep_graph": dep_graph,
        "crisis_state": prev_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assembled-Trading-AI intel cycle runner (GDELT 15-min polling)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously on a fixed interval",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=_DEFAULT_INTERVAL,
        help=f"Seconds between cycles when --loop is set (default: {_DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process but do not write output files",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for output artifacts (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--state-dir",
        default=_DEFAULT_STATE_DIR,
        help=f"Directory for fetch state files (default: {_DEFAULT_STATE_DIR})",
    )
    parser.add_argument(
        "--graph-path",
        default=_DEFAULT_GRAPH_PATH,
        help=f"Path to dependency graph YAML (default: {_DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    if _is_kill_switch_active():
        logger.info("[SKIP] kill_switch_active — intel_cycle halted by policy.")
        return

    config = _build_config(args)

    if args.loop:
        logger.info(
            "[START] Intel loop: interval=%ds, dry_run=%s",
            args.interval,
            args.dry_run,
        )
        while True:
            try:
                summary = run_single_cycle(config)
                logger.info(
                    "[OK] Cycle complete in %.1fs. Next in %ds.",
                    summary["elapsed_s"],
                    args.interval,
                )
            except KeyboardInterrupt:
                logger.info("[OK] Intel loop stopped by user.")
                break
            except Exception as exc:
                logger.error("[ERROR] Cycle failed: %s", exc)
            time.sleep(args.interval)
    else:
        run_single_cycle(config)


if __name__ == "__main__":
    main()
