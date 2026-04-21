#!/usr/bin/env python3
"""Pre-market news digest generator (Point 34).

Reads the latest intel artifacts and produces a structured pre-market
summary covering top risks, sector exposure, and high-confidence clusters.

Usage:
    python scripts/run_premarket_digest.py
    python scripts/run_premarket_digest.py --output-dir data/intel
    python scripts/run_premarket_digest.py --hours 8 --min-confidence 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = "data/intel"
_DEFAULT_HOURS = 12
_DEFAULT_MIN_CONFIDENCE = 0.35


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def generate_premarket_digest(
    intel_dir: Path,
    *,
    lookback_hours: float = _DEFAULT_HOURS,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> dict:
    """Generate a pre-market digest from intel artifacts.

    Args:
        intel_dir: Directory containing intel JSON artifacts.
        lookback_hours: Hours of news to consider.
        min_confidence: Minimum confidence for high-priority signals.

    Returns:
        Digest dict with sections: summary, top_risks, sector_exposure,
        active_clusters, feed_health.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)  # noqa: F841 — kept for future time filters

    triggers = _load_json(intel_dir / "triggers_latest.json")
    crisis_state = _load_json(intel_dir / "crisis_state.json")
    dep_signal = _load_json(intel_dir / "dependency_signal.json")
    feed_health = _load_json(intel_dir / "feed_health.json")
    intel_signal = _load_json(intel_dir / "intel_signal.json")
    intel_alerts = _load_json(intel_dir / "intel_alerts.json")
    intel_sentiment = _load_json(intel_dir / "intel_sentiment.json")

    # --- Top risks from triggers ---
    trigger_list: list[dict] = triggers.get("triggers", [])
    top_risks = [
        t for t in trigger_list
        if float(t.get("confidence", 0)) >= min_confidence
    ]
    top_risks.sort(key=lambda t: -float(t.get("severity", 0)))

    # --- Sector exposure from dependency signal ---
    sector_exposure: dict[str, dict] = {}
    if dep_signal:
        transmissions = dep_signal.get("transmissions", [])
        for tx in transmissions:
            sector = tx.get("sector") or tx.get("target_sector", "unknown")
            magnitude = float(tx.get("magnitude", 0))
            if sector not in sector_exposure:
                sector_exposure[sector] = {"magnitude": 0.0, "count": 0, "direction": "unknown"}
            sector_exposure[sector]["magnitude"] += magnitude
            sector_exposure[sector]["count"] += 1
            if magnitude < 0:
                sector_exposure[sector]["direction"] = "bearish"
            elif magnitude > 0:
                sector_exposure[sector]["direction"] = "bullish"

    # --- Crisis mode summary ---
    crisis_mode = crisis_state.get("mode", "NORMAL")
    geo_score = int(crisis_state.get("geo_score", 0))
    risk_posture = crisis_state.get("risk_posture", {})

    # --- Feed health summary ---
    total_sources = feed_health.get("total_sources_tracked", 0)
    silent_feeds = feed_health.get("silent_feeds", [])
    feed_summary = {
        "total_sources": total_sources,
        "silent_feeds": silent_feeds,
        "silent_count": len(silent_feeds),
    }

    # --- Intel signal summary ---
    intel_direction = intel_signal.get("net_direction", "neutral")
    intel_risk = intel_signal.get("risk_level", "LOW")
    intel_conf = float(intel_signal.get("aggregate_confidence", 0.0))
    sector_overlay = intel_signal.get("sector_overlay", {})
    macro_info = intel_signal.get("macro", {})
    ticker_surges = intel_signal.get("ticker_surges", [])

    # --- Alerts ---
    recent_alerts = intel_alerts.get("alerts", [])

    # --- Sentiment drift: deteriorating names ---
    deteriorating = [
        e for e in intel_sentiment.get("entries", [])
        if e.get("drift_direction") == "deteriorating"
    ]
    deteriorating.sort(key=lambda e: e.get("slope", 0.0))  # most negative slope first

    # --- Build digest ---
    summary_lines = []
    if crisis_mode in ("CRISIS", "ACTIVE"):
        summary_lines.append(f"ELEVATED RISK: crisis_mode={crisis_mode}, geo_score={geo_score}")
    elif geo_score >= 2:
        summary_lines.append(f"WATCH: geo_score={geo_score}, {len(top_risks)} active risk signals")
    else:
        summary_lines.append(f"NORMAL: geo_score={geo_score}, {len(top_risks)} signals above threshold")

    if intel_risk in ("HIGH", "CRITICAL"):
        summary_lines.append(f"INTEL: {intel_direction.upper()} risk={intel_risk} conf={intel_conf:.2f}")

    if macro_info.get("blackout_active"):
        kinds = macro_info.get("blackout_kinds", [])
        summary_lines.append(f"MACRO BLACKOUT: {', '.join(kinds)} — reduce position sizing")

    if ticker_surges:
        names = [t["ticker"] for t in ticker_surges[:3]]
        summary_lines.append(f"TICKER SURGE: {', '.join(names)}")

    if recent_alerts:
        summary_lines.append(f"ALERTS: {len(recent_alerts)} active (top: {recent_alerts[0]['kind']})")

    if silent_feeds:
        summary_lines.append(f"WARNING: {len(silent_feeds)} feeds silent >2h: {', '.join(silent_feeds[:3])}")

    digest = {
        "schema_version": "premarket.digest.v2",
        "generated_utc": now.isoformat(),
        "lookback_hours": lookback_hours,
        "min_confidence": min_confidence,
        "summary": " | ".join(summary_lines),
        "crisis_mode": crisis_mode,
        "geo_score": geo_score,
        "risk_posture": risk_posture,
        "intel": {
            "direction": intel_direction,
            "risk_level": intel_risk,
            "confidence": intel_conf,
        },
        "top_risks": top_risks[:10],
        "sector_exposure": sector_exposure,
        "sector_overlay": sector_overlay,
        "macro": macro_info,
        "ticker_surges": ticker_surges,
        "alerts": recent_alerts,
        "sentiment_deteriorating": deteriorating[:10],
        "feed_health": feed_summary,
        "active_trigger_count": len(trigger_list),
        "high_confidence_count": len(top_risks),
    }

    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-market news digest generator")
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Intel artifacts directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=_DEFAULT_HOURS,
        help=f"Lookback hours (default: {_DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=_DEFAULT_MIN_CONFIDENCE,
        help=f"Min confidence for high-priority signals (default: {_DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write digest to premarket_digest.json in --output-dir",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    intel_dir = Path(args.output_dir)
    digest = generate_premarket_digest(
        intel_dir,
        lookback_hours=args.hours,
        min_confidence=args.min_confidence,
    )

    if args.write:
        out_path = intel_dir / "premarket_digest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(digest, fh, indent=2, default=str)
        logger.info("[OK] Pre-market digest written to %s", out_path)
    else:
        print(json.dumps(digest, indent=2, default=str))


if __name__ == "__main__":
    main()
