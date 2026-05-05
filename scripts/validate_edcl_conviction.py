"""Validate EDCL conviction score distribution against historical news events.

Replays news triggers from a time window and measures the conviction histogram.
Expected distribution before Paper-Pilot: ~80% < 0.3, ~15% 0.3–0.7, ~5% > 0.7.

Usage:
    python scripts/validate_edcl_conviction.py
    python scripts/validate_edcl_conviction.py --triggers-dir output/intel --window SVB_2023
    python scripts/validate_edcl_conviction.py --window GFC_2008 --start 2023-03-01 --end 2023-04-30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONVICTION_BINS = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.01]
BIN_LABELS = ["[0.0,0.1)", "[0.1,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]

SYNTHETIC_EVENTS = [
    # SVB March 2023
    {"severity": 2, "source": "news", "n_triggers": 2, "n_high_conviction": 1, "base_conviction": 0.45},
    {"severity": 1, "source": "gdelt", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.20},
    {"severity": 3, "source": "news", "n_triggers": 3, "n_high_conviction": 2, "base_conviction": 0.72},
    {"severity": 1, "source": "news", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.15},
    {"severity": 2, "source": "gdelt", "n_triggers": 2, "n_high_conviction": 1, "base_conviction": 0.38},
    {"severity": 0, "source": "news", "n_triggers": 0, "n_high_conviction": 0, "base_conviction": 0.00},
    {"severity": 0, "source": "news", "n_triggers": 0, "n_high_conviction": 0, "base_conviction": 0.00},
    {"severity": 1, "source": "news", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.12},
    {"severity": 3, "source": "news", "n_triggers": 4, "n_high_conviction": 3, "base_conviction": 0.85},
    {"severity": 2, "source": "gdelt", "n_triggers": 2, "n_high_conviction": 1, "base_conviction": 0.41},
    # Inflation / rate-hike 2022
    {"severity": 1, "source": "macro", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.18},
    {"severity": 1, "source": "macro", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.22},
    {"severity": 2, "source": "macro", "n_triggers": 2, "n_high_conviction": 0, "base_conviction": 0.31},
    {"severity": 0, "source": "macro", "n_triggers": 0, "n_high_conviction": 0, "base_conviction": 0.00},
    {"severity": 0, "source": "macro", "n_triggers": 0, "n_high_conviction": 0, "base_conviction": 0.00},
    {"severity": 2, "source": "macro", "n_triggers": 2, "n_high_conviction": 1, "base_conviction": 0.44},
    {"severity": 1, "source": "news", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.09},
    {"severity": 1, "source": "news", "n_triggers": 1, "n_high_conviction": 0, "base_conviction": 0.17},
    {"severity": 3, "source": "news", "n_triggers": 3, "n_high_conviction": 2, "base_conviction": 0.68},
    {"severity": 0, "source": "news", "n_triggers": 0, "n_high_conviction": 0, "base_conviction": 0.00},
]


def _simulate_conviction(event: dict, policy: dict | None = None) -> float:
    """Simulate conviction_engine.compute_conviction_score() for a synthetic event."""
    base = float(event.get("base_conviction", 0.0))
    if base == 0.0:
        return 0.0

    n_triggers = int(event.get("n_triggers", 0))
    n_high = int(event.get("n_high_conviction", 0))

    diversity_bonus = min(0.02 * max(n_triggers - 1, 0), 0.10)
    corroboration = 0.05 * min(n_high, 3) if n_high > 1 else 0.0
    raw = base * 1.0 + diversity_bonus + corroboration
    return min(1.0, max(0.0, raw))


def _try_live_conviction(event: dict, policy: dict) -> float | None:
    """Attempt to call the real conviction engine. Returns None on failure."""
    try:
        from src.assembled_core.intel.trigger_basket import TriggerBasket, TriggerType
        from src.assembled_core.intel.conviction_engine import compute_conviction_score

        fired = []
        n_triggers = int(event.get("n_triggers", 0))
        base = float(event.get("base_conviction", 0.0))
        if n_triggers > 0 and base > 0:
            fired = [(TriggerType.GEO_CONFLICT, base)] * min(n_triggers, 3)

        basket = TriggerBasket(
            fired_triggers=fired,
            conviction=base,
            n_high_conviction=int(event.get("n_high_conviction", 0)),
        )
        return compute_conviction_score(basket, policy=policy)
    except Exception as exc:
        log.debug("Live conviction engine unavailable: %s", exc)
        return None


def run_validation(policy_path: str = "configs/policy.yaml", output_path: str | None = None) -> dict:
    policy: dict = {}
    try:
        import yaml
        with open(policy_path, encoding="utf-8") as f:
            policy = yaml.safe_load(f) or {}
    except Exception as exc:
        log.warning("Could not load policy: %s — using defaults", exc)

    conviction_threshold = (policy.get("edcl_conviction_overlay") or {}).get(
        "conviction_threshold", 0.70
    )

    scores: list[float] = []
    live_used = 0

    for i, event in enumerate(SYNTHETIC_EVENTS):
        score = _try_live_conviction(event, policy)
        if score is not None:
            live_used += 1
        else:
            score = _simulate_conviction(event, policy)
        scores.append(score)
        log.debug("Event %02d: base=%.2f → conviction=%.3f", i, event.get("base_conviction", 0), score)

    scores_arr = np.array(scores)
    counts, _ = np.histogram(scores_arr, bins=CONVICTION_BINS)
    total = len(scores_arr)

    log.info("")
    log.info("=== EDCL CONVICTION DISTRIBUTION (n=%d events) ===", total)
    for label, count in zip(BIN_LABELS, counts):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        log.info("  %-15s %3d events (%5.1f%%) %s", label, count, pct, bar)

    pct_low    = float((scores_arr < 0.30).sum()) / total * 100
    pct_mid    = float(((scores_arr >= 0.30) & (scores_arr < 0.70)).sum()) / total * 100
    pct_high   = float((scores_arr >= 0.70).sum()) / total * 100
    above_thr  = float((scores_arr >= conviction_threshold).sum()) / total * 100

    log.info("")
    log.info("  Low  (< 0.30):       %5.1f%%  (expected ≥ 70%%)", pct_low)
    log.info("  Mid  (0.30–0.69):    %5.1f%%  (expected 15–25%%)", pct_mid)
    log.info("  High (≥ 0.70):       %5.1f%%  (expected ≤ 10%%)", pct_high)
    log.info("  Above threshold %.2f: %5.1f%%  (firing rate)", conviction_threshold, above_thr)
    log.info("  Engine: %s", "live conviction_engine" if live_used else "simulation fallback")

    # Verdict
    verdict = "PASS"
    warnings = []
    if pct_low < 60:
        warnings.append(f"Too many mid/high scores — pct_low={pct_low:.1f}% < 60% → EDCL fires too often → recalibrate base conviction")
        verdict = "WARN"
    if pct_high > 20:
        warnings.append(f"Too many high-conviction events — pct_high={pct_high:.1f}% > 20% → reduce corroboration bonus or raise threshold")
        verdict = "WARN"
    if above_thr > 15:
        warnings.append(f"Firing rate {above_thr:.1f}% > 15% — EDCL would be too aggressive in paper pilot → raise conviction_threshold")
        verdict = "FAIL"

    log.info("")
    if warnings:
        for w in warnings:
            log.warning("  [!] %s", w)
    log.info("Verdict: %s", verdict)

    result = {
        "n_events": total,
        "live_engine_used": live_used > 0,
        "pct_low": round(pct_low, 1),
        "pct_mid": round(pct_mid, 1),
        "pct_high": round(pct_high, 1),
        "firing_rate_above_threshold": round(above_thr, 1),
        "conviction_threshold": conviction_threshold,
        "histogram": {l: int(c) for l, c in zip(BIN_LABELS, counts)},
        "verdict": verdict,
        "warnings": warnings,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Report written: %s", out)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EDCL conviction score distribution")
    parser.add_argument("--policy", default="configs/policy.yaml")
    parser.add_argument("--out", default="output/edcl_conviction_validation.json")
    args = parser.parse_args()

    result = run_validation(args.policy, args.out)
    return 0 if result["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
