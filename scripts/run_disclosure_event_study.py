"""Disclosure Event Study (T6.2).

Runs an event study over disclosure events (Form-4 insider, SEC filings) to measure
forward-return IC by severity tier and source tier. Outputs a JSON report artifact.

Usage:
    python scripts/run_disclosure_event_study.py \
        --disclosures data/disclosures/events.csv \
        --prices data/prices/panel.parquet \
        --output output/intel/event_study \
        --window-before 5 \
        --window-after 20

Kill-switch: reads policy.yaml; exits if intel.kill_switch.enabled=true.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_disclosure_event_study")

_POLICY_PATH = ROOT / "configs" / "policy.yaml"


def _is_kill_switch_active(policy_path: Path = _POLICY_PATH) -> bool:
    try:
        import yaml
        with open(policy_path, "r", encoding="utf-8") as fh:
            policy = yaml.safe_load(fh)
        return bool((policy or {}).get("intel", {}).get("kill_switch", {}).get("enabled", False))
    except Exception as exc:
        logger.warning("[WARN] Could not read kill_switch from policy: %s", exc)
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disclosure Event Study (T6.2)")
    p.add_argument("--disclosures", required=True, help="Path to disclosures CSV/Parquet")
    p.add_argument("--prices", required=True, help="Path to price panel CSV/Parquet")
    p.add_argument("--output", default="output/intel/event_study", help="Output directory")
    p.add_argument("--window-before", type=int, default=5)
    p.add_argument("--window-after", type=int, default=20)
    p.add_argument("--min-severity", default=None, help="Filter by minimum severity label")
    p.add_argument("--tier", default=None, help="Filter by source tier (T0/T1/T2/T3)")
    return p.parse_args()


def _load_frame(path: str) -> "pd.DataFrame":
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _compute_ic(windows: "pd.DataFrame", window_after: int) -> dict:
    """Compute forward-return IC between event day return and post-event drift."""
    import numpy as np
    try:
        event_day = windows[windows["rel_day"] == 0][["event_id", "event_return"]].rename(
            columns={"event_return": "event_day_return"}
        )
        forward = windows[windows["rel_day"] == window_after][["event_id", "event_return"]].rename(
            columns={"event_return": "forward_return"}
        )
        merged = event_day.merge(forward, on="event_id")
        if len(merged) < 5:
            return {"ic": None, "n_events": len(merged), "note": "too few events"}
        x = merged["event_day_return"].values
        y = merged["forward_return"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            return {"ic": None, "n_events": int(mask.sum()), "note": "too few valid pairs"}
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        return {"ic": round(corr, 4), "n_events": int(mask.sum())}
    except Exception as exc:
        return {"ic": None, "error": str(exc)}


def main() -> None:
    if _is_kill_switch_active():
        logger.info("[SKIP] kill_switch_active — disclosure_event_study halted by policy.")
        return

    args = _parse_args()

    import pandas as pd
    from src.assembled_core.qa.event_study import (
        build_event_window_prices,
        compute_event_returns,
    )

    logger.info("[START] Loading disclosures from %s", args.disclosures)
    disclosures = _load_frame(args.disclosures)

    logger.info("[START] Loading prices from %s", args.prices)
    prices = _load_frame(args.prices)

    # Apply filters
    if args.min_severity and "severity" in disclosures.columns:
        disclosures = disclosures[disclosures["severity"] == args.min_severity]
        logger.info("[OK] Filtered to severity=%s: %d events", args.min_severity, len(disclosures))

    if args.tier and "source_tier" in disclosures.columns:
        disclosures = disclosures[disclosures["source_tier"] == args.tier]
        logger.info("[OK] Filtered to tier=%s: %d events", args.tier, len(disclosures))

    if disclosures.empty:
        logger.warning("[WARN] No disclosure events after filtering — nothing to study.")
        return

    # Rename/normalize columns for event_study
    if "event_type" not in disclosures.columns:
        disclosures = disclosures.copy()
        disclosures["event_type"] = disclosures.get("severity", pd.Series(["disclosure"] * len(disclosures))).fillna("disclosure")

    logger.info("[START] Building event windows (before=%d, after=%d)", args.window_before, args.window_after)
    windows = build_event_window_prices(
        prices,
        disclosures,
        window_before=args.window_before,
        window_after=args.window_after,
    )

    if windows.empty:
        logger.warning("[WARN] No event windows built — check timestamp alignment.")
        return

    returns_df = compute_event_returns(windows)
    ic_result = _compute_ic(returns_df, args.window_after)

    # Group IC by event_type
    group_ic: dict[str, dict] = {}
    for etype, grp in returns_df.groupby("event_type"):
        sub_windows = grp
        sub_ic = _compute_ic(sub_windows, args.window_after)
        group_ic[str(etype)] = sub_ic

    report = {
        "schema_version": "disclosure_event_study.v1",
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "params": {
            "disclosures": args.disclosures,
            "prices": args.prices,
            "window_before": args.window_before,
            "window_after": args.window_after,
            "min_severity": args.min_severity,
            "tier": args.tier,
        },
        "n_events": len(disclosures),
        "n_windows": len(windows),
        "overall_ic": ic_result,
        "ic_by_event_type": group_ic,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"event_study_{run_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("[OK] Event study report written: %s", out_path)
    logger.info("[OK] Overall IC: %s (n=%s)", ic_result.get("ic"), ic_result.get("n_events"))


if __name__ == "__main__":
    main()
