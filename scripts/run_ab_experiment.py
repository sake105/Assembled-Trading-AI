#!/usr/bin/env python
"""A/B experiment runner for paper track.

Runs the existing paper track runner twice (gate_off vs gate_on) over the
same date range with isolated output directories, then produces per-run
summaries and a compare file.

Usage:
    # Run both arms
    python scripts/run_ab_experiment.py run \\
        --config-file configs/paper_track/trend_baseline.yaml \\
        --start-date 2025-10-16 --end-date 2025-10-17 \\
        --output-root output/experiments/ab_test_1

    # Compare two existing summaries
    python scripts/run_ab_experiment.py compare \\
        --summary-a output/experiments/ab_test_1/gate_off/summary.json \\
        --summary-b output/experiments/ab_test_1/gate_on/summary.json \\
        --output output/experiments/ab_test_1/compare.json

    # Build summary from an existing run
    python scripts/run_ab_experiment.py summarize \\
        --run-dir output/experiments/ab_test_1/gate_off
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary_from_run(run_dir: Path) -> dict[str, Any]:
    """Build a compact summary from a paper track run directory.

    Reads paper_track_run_summary.json and intel_summary.json (if present).
    """
    run_summary_path = run_dir / "paper_track_run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {run_summary_path}")

    with open(run_summary_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    per_day = run_data.get("per_day_statuses", [])
    success_days = [d for d in per_day if d.get("status") == "success"]

    if not success_days:
        return _empty_summary(run_data)

    equities = [d["equity"] for d in success_days if d.get("equity") is not None]
    returns = [d["daily_return_pct"] for d in success_days if d.get("daily_return_pct") is not None]

    start_equity = equities[0] if equities else 0.0
    end_equity = equities[-1] if equities else 0.0
    total_return = (end_equity / start_equity - 1.0) if start_equity > 0 else 0.0

    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= (1.0 + r / 100.0)
        if cumulative > peak:
            peak = cumulative
        dd = (cumulative - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    total_trades = sum(d.get("trades_count", 0) or 0 for d in success_days)

    # Intel metrics
    intel_summary_path = run_dir / "intel_summary.json"
    intel_data: dict[str, Any] | None = None
    if intel_summary_path.exists():
        try:
            with open(intel_summary_path, "r", encoding="utf-8") as f:
                intel_data = json.load(f)
        except Exception:
            intel_data = None

    gate_info = _extract_gate_info(intel_data, run_data)

    return {
        "schema_version": "paper.track.summary.v1",
        "run_name": run_data.get("strategy_name", ""),
        "start_date": run_data.get("date_range", {}).get("start", ""),
        "end_date": run_data.get("date_range", {}).get("end", ""),
        "n_days": len(success_days),
        "n_days_failed": run_data.get("days_failed", 0),
        "n_days_skipped": run_data.get("days_skipped", 0),
        "equity": {
            "start": round(start_equity, 2),
            "end": round(end_equity, 2),
            "total_return": round(total_return, 6),
            "max_drawdown": round(max_dd, 6),
        },
        "trading": {
            "total_trades": total_trades,
        },
        "intel": gate_info,
    }


def _extract_gate_info(
    intel_data: dict[str, Any] | None,
    run_data: dict[str, Any],
) -> dict[str, Any]:
    """Extract gate/intel metrics from available data."""
    result: dict[str, Any] = {
        "mode": "none",
        "avg_multiplier_applied": 1.0,
        "days_active_hint": 0,
        "days_watch_hint": 0,
        "active_pct": 0.0,
        "watch_pct": 1.0,
    }

    orch = run_data.get("intel_orchestration") or {}
    mode = orch.get("mode", "none")
    result["mode"] = mode

    if intel_data:
        orch_i = intel_data.get("intel_orchestration") or {}
        mode = orch_i.get("mode", mode)
        result["mode"] = mode

        gate = (
            intel_data.get("georisk_gate")
            or orch_i.get("georisk_gate")
            or orch.get("georisk_gate")
            or {}
        )
        if gate.get("enabled"):
            mult = float(gate.get("multiplier_applied", 1.0))
            result["avg_multiplier_applied"] = mult
            hint = str(gate.get("state_hint", "WATCH")).upper()
            if hint == "ACTIVE":
                result["days_active_hint"] = 1
                result["active_pct"] = 1.0
                result["watch_pct"] = 0.0
            else:
                result["days_watch_hint"] = 1
        else:
            result["days_watch_hint"] = 1

    return result


def _empty_summary(run_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "paper.track.summary.v1",
        "run_name": run_data.get("strategy_name", ""),
        "start_date": run_data.get("date_range", {}).get("start", ""),
        "end_date": run_data.get("date_range", {}).get("end", ""),
        "n_days": 0,
        "n_days_failed": run_data.get("days_failed", 0),
        "n_days_skipped": run_data.get("days_skipped", 0),
        "equity": {"start": 0, "end": 0, "total_return": 0, "max_drawdown": 0},
        "trading": {"total_trades": 0},
        "intel": {
            "mode": "none", "avg_multiplier_applied": 1.0,
            "days_active_hint": 0, "days_watch_hint": 0,
            "active_pct": 0.0, "watch_pct": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare_summaries(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two run summaries and produce delta metrics."""
    def _g(s: dict, *keys: str, default: float = 0.0) -> float:
        v = s
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, default)
            else:
                return default
        return float(v) if v is not None else default

    a_ret = _g(summary_a, "equity", "total_return")
    b_ret = _g(summary_b, "equity", "total_return")
    a_dd = _g(summary_a, "equity", "max_drawdown")
    b_dd = _g(summary_b, "equity", "max_drawdown")
    a_mult = _g(summary_a, "intel", "avg_multiplier_applied", default=1.0)
    b_mult = _g(summary_b, "intel", "avg_multiplier_applied", default=1.0)
    a_active = _g(summary_a, "intel", "active_pct")
    b_active = _g(summary_b, "intel", "active_pct")
    a_watch = _g(summary_a, "intel", "watch_pct", default=1.0)
    b_watch = _g(summary_b, "intel", "watch_pct", default=1.0)

    return {
        "schema_version": "paper.track.compare.v1",
        "a": {
            "run_name": summary_a.get("run_name", ""),
            "total_return": a_ret,
            "max_drawdown": a_dd,
            "avg_multiplier": a_mult,
            "active_pct": a_active,
            "n_days": summary_a.get("n_days", 0),
            "total_trades": _g(summary_a, "trading", "total_trades"),
        },
        "b": {
            "run_name": summary_b.get("run_name", ""),
            "total_return": b_ret,
            "max_drawdown": b_dd,
            "avg_multiplier": b_mult,
            "active_pct": b_active,
            "n_days": summary_b.get("n_days", 0),
            "total_trades": _g(summary_b, "trading", "total_trades"),
        },
        "delta": {
            "total_return": round(b_ret - a_ret, 8),
            "max_drawdown": round(b_dd - a_dd, 8),
            "avg_multiplier_applied": round(b_mult - a_mult, 4),
            "active_pct": round(b_active - a_active, 4),
            "watch_pct": round(b_watch - a_watch, 4),
            "total_trades": int(_g(summary_b, "trading", "total_trades") - _g(summary_a, "trading", "total_trades")),
        },
    }


# ---------------------------------------------------------------------------
# Run A/B experiment
# ---------------------------------------------------------------------------

def run_ab_experiment(
    config_file: Path,
    start_date: str,
    end_date: str,
    output_root: Path,
    rerun: bool = False,
    active_multiplier: float = 0.70,
    rebalance_filter: bool = False,
    rebalance_min_notional: float = 500.0,
    deadzone: bool = False,
    deadzone_pct: float = 0.05,
) -> int:
    """Run gate_off and gate_on arms, write summaries and compare."""
    from scripts.run_paper_track import run_paper_track_from_cli, load_paper_track_config

    output_root.mkdir(parents=True, exist_ok=True)

    arms = [
        {
            "name": "gate_off",
            "intel_mode": "none",
        },
        {
            "name": "gate_on",
            "intel_mode": "real",
        },
    ]

    summaries: dict[str, dict[str, Any]] = {}

    for arm in arms:
        arm_name = arm["name"]
        arm_dir = output_root / arm_name
        logger.info(f"=== Running arm: {arm_name} ===")

        # Write arm-specific config overlay
        base_config = load_paper_track_config(config_file)
        from dataclasses import replace
        arm_config = replace(
            base_config,
            output_root=arm_dir,
            intel_mode=arm["intel_mode"],
            georisk_gate_enabled=(arm["intel_mode"] == "real"),
            georisk_active_multiplier=active_multiplier,
        )

        # Write arm config to disk so the runner can load it
        arm_config_path = arm_dir / "config_overlay.json"
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_config_dict = {
            "strategy_name": arm_config.strategy_name,
            "strategy_type": arm_config.strategy_type,
            "universe": {"file": str(arm_config.universe_file)},
            "trading": {"freq": arm_config.freq},
            "portfolio": {"seed_capital": arm_config.seed_capital},
            "costs": {
                "commission_bps": arm_config.commission_bps,
                "spread_w": arm_config.spread_w,
                "impact_w": arm_config.impact_w,
            },
            "output": {
                "root": str(arm_dir),
                "strategy_dir": ".",
                "format": arm_config.output_format,
            },
            "intel": {
                "mode": arm["intel_mode"],
                "georisk_gate": {
                    "enabled": arm["intel_mode"] == "real",
                    "active_multiplier": active_multiplier,
                },
                "rebalance_filter": {
                    "enabled": rebalance_filter and arm["intel_mode"] == "real",
                    "min_notional": rebalance_min_notional,
                },
                "deadzone": {
                    "enabled": deadzone and arm["intel_mode"] == "real",
                    "pct": deadzone_pct,
                },
            },
            "strategy": {"params": arm_config.strategy_params},
            "integration": {"enable_pit_checks": arm_config.enable_pit_checks},
        }
        if arm_config.random_seed is not None:
            arm_config_dict["random_seed"] = arm_config.random_seed

        with open(arm_config_path, "w", encoding="utf-8") as f:
            json.dump(arm_config_dict, f, indent=2)

        exit_code = run_paper_track_from_cli(
            config_file=arm_config_path,
            start_date=start_date,
            end_date=end_date,
            rerun=rerun,
            intel_mode=arm["intel_mode"],
        )

        logger.info(f"Arm {arm_name} finished with exit code {exit_code}")

        # Build summary
        try:
            summary = build_summary_from_run(arm_dir)
            summary["run_name"] = arm_name
            summary_path = arm_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary written: {summary_path}")
            summaries[arm_name] = summary
        except Exception as exc:
            logger.warning(f"Failed to build summary for {arm_name}: {exc}")

    # Compare
    if "gate_off" in summaries and "gate_on" in summaries:
        comp = compare_summaries(summaries["gate_off"], summaries["gate_on"])
        comp_path = output_root / "compare_gate_off_vs_gate_on.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        logger.info(f"Compare written: {comp_path}")
        logger.info(
            f"Delta total_return: {comp['delta']['total_return']:.6f}, "
            f"Delta max_drawdown: {comp['delta']['max_drawdown']:.6f}"
        )
    else:
        logger.warning("Could not compare: one or both summaries missing")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B experiment runner for paper track",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run gate_off + gate_on A/B experiment")
    run_p.add_argument("--config-file", type=Path, required=True)
    run_p.add_argument("--start-date", type=str, required=True)
    run_p.add_argument("--end-date", type=str, required=True)
    run_p.add_argument("--output-root", type=Path, required=True)
    run_p.add_argument("--rerun", action="store_true", default=False)
    run_p.add_argument("--active-multiplier", type=float, default=0.70)
    run_p.add_argument("--rebalance-filter", action="store_true", default=False,
                        help="Enable rebalance filter on gate_on arm")
    run_p.add_argument("--rebalance-min-notional", type=float, default=500.0,
                        help="Minimum order notional to keep (default: 500)")
    run_p.add_argument("--deadzone", action="store_true", default=False,
                        help="Enable dead-zone rebalance filter on gate_on arm")
    run_p.add_argument("--deadzone-pct", type=float, default=0.05,
                        help="Dead-zone threshold as fraction (default: 0.05 = 5%%)")

    # summarize
    sum_p = sub.add_parser("summarize", help="Build summary from existing run")
    sum_p.add_argument("--run-dir", type=Path, required=True)

    # compare
    cmp_p = sub.add_parser("compare", help="Compare two summaries")
    cmp_p.add_argument("--summary-a", type=Path, required=True)
    cmp_p.add_argument("--summary-b", type=Path, required=True)
    cmp_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "run":
        code = run_ab_experiment(
            config_file=args.config_file,
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root,
            rerun=args.rerun,
            active_multiplier=args.active_multiplier,
            rebalance_filter=args.rebalance_filter,
            rebalance_min_notional=args.rebalance_min_notional,
            deadzone=args.deadzone,
            deadzone_pct=args.deadzone_pct,
        )
        sys.exit(code)

    elif args.command == "summarize":
        summary = build_summary_from_run(args.run_dir)
        out_path = args.run_dir / "summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    elif args.command == "compare":
        with open(args.summary_a, "r", encoding="utf-8") as f:
            sa = json.load(f)
        with open(args.summary_b, "r", encoding="utf-8") as f:
            sb = json.load(f)
        comp = compare_summaries(sa, sb)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
