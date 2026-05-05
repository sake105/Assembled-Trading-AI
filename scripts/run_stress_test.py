"""Stress-test the backtest strategy across historical crisis windows (Plan 11/10 §2.3).

Reads configs/stress_windows.yaml, runs run_backtest_strategy.py for each window,
aggregates results and emits a stress report to output/stress/aggregate.json.

Usage:
    python scripts/run_stress_test.py [--policy configs/policy.yaml]
    python scripts/run_stress_test.py --windows COVID_2020 Inflation_2022
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _run_window(
    window: dict,
    policy: str,
    out_dir: Path,
    price_file: str | None = None,
    strategy: str = "multifactor_v2",
    bundle_path: str | None = None,
) -> dict | None:
    name = window["name"]
    out_path = out_dir / name
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_backtest_strategy.py",
        "--start-date",
        window["start"],
        "--end-date",
        window["end"],
        "--out",
        str(out_path),
        "--policy",
        policy,
        "--freq",
        "1d",
        "--strategy",
        strategy,
    ]
    if bundle_path:
        cmd += ["--bundle-path", bundle_path]
    if price_file:
        cmd += ["--price-file", price_file]

    logger.info("[stress] %s  %s → %s", name, window["start"], window["end"])
    result = subprocess.run(cmd, capture_output=True, text=True)

    metrics_path = out_path / "reports" / "metrics.json"
    if not metrics_path.exists():
        logger.warning("[stress] %s — metrics.json missing (stdout below)", name)
        logger.warning(result.stdout[-2000:] if result.stdout else "(no stdout)")
        logger.warning(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    # max_drawdown_pct is already a percentage (e.g. -39.1 means -39.1%)
    # convert to fraction for consistency with threshold checks
    mdd_raw = metrics.get("max_drawdown_pct", None)
    if mdd_raw is None:
        mdd_raw = metrics.get("max_drawdown", 0.0)
        # if it's in absolute dollar terms (>1 or <-1), skip using it
        if abs(mdd_raw) > 1:
            mdd_raw = None
    mdd = float(mdd_raw) / 100.0 if mdd_raw is not None else 0.0

    cagr_raw = metrics.get("cagr", None)
    if cagr_raw is not None:
        cagr = float(cagr_raw)
    else:
        # Annualize total_return for short windows where cagr is null
        try:
            import datetime as _dt

            s = _dt.date.fromisoformat(window["start"])
            e = _dt.date.fromisoformat(window["end"])
            n_years = max((e - s).days / 365.25, 1 / 365.25)
            tr = metrics.get("total_return", 0.0) or 0.0
            cagr = float((1 + tr) ** (1 / n_years) - 1)
        except Exception:
            cagr = 0.0

    return {
        "window": name,
        "description": window.get("description", ""),
        "start": window["start"],
        "end": window["end"],
        "cagr": cagr,
        "sharpe": metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)),
        "mdd": mdd,
        "n_trades": metrics.get(
            "n_trades", metrics.get("total_trades", metrics.get("trades", 0))
        ),
        "total_return": metrics.get("total_return", 0.0),
        "worst_day": metrics.get(
            "worst_day_return", metrics.get("min_daily_return", None)
        ),
    }


def _check_thresholds(results: list[dict], thresholds: dict) -> dict:
    checks: dict[str, bool] = {}
    mdds = [r["mdd"] for r in results if r["mdd"] is not None]
    worst_days = [r["worst_day"] for r in results if r.get("worst_day") is not None]

    checks["stress_score_cagr"] = True  # computed below after stress_score
    checks["worst_mdd"] = (
        (min(mdds) >= thresholds.get("worst_mdd_max", -0.25)) if mdds else True
    )
    checks["worst_single_day"] = (
        (min(worst_days) >= thresholds.get("worst_single_day_max", -0.08))
        if worst_days
        else True
    )

    gfc = next((r for r in results if r["window"] == "GFC_2008"), None)
    if gfc:
        checks["gfc_survived"] = gfc["total_return"] >= (
            thresholds.get("gfc_final_equity_min_pct", 0.50) - 1.0
        )

    inflation = next((r for r in results if r["window"] == "Inflation_2022"), None)
    if inflation:
        checks["inflation_2022_mdd"] = inflation["mdd"] >= thresholds.get(
            "inflation_2022_mdd_max", -0.20
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stress tests across crisis windows"
    )
    parser.add_argument("--policy", default="configs/policy.yaml")
    parser.add_argument("--stress-config", default="configs/stress_windows.yaml")
    parser.add_argument("--out-dir", default="output/stress")
    parser.add_argument(
        "--price-file",
        default="data/sample/watchlist_2007_2026.parquet",
        help="Explicit price parquet to pass to backtest",
    )
    parser.add_argument("--windows", nargs="*", help="Subset of window names to run")
    parser.add_argument(
        "--strategy",
        default="multifactor_v2",
        help="Strategy to run for each stress window (default: multifactor_v2)",
    )
    parser.add_argument(
        "--bundle-path",
        default="configs/factor_bundles/ai_tech_core_ml_bundle.yaml",
        help="Factor bundle for multifactor_v2",
    )
    args = parser.parse_args()

    cfg_path = Path(args.stress_config)
    if not cfg_path.exists():
        logger.error("Stress config not found: %s", cfg_path)
        return 1

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    windows = cfg.get("stress_windows", [])
    if args.windows:
        windows = [w for w in windows if w["name"] in args.windows]
        if not windows:
            logger.error("No matching windows found for: %s", args.windows)
            return 1

    thresholds = cfg.get("live_activation_thresholds", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price_file = args.price_file or None
    results = []
    for window in windows:
        r = _run_window(
            window,
            args.policy,
            out_dir,
            price_file=price_file,
            strategy=args.strategy,
            bundle_path=args.bundle_path,
        )
        if r:
            results.append(r)
            logger.info(
                "  → CAGR %.2f%%  Sharpe %.3f  MDD %.2f%%  Trades %d",
                (r["cagr"] or 0) * 100,
                r["sharpe"] or 0,
                (r["mdd"] or 0) * 100,
                r["n_trades"],
            )

    if not results:
        logger.error("No windows produced results")
        return 1

    # Aggregate
    valid_cagrs = [r["cagr"] for r in results if r["cagr"] is not None]
    stress_score = (
        float(np.exp(np.mean([np.log(1 + c) for c in valid_cagrs])) - 1)
        if valid_cagrs
        else 0.0
    )
    worst_mdd = min((r["mdd"] for r in results if r["mdd"] is not None), default=0.0)
    worst_day = min(
        (r["worst_day"] for r in results if r.get("worst_day") is not None),
        default=None,
    )

    threshold_checks = _check_thresholds(results, thresholds)
    threshold_checks["stress_score_cagr"] = stress_score >= thresholds.get(
        "stress_score_cagr_min", 0.0
    )
    all_pass = all(threshold_checks.values())

    summary = {
        "policy": args.policy,
        "windows_run": len(results),
        "windows": results,
        "aggregate": {
            "stress_score_cagr": round(stress_score, 4),
            "worst_mdd": round(worst_mdd, 4),
            "worst_single_day": round(worst_day, 4) if worst_day is not None else None,
        },
        "threshold_checks": threshold_checks,
        "live_activation_verdict": "PASS" if all_pass else "FAIL",
    }

    agg_path = out_dir / "aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=== STRESS TEST AGGREGATE ===")
    logger.info("Stress-Score CAGR (geom mean): %.2f%%", stress_score * 100)
    logger.info("Worst MDD across crises:       %.2f%%", worst_mdd * 100)
    if worst_day is not None:
        logger.info("Worst single day:              %.2f%%", worst_day * 100)
    logger.info("Live-activation verdict:       %s", summary["live_activation_verdict"])
    logger.info("Report: %s", agg_path)

    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
