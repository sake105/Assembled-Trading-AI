"""CLI: Monte-Carlo Stress-Test auf Portfolio-Returns.

Verwendung:
    python scripts/analysis/run_stress_test.py \\
        --returns output/backtests/baseline_returns.csv \\
        --out output/ops/stress_test_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_returns(path: Path) -> pd.Series:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix == ".json":
        df = pd.read_json(path, orient="records")
    else:
        raise ValueError(f"Unbekanntes Format: {path.suffix}")

    if "returns" not in df.columns:
        raise ValueError(f"'returns' Spalte fehlt in {path}")
    return df["returns"].dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte-Carlo Stress-Test")
    parser.add_argument("--returns", type=Path, required=True, help="Path to portfolio returns (CSV/Parquet/JSON)")
    parser.add_argument(
        "--portfolio-returns", type=Path, default=None,
        help="Optional: per-asset returns DataFrame for correlation scenarios",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(f"output/ops/stress_test_{pd.Timestamp.now().strftime('%Y%m%d')}.json"),
    )
    args = parser.parse_args()

    if not args.returns.exists():
        logger.error("Returns nicht gefunden: %s", args.returns)
        return 1

    try:
        returns = _load_returns(args.returns)
    except Exception as exc:
        logger.error("Load failed: %s", exc)
        return 1

    portfolio_df = None
    if args.portfolio_returns and args.portfolio_returns.exists():
        portfolio_df = pd.read_parquet(args.portfolio_returns)

    try:
        from src.assembled_core.qa.scenario_simulator import run_stress_test
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    report = run_stress_test(
        baseline_returns=returns,
        portfolio_returns=portfolio_df,
    )

    logger.info("=" * 60)
    logger.info("BASELINE:")
    for k, v in report.baseline_metrics.items():
        logger.info("  %s: %s", k, v)

    logger.info("=" * 60)
    logger.info("SCENARIOS:")
    for sc in report.scenarios:
        logger.info(
            "  %s: CVaR95=%.4f MaxDD=%.4f σ=%.4f",
            sc.scenario_name, sc.cvar_95, sc.max_drawdown, sc.std_return,
        )

    logger.info("=" * 60)
    logger.info("WORST: %s (CVaR=%.4f)", report.worst_scenario, report.worst_cvar)

    report_dict = {
        "baseline_metrics": report.baseline_metrics,
        "scenarios": [vars(s) for s in report.scenarios],
        "worst_scenario": report.worst_scenario,
        "worst_cvar": report.worst_cvar,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
    logger.info("[OK] Saved: %s", args.out)
    try:
        from src.assembled_core.ops.report_retention import purge_old_dated_reports
        purge_old_dated_reports(args.out.parent, "stress_test_", ".json", keep_last_n=60)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
