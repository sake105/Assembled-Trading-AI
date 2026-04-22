"""CLI: Compare N Backtests gegeneinander.

Liest mehrere Backtest-Returns-Files (JSON/CSV/Parquet) und gibt Vergleichs-Report aus.

Verwendung:
    python scripts/analysis/compare_backtests.py \\
        --strategy "baseline:output/backtests/baseline_returns.csv" \\
        --strategy "v2:output/backtests/v2_returns.csv" \\
        --strategy "meta:output/backtests/meta_returns.csv" \\
        --out output/ops/backtest_comparison.json

Return-Files: jede Datei braucht eine 'returns' Spalte und optional 'timestamp' Index.
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

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df["returns"].dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Strategy Backtest Comparison")
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help="name:path (z.B. 'baseline:output/backtests/bl.csv'). Mehrfach verwendbar.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/ops/backtest_comparison.json"),
    )
    args = parser.parse_args()

    strategies: dict[str, pd.Series] = {}
    for entry in args.strategy:
        if ":" not in entry:
            logger.error("Ungültiges --strategy Format (name:path): %s", entry)
            return 1
        name, path_str = entry.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            logger.error("Datei nicht gefunden: %s", path)
            return 1
        try:
            strategies[name.strip()] = _load_returns(path)
            logger.info("Loaded %s: n=%d", name, len(strategies[name.strip()]))
        except Exception as exc:
            logger.error("Load failed für %s: %s", path, exc)
            return 1

    if len(strategies) < 2:
        logger.error("Mindestens 2 Strategien erforderlich")
        return 1

    try:
        from src.assembled_core.qa.backtest_comparison import compare_backtests
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    report = compare_backtests(strategies)

    logger.info("=" * 60)
    logger.info("RANKING:")
    for rank, (name, sharpe) in enumerate(report.ranking, 1):
        logger.info("  %d. %s (Sharpe=%.2f)", rank, name, sharpe)

    logger.info("=" * 60)
    logger.info("PAIRWISE SIGNIFICANCE (Bonferroni-korrigiert):")
    for pc in report.pairwise:
        marker = "***" if pc.bonferroni_pvalue < 0.05 else "   "
        logger.info(
            "  %s %s vs %s: ΔSharpe=%+.2f DM p=%.4f (corr %.4f)",
            marker, pc.strategy_a, pc.strategy_b,
            pc.sharpe_diff, pc.dm_pvalue, pc.bonferroni_pvalue,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("[OK] Saved: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
