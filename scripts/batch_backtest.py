#!/usr/bin/env python
"""
Batch runner for strategy backtests (P4 – Batch Runner & Parallelisierung).

This script executes multiple backtests defined in a YAML/JSON configuration
file. It currently runs backtests seriell (max_workers is validated but not
yet used for echte Parallelisierung), mit sauberer Output-Struktur und
Batch-Summary.

Design-Referenz:
- docs/BATCH_BACKTEST_P4_DESIGN.md

Wichtige Eigenschaften:
- Nur lokale Daten (data_source="local").
- Nutzt bestehendes Backtest-Script `scripts/run_backtest_strategy.py`.
- Kein Eingriff in Finanzlogik, nur Orchestrierung und Logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess
import datetime as dt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SingleRunConfig:
    """Configuration for a single backtest run in a batch."""

    id: str
    bundle_path: Path
    start_date: str
    end_date: str

    # Optional overrides / parameters
    freq: str = "1d"
    data_source: str = "local"
    strategy: str = "multifactor_long_short"
    rebalance_freq: str = "M"
    max_gross_exposure: float = 1.0
    start_capital: float = 100000.0
    generate_report: bool = True
    generate_risk_report: bool = False
    generate_tca_report: bool = False
    symbols_file: Optional[Path] = None
    universe: Optional[Path] = None

    # Raw extras (for future extension; not passed to CLI directly by default)
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for an entire batch of backtests."""

    batch_name: str
    description: str
    output_root: Path
    runs: List[SingleRunConfig]


@dataclass
class SingleRunResult:
    """Result metadata for a single backtest run."""

    run_id: str
    status: str
    backtest_dir: Optional[Path]
    runtime_sec: float
    exit_code: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with optional PyYAML dependency."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - simple import guard
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install via 'pip install pyyaml' or use JSON config instead."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_batch_config(path: Path) -> BatchConfig:
    """
    Load and validate batch config from YAML/JSON.

    Args:
        path: Path to YAML/JSON config file.

    Returns:
        Parsed BatchConfig instance.
    """
    if not path.exists():
        raise FileNotFoundError(f"Batch config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        raw = _load_yaml(path)
    elif suffix == ".json":
        raw = _load_json(path)
    else:
        raise ValueError(f"Unsupported config file extension: {path.suffix}")

    if not isinstance(raw, dict):
        raise ValueError("Batch config root must be a mapping/object")

    batch_name = str(raw.get("batch_name") or "").strip()
    if not batch_name:
        raise ValueError("batch_name must be set in batch config")

    description = str(raw.get("description") or "").strip()
    if not description:
        description = f"Batch backtests for {batch_name}"

    output_root_raw = raw.get("output_root") or "output/backtests/batch"
    output_root = (ROOT / output_root_raw).resolve()

    defaults: Dict[str, Any] = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping/object if provided")

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("runs must be a non-empty list in batch config")

    runs: List[SingleRunConfig] = []

    for idx, item in enumerate(runs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"runs[{idx}] must be an object")

        merged: Dict[str, Any] = {}
        merged.update(defaults)
        merged.update(item)

        run_id = str(merged.get("id") or "").strip()
        if not run_id:
            raise ValueError(f"runs[{idx}]: id must be set")

        bundle_path_raw = merged.get("bundle_path")
        if not bundle_path_raw:
            raise ValueError(f"runs[{idx}]: bundle_path must be set")

        start_date = str(merged.get("start_date") or "").strip()
        end_date = str(merged.get("end_date") or "").strip()
        if not start_date or not end_date:
            raise ValueError(f"runs[{idx}]: start_date and end_date must be set")

        freq = str(merged.get("freq") or "1d")
        data_source = str(merged.get("data_source") or "local")
        if data_source != "local":
            raise ValueError(
                f"runs[{idx}]: data_source must be 'local' to comply with offline-only workflow"
            )

        strategy = str(merged.get("strategy") or "multifactor_long_short")
        rebalance_freq = str(merged.get("rebalance_freq") or "M")
        max_gross_exposure = float(merged.get("max_gross_exposure") or 1.0)
        start_capital = float(merged.get("start_capital") or 100000.0)
        generate_report = bool(merged.get("generate_report", True))
        generate_risk_report = bool(merged.get("generate_risk_report", False))
        generate_tca_report = bool(merged.get("generate_tca_report", False))

        symbols_file_raw = merged.get("symbols_file")
        universe_raw = merged.get("universe")

        symbols_file = (
            (ROOT / symbols_file_raw).resolve() if symbols_file_raw else None
        )
        universe = (ROOT / universe_raw).resolve() if universe_raw else None

        bundle_path = (ROOT / bundle_path_raw).resolve()

        extra_args = dict(merged.get("extra_args") or {})

        run_cfg = SingleRunConfig(
            id=run_id,
            bundle_path=bundle_path,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            data_source=data_source,
            strategy=strategy,
            rebalance_freq=rebalance_freq,
            max_gross_exposure=max_gross_exposure,
            start_capital=start_capital,
            generate_report=generate_report,
            generate_risk_report=generate_risk_report,
            generate_tca_report=generate_tca_report,
            symbols_file=symbols_file,
            universe=universe,
            extra_args=extra_args,
        )
        runs.append(run_cfg)

    return BatchConfig(
        batch_name=batch_name,
        description=description,
        output_root=output_root,
        runs=runs,
    )


# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_backtest_command(
    run_cfg: SingleRunConfig,
    run_output_dir: Path,
) -> List[str]:
    """
    Build subprocess command for a single backtest run.

    Uses scripts/run_backtest_strategy.py directly to avoid duplicating logic.
    """
    script_path = ROOT / "scripts" / "run_backtest_strategy.py"

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--freq",
        run_cfg.freq,
        "--strategy",
        run_cfg.strategy,
        "--start-date",
        run_cfg.start_date,
        "--end-date",
        run_cfg.end_date,
        "--data-source",
        run_cfg.data_source,
        "--start-capital",
        str(run_cfg.start_capital),
        "--rebalance-freq",
        run_cfg.rebalance_freq,
        "--max-gross-exposure",
        str(run_cfg.max_gross_exposure),
        "--bundle-path",
        str(run_cfg.bundle_path),
        "--out",
        str(run_output_dir),
    ]

    if run_cfg.symbols_file is not None:
        cmd.extend(["--symbols-file", str(run_cfg.symbols_file)])
    elif run_cfg.universe is not None:
        cmd.extend(["--universe", str(run_cfg.universe)])

    if run_cfg.generate_report:
        cmd.append("--generate-report")

    # Extra args (simple key/value pairs) – conservative mapping
    for key, value in run_cfg.extra_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def run_single_backtest(
    run_cfg: SingleRunConfig,
    base_output_dir: Path,
    dry_run: bool = False,
) -> SingleRunResult:
    """
    Execute a single backtest run using run_backtest_strategy.py.

    Args:
        run_cfg: Single run configuration.
        base_output_dir: Base directory for batch outputs.
        dry_run: If True, only log the command without executing.

    Returns:
        SingleRunResult with status and runtime.
    """
    run_dir = _ensure_dir(base_output_dir / run_cfg.id)
    backtest_output_dir = run_dir / "backtest"
    _ensure_dir(backtest_output_dir)

    cmd = _build_backtest_command(run_cfg, backtest_output_dir)
    logger.info("Running backtest run_id=%s", run_cfg.id)
    logger.info("  Output dir: %s", backtest_output_dir)
    logger.info("  Command: %s", " ".join(str(c) for c in cmd))

    if dry_run:
        logger.info("Dry-run mode enabled – command not executed.")
        return SingleRunResult(
            run_id=run_cfg.id,
            status="skipped",
            backtest_dir=None,
            runtime_sec=0.0,
            exit_code=0,
            error=None,
        )

    start = dt.datetime.utcnow()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=False,
        )
        end = dt.datetime.utcnow()
        runtime_sec = (end - start).total_seconds()
        status = "success" if proc.returncode == 0 else "failed"
        error: Optional[str] = None
        if proc.returncode != 0:
            error = f"Backtest exited with code {proc.returncode}"
            logger.warning("Run %s failed: %s", run_cfg.id, error)
        else:
            logger.info("Run %s completed successfully in %.2f sec", run_cfg.id, runtime_sec)

        return SingleRunResult(
            run_id=run_cfg.id,
            status=status,
            backtest_dir=backtest_output_dir,
            runtime_sec=runtime_sec,
            exit_code=proc.returncode,
            error=error,
        )
    except Exception as exc:  # pragma: no cover - defensive
        end = dt.datetime.utcnow()
        runtime_sec = (end - start).total_seconds()
        logger.error("Run %s raised an exception: %s", run_cfg.id, exc, exc_info=True)
        return SingleRunResult(
            run_id=run_cfg.id,
            status="failed",
            backtest_dir=backtest_output_dir,
            runtime_sec=runtime_sec,
            exit_code=1,
            error=str(exc),
        )


def run_batch(
    batch_cfg: BatchConfig,
    max_workers: int = 1,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> List[SingleRunResult]:
    """
    Execute all runs in the batch (currently serial execution).

    Args:
        batch_cfg: Parsed batch configuration.
        max_workers: Maximum number of parallel workers (currently only validated).
        dry_run: If True, do not actually execute backtests.
        fail_fast: If True, abort batch on first failure.

    Returns:
        List of SingleRunResult.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    batch_output_root = _ensure_dir(batch_cfg.output_root / batch_cfg.batch_name)
    runs_output_root = _ensure_dir(batch_output_root / "runs")

    logger.info("=" * 60)
    logger.info("Batch Backtests")
    logger.info("=" * 60)
    logger.info("Batch name: %s", batch_cfg.batch_name)
    logger.info("Description: %s", batch_cfg.description)
    logger.info("Output root: %s", batch_output_root)
    logger.info("Runs: %d", len(batch_cfg.runs))
    logger.info("max_workers (planned): %d", max_workers)
    if dry_run:
        logger.info("Dry-run mode: commands will not be executed.")
    logger.info("")

    results: List[SingleRunResult] = []

    for run_cfg in batch_cfg.runs:
        result = run_single_backtest(
            run_cfg=run_cfg,
            base_output_dir=runs_output_root,
            dry_run=dry_run,
        )
        results.append(result)

        if fail_fast and result.status == "failed":
            logger.warning("Fail-fast enabled and run %s failed – aborting batch.", run_cfg.id)
            break

    # Write detailed summary CSV
    summary_csv = batch_output_root / "batch_summary.csv"
    try:
        import csv

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "strategy",
                    "bundle_path",
                    "start_date",
                    "end_date",
                    "status",
                    "exit_code",
                    "runtime_sec",
                    "backtest_dir",
                    "equity_curve_path",
                    "performance_report_path",
                    "error",
                ]
            )
            for cfg, r in zip(batch_cfg.runs, results):
                backtest_dir = r.backtest_dir or (runs_output_root / cfg.id / "backtest")
                equity_curve_path = backtest_dir / "equity_curve.parquet"
                performance_report_path = backtest_dir / f"performance_report_{cfg.freq}.md"
                writer.writerow(
                    [
                        r.run_id,
                        cfg.strategy,
                        str(cfg.bundle_path),
                        cfg.start_date,
                        cfg.end_date,
                        r.status,
                        r.exit_code,
                        f"{r.runtime_sec:.3f}",
                        str(backtest_dir),
                        str(equity_curve_path),
                        str(performance_report_path),
                        r.error or "",
                    ]
                )
        logger.info("Batch summary written to %s", summary_csv)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write batch summary CSV: %s", exc, exc_info=True)

    # Write optional Markdown summary
    summary_md = batch_output_root / "batch_summary.md"
    try:
        with summary_md.open("w", encoding="utf-8") as f_md:
            f_md.write(f"# Batch Summary: {batch_cfg.batch_name}\n\n")
            f_md.write(f"**Description:** {batch_cfg.description}\n\n")
            f_md.write("| run_id | strategy | bundle | start_date | end_date | status | runtime_sec |\n")
            f_md.write("|--------|----------|--------|------------|----------|--------|-------------|\n")
            for cfg, r in zip(batch_cfg.runs, results):
                f_md.write(
                    f"| {r.run_id} | {cfg.strategy} | {cfg.bundle_path.name} | "
                    f"{cfg.start_date} | {cfg.end_date} | {r.status} | {r.runtime_sec:.3f} |\n"
                )
        logger.info("Batch Markdown summary written to %s", summary_md)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write batch Markdown summary: %s", exc, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(verbosity: int) -> None:
    """Setup basic logging configuration."""
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for batch backtests."""
    parser = argparse.ArgumentParser(
        description="Batch runner for strategy backtests (P4 – Batch Runner & Parallelisierung)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/batch_backtest.py --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml

  python scripts/batch_backtest.py --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.json --dry-run

Notes:
  - Currently runs backtests serially (max_workers is validated but not yet used).
  - All backtests must use data_source="local" (no live APIs).
        """,
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to batch config file (YAML or JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output root directory for batch (default: from config.output_root).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of workers (currently only validated, execution is serial).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort batch on first failed run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing backtests.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for batch backtest runner."""
    args = parse_args(argv)
    _setup_logging(args.verbose)

    try:
        batch_cfg = load_batch_config(args.config_file)
    except Exception as exc:
        logger.error("Failed to load batch config: %s", exc, exc_info=True)
        return 1

    if args.output_dir is not None:
        batch_cfg.output_root = args.output_dir.resolve()

    try:
        results = run_batch(
            batch_cfg=batch_cfg,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Batch execution failed: %s", exc, exc_info=True)
        return 1

    # Determine exit code: 0 if at least one success, 1 otherwise
    any_success = any(r.status == "success" for r in results)
    if not results:
        return 1
    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())


