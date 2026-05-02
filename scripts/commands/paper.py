# scripts/commands/paper.py
"""Paper trading subcommands."""
from __future__ import annotations

import argparse
from datetime import timezone
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_config import generate_run_id, setup_logging


def run_paper_daily_subcommand(args: argparse.Namespace) -> int:
    """Run a single trading_cycle in shadow/paper mode and write KPI artifacts."""
    from src.assembled_core.data.prices_ingest import load_eod_prices
    from src.assembled_core.ops.paper_runner import run_paper_daily_one

    run_id = generate_run_id(prefix="paper")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    mode = args.mode
    logger.info("=" * 60)
    logger.info("Paper/Shadow Harness (run_paper_daily)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 60)

    if args.as_of:
        as_of_ts = pd.to_datetime(args.as_of, utc=True)
    else:
        as_of_ts = pd.Timestamp.now(timezone.utc).normalize()

    if args.output is not None:
        output_dir = args.output
    else:
        date_str = as_of_ts.date().isoformat()
        output_dir = ROOT / "output" / "runs" / date_str

    app_cfg_path = ROOT / "configs" / "app.yaml"
    app_cfg = {}
    if app_cfg_path.exists():
        try:
            import yaml

            with open(app_cfg_path, "r", encoding="utf-8") as f:
                app_cfg = yaml.safe_load(f) or {}
        except Exception as _exc:
            logger.warning("[paper] Failed to load app.yaml: %s", _exc)
    if args.include_news_pipeline:
        logger.info(
            "include_news_pipeline=True, but NEWS pipeline integration is not wired in v1. Skipping."
        )

    try:
        prices = load_eod_prices(freq="1d")
    except Exception as e:
        logger.error(f"Failed to load EOD prices: {e}")
        return 1

    exit_code, _ = run_paper_daily_one(
        as_of_ts, output_dir, mode, app_cfg, prices, root=ROOT
    )
    logger.info(f"KPI and OPS artifacts written to {output_dir}")
    return exit_code


def run_paper_range_subcommand(args: argparse.Namespace) -> int:
    """Run paper/shadow daily for each trading day in [start, end] and write summary."""
    from src.assembled_core.data.prices_ingest import load_eod_prices
    from src.assembled_core.ops.paper_runner import run_paper_daily_one
    from src.assembled_core.ops.paper_summary import (
        build_paper_summary,
        write_paper_summary,
    )

    run_id = generate_run_id(prefix="paper_range")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    mode = getattr(args, "mode", "paper")
    start_str = getattr(args, "start", "")
    end_str = getattr(args, "end", "")
    output_root = getattr(args, "output_root", None) or ROOT / "output" / "runs"

    logger.info("Paper range runner: start=%s end=%s mode=%s", start_str, end_str, mode)

    try:
        start_ts = pd.to_datetime(start_str, utc=True).normalize()
        end_ts = pd.to_datetime(end_str, utc=True).normalize()
    except Exception as e:
        logger.error("Invalid start/end dates: %s", e)
        return 1
    if start_ts > end_ts:
        logger.error("start must be <= end")
        return 1

    try:
        prices = load_eod_prices(freq="1d")
    except Exception as e:
        logger.error("Failed to load EOD prices: %s", e)
        return 1
    if prices.empty or "timestamp" not in prices.columns:
        logger.error("No price data")
        return 1

    ts = pd.to_datetime(prices["timestamp"], utc=True)
    dates_sorted = sorted(ts.dt.normalize().dt.date.unique())
    start_date = start_ts.date()
    end_date = end_ts.date()
    trading_dates = [d for d in dates_sorted if start_date <= d <= end_date]
    date_strs = [d.isoformat() for d in trading_dates]

    app_cfg_path = ROOT / "configs" / "app.yaml"
    app_cfg = {}
    if app_cfg_path.exists():
        try:
            import yaml

            with open(app_cfg_path, "r", encoding="utf-8") as f:
                app_cfg = yaml.safe_load(f) or {}
        except Exception as _exc:
            logger.warning("[paper] Failed to load app.yaml: %s", _exc)

    for i, d in enumerate(trading_dates):
        date_str = date_strs[i]
        day_ts = pd.Timestamp(d, tz="UTC")
        out_dir = Path(output_root) / date_str
        logger.info(
            "Run paper daily for %s (%d/%d)", date_str, i + 1, len(trading_dates)
        )
        exit_code, _ = run_paper_daily_one(
            day_ts, out_dir, mode, app_cfg, prices, root=ROOT
        )
        if exit_code != 0:
            logger.warning(
                "run_paper_daily failed for %s (exit_code=%d)", date_str, exit_code
            )

    summary = build_paper_summary(output_root, date_strs)
    path = write_paper_summary(output_root, start_str, end_str, summary)
    logger.info("Summary written to %s", path)
    return 0


def run_paper_experiment_subcommand(args: argparse.Namespace) -> int:
    """Run A/B paper experiment with policy overrides."""
    import json
    from pathlib import Path

    from src.assembled_core.ops.experiment_runner import run_experiment

    setup_logging(run_id=generate_run_id(prefix="paper_exp"), level="INFO")
    logger = logging.getLogger(__name__)

    name = getattr(args, "name", "").strip()
    start = getattr(args, "start", "")
    end = getattr(args, "end", "")
    mode = getattr(args, "mode", "paper")
    output_root = getattr(args, "output_root", None) or ROOT / "output" / "runs"
    overrides_raw = getattr(args, "overrides", None) or "{}"
    app_overrides_raw = getattr(args, "app_overrides", None) or "{}"
    if not name:
        logger.error("--name is required")
        return 1
    if not start or not end:
        logger.error("--start and --end are required")
        return 1
    overrides = {}
    if overrides_raw:
        overrides_path = Path(overrides_raw)
        if overrides_path.exists():
            try:
                text = overrides_path.read_text(encoding="utf-8")
                if overrides_path.suffix.lower() in (".yaml", ".yml"):
                    import yaml

                    overrides = yaml.safe_load(text) or {}
                else:
                    overrides = json.loads(text)
            except Exception as e:
                logger.error("Failed to load overrides from file: %s", e)
                return 1
        else:
            try:
                overrides = json.loads(overrides_raw)
            except Exception as e:
                logger.error("Invalid --overrides JSON: %s", e)
                return 1
    if not isinstance(overrides, dict):
        overrides = {}

    app_overrides: dict = {}
    if app_overrides_raw:
        app_path = Path(app_overrides_raw)
        if app_path.exists():
            try:
                text = app_path.read_text(encoding="utf-8")
                if app_path.suffix.lower() in (".yaml", ".yml"):
                    import yaml

                    app_overrides = yaml.safe_load(text) or {}
                else:
                    app_overrides = json.loads(text)
            except Exception as e:
                logger.error("Failed to load app-overrides from file: %s", e)
                return 1
        else:
            try:
                app_overrides = json.loads(app_overrides_raw)
            except Exception as e:
                logger.error("Invalid --app-overrides JSON: %s", e)
                return 1
    if not isinstance(app_overrides, dict):
        app_overrides = {}

    try:
        experiment_root = run_experiment(
            name=name,
            start_date=start,
            end_date=end,
            mode=mode,
            output_root=output_root,
            policy_overrides=overrides,
            app_overrides=app_overrides,
            root=ROOT,
        )
        logger.info("Experiment summary at %s", experiment_root / "summary.json")
        return 0
    except Exception as e:
        logger.exception("Experiment failed: %s", e)
        return 1


def compare_paper_experiments_subcommand(args: argparse.Namespace) -> int:
    """Compare two paper experiment summaries (A vs B) and write compare.json."""
    import json

    from src.assembled_core.ops.compare import compare_summaries

    logger = logging.getLogger(__name__)

    name_a = getattr(args, "a", "").strip()
    name_b = getattr(args, "b", "").strip()
    output_root = getattr(args, "output_root", None) or ROOT / "output" / "runs"
    output_root = Path(output_root)
    if not name_a or not name_b:
        logger.error("--a and --b (experiment names) are required")
        return 1
    path_a = output_root / "_experiments" / name_a / "summary.json"
    path_b = output_root / "_experiments" / name_b / "summary.json"
    try:
        result = compare_summaries(path_a, path_b)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    out_path = output_root / "_experiments" / f"compare_{name_a}_vs_{name_b}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info("Compare report written to %s", out_path)
    return 0


def summarize_intel_activity_subcommand(args: argparse.Namespace) -> int:
    """Build intel activity summary for an experiment; write intel_activity_summary.json."""
    import json

    from src.assembled_core.ops.intel_activity_summary import (
        build_intel_activity_summary,
    )

    logger = logging.getLogger(__name__)

    name = getattr(args, "experiment", "").strip()
    output_root = getattr(args, "output_root", None) or ROOT / "output" / "runs"
    output_root = Path(output_root)
    if not name:
        logger.error("--experiment is required")
        return 1
    experiment_root = output_root / "_experiments" / name
    runs_root = experiment_root / "runs"
    if not runs_root.exists():
        logger.error("Experiment runs dir not found: %s", runs_root)
        return 1
    summary = build_intel_activity_summary(
        runs_root, intel_output_root=output_root.parent / "intel"
    )
    out_path = experiment_root / "intel_activity_summary.json"
    experiment_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info("Intel activity summary written to %s", out_path)
    return 0


def inspect_eod_range_subcommand(args: argparse.Namespace) -> int:
    """Inspect EOD price coverage; print min/max and optional JSON report."""
    from src.assembled_core.data.prices_ingest import load_eod_prices
    from src.assembled_core.ops.inspect_data import inspect_eod_prices

    logger = logging.getLogger(__name__)

    freq = getattr(args, "freq", "1d")
    write_json = (getattr(args, "write_json", "false") or "").strip().lower() == "true"
    output_root = getattr(args, "output_root", None) or ROOT / "output" / "runs"
    output_root = Path(output_root)

    try:
        prices = load_eod_prices(freq=freq)
    except Exception as e:
        logger.error("Failed to load EOD prices: %s", e)
        return 1
    report = inspect_eod_prices(prices)
    n_rows = report.get("n_rows", 0)
    n_symbols = report.get("n_symbols")
    n_unique_days = report.get("n_unique_days", 0)
    min_utc = report.get("min_utc")
    max_utc = report.get("max_utc")
    last_30 = report.get("last_30_trading_days")
    last_90 = report.get("last_90_trading_days")
    print("EOD coverage (freq=%s):" % freq)
    print("  rows: %d" % n_rows)
    print("  symbols: %s" % (n_symbols if n_symbols is not None else "n/a"))
    print("  unique_days: %d" % n_unique_days)
    print("  min_utc: %s" % (min_utc or "n/a"))
    print("  max_utc: %s" % (max_utc or "n/a"))
    if last_30:
        print(
            "  last_30_trading_days: start=%s end=%s"
            % (last_30["start"], last_30["end"])
        )
    if last_90:
        print(
            "  last_90_trading_days: start=%s end=%s"
            % (last_90["start"], last_90["end"])
        )
    if write_json:
        summaries_dir = output_root / "_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        out_path = summaries_dir / "eod_coverage.json"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(
            __import__("json").dumps(report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)
        logger.info("EOD coverage written to %s", out_path)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register paper trading subcommands."""
    # run_paper_daily
    paper_daily_parser = subparsers.add_parser(
        "run_paper_daily",
        help="Run paper/shadow trading cycle once and write KPI artifacts",
        description=(
            "Runs a single trading_cycle in 'shadow' or 'paper' mode for a given as_of date, "
            "and writes KPI artifacts explaining risk overlays and gates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cli.py run_paper_daily --mode shadow\n"
            "  python scripts/cli.py run_paper_daily --mode paper --as-of 2026-03-04 "
            "--output output/runs/2026-03-04\n"
        ),
    )
    paper_daily_parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "paper"],
        default="shadow",
        help="Runner mode: 'shadow' (no additional execution) or 'paper' (reserved for future ledger sim).",
    )
    paper_daily_parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Optional as_of timestamp (YYYY-MM-DD or full ISO). Default: current UTC date.",
    )
    paper_daily_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for KPI artifacts (default: output/runs/<YYYY-MM-DD>).",
    )
    paper_daily_parser.add_argument(
        "--include-news-pipeline",
        action="store_true",
        default=False,
        help="If set, will attempt to run the NEWS pipeline before trading_cycle (v1: placeholder, logs only).",
    )
    paper_daily_parser.set_defaults(func=run_paper_daily_subcommand)

    # run_paper_range
    paper_range_parser = subparsers.add_parser(
        "run_paper_range",
        help="Run paper/shadow daily for a date range and write summary",
        description=(
            "Runs run_paper_daily for each trading day in [start, end], "
            "then writes a summary JSON to output_root/_summaries/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cli.py run_paper_range --start 2026-02-01 --end 2026-03-01 --mode paper\n"
        ),
    )
    paper_range_parser.add_argument(
        "--start",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive).",
    )
    paper_range_parser.add_argument(
        "--end",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="End date (inclusive).",
    )
    paper_range_parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "paper"],
        default="paper",
        help="Runner mode (default: paper).",
    )
    paper_range_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output root for run dirs (default: output/runs).",
    )
    paper_range_parser.add_argument(
        "--include-news-pipeline",
        action="store_true",
        default=False,
        help="Include NEWS pipeline (v1: placeholder, no-op).",
    )
    paper_range_parser.set_defaults(func=run_paper_range_subcommand)

    # run_paper_experiment
    paper_exp_parser = subparsers.add_parser(
        "run_paper_experiment",
        help="Run A/B paper experiment with policy overrides",
        description="Runs paper range for [start,end] with merged policy overrides; writes summary to output_root/_experiments/<name>/.",
    )
    paper_exp_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name (e.g. baseline, treatment).",
    )
    paper_exp_parser.add_argument(
        "--start",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive).",
    )
    paper_exp_parser.add_argument(
        "--end",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="End date (inclusive).",
    )
    paper_exp_parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "paper"],
        default="paper",
        help="Runner mode.",
    )
    paper_exp_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: output/runs).",
    )
    paper_exp_parser.add_argument(
        "--overrides",
        type=str,
        default="",
        help='Policy overrides as JSON string or path to .json/.yaml file (e.g. \'{"risk_state_machine":{"hysteresis":{"require_disclosures_confirm":true}}}\').',
    )
    paper_exp_parser.add_argument(
        "--app-overrides",
        type=str,
        default="",
        help='App config overrides as JSON string or path to .json/.yaml (e.g. \'{"paper_runner":{"intel":{"mode":"real"}}}\'). Enables intel mode none|sim|real without editing configs/app.yaml.',
    )
    paper_exp_parser.set_defaults(func=run_paper_experiment_subcommand)

    # compare_paper_experiments
    compare_exp_parser = subparsers.add_parser(
        "compare_paper_experiments",
        help="Compare two paper experiment summaries (A vs B)",
        description="Compares summary.json of two experiments and writes compare_<a>_vs_<b>.json.",
    )
    compare_exp_parser.add_argument(
        "--a",
        type=str,
        required=True,
        metavar="NAME",
        help="First experiment name (baseline).",
    )
    compare_exp_parser.add_argument(
        "--b",
        type=str,
        required=True,
        metavar="NAME",
        help="Second experiment name (treatment).",
    )
    compare_exp_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root where _experiments/ lives (default: output/runs).",
    )
    compare_exp_parser.set_defaults(func=compare_paper_experiments_subcommand)

    # summarize_intel_activity
    intel_activity_parser = subparsers.add_parser(
        "summarize_intel_activity",
        help="Build intel activity summary for an experiment",
        description="Reads run_kpis.json per day under the experiment runs/ and writes intel_activity_summary.json.",
    )
    intel_activity_parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        metavar="NAME",
        help="Experiment name (e.g. real_gate_off).",
    )
    intel_activity_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root where _experiments/ lives (default: output/runs).",
    )
    intel_activity_parser.set_defaults(func=summarize_intel_activity_subcommand)

    # inspect_eod_range
    inspect_eod_parser = subparsers.add_parser(
        "inspect_eod_range",
        help="Inspect EOD price coverage and get recommended experiment start/end",
        description="Loads EOD prices (same as paper runner), prints min/max and recommends last_30 / last_90 trading-day windows.",
    )
    inspect_eod_parser.add_argument(
        "--freq",
        type=str,
        default="1d",
        choices=["1d", "5min"],
        help="Price frequency (default: 1d).",
    )
    inspect_eod_parser.add_argument(
        "--write-json",
        type=str,
        default="false",
        choices=["true", "false"],
        help="If true, write output_root/_summaries/eod_coverage.json (default: false).",
    )
    inspect_eod_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for --write-json (default: output/runs).",
    )
    inspect_eod_parser.set_defaults(func=inspect_eod_range_subcommand)
