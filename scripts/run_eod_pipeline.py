# scripts/run_eod_pipeline.py
"""EOD pipeline orchestration script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.logging_config import generate_run_id, setup_logging
from src.assembled_core.pipeline.orchestrator import run_eod_pipeline

import logging


def parse_eod_args() -> argparse.Namespace:
    """Parse command-line arguments for EOD pipeline.

    Returns:
        Parsed arguments
    """
    # Get default cost model for help text
    default_costs = get_default_cost_model()

    p = argparse.ArgumentParser(
        description="EOD Pipeline Orchestration - Runs full pipeline (execute, backtest, portfolio, QA)"
    )
    p.add_argument(
        "--freq", choices=SUPPORTED_FREQS, required=True, help="Trading frequency"
    )
    p.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000.0)",
    )
    p.add_argument("--skip-backtest", action="store_true", help="Skip backtest step")
    p.add_argument("--skip-portfolio", action="store_true", help="Skip portfolio step")
    p.add_argument("--skip-qa", action="store_true", help="Skip QA step")
    p.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe file (default: watchlist.txt in repo root)",
    )
    p.add_argument(
        "--price-file",
        type=str,
        default=None,
        help="Optional explicit path to price file",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for price data (YYYY-MM-DD or 'today', optional)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for price data (YYYY-MM-DD or 'today', optional). Use 'today' for live data.",
    )
    p.add_argument(
        "--data-source",
        type=str,
        choices=["local", "yahoo"],
        default=None,
        help="Data source type: 'local' (Parquet files) or 'yahoo' (Yahoo Finance API). Default: from settings.data_source",
    )
    p.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="List of symbols to load (e.g., --symbols AAPL MSFT GOOGL). Overrides universe file.",
    )
    p.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help=f"Commission in basis points (default: {default_costs.commission_bps} from cost model)",
    )
    p.add_argument(
        "--spread-w",
        type=float,
        default=None,
        help=f"Spread weight (default: {default_costs.spread_w} from cost model)",
    )
    p.add_argument(
        "--impact-w",
        type=float,
        default=None,
        help=f"Impact weight (default: {default_costs.impact_w} from cost model)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory (default: from config)",
    )

    return p.parse_args()


def run_eod_from_args(args: argparse.Namespace) -> dict:
    """Run EOD pipeline from parsed arguments.

    This function can be called from the central CLI or from the standalone script.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with run manifest data
    """
    # Generate Run-ID and setup logging
    run_id = generate_run_id(prefix="eod")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EOD Pipeline")
    logger.info(f"Run-ID: {run_id}")
    logger.info("=" * 60)
    logger.info(f"Frequency: {args.freq}")
    logger.info(f"Start capital: {args.start_capital}")

    # Determine output directory (use settings if not provided)
    from src.assembled_core.config.settings import get_settings

    settings = get_settings()
    output_dir = args.out if args.out else settings.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Handle --price-file as Path or str
    price_file = args.price_file
    if price_file and isinstance(price_file, str):
        price_file = Path(price_file)
    elif price_file and isinstance(price_file, Path):
        price_file = price_file
    else:
        price_file = None

    # Handle universe file and symbols
    symbols = getattr(args, "symbols", None)
    if symbols:
        logger.info(f"Using symbols from CLI: {symbols}")
    elif args.universe:
        # Read symbols from universe file
        try:
            with open(args.universe, "r", encoding="utf-8") as f:
                symbols = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        symbols.append(line.upper())
            logger.info(
                f"Loaded {len(symbols)} symbols from universe file: {args.universe}"
            )
        except Exception as e:
            logger.warning(f"Failed to read universe file {args.universe}: {e}")
            symbols = None

    manifest = run_eod_pipeline(
        freq=args.freq,
        start_capital=args.start_capital,
        skip_backtest=args.skip_backtest,
        skip_portfolio=args.skip_portfolio,
        skip_qa=args.skip_qa,
        output_dir=Path(output_dir),
        price_file=str(price_file) if price_file else None,
        commission_bps=args.commission_bps,
        spread_w=args.spread_w,
        impact_w=args.impact_w,
        data_source=getattr(args, "data_source", None),
        symbols=symbols,
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
    )

    logger.info("=" * 60)
    logger.info("Pipeline completed")
    logger.info(f"Completed steps: {', '.join(manifest['completed_steps'])}")
    logger.info("=" * 60)

    if manifest.get("qa_overall_status"):
        qa_status = manifest["qa_overall_status"]
        logger.info(f"QA overall_status: {qa_status}")

        if qa_status == "error":
            logger.error("QA overall_status is 'error' - pipeline may have issues")
        elif qa_status == "warning":
            logger.warning("QA overall_status is 'warning' - some checks failed")

    # Log QA report path if available
    if manifest.get("qa_report_path"):
        logger.info(f"QA Report: {manifest['qa_report_path']}")

    if manifest.get("failure"):
        logger.error("Some pipeline steps failed")
        # Don't raise here - return manifest so caller can handle it
        # raise RuntimeError("Pipeline steps failed")

    return manifest


def main() -> int:
    """Main entry point for EOD pipeline CLI (standalone script).

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Generate Run-ID and setup logging early
    run_id = generate_run_id(prefix="eod")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    try:
        args = parse_eod_args()
        run_eod_from_args(args)
        logger.info("=" * 60)
        logger.info("EOD Pipeline completed successfully")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
