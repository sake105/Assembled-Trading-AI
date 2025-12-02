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
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.pipeline.orchestrator import run_eod_pipeline


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
        "--freq",
        choices=SUPPORTED_FREQS,
        required=True,
        help="Trading frequency"
    )
    p.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000.0)"
    )
    p.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest step"
    )
    p.add_argument(
        "--skip-portfolio",
        action="store_true",
        help="Skip portfolio step"
    )
    p.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA step"
    )
    p.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe file (default: watchlist.txt in repo root)"
    )
    p.add_argument(
        "--price-file",
        type=str,
        default=None,
        help="Optional explicit path to price file"
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for price data (YYYY-MM-DD, optional)"
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for price data (YYYY-MM-DD, optional)"
    )
    p.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help=f"Commission in basis points (default: {default_costs.commission_bps} from cost model)"
    )
    p.add_argument(
        "--spread-w",
        type=float,
        default=None,
        help=f"Spread weight (default: {default_costs.spread_w} from cost model)"
    )
    p.add_argument(
        "--impact-w",
        type=float,
        default=None,
        help=f"Impact weight (default: {default_costs.impact_w} from cost model)"
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory (default: from config)"
    )
    
    return p.parse_args()


def run_eod_from_args(args: argparse.Namespace) -> None:
    """Run EOD pipeline from parsed arguments.
    
    This function can be called from the central CLI or from the standalone script.
    
    Args:
        args: Parsed command-line arguments
    """
    # Setup logging
    logger = setup_logging(level="INFO")
    
    logger.info(f"Starting EOD pipeline for {args.freq}")
    logger.info(f"Start capital: {args.start_capital}")
    logger.info(f"Output directory: {args.out}")
    
    # Handle --price-file as Path or str
    price_file = args.price_file
    if price_file and isinstance(price_file, str):
        price_file = Path(price_file)
    elif price_file and isinstance(price_file, Path):
        price_file = price_file
    else:
        price_file = None
    
    # Handle universe file (if provided, pass to orchestrator if it supports it)
    # Note: These are currently not passed to run_eod_pipeline, but kept for future use
    _ = getattr(args, "universe", None)  # universe_file
    _ = getattr(args, "start_date", None)  # start_date
    _ = getattr(args, "end_date", None)  # end_date
    
    manifest = run_eod_pipeline(
        freq=args.freq,
        start_capital=args.start_capital,
        skip_backtest=args.skip_backtest,
        skip_portfolio=args.skip_portfolio,
        skip_qa=args.skip_qa,
        output_dir=Path(args.out),
        price_file=price_file,
        commission_bps=args.commission_bps,
        spread_w=args.spread_w,
        impact_w=args.impact_w
    )
    
    logger.info("Pipeline completed")
    logger.info(f"Completed steps: {', '.join(manifest['completed_steps'])}")
    
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
        raise RuntimeError("Pipeline steps failed")


def main() -> None:
    """Main entry point for EOD pipeline CLI (standalone script)."""
    try:
        args = parse_eod_args()
        run_eod_from_args(args)
        sys.exit(0)
    except Exception as e:
        logger = setup_logging(level="INFO")
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

