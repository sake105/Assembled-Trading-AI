# scripts/sprint10_portfolio.py
# LEGACY: Dieses Script gehört zur alten Sprint-Phase.
# Für neue Entwicklungen bitte src/assembled_core/* und scripts/run_*.py verwenden.
# Dieses Script wird noch von run_all_sprint10.ps1 verwendet, ist aber deprecated.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import core modules
sys.path.insert(0, str(ROOT))
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.pipeline.io import load_orders
from src.assembled_core.pipeline.portfolio import (
    simulate_with_costs,
    write_portfolio_report,
)


def main():
    """CLI wrapper for portfolio simulation with costs."""
    # Get default cost model from central configuration
    default_costs = get_default_cost_model()

    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["1d", "5min"], required=True)
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument(
        "--commission-bps",
        type=float,
        default=default_costs.commission_bps,
        help=f"Commission in basis points (default: {default_costs.commission_bps} from cost model)",
    )
    ap.add_argument(
        "--spread-w",
        type=float,
        default=default_costs.spread_w,
        help=f"Spread weight (default: {default_costs.spread_w} from cost model)",
    )
    ap.add_argument(
        "--impact-w",
        type=float,
        default=default_costs.impact_w,
        help=f"Impact weight (default: {default_costs.impact_w} from cost model)",
    )
    a = ap.parse_args()

    # Load orders
    orders = load_orders(a.freq, output_dir=OUTPUT_DIR, strict=True)

    # Simulate with costs
    eq, rep = simulate_with_costs(
        orders,
        a.start_capital,
        a.commission_bps,
        a.spread_w,
        a.impact_w,
        a.freq,
    )

    # Write results
    eq_path, rep_path = write_portfolio_report(eq, rep, a.freq, output_dir=OUTPUT_DIR)
    print(
        f"[PF10] DONE | PF={rep['final_pf']:.4f} Sharpe={rep['sharpe']} Trades={rep['trades']}"
    )


if __name__ == "__main__":
    main()
