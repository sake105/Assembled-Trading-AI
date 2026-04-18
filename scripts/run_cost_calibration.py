"""E5-loop — offline cost-model calibration runner.

Reads TCA aggregates produced by the paper engine and writes a
shrinkage-blended recommendation file. The recommendation is ``deploy: False``
by default; adopting new parameters is a manual config change (CLAUDE.md §5.4,
§10.4).

Usage::

    python scripts/run_cost_calibration.py
    python scripts/run_cost_calibration.py --tca-dir output/paper_tca \\
        --out output/qa/cost_calibration.yaml --shrinkage 0.3

The runner intentionally does not touch ``UnifiedPaperConfig``; see
Ultra-Plan part E5 for the deployment ceremony.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.cost_model_calibrator import (  # noqa: E402
    CostModelPriors,
    calibrate_cost_model,
    write_calibration_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cost_calibration")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tca-dir", type=Path, default=ROOT / "output" / "paper_tca")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "qa" / "cost_calibration.yaml",
    )
    p.add_argument("--shrinkage", type=float, default=0.30)
    p.add_argument("--min-runs", type=int, default=1)
    p.add_argument("--half-spread-prior", type=float, default=5.0)
    p.add_argument("--impact-prior", type=float, default=10.0)
    p.add_argument("--participation-cap-prior", type=float, default=0.05)
    args = p.parse_args(argv)

    priors = CostModelPriors(
        half_spread_bps=args.half_spread_prior,
        impact_bps_per_pct_adv=args.impact_prior,
        participation_cap=args.participation_cap_prior,
    )
    logger.info("[CALIBRATOR] scanning TCA dir: %s", args.tca_dir)
    result = calibrate_cost_model(
        args.tca_dir,
        priors=priors,
        shrinkage=args.shrinkage,
        min_runs=args.min_runs,
    )
    path = write_calibration_report(result, args.out)
    logger.info(
        "[CALIBRATOR] report written to %s (n_runs=%d, deploy=False)",
        path,
        result.n_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
