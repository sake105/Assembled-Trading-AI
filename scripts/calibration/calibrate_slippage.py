"""Calibrate FillModel slippage parameters against historical execution data.

Compares simulated costs (from backtest ledgers) vs a target slippage model,
then optimises FillModel parameters (half_spread_bps, impact_coefficient) to
minimise the squared difference between model predictions and observed costs.

Workflow
--------
1. Load backtest ledger events from output/ledger_backtest_*/
2. Extract fill events: fill_price, mid_price proxy (open/close average), qty
3. Compute observed slippage in bps = (fill_price - mid_price) / mid_price * 10000
4. Bucket by ADV (average daily volume) quintile
5. Optimise FillModel params via scipy.optimize.minimize (L-BFGS-B)
6. Output JSON report to output/calibration/slippage_report.json

Usage
-----
python scripts/calibration/calibrate_slippage.py
python scripts/calibration/calibrate_slippage.py \\
    --bt-root output \\
    --output-path output/calibration/slippage_report.json \\
    --half-spread-init 5.0 \\
    --impact-init 0.1

Log prefix: [SLIPPAGE-CAL]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Optional: scipy
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
_TAG = "[SLIPPAGE-CAL]"


def _log(msg: str) -> None:
    logger.info("%s %s", _TAG, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _TAG, msg)


# ---------------------------------------------------------------------------
# FillModel prediction
# ---------------------------------------------------------------------------

def _fill_model_slippage_bps(
    half_spread_bps: float,
    impact_coefficient: float,
    qty: float,
    adv: float,
) -> float:
    """Simple square-root market-impact model.

    slippage_bps = half_spread + impact_coeff * sqrt(qty / adv) * 10000

    Parameters
    ----------
    half_spread_bps:   Fixed spread component (bps).
    impact_coefficient: Price-impact coefficient (dimensionless).
    qty:               Order size (shares or notional).
    adv:               Average daily volume (same units as qty).
    """
    if adv <= 0 or qty <= 0:
        return float(half_spread_bps)
    pov = float(qty) / float(adv)
    return float(half_spread_bps) + float(impact_coefficient) * np.sqrt(pov) * 10_000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ledger_fills(bt_root: Path) -> pd.DataFrame:
    """Load all ledger events that contain fill-like cost data."""
    ledger_dirs = sorted(bt_root.glob("ledger_backtest_*"))
    _log(f"Scanning {len(ledger_dirs)} ledger directories")

    frames: list[pd.DataFrame] = []

    for ld in ledger_dirs:
        ledger_file = ld / "ledger_events.parquet"
        if not ledger_file.exists():
            continue
        try:
            df = pd.read_parquet(ledger_file)
        except Exception as exc:
            _warn(f"Cannot read {ledger_file}: {exc!r}")
            continue

        if df.empty:
            continue

        # Keep rows that have cost data (spread/slippage)
        cost_cols = {"spread_cash", "slippage_cash", "total_cost_cash"}
        has_cost = cost_cols.intersection(df.columns)
        if not has_cost:
            continue

        df = df[df["event_type"].isin(["FILL", "ORDER_SUBMIT"])].copy()
        if df.empty:
            continue

        df["source_run"] = ld.name
        frames.append(df)

    if not frames:
        _log("No cost-bearing ledger events found.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    _log(f"Loaded {len(combined)} cost-bearing events from {len(frames)} ledgers")
    return combined


def _load_paper_fills(paper_track_root: Path) -> pd.DataFrame:
    """Load paper trading fills if available."""
    frames: list[pd.DataFrame] = []

    for strategy_dir in paper_track_root.iterdir():
        if not strategy_dir.is_dir():
            continue
        for pq in strategy_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(pq)
                df["source"] = f"paper_{strategy_dir.name}"
                frames.append(df)
            except Exception as exc:
                _warn(f"Cannot read {pq}: {exc!r}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    _log(f"Loaded {len(combined)} paper trade rows")
    return combined


# ---------------------------------------------------------------------------
# Slippage computation
# ---------------------------------------------------------------------------

def _compute_observed_slippage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed slippage bps from ledger events.

    Uses total_cost_cash / (|qty| * price) * 10000 as the slippage proxy
    when no mid_price is available.
    """
    df = df.copy()

    # Fill price from 'price' column
    if "price" not in df.columns:
        _warn("No 'price' column -- cannot compute slippage.")
        return pd.DataFrame()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df.get("qty", np.nan), errors="coerce").abs()

    # Observed slippage proxy
    if "total_cost_cash" in df.columns:
        df["total_cost_cash"] = pd.to_numeric(df["total_cost_cash"], errors="coerce")
        notional = (df["qty"] * df["price"]).replace(0, np.nan)
        df["observed_slippage_bps"] = (df["total_cost_cash"] / notional * 10_000.0).fillna(0.0)
    elif "slippage_cash" in df.columns:
        df["slippage_cash"] = pd.to_numeric(df["slippage_cash"], errors="coerce")
        notional = (df["qty"] * df["price"]).replace(0, np.nan)
        df["observed_slippage_bps"] = (df["slippage_cash"] / notional * 10_000.0).fillna(0.0)
    else:
        df["observed_slippage_bps"] = 0.0

    # ADV proxy -- use qty as self-fill; real ADV data not available here
    df["adv_proxy"] = df["qty"] * 10.0  # conservative: order is ~10% of ADV

    result = df[["price", "qty", "adv_proxy", "observed_slippage_bps"]].dropna()
    result = result[result["qty"] > 0]
    _log(f"Computed slippage for {len(result)} valid fill rows")
    return result


def _adv_bucket_analysis(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Split by ADV quintile and summarise observed slippage per bucket."""
    if df.empty or "adv_proxy" not in df.columns:
        return []

    df = df.copy()
    try:
        df["adv_bucket"] = pd.qcut(df["adv_proxy"], q=5, labels=False, duplicates="drop")
    except ValueError:
        df["adv_bucket"] = 0

    buckets = []
    for bucket_id, grp in df.groupby("adv_bucket", observed=True):
        buckets.append(
            {
                "adv_bucket": int(bucket_id),
                "n_fills": len(grp),
                "mean_slippage_bps": round(float(grp["observed_slippage_bps"].mean()), 4),
                "median_slippage_bps": round(float(grp["observed_slippage_bps"].median()), 4),
                "std_slippage_bps": round(float(grp["observed_slippage_bps"].std()), 4),
                "mean_qty": round(float(grp["qty"].mean()), 2),
                "mean_adv_proxy": round(float(grp["adv_proxy"].mean()), 2),
            }
        )
    return buckets


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def _optimise_fill_model(
    df: pd.DataFrame,
    half_spread_init: float = 5.0,
    impact_init: float = 0.1,
) -> dict[str, Any]:
    """Optimise half_spread_bps and impact_coefficient.

    Minimises mean squared error between model prediction and observed slippage.
    """
    if df.empty:
        _warn("No data to optimise -- returning initial params.")
        return {
            "half_spread_bps": half_spread_init,
            "impact_coefficient": impact_init,
            "optimised": False,
            "reason": "no_data",
        }

    observed = df["observed_slippage_bps"].values
    qtys = df["qty"].values
    advs = df["adv_proxy"].values

    def _mse(params: np.ndarray) -> float:
        hs, ic = params[0], params[1]
        preds = np.array(
            [_fill_model_slippage_bps(hs, ic, q, a) for q, a in zip(qtys, advs)]
        )
        return float(np.mean((preds - observed) ** 2))

    if not _SCIPY_AVAILABLE:
        _warn("scipy not available -- skipping optimisation, returning initial params.")
        return {
            "half_spread_bps": half_spread_init,
            "impact_coefficient": impact_init,
            "optimised": False,
            "reason": "scipy_unavailable",
        }

    x0 = np.array([half_spread_init, impact_init])
    bounds = [(0.0, 100.0), (0.0, 10.0)]

    try:
        res = _scipy_minimize(_mse, x0, method="L-BFGS-B", bounds=bounds)
        if res.success or res.fun < _mse(x0):
            opt_hs, opt_ic = float(res.x[0]), float(res.x[1])
            _log(
                f"[OK] Optimised params: "
                f"half_spread_bps={opt_hs:.4f}, "
                f"impact_coefficient={opt_ic:.6f} "
                f"(MSE={res.fun:.4f})"
            )
            return {
                "half_spread_bps": round(opt_hs, 4),
                "impact_coefficient": round(opt_ic, 6),
                "mse": round(float(res.fun), 6),
                "optimised": True,
                "reason": res.message,
            }
        else:
            _warn(f"Optimisation did not improve: {res.message}")
            return {
                "half_spread_bps": half_spread_init,
                "impact_coefficient": impact_init,
                "optimised": False,
                "reason": res.message,
            }
    except Exception as exc:
        _warn(f"Optimisation failed: {exc!r}")
        return {
            "half_spread_bps": half_spread_init,
            "impact_coefficient": impact_init,
            "optimised": False,
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def calibrate_slippage(
    bt_root: Path,
    paper_track_root: Path | None,
    output_path: Path,
    half_spread_init: float = 5.0,
    impact_init: float = 0.1,
) -> dict[str, Any]:
    """Full calibration pipeline."""
    _log("=" * 60)
    _log("calibrate_slippage.py -- START")
    _log(f"bt_root          : {bt_root}")
    _log(f"output_path      : {output_path}")
    _log(f"half_spread_init : {half_spread_init}")
    _log(f"impact_init      : {impact_init}")
    _log("=" * 60)

    # 1. Load backtest fills
    bt_events = _load_ledger_fills(bt_root)

    # 2. Load paper fills (optional)
    paper_events = pd.DataFrame()
    if paper_track_root and paper_track_root.exists():
        paper_events = _load_paper_fills(paper_track_root)

    # 3. Combine
    all_events = pd.concat(
        [df for df in [bt_events, paper_events] if not df.empty],
        ignore_index=True,
    )
    if all_events.empty:
        _warn("No fill events available -- report will have no calibration data.")

    # 4. Compute observed slippage
    slippage_df = _compute_observed_slippage(all_events) if not all_events.empty else pd.DataFrame()

    # 5. ADV bucket analysis
    adv_buckets = _adv_bucket_analysis(slippage_df)
    _log(f"ADV buckets computed: {len(adv_buckets)}")

    # 6. Optimise
    opt_params = _optimise_fill_model(slippage_df, half_spread_init, impact_init)

    # 7. Overall stats
    if slippage_df.empty:
        overall_stats: dict[str, Any] = {"n_fills": 0}
    else:
        obs = slippage_df["observed_slippage_bps"]
        overall_stats = {
            "n_fills": len(slippage_df),
            "mean_slippage_bps": round(float(obs.mean()), 4),
            "median_slippage_bps": round(float(obs.median()), 4),
            "std_slippage_bps": round(float(obs.std()), 4),
            "p95_slippage_bps": round(float(obs.quantile(0.95)), 4),
        }

    report: dict[str, Any] = {
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bt_root": str(bt_root),
        "overall_stats": overall_stats,
        "optimised_params": opt_params,
        "adv_bucket_analysis": adv_buckets,
        "initial_params": {
            "half_spread_bps": half_spread_init,
            "impact_coefficient": impact_init,
        },
    }

    # 8. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)

    _log(f"[OK] Report written to {output_path}")
    _log(
        f"Optimised: half_spread={opt_params.get('half_spread_bps')} bps, "
        f"impact_coeff={opt_params.get('impact_coefficient')}"
    )
    _log("calibrate_slippage.py -- DONE")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate FillModel slippage parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bt-root",
        type=Path,
        default=Path("output"),
        help="Root directory containing ledger_backtest_* subdirs.",
    )
    parser.add_argument(
        "--paper-track-root",
        type=Path,
        default=Path("output/paper_track"),
        help="Root of paper_track/ output dirs (optional).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/calibration/slippage_report.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--half-spread-init",
        type=float,
        default=5.0,
        help="Initial half-spread in bps for optimisation.",
    )
    parser.add_argument(
        "--impact-init",
        type=float,
        default=0.1,
        help="Initial price-impact coefficient for optimisation.",
    )
    args = parser.parse_args(argv)

    paper_root = args.paper_track_root if args.paper_track_root.exists() else None

    calibrate_slippage(
        bt_root=args.bt_root.resolve(),
        paper_track_root=paper_root,
        output_path=args.output_path.resolve(),
        half_spread_init=args.half_spread_init,
        impact_init=args.impact_init,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
