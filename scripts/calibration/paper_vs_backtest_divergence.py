"""Paper vs backtest equity curve divergence analysis.

Loads the paper trading equity curve and a reference backtest equity curve,
then computes:
  - Return correlation
  - Annualised tracking error
  - Systematic bias (mean of paper_return - backtest_return)
  - Cost attribution (difference in total_cost_cash)

Gates (logged as WARN if failed):
  - Correlation > 0.80
  - Tracking error < 5 % annualised

Output: output/calibration/divergence_report.json

Usage
-----
python scripts/calibration/paper_vs_backtest_divergence.py
python scripts/calibration/paper_vs_backtest_divergence.py \\
    --paper-equity output/paper_track/test_strategy/equity.parquet \\
    --bt-equity    output/backtest_equity.parquet \\
    --output-path  output/calibration/divergence_report.json

Log prefix: [DIVERGENCE]
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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
_TAG = "[DIVERGENCE]"


def _log(msg: str) -> None:
    logger.info("%s %s", _TAG, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _TAG, msg)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
GATE_MIN_CORRELATION: float = 0.80
GATE_MAX_TE_ANNUALISED: float = 0.05  # 5 %


# ---------------------------------------------------------------------------
# Equity curve loading
# ---------------------------------------------------------------------------

def _try_load_equity(path: Path, label: str) -> pd.DataFrame | None:
    """Load a parquet or CSV equity curve.  Returns None on failure."""
    if not path.exists():
        _warn(f"{label} equity path not found: {path}")
        return None

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"Cannot load {label} equity from {path}: {exc!r}")
        return None

    # Normalise date column
    for col in ("date", "timestamp", "event_ts"):
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            break

    if "date" not in df.columns:
        _warn(f"{label}: no date column found in {path}")
        return None

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Equity column
    for col in ("equity", "total_equity", "nav", "portfolio_value", "cash_end", "cash"):
        if col in df.columns:
            df = df.rename(columns={col: "equity"})
            break

    if "equity" not in df.columns:
        _warn(f"{label}: no equity column found; columns = {df.columns.tolist()}")
        return None

    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["equity"])
    _log(f"{label}: loaded {len(df)} rows from {path}")
    return df[["date", "equity"]]


def _load_backtest_equity_from_accounting(bt_root: Path) -> pd.DataFrame | None:
    """Reconstruct an equity series from accounting JSON files.

    Reads all accounting_report_backtest_*/accounting_*.json files and
    extracts cash_end + unrealized_pnl as a proxy equity series.
    """
    rows = []
    for bt_dir in sorted(bt_root.glob("accounting_report_backtest_*")):
        for json_file in sorted(bt_dir.glob("accounting_*.json")):
            try:
                with open(json_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                as_of = pd.to_datetime(data.get("as_of_date"), utc=True, errors="coerce")
                cash_end = float(data.get("cash", {}).get("end", np.nan))
                unreal = float(data.get("pnl", {}).get("total_unrealized", 0.0) or 0.0)
                if pd.isna(as_of) or np.isnan(cash_end):
                    continue
                rows.append({"date": as_of, "equity": cash_end + unreal})
            except Exception:
                continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Aggregate by date (median if multiple runs on same day)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.groupby("date", as_index=False)["equity"].median()
    _log(f"Reconstructed backtest equity from accounting: {len(df)} dates")
    return df


def _load_paper_equity_from_track(paper_track_root: Path) -> pd.DataFrame | None:
    """Load paper equity from paper_track/ directory (parquet/csv files)."""
    if not paper_track_root.exists():
        return None

    frames: list[pd.DataFrame] = []
    for strategy_dir in sorted(paper_track_root.iterdir()):
        if not strategy_dir.is_dir():
            continue
        for pq in sorted(strategy_dir.glob("*.parquet")) + sorted(strategy_dir.glob("*.csv")):
            result = _try_load_equity(pq, f"paper/{strategy_dir.name}/{pq.name}")
            if result is not None:
                frames.append(result)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.groupby("date", as_index=False)["equity"].mean()
    _log(f"Paper equity from track: {len(combined)} dates")
    return combined


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

def _compute_divergence_metrics(
    paper_eq: pd.DataFrame,
    bt_eq: pd.DataFrame,
) -> dict[str, Any]:
    """Compute divergence metrics between two equity series.

    Both DataFrames must have 'date' and 'equity' columns.
    Inner-joins on date to ensure aligned series.
    """
    paper_eq = paper_eq.copy()
    bt_eq = bt_eq.copy()
    paper_eq["date"] = pd.to_datetime(paper_eq["date"]).dt.normalize()
    bt_eq["date"] = pd.to_datetime(bt_eq["date"]).dt.normalize()

    merged = paper_eq.rename(columns={"equity": "paper_equity"}).merge(
        bt_eq.rename(columns={"equity": "bt_equity"}),
        on="date",
        how="inner",
    )

    if len(merged) < 5:
        _warn(f"Only {len(merged)} overlapping dates -- metrics may be unreliable.")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Daily returns
    merged["paper_ret"] = merged["paper_equity"].pct_change(fill_method=None).fillna(0.0)
    merged["bt_ret"] = merged["bt_equity"].pct_change(fill_method=None).fillna(0.0)
    merged["ret_diff"] = merged["paper_ret"] - merged["bt_ret"]

    n = len(merged)
    trading_days_per_year = 252.0

    # Correlation
    if n >= 2 and merged["paper_ret"].std() > 0 and merged["bt_ret"].std() > 0:
        correlation = float(merged["paper_ret"].corr(merged["bt_ret"]))
    else:
        correlation = float("nan")

    # Tracking error (annualised std of return differences)
    if n >= 2:
        te_daily = float(merged["ret_diff"].std(ddof=1))
        te_annualised = te_daily * np.sqrt(trading_days_per_year)
    else:
        te_daily = float("nan")
        te_annualised = float("nan")

    # Systematic bias = mean daily return difference
    systematic_bias = float(merged["ret_diff"].mean())
    bias_annualised = systematic_bias * trading_days_per_year

    # Final equity comparison
    paper_final = float(merged["paper_equity"].iloc[-1]) if n > 0 else float("nan")
    bt_final = float(merged["bt_equity"].iloc[-1]) if n > 0 else float("nan")

    # Total return
    paper_total_ret = (
        float(merged["paper_equity"].iloc[-1] / merged["paper_equity"].iloc[0] - 1)
        if n > 0 and merged["paper_equity"].iloc[0] != 0
        else float("nan")
    )
    bt_total_ret = (
        float(merged["bt_equity"].iloc[-1] / merged["bt_equity"].iloc[0] - 1)
        if n > 0 and merged["bt_equity"].iloc[0] != 0
        else float("nan")
    )

    metrics: dict[str, Any] = {
        "n_common_dates": n,
        "date_range_start": str(merged["date"].iloc[0].date()) if n > 0 else None,
        "date_range_end": str(merged["date"].iloc[-1].date()) if n > 0 else None,
        "return_correlation": round(correlation, 4) if not np.isnan(correlation) else None,
        "tracking_error_daily": round(te_daily, 6) if not np.isnan(te_daily) else None,
        "tracking_error_annualised": round(te_annualised, 6) if not np.isnan(te_annualised) else None,
        "systematic_bias_daily": round(systematic_bias, 6),
        "systematic_bias_annualised": round(bias_annualised, 4),
        "paper_total_return": round(paper_total_ret, 4) if not np.isnan(paper_total_ret) else None,
        "bt_total_return": round(bt_total_ret, 4) if not np.isnan(bt_total_ret) else None,
        "paper_final_equity": round(paper_final, 2) if not np.isnan(paper_final) else None,
        "bt_final_equity": round(bt_final, 2) if not np.isnan(bt_final) else None,
    }

    return metrics


def _cost_attribution(
    bt_root: Path,
) -> dict[str, Any]:
    """Summarise total cost differences across backtest runs."""
    total_commission = 0.0
    total_slippage = 0.0
    total_spread = 0.0
    n_runs = 0

    for bt_dir in sorted(bt_root.glob("accounting_report_backtest_*")):
        for json_file in sorted(bt_dir.glob("accounting_*.json")):
            try:
                with open(json_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                costs = data.get("costs", {})
                total_commission += float(costs.get("commission_cash", 0) or 0)
                total_slippage += float(costs.get("slippage_cash", 0) or 0)
                total_spread += float(costs.get("spread_cash", 0) or 0)
                n_runs += 1
            except Exception:
                continue

    return {
        "n_accounting_snapshots": n_runs,
        "total_commission_cash": round(total_commission, 4),
        "total_slippage_cash": round(total_slippage, 4),
        "total_spread_cash": round(total_spread, 4),
        "total_cost_cash": round(total_commission + total_slippage + total_spread, 4),
    }


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def _check_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate gate thresholds and return gate status."""
    gates: dict[str, Any] = {}

    corr = metrics.get("return_correlation")
    if corr is None:
        gates["correlation_gate"] = {
            "passed": None,
            "value": None,
            "threshold": GATE_MIN_CORRELATION,
            "reason": "insufficient_data",
        }
    else:
        passed = corr >= GATE_MIN_CORRELATION
        gates["correlation_gate"] = {
            "passed": bool(passed),
            "value": corr,
            "threshold": GATE_MIN_CORRELATION,
        }
        if not passed:
            _warn(
                f"GATE FAIL: return_correlation={corr:.4f} < "
                f"{GATE_MIN_CORRELATION} (required)"
            )

    te = metrics.get("tracking_error_annualised")
    if te is None:
        gates["tracking_error_gate"] = {
            "passed": None,
            "value": None,
            "threshold": GATE_MAX_TE_ANNUALISED,
            "reason": "insufficient_data",
        }
    else:
        passed = te <= GATE_MAX_TE_ANNUALISED
        gates["tracking_error_gate"] = {
            "passed": bool(passed),
            "value": round(te, 4),
            "threshold": GATE_MAX_TE_ANNUALISED,
        }
        if not passed:
            _warn(
                f"GATE FAIL: tracking_error_annualised={te:.4f} > "
                f"{GATE_MAX_TE_ANNUALISED:.2%} (required)"
            )

    all_passed = all(
        v.get("passed") is True for v in gates.values() if v.get("passed") is not None
    )
    gates["all_gates_passed"] = all_passed
    if all_passed:
        _log("[OK] All gates passed.")
    return gates


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_divergence(
    paper_equity_path: Path | None,
    bt_equity_path: Path | None,
    bt_root: Path,
    paper_track_root: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Full divergence pipeline."""
    _log("=" * 60)
    _log("paper_vs_backtest_divergence.py -- START")
    _log(f"paper_equity_path : {paper_equity_path}")
    _log(f"bt_equity_path    : {bt_equity_path}")
    _log(f"bt_root           : {bt_root}")
    _log("=" * 60)

    # --- Load paper equity ---
    paper_eq: pd.DataFrame | None = None
    if paper_equity_path and paper_equity_path.exists():
        paper_eq = _try_load_equity(paper_equity_path, "paper")
    if paper_eq is None and paper_track_root:
        paper_eq = _load_paper_equity_from_track(paper_track_root)
    if paper_eq is None:
        _warn("Could not load paper equity -- divergence metrics will be empty.")

    # --- Load backtest equity ---
    bt_eq: pd.DataFrame | None = None
    if bt_equity_path and bt_equity_path.exists():
        bt_eq = _try_load_equity(bt_equity_path, "backtest")
    if bt_eq is None:
        bt_eq = _load_backtest_equity_from_accounting(bt_root)
    if bt_eq is None:
        _warn("Could not load backtest equity -- divergence metrics will be empty.")

    # --- Divergence metrics ---
    if paper_eq is not None and bt_eq is not None:
        metrics = _compute_divergence_metrics(paper_eq, bt_eq)
    else:
        metrics = {
            "n_common_dates": 0,
            "return_correlation": None,
            "tracking_error_annualised": None,
            "systematic_bias_daily": None,
            "reason": "equity_data_unavailable",
        }

    # --- Gate checks ---
    gates = _check_gates(metrics)

    # --- Cost attribution ---
    cost_attr = _cost_attribution(bt_root)

    report: dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "divergence_metrics": metrics,
        "gate_results": gates,
        "cost_attribution": cost_attr,
        "data_sources": {
            "paper_equity_path": str(paper_equity_path) if paper_equity_path else None,
            "bt_equity_path": str(bt_equity_path) if bt_equity_path else None,
            "bt_root": str(bt_root),
        },
    }

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True, default=str)

    _log(f"[OK] Report written to {output_path}")
    _log(
        f"Correlation={metrics.get('return_correlation')} | "
        f"TE_ann={metrics.get('tracking_error_annualised')} | "
        f"gates_passed={gates.get('all_gates_passed')}"
    )
    _log("paper_vs_backtest_divergence.py -- DONE")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare paper vs backtest equity curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-equity",
        type=Path,
        default=None,
        help="Path to paper equity parquet/CSV (optional; auto-detected from paper_track/).",
    )
    parser.add_argument(
        "--bt-equity",
        type=Path,
        default=None,
        help="Path to backtest equity parquet/CSV (optional; auto-reconstructed).",
    )
    parser.add_argument(
        "--bt-root",
        type=Path,
        default=Path("output"),
        help="Root dir containing accounting_report_backtest_* subdirs.",
    )
    parser.add_argument(
        "--paper-track-root",
        type=Path,
        default=Path("output/paper_track"),
        help="Root of paper_track/ output dirs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/calibration/divergence_report.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    compute_divergence(
        paper_equity_path=args.paper_equity,
        bt_equity_path=args.bt_equity,
        bt_root=args.bt_root.resolve(),
        paper_track_root=args.paper_track_root if args.paper_track_root.exists() else None,
        output_path=args.output_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
