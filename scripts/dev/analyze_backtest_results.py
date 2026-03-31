#!/usr/bin/env python3
"""Load backtest output dir and write metrics_summary.json + .csv (deterministic, ASCII).

Reads either reports/metrics.json or equity_curve_*.csv + trades_*.csv and computes
total return, CAGR, vol, Sharpe, Sortino, max drawdown, #trades, win rate, avg win/loss,
profit factor. No new deps (uses pandas, qa.metrics, reports.metrics_export).

Usage:
    py -3 scripts/dev/analyze_backtest_results.py --out output/analysis_run/baseline_a
    py -3 scripts/dev/analyze_backtest_results.py --out output/analysis_run/smoke --summary-dir output/analysis_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _normalize_float(v: float | None) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _load_metrics_from_json(reports_dir: Path) -> dict[str, object] | None:
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_equity_and_trades(out_dir: Path, freq: str = "1d") -> tuple | None:
    import pandas as pd

    equity_path = out_dir / f"equity_curve_{freq}.csv"
    trades_path = out_dir / f"trades_{freq}.csv"
    if not equity_path.exists():
        return None
    equity = pd.read_csv(equity_path)
    if "timestamp" not in equity.columns or "equity" not in equity.columns:
        return None
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    trades = None
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        if "timestamp" in trades.columns:
            trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    return (equity, trades)


def _metrics_to_summary_row(metrics: dict, run_id: str) -> dict:
    """Build a flat row for CSV/JSON summary (ASCII keys, stable order)."""
    row = {
        "run_id": run_id,
        "total_return": _normalize_float(metrics.get("total_return")),
        "cagr": _normalize_float(metrics.get("cagr")),
        "volatility": _normalize_float(metrics.get("volatility")),
        "sharpe_ratio": _normalize_float(
            metrics.get("sharpe_ratio") or metrics.get("sharpe")
        ),
        "sortino_ratio": _normalize_float(metrics.get("sortino_ratio")),
        "max_drawdown_pct": _normalize_float(metrics.get("max_drawdown_pct")),
        "total_trades": metrics.get("total_trades") or metrics.get("trades"),
        "hit_rate": _normalize_float(metrics.get("hit_rate")),
        "profit_factor": _normalize_float(metrics.get("profit_factor")),
        "avg_win": _normalize_float(metrics.get("avg_win")),
        "avg_loss": _normalize_float(metrics.get("avg_loss")),
        "turnover": _normalize_float(metrics.get("turnover")),
        "start_date": metrics.get("start_date"),
        "end_date": metrics.get("end_date"),
        "start_capital": _normalize_float(metrics.get("start_capital")),
        "end_equity": _normalize_float(metrics.get("end_equity")),
    }
    if row["total_trades"] is not None:
        row["total_trades"] = int(row["total_trades"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze backtest output and write metrics summary."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Backtest output directory"
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Directory for metrics_summary.json/csv (default: same as --out)",
    )
    parser.add_argument(
        "--freq", type=str, default="1d", help="Frequency for equity/trades filenames"
    )
    args = parser.parse_args()
    out_dir = args.out.resolve()
    summary_dir = (args.summary_dir or out_dir).resolve()
    run_id = out_dir.name or "run"

    metrics_dict: dict | None = None

    # Prefer reports/metrics.json
    reports_dir = out_dir / "reports"
    metrics_dict = _load_metrics_from_json(reports_dir)

    # Else compute from equity + trades
    if metrics_dict is None:
        loaded = _load_equity_and_trades(out_dir, args.freq)
        if loaded is not None:
            equity, trades = loaded
            from src.assembled_core.qa.metrics import compute_all_metrics

            start_capital = 10000.0
            metrics = compute_all_metrics(
                equity=equity,
                trades=trades,
                start_capital=start_capital,
                freq=args.freq,
                risk_free_rate=0.0,
            )
            metrics_dict = {
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_trades": metrics.total_trades,
                "hit_rate": metrics.hit_rate,
                "profit_factor": metrics.profit_factor,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "turnover": metrics.turnover,
                "start_date": (
                    metrics.start_date.isoformat()
                    if metrics.start_date is not None
                    else None
                ),
                "end_date": (
                    metrics.end_date.isoformat()
                    if metrics.end_date is not None
                    else None
                ),
                "start_capital": metrics.start_capital,
                "end_equity": metrics.end_equity,
            }
        else:
            print(
                "No metrics.json or equity_curve_*.csv found under --out",
                file=sys.stderr,
            )
            return 1

    row = _metrics_to_summary_row(metrics_dict, run_id)

    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "metrics_summary.json"
    csv_path = summary_dir / "metrics_summary.csv"

    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(row, f, indent=2, sort_keys=True, ensure_ascii=True)

    import csv as csv_module

    with csv_path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv_module.DictWriter(f, fieldnames=sorted(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
