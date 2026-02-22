"""Turnover and cost diagnosis from a run directory (trades_1d.csv).

Usage:
  py -3 scripts/dev/turnover_cost_diagnosis.py --run output/system_run/benchmark/trend_baseline/1y
  py -3 scripts/dev/turnover_cost_diagnosis.py --run output/system_run/debug_postfix

Prints: status counts, total_cost_cash sum, gross_traded_notional, cost_bps_est.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Turnover/cost diagnosis from run dir")
    ap.add_argument("--run", type=Path, required=True, help="Run directory (contains trades_1d.csv)")
    args = ap.parse_args()
    run = args.run.resolve()
    tr_path = run / "trades_1d.csv"
    if not tr_path.exists():
        print(f"Missing: {tr_path}")
        return 1
    with open(tr_path) as f:
        rows = list(csv.DictReader(f))
    # Status counts
    status_count: dict[str, int] = {}
    for r in rows:
        s = r.get("status", "")
        status_count[s] = status_count.get(s, 0) + 1
    print("status counts:", dict(sorted(status_count.items(), key=lambda x: -x[1])))
    filled = [r for r in rows if r.get("status") == "filled"]
    if not filled:
        print("total_cost_cash_sum=0")
        print("gross_traded_notional=0")
        print("cost_bps_est=N/A")
        return 0
    cost_sum = sum(float(r.get("total_cost_cash") or 0) for r in filled)
    turn = sum(abs(float(r.get("fill_qty") or 0) * float(r.get("fill_price") or 0)) for r in filled)
    print(f"total_cost_cash_sum={cost_sum}")
    print(f"gross_traded_notional={turn}")
    if turn and turn > 0:
        # ~10k start capital -> cost in bps of traded notional
        cost_bps_est = 10000 * cost_sum / turn
        print(f"cost_bps_est={cost_bps_est}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
