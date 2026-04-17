"""B0 — 1e-9 equity-curve regression diff.

Compares two equity JSON files (produced by the golden-baseline path or by any
`scripts/run_reference_backtest_for_golden.py`-style caller) and exits non-zero
if either `max_abs_diff > --abs-tol` (default 1e-9) or `max_rel_diff > --rel-tol`
(default 1e-12).

Used after each B1-B5 speed fix to prove "equity 1e-9 identical".

JSON schema (both files):
    {
        "config": {"n_symbols": int, "n_days": int, "seed": int, ...},
        "equity": [{"timestamp": "2024-01-02T00:00:00+00:00", "equity": 100000.0}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_equity(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "equity" not in data:
        raise ValueError(f"{path}: missing 'equity' key")
    df = pd.DataFrame(data["equity"])
    if "equity" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"{path}: each equity row needs 'timestamp' and 'equity'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--rel-tol", type=float, default=1e-12)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write a structured diff report",
    )
    args = parser.parse_args()

    base = load_equity(args.baseline)
    cand = load_equity(args.candidate)

    if len(base) != len(cand):
        print(
            f"[COMPARE] LENGTH MISMATCH: baseline={len(base)} candidate={len(cand)}",
            file=sys.stderr,
        )
        return 2

    if not (base["timestamp"].values == cand["timestamp"].values).all():
        print("[COMPARE] TIMESTAMP MISMATCH", file=sys.stderr)
        return 2

    base_eq = base["equity"].to_numpy(dtype=np.float64)
    cand_eq = cand["equity"].to_numpy(dtype=np.float64)
    diff = cand_eq - base_eq
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if len(abs_diff) else 0.0
    denom = np.where(np.abs(base_eq) > 0, np.abs(base_eq), 1.0)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max()) if len(rel_diff) else 0.0

    worst_idx = int(np.argmax(abs_diff)) if len(abs_diff) else 0
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "n_points": int(len(base)),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "worst_index": worst_idx,
        "worst_timestamp": str(base["timestamp"].iloc[worst_idx]) if len(base) else None,
        "worst_baseline_equity": float(base_eq[worst_idx]) if len(base) else None,
        "worst_candidate_equity": float(cand_eq[worst_idx]) if len(base) else None,
        "final_baseline_equity": float(base_eq[-1]) if len(base) else None,
        "final_candidate_equity": float(cand_eq[-1]) if len(base) else None,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    abs_ok = max_abs <= args.abs_tol
    rel_ok = max_rel <= args.rel_tol
    status = "OK" if (abs_ok and rel_ok) else "FAIL"
    print(
        f"[COMPARE] {status} max_abs={max_abs:.3e} (tol {args.abs_tol:.1e}) "
        f"max_rel={max_rel:.3e} (tol {args.rel_tol:.1e}) n={report['n_points']}"
    )
    if not abs_ok or not rel_ok:
        print(
            f"[COMPARE]   worst at {report['worst_timestamp']}: "
            f"base={report['worst_baseline_equity']:.9f} "
            f"cand={report['worst_candidate_equity']:.9f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
