#!/usr/bin/env python3
"""Mini-Check: Prueft ob ein Parquet-Dataset fuer den Strategy-Benchmark brauchbar ist.

Aufruf:
  py -3 scripts/dev/check_parquet_dataset.py
  py -3 scripts/dev/check_parquet_dataset.py <PFAD_ZUM_PARQUET>
  py -3 scripts/dev/check_parquet_dataset.py output/aggregates/1d.parquet

Zeigt: rows, unique_days, min/max timestamp, Spalten-Check (timestamp, symbol, close).
Damit siehst du sofort, ob das Dataset fuer realistische Baseline-Performance taugt.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not path_arg:
        print(
            "Usage: py -3 scripts/dev/check_parquet_dataset.py <DEIN_PARQUET>",
            file=sys.stderr,
        )
        print(
            "Example: py -3 scripts/dev/check_parquet_dataset.py output/aggregates/1d.parquet",
            file=sys.stderr,
        )
        return 1

    p = Path(path_arg)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        return 1

    import pandas as pd

    df = pd.read_parquet(p, columns=["timestamp"] if path_arg else None)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    valid = ts.notna()
    n = len(df)
    n_valid = valid.sum()
    unique_days = ts[valid].dt.date.nunique() if n_valid else 0
    t_min = ts[valid].min() if n_valid else None
    t_max = ts[valid].max() if n_valid else None

    print("rows", n)
    print("rows_valid_ts", int(n_valid))
    print("unique_days", unique_days)
    print("min", t_min)
    print("max", t_max)

    # Schema-Check (Benchmark braucht timestamp, symbol, close)
    full = pd.read_parquet(p, columns=None)
    has_ts = "timestamp" in full.columns
    has_sym = "symbol" in full.columns
    has_close = "close" in full.columns
    print("has_timestamp", has_ts)
    print("has_symbol", has_sym)
    print("has_close", has_close)
    if has_ts and has_sym and has_close:
        print("schema_ok", True)
        if unique_days >= 252:
            print("note", ">=252 trading days -> OOS-sweep sinnvoll")
        else:
            print(
                "note", "<252 days -> OOS-sweep ggf. uebersprungen (MIN_TRADING_DAYS)"
            )
    else:
        print("schema_ok", False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
