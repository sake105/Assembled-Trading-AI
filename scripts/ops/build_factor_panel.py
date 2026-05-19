"""Build a factor-enriched panel from a raw OHLCV panel.

Closes the 2026-05-19 audit finding that multifactor_v2 was running degraded
because its bundle (configs/factor_bundles/macro_world_etfs_core_bundle.yaml)
expected 5 factors that were never persisted:
  - trailing_momentum_12m_excl_1m
  - trend_strength_50, trend_strength_200
  - trailing_returns_12m
  - rv_20

Both feature builders already exist in src/assembled_core/features/. This
script applies them and writes a persistable factor panel.

Usage::

    python -m scripts.ops.build_factor_panel \\
        --in data/sample/master_universe_panel.parquet \\
        --out output/factor_panels/master_universe_factor_panel.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to raw OHLCV parquet (must have timestamp, symbol, close).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        required=True,
        help="Where to write the factor-enriched panel.",
    )
    parser.add_argument(
        "--rv-windows",
        type=int,
        nargs="+",
        default=[20, 60],
        help="Realized-volatility windows in trading days (default: 20 60).",
    )
    parser.add_argument(
        "--with-gpr",
        action="store_true",
        help="Merge gpr_index from output/macro_gpr.parquet (run "
        "fetch_caldara_iacoviello_gpr.py to populate).",
    )
    parser.add_argument(
        "--gpr-path",
        default=str(_REPO_ROOT / "output" / "macro_gpr.parquet"),
        help="Override GPR parquet path.",
    )
    args = parser.parse_args()

    print(f"[START] reading {args.input_path}")
    df = pd.read_parquet(args.input_path)
    # Yfinance-derived panels (download_master_universe_data.py) write `date`;
    # earlier hand-built panels used `timestamp`. Normalize to `timestamp`.
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    print(
        f"  loaded: rows={len(df)} cols={len(df.columns)} syms={df['symbol'].nunique()}"
    )
    print(f"  range: {df['timestamp'].min()} .. {df['timestamp'].max()}")

    # Normalize timestamp dtype (some panels store tz-naive)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    print("[STEP] build_core_ta_factors")
    df = build_core_ta_factors(df)
    new_factors = [
        "trailing_momentum_12m_excl_1m",
        "trend_strength_50",
        "trend_strength_200",
        "trailing_returns_12m",
    ]
    missing = [c for c in new_factors if c not in df.columns]
    if missing:
        print(f"[ERROR] still missing after core build: {missing}")
        return 1
    for c in new_factors:
        nan_pct = df[c].isna().mean() * 100
        print(f"  {c}: nan%={nan_pct:.1f}")

    print(f"[STEP] add_realized_volatility windows={args.rv_windows}")
    df = add_realized_volatility(df, windows=args.rv_windows)
    rv_cols = [f"rv_{w}" for w in args.rv_windows]
    for c in rv_cols:
        if c in df.columns:
            nan_pct = df[c].isna().mean() * 100
            print(f"  {c}: nan%={nan_pct:.1f}")

    # Optional: merge Caldara-Iacoviello GPR (monthly → ffill to daily).
    # Populates the `gpr_index` column that _compute_geo_risk_composite reads
    # (Path 1) — replaces the dead FRED GPRC fetch removed in 6be8ce3.
    if args.with_gpr:
        gpr_path = Path(args.gpr_path)
        if not gpr_path.exists():
            print(
                f"[WARN] --with-gpr set but {gpr_path} missing; "
                "run scripts/ops/fetch_caldara_iacoviello_gpr.py first."
            )
        else:
            print(f"[STEP] merging gpr_index from {gpr_path}")
            gpr = pd.read_parquet(gpr_path)[["timestamp", "gpr_index"]].copy()
            # GPR is month-start; panel is daily. Forward-fill within each
            # calendar month so every trading day inherits the prior month's
            # value at month-start (matches PIT semantics — month t value is
            # released during month t+1).
            gpr["timestamp"] = pd.to_datetime(gpr["timestamp"], utc=True)
            gpr = gpr.sort_values("timestamp")
            # Use merge_asof on a sorted single-column key.
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = pd.merge_asof(
                df,
                gpr,
                on="timestamp",
                direction="backward",
            )
            # Re-establish (symbol, timestamp) sort for downstream consumers.
            df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            nan_pct = df["gpr_index"].isna().mean() * 100
            print(f"  gpr_index: nan%={nan_pct:.1f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"[OK] wrote -> {args.output_path}  rows={len(df)} cols={len(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
