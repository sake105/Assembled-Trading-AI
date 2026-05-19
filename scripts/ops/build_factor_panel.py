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
    args = parser.parse_args()

    print(f"[START] reading {args.input_path}")
    df = pd.read_parquet(args.input_path)
    print(
        f"  loaded: rows={len(df)} cols={len(df.columns)} syms={df['symbol'].nunique()}"
    )
    print(f"  range: {df['timestamp'].min()} .. {df['timestamp'].max()}")

    # Normalize timestamp dtype (some panels store tz-naive)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"[OK] wrote -> {args.output_path}  rows={len(df)} cols={len(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
