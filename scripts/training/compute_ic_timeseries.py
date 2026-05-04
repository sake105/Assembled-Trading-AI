"""
compute_ic_timeseries.py
------------------------
Compute Information Coefficient (IC) timeseries for all factors in the
Assembled-Trading-AI factor panel.

IC = Spearman rank correlation between factor values and forward returns,
computed cross-sectionally per date.

Usage:
    python scripts/training/compute_ic_timeseries.py
    python scripts/training/compute_ic_timeseries.py \
        --panel-path output/factor_panels/full_panel_7y.parquet \
        --output-dir output/factor_analysis/ic_timeseries
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
LOG_PREFIX = "[IC]"


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"{LOG_PREFIX} [WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
_META_COLS = {"date", "symbol", "timestamp", "label"}
_FWD_RETURN_PREFIX = "fwd_return_"
_EXCLUDED_PREFIXES = ("fwd_return_", "returns_")


def _identify_factor_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns that are actual factors.

    Excludes:
    - Meta columns (date, symbol, timestamp, label)
    - Forward return columns (fwd_return_*) -- these ARE the target
    - Raw return columns (returns_*) -- also future-looking
    - Any non-numeric columns
    """
    return [
        c
        for c in df.columns
        if c not in _META_COLS
        and not any(c.startswith(p) for p in _EXCLUDED_PREFIXES)
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]


def _fwd_return_col(horizon: int) -> str:
    return f"{_FWD_RETURN_PREFIX}{horizon}d"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_ic_timeseries_all_factors(
    panel_path: Path = Path("output/factor_panels/full_panel_7y.parquet"),
    horizons: list[int] | None = None,
    method: str = "spearman",
    output_dir: Path = Path("output/factor_analysis/ic_timeseries"),
    min_symbols_per_date: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compute IC timeseries for all factors across all horizons.

    Per date, per factor:
        IC = spearman_corr(factor_values_cross_section, fwd_return_cross_section)

    Parameters
    ----------
    panel_path:
        Path to the factor panel Parquet file.
        Schema: date | symbol | factor_col_1 | ... | fwd_return_5d | fwd_return_10d | ...
    horizons:
        Forward-return horizons in days to evaluate (default [5, 10, 20]).
    method:
        Correlation method; only "spearman" is officially supported here.
    output_dir:
        Directory where IC timeseries parquets and summary JSON are saved.
    min_symbols_per_date:
        Minimum number of valid (non-NaN) symbols required to compute IC for a date.

    Returns
    -------
    dict keyed by horizon label, e.g. {"5d": DataFrame, "10d": DataFrame, "20d": DataFrame}.
    Each DataFrame has columns: date | ic_<factor_1> | ic_<factor_2> | ...
    """
    if horizons is None:
        horizons = [5, 10, 20]

    # ------------------------------------------------------------------
    # Load panel
    # ------------------------------------------------------------------
    panel_path = Path(panel_path)
    if not panel_path.exists():
        _warn(f"Panel not found: {panel_path}")
        sys.exit(1)

    _log(f"Loading panel from {panel_path} ...")
    df = pd.read_parquet(panel_path)
    _log(f"Panel loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Normalise date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        _warn("No 'date' column found in panel -- aborting.")
        sys.exit(1)

    factor_cols = _identify_factor_cols(df)
    if not factor_cols:
        _warn("No factor columns identified in panel -- aborting.")
        sys.exit(1)

    _log(f"Factor columns identified: {len(factor_cols)}")
    _log(f"Horizons: {horizons}")
    _log(f"Method: {method}")
    _log(f"Min symbols per date: {min_symbols_per_date}")

    # Verify forward-return columns exist
    available_horizons: list[int] = []
    for h in horizons:
        col = _fwd_return_col(h)
        if col in df.columns:
            available_horizons.append(h)
        else:
            _warn(
                f"Forward-return column '{col}' not in panel -- skipping horizon {h}d"
            )

    if not available_horizons:
        _warn("No valid forward-return columns found -- aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute IC per date per factor per horizon
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted(df["date"].unique())
    _log(f"Dates in panel: {len(dates)}")

    results: dict[str, dict] = {f"{h}d": {} for h in available_horizons}

    skipped_dates = 0
    processed_dates = 0

    for date in dates:
        day_df = df[df["date"] == date].copy()

        # Drop rows where symbol is missing
        day_df = day_df.dropna(subset=["symbol"])

        if len(day_df) < min_symbols_per_date:
            skipped_dates += 1
            continue

        processed_dates += 1

        for h in available_horizons:
            fwd_col = _fwd_return_col(h)
            horizon_key = f"{h}d"

            for factor in factor_cols:
                # Drop rows where either factor or fwd_return is NaN
                valid = day_df[[factor, fwd_col]].dropna()

                if len(valid) < min_symbols_per_date:
                    # Not enough valid observations for this factor on this date
                    continue

                if method == "spearman":
                    corr, _ = spearmanr(valid[factor].values, valid[fwd_col].values)
                else:
                    # Fallback: Pearson
                    corr = float(
                        np.corrcoef(valid[factor].values, valid[fwd_col].values)[0, 1]
                    )

                ic_col = f"ic_{factor}"
                if ic_col not in results[horizon_key]:
                    results[horizon_key][ic_col] = {}

                results[horizon_key][ic_col][date] = (
                    float(corr) if not math.isnan(corr) else np.nan
                )

    _log(
        f"Processed {processed_dates} dates, skipped {skipped_dates} "
        f"(< {min_symbols_per_date} symbols)"
    )

    # ------------------------------------------------------------------
    # Build DataFrames and save
    # ------------------------------------------------------------------
    ic_timeseries: dict[str, pd.DataFrame] = {}

    for h in available_horizons:
        horizon_key = f"{h}d"
        data = results[horizon_key]

        if not data:
            _warn(f"No IC data for horizon {horizon_key}")
            continue

        ic_df = pd.DataFrame(data)
        ic_df.index.name = "date"
        ic_df = ic_df.reset_index().sort_values("date")

        out_path = output_dir / f"ic_timeseries_{horizon_key}.parquet"
        ic_df.to_parquet(out_path, index=False)
        _log(
            f"[OK] Saved IC timeseries {horizon_key} -> {out_path} ({len(ic_df)} rows, {len(ic_df.columns)-1} factors)"
        )

        ic_timeseries[horizon_key] = ic_df

    return ic_timeseries


# ---------------------------------------------------------------------------
# IC Summary
# ---------------------------------------------------------------------------


def compute_ic_summary(
    ic_timeseries: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict]]:
    """
    Compute per-factor per-horizon summary statistics.

    Returns
    -------
    Nested dict:
        { horizon_key: { factor_name: { mean_ic, ic_ir, hit_ratio, t_stat,
                                        n_periods, max_ic, min_ic } } }

    Statistics
    ----------
    mean_ic   : average IC across all valid dates
    ic_ir     : mean_ic / std_ic  (Information Ratio -- measures IC stability)
    hit_ratio : fraction of dates where IC > 0
    t_stat    : mean_ic / (std_ic / sqrt(n_periods))
    n_periods : number of dates with valid IC
    max_ic    : maximum IC value across all dates
    min_ic    : minimum IC value across all dates
    """
    summary: dict[str, dict[str, dict]] = {}

    for horizon_key, ic_df in ic_timeseries.items():
        ic_cols = [c for c in ic_df.columns if c.startswith("ic_")]
        horizon_summary: dict[str, dict] = {}

        for ic_col in ic_cols:
            factor_name = ic_col[len("ic_") :]  # strip "ic_" prefix
            series = ic_df[ic_col].dropna()
            n = len(series)

            if n < 2:
                horizon_summary[factor_name] = {
                    "mean_ic": np.nan,
                    "ic_ir": np.nan,
                    "hit_ratio": np.nan,
                    "t_stat": np.nan,
                    "n_periods": n,
                    "max_ic": np.nan,
                    "min_ic": np.nan,
                }
                continue

            mean_ic = float(series.mean())
            std_ic = float(series.std(ddof=1))

            ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
            hit_ratio = float((series > 0).mean())
            t_stat = mean_ic / (std_ic / math.sqrt(n)) if std_ic > 0 else np.nan

            horizon_summary[factor_name] = {
                "mean_ic": round(mean_ic, 6),
                "ic_ir": round(ic_ir, 4) if not math.isnan(ic_ir) else np.nan,
                "hit_ratio": round(hit_ratio, 4),
                "t_stat": round(t_stat, 4) if not math.isnan(t_stat) else np.nan,
                "n_periods": n,
                "max_ic": round(float(series.max()), 6),
                "min_ic": round(float(series.min()), 6),
            }

        summary[horizon_key] = horizon_summary

    return summary


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_factor_table(
    summary: dict[str, dict[str, dict]],
    horizon_key: str = "20d",
) -> None:
    """Print ranked factor table for the given horizon to stdout."""
    if horizon_key not in summary:
        _warn(f"Horizon '{horizon_key}' not in summary -- cannot print table")
        return

    horizon_data = summary[horizon_key]
    rows = []
    for factor, stats in horizon_data.items():
        rows.append(
            {
                "factor": factor,
                "mean_ic": stats.get("mean_ic", np.nan),
                "ic_ir": stats.get("ic_ir", np.nan),
                "hit_ratio": stats.get("hit_ratio", np.nan),
                "t_stat": stats.get("t_stat", np.nan),
                "n_periods": stats.get("n_periods", 0),
            }
        )

    if not rows:
        _warn("No rows to display")
        return

    table = pd.DataFrame(rows).sort_values("ic_ir", ascending=False, na_position="last")

    print()
    print(f"{'='*80}")
    print(f"  Factor IC Rankings -- Horizon: {horizon_key}")
    print(f"{'='*80}")
    header = f"{'Factor':<35} {'MeanIC':>8} {'IC-IR':>7} {'HitRatio':>9} {'t-stat':>8} {'N':>6}"
    print(header)
    print(f"{'-'*80}")

    for row in table.itertuples(index=False):
        mean_ic_str = (
            f"{row.mean_ic:8.4f}" if not math.isnan(row.mean_ic) else "     NaN"
        )
        ic_ir_str = f"{row.ic_ir:7.3f}" if not math.isnan(row.ic_ir) else "    NaN"
        hit_str = (
            f"{row.hit_ratio:9.3f}" if not math.isnan(row.hit_ratio) else "      NaN"
        )
        t_str = f"{row.t_stat:8.3f}" if not math.isnan(row.t_stat) else "     NaN"
        print(
            f"{row.factor:<35} {mean_ic_str} {ic_ir_str} {hit_str} {t_str} {int(row.n_periods):>6}"
        )

    print(f"{'='*80}")
    print()


def _gate_check(
    summary: dict[str, dict[str, dict]],
    horizon_key: str = "20d",
    required: int = 15,
) -> None:
    """Report how many factors have IC-IR > 0 at the given horizon."""
    if horizon_key not in summary:
        _warn(f"Gate check skipped -- horizon '{horizon_key}' not in summary")
        return

    horizon_data = summary[horizon_key]
    total = len(horizon_data)
    positive_ir = sum(
        1
        for stats in horizon_data.values()
        if not math.isnan(stats.get("ic_ir", math.nan))
        and stats.get("ic_ir", math.nan) > 0
    )

    status = "PASS" if positive_ir >= required else "FAIL"
    _log(
        f"[GATE] Factors with IC-IR > 0 at {horizon_key}: "
        f"{positive_ir}/{total} (need >= {required}) -- {status}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IC timeseries for all factors in the factor panel."
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("output/factor_panels/full_panel_7y.parquet"),
        help="Path to factor panel Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/factor_analysis/ic_timeseries"),
        help="Directory where IC timeseries parquets and summary JSON are saved.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Forward-return horizons in days (default: 5 10 20).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="spearman",
        choices=["spearman", "pearson"],
        help="Correlation method (default: spearman).",
    )
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=10,
        dest="min_symbols_per_date",
        help="Minimum symbols per date to compute IC (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    _log("Starting IC timeseries computation ...")
    _log(f"Panel path    : {args.panel_path}")
    _log(f"Output dir    : {args.output_dir}")
    _log(f"Horizons      : {args.horizons}")
    _log(f"Method        : {args.method}")
    _log(f"Min symbols   : {args.min_symbols_per_date}")

    # Step 1: Compute IC timeseries
    ic_timeseries = compute_ic_timeseries_all_factors(
        panel_path=args.panel_path,
        horizons=args.horizons,
        method=args.method,
        output_dir=args.output_dir,
        min_symbols_per_date=args.min_symbols_per_date,
    )

    if not ic_timeseries:
        _warn("No IC timeseries produced -- check panel and horizons.")
        sys.exit(1)

    # Step 2: Compute summary statistics
    _log("Computing IC summary statistics ...")
    summary = compute_ic_summary(ic_timeseries)

    # Step 3: Save summary JSON
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "ic_summary.json"

    # Convert any NaN to None for JSON serialisation
    def _sanitise(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(v) for v in obj]
        return obj

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitise(summary), fh, indent=2)
    _log(f"[OK] Summary saved -> {summary_path}")

    # Step 4: Print ranked factor table for primary horizon (20d)
    primary_horizon = (
        "20d" if "20d" in ic_timeseries else sorted(ic_timeseries.keys())[-1]
    )
    _print_factor_table(summary, horizon_key=primary_horizon)

    # Step 5: Gate check
    _gate_check(summary, horizon_key=primary_horizon, required=15)

    _log("Done.")
