"""Train regime-conditional factor weights from real IC data.

Loads IC timeseries (parquet, columns: date | ic_<factor>...) produced by
compute_ic_timeseries.py, joins with HMM regime labels, and computes
per-regime factor weights using one of three methods:

  ic_ir_weighted   -- rank by IC-IR, clip negatives, normalise
  optimization     -- scipy.optimize maximise w'@mean_ic s.t. sum=1, 0<=w<=0.5
  shrinkage        -- (1-lambda)*data_weights + lambda*equal_weights  [default]

Output: configs/factor_weights_by_regime.json (compatible with multifactor_v2.py)

Usage
-----
python scripts/training/train_regime_weights.py \\
    --ic-dir output/factor_analysis/ic_timeseries \\
    --panel-path output/factor_panel.parquet \\
    --output-path configs/factor_weights_by_regime.json \\
    --method shrinkage \\
    --shrinkage-lambda 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PREFIX = "[REGIME-WT]"


def _log(msg: str) -> None:
    logger.info("%s %s", _PREFIX, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _PREFIX, msg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACTOR_CATEGORIES = [
    "momentum",
    "quality",
    "value",
    "volatility",
    "breadth",
    "mean_reversion",
    "macro",
    "intermarket",
    "ml_alpha",
]

KNOWN_REGIMES = ["bull", "bear", "sideways"]

_EQUAL_WEIGHT = 1.0 / len(FACTOR_CATEGORIES)

MAX_SINGLE_WEIGHT = 0.50
MIN_SINGLE_WEIGHT = 0.0


# ---------------------------------------------------------------------------
# Regime label generation
# ---------------------------------------------------------------------------

def generate_regime_labels(
    prices_df: pd.DataFrame,
    method: str = "hmm",
    n_regimes: int = 3,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    """Generate regime labels from price data.

    Fits a Gaussian HMM on the log-return series of `benchmark_symbol` (or
    the first column if the symbol is absent) and returns a tidy DataFrame.

    Args:
        prices_df: Wide DataFrame with DatetimeIndex or 'date' column, one
                   column per symbol (or a single-column returns series).
        method:    'hmm' (only supported method; reserved for future rule-based
                   fallback).
        n_regimes: Number of HMM states.
        benchmark_symbol: Column to use as the regime signal.

    Returns:
        DataFrame with columns: date | regime_label
        Falls back to 'sideways' for every row when HMM cannot be fit.
    """
    _log(f"Generating regime labels via {method} (n_regimes={n_regimes}, symbol={benchmark_symbol})")

    # ---- resolve date index ------------------------------------------------
    if "date" in prices_df.columns:
        prices_df = prices_df.set_index("date")
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()

    # ---- pick price series -------------------------------------------------
    if benchmark_symbol in prices_df.columns:
        price_series = prices_df[benchmark_symbol].dropna()
    elif prices_df.shape[1] == 1:
        price_series = prices_df.iloc[:, 0].dropna()
        _warn(f"Symbol '{benchmark_symbol}' not found; using first column: {prices_df.columns[0]}")
    else:
        _warn(f"Symbol '{benchmark_symbol}' not found; using row-mean as proxy.")
        price_series = prices_df.mean(axis=1).dropna()

    returns = np.log((price_series / price_series.shift(1)).clip(lower=1e-10)).dropna()

    if len(returns) < 60:
        _warn("Too few observations for HMM -- falling back to equal 'sideways' labels.")
        return _fallback_labels(returns.index)

    # ---- fit HMM -----------------------------------------------------------
    try:
        _log(f"[START] HMM fit on {len(returns)} observations")
        # Import here so environments without hmmlearn still get a graceful fallback
        from src.assembled_core.ml.regime_hmm import RegimeHMM  # type: ignore

        model = RegimeHMM(n_regimes=n_regimes, random_state=42)
        model.fit(returns)
        regime_series = model.predict_regime(returns)
        _log(f"[OK] HMM fit complete -- value counts: {regime_series.value_counts().to_dict()}")
    except Exception as exc:
        _warn(f"HMM fit failed ({exc!r}) -- falling back to 'sideways' for all rows.")
        return _fallback_labels(returns.index)

    result = regime_series.rename("regime_label").reset_index()
    # The index column name may vary (timestamp, date, index, etc.) -- normalize
    idx_col = [c for c in result.columns if c != "regime_label"][0]
    result = result.rename(columns={idx_col: "date"})
    result["date"] = pd.to_datetime(result["date"])
    return result[["date", "regime_label"]]


def _fallback_labels(index: pd.Index) -> pd.DataFrame:
    """Return a DataFrame assigning 'sideways' to every date."""
    dates = pd.to_datetime(index)
    return pd.DataFrame({"date": dates, "regime_label": "sideways"})


# ---------------------------------------------------------------------------
# Per-regime IC statistics
# ---------------------------------------------------------------------------

def compute_per_regime_ic(
    ic_timeseries_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    factor_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Join IC timeseries with HMM regime labels; compute per-regime IC stats.

    For each regime computes:
        mean_ic     -- time-series mean of the daily IC
        ic_ir       -- mean_ic / std_ic  (information ratio of the IC)
        hit_ratio   -- fraction of days with IC > 0
        t_stat      -- one-sample t-test vs zero (scipy)

    Args:
        ic_timeseries_df: DataFrame with 'date' column and IC columns
                          (e.g. 'ic_momentum', 'ic_quality', ...).
        regime_state_df:  DataFrame with 'date' and 'regime_label' columns.
        factor_cols:      IC column names to include (e.g. ['ic_momentum']).

    Returns:
        Dict mapping regime name -> summary DataFrame with rows = factors and
        columns = [mean_ic, ic_ir, hit_ratio, t_stat, n_obs].
    """
    _log("[START] compute_per_regime_ic")

    ic_df = ic_timeseries_df.copy()
    if "date" not in ic_df.columns:
        if ic_df.index.name == "date":
            ic_df = ic_df.reset_index()
        else:
            raise ValueError("ic_timeseries_df must have a 'date' column or index.")

    ic_df["date"] = pd.to_datetime(ic_df["date"]).dt.normalize()

    reg_df = regime_state_df.copy()
    reg_df["date"] = pd.to_datetime(reg_df["date"]).dt.normalize()

    # keep only factor cols that actually exist
    available = [c for c in factor_cols if c in ic_df.columns]
    missing = set(factor_cols) - set(available)
    if missing:
        _warn(f"Skipping {len(missing)} factor cols not in IC timeseries: {sorted(missing)[:10]}")
    if not available:
        _warn("No valid factor columns -- returning empty dict.")
        return {}

    merged = ic_df[["date"] + available].merge(
        reg_df[["date", "regime_label"]], on="date", how="inner"
    )
    _log(f"Merged rows: {len(merged)} across {merged['regime_label'].nunique()} regimes")

    results: dict[str, pd.DataFrame] = {}

    for regime, group in merged.groupby("regime_label"):
        rows = []
        for col in available:
            series = group[col].dropna()
            n = len(series)
            if n < 2:
                rows.append(
                    {"factor": col, "mean_ic": np.nan, "ic_ir": np.nan,
                     "hit_ratio": np.nan, "t_stat": np.nan, "n_obs": n}
                )
                continue
            mean_ic = float(series.mean())
            std_ic = float(series.std(ddof=1))
            ic_ir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
            hit_ratio = float((series > 0).mean())
            t_stat_val, _ = ttest_1samp(series, popmean=0.0)
            rows.append({
                "factor": col,
                "mean_ic": round(mean_ic, 6),
                "ic_ir": round(ic_ir, 6),
                "hit_ratio": round(hit_ratio, 4),
                "t_stat": round(float(t_stat_val), 4),
                "n_obs": n,
            })
        df_out = pd.DataFrame(rows).set_index("factor")
        results[str(regime)] = df_out
        _log(f"[OK] Regime '{regime}': {len(group)} obs, {len(available)} factors computed")

    return results


# ---------------------------------------------------------------------------
# Weight computation helpers
# ---------------------------------------------------------------------------

def _equal_weights(factor_cols: list[str]) -> dict[str, float]:
    """Return equal positive weights summing to 1."""
    n = len(factor_cols)
    if n == 0:
        return {}
    w = round(1.0 / n, 8)
    return {c: w for c in factor_cols}


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    """Clip to [MIN, MAX] and normalise to sum = 1. Fallback to equal if all zero."""
    clipped = {k: max(MIN_SINGLE_WEIGHT, min(MAX_SINGLE_WEIGHT, v)) for k, v in raw.items()}
    total = sum(clipped.values())
    if total < 1e-12:
        return _equal_weights(list(raw.keys()))
    return {k: round(v / total, 8) for k, v in clipped.items()}


def _ic_ir_weighted(stats_df: pd.DataFrame, factor_cols: list[str]) -> dict[str, float]:
    """Compute IC-IR weighted factor weights for a single regime."""
    raw: dict[str, float] = {}
    for col in factor_cols:
        if col in stats_df.index:
            ic_ir = stats_df.loc[col, "ic_ir"]
            raw[col] = float(ic_ir) if (not np.isnan(ic_ir) and ic_ir > 0) else 0.0
        else:
            raw[col] = 0.0
    return _normalise(raw)


def _optimisation_weights(stats_df: pd.DataFrame, factor_cols: list[str]) -> dict[str, float]:
    """Scipy optimisation: maximise w'@mean_ic s.t. sum(w)=1, 0<=w<=0.5."""
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError:
        _warn("scipy not available -- falling back to IC-IR weighted method.")
        return _ic_ir_weighted(stats_df, factor_cols)

    mean_ics = np.array([
        float(stats_df.loc[c, "mean_ic"]) if c in stats_df.index else 0.0
        for c in factor_cols
    ])
    mean_ics = np.nan_to_num(mean_ics, nan=0.0)

    n = len(factor_cols)
    if n == 0:
        return {}

    w0 = np.full(n, 1.0 / n)

    # objective: minimise -w'@mean_ic
    def neg_portfolio_ic(w: np.ndarray) -> float:
        return -float(w @ mean_ics)

    def neg_portfolio_ic_grad(w: np.ndarray) -> np.ndarray:
        return -mean_ics

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(MIN_SINGLE_WEIGHT, MAX_SINGLE_WEIGHT)] * n

    res = minimize(
        neg_portfolio_ic,
        w0,
        jac=neg_portfolio_ic_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if res.success:
        w_opt = np.clip(res.x, MIN_SINGLE_WEIGHT, MAX_SINGLE_WEIGHT)
        w_sum = w_opt.sum()
        w_opt = w_opt / w_sum if w_sum > 1e-12 else np.full_like(w_opt, 1.0 / len(w_opt))
        return {c: round(float(w), 8) for c, w in zip(factor_cols, w_opt)}
    else:
        _warn(f"Optimisation did not converge ({res.message}) -- falling back to IC-IR method.")
        return _ic_ir_weighted(stats_df, factor_cols)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_regime_weights(
    ic_timeseries_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    factor_cols: list[str],
    method: str = "shrinkage",
    shrinkage_to_equal: float = 0.3,
    min_days_per_regime: int = 60,
) -> dict[str, dict[str, float]]:
    """Train regime-conditional factor weights from IC data.

    Methods
    -------
    ic_ir_weighted
        w_f = max(0, IC_IR_f) / sum(max(0, IC_IR_f))
    optimization
        Maximise w'@mean_ic subject to: sum(w)=1, 0<=w<=0.5 (scipy SLSQP).
    shrinkage  [recommended]
        w_final = (1-lambda)*w_data + lambda*w_equal

    Guardrails
    ----------
    * No factor weight > 50 %.
    * Minimum `min_days_per_regime` observations per regime; else equal weights.
    * Weights always sum to 1.
    * Factor cols missing from IC timeseries are silently skipped.

    Args:
        ic_timeseries_df:  IC timeseries (date | ic_<factor>...).
        regime_state_df:   Regime labels (date | regime_label).
        factor_cols:       IC column names to use (e.g. ['ic_momentum', ...]).
        method:            'ic_ir_weighted' | 'optimization' | 'shrinkage'.
        shrinkage_to_equal: Lambda for shrinkage (0=pure data, 1=equal weights).
        min_days_per_regime: Minimum observations; regimes below this get equal weights.

    Returns:
        Dict: regime -> {factor_col: weight, ...}
    """
    _log(f"[START] train_regime_weights | method={method} | lambda={shrinkage_to_equal}")

    per_regime_stats = compute_per_regime_ic(
        ic_timeseries_df, regime_state_df, factor_cols
    )

    # ensure all KNOWN_REGIMES are present (even if empty)
    for r in KNOWN_REGIMES:
        if r not in per_regime_stats:
            _warn(f"Regime '{r}' has no IC data -- assigning equal weights.")

    # valid factor cols (intersection with what was computed)
    computed_cols = (
        list(next(iter(per_regime_stats.values())).index)
        if per_regime_stats else factor_cols
    )

    regime_weights: dict[str, dict[str, float]] = {}

    target_regimes = list(set(KNOWN_REGIMES) | set(per_regime_stats.keys()))

    for regime in target_regimes:
        eq = _equal_weights(computed_cols)

        if regime not in per_regime_stats:
            regime_weights[regime] = eq
            continue

        stats_df = per_regime_stats[regime]
        n_obs = int(stats_df["n_obs"].max()) if "n_obs" in stats_df.columns else 0

        if n_obs < min_days_per_regime:
            _warn(
                f"Regime '{regime}' has only {n_obs} obs "
                f"(< min {min_days_per_regime}) -- using equal weights."
            )
            regime_weights[regime] = eq
            continue

        # ---- compute data-driven weights ----------------------------------
        if method == "ic_ir_weighted":
            w_data = _ic_ir_weighted(stats_df, computed_cols)
        elif method == "optimization":
            w_data = _optimisation_weights(stats_df, computed_cols)
        elif method == "shrinkage":
            w_raw = _ic_ir_weighted(stats_df, computed_cols)
            lam = float(shrinkage_to_equal)
            w_data = {
                c: round((1.0 - lam) * w_raw.get(c, 0.0) + lam * eq.get(c, 0.0), 8)
                for c in computed_cols
            }
            w_data = _normalise(w_data)
        else:
            _warn(f"Unknown method '{method}' -- falling back to shrinkage.")
            w_raw = _ic_ir_weighted(stats_df, computed_cols)
            lam = float(shrinkage_to_equal)
            w_data = {
                c: round((1.0 - lam) * w_raw.get(c, 0.0) + lam * eq.get(c, 0.0), 8)
                for c in computed_cols
            }
            w_data = _normalise(w_data)

        regime_weights[regime] = w_data
        _log(
            f"[OK] Regime '{regime}': {n_obs} obs | "
            f"top-3 = {sorted(w_data.items(), key=lambda x: -x[1])[:3]}"
        )

    return regime_weights


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def validate_regime_weights_wf(
    panel_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Walk-forward validation of regime weights vs a static equal-weight baseline.

    Splits the IC timeseries chronologically into `n_splits` folds. In each
    fold:
      * Train: all data up to fold boundary.
      * Test: next fold window.
      * Compute weighted-average IC (using trained weights) vs equal-weight IC.
      * Record out-of-sample improvement (wt_avg_ic - eq_ic).

    Args:
        panel_df:        IC timeseries DataFrame (date | ic_<factor>...).
        regime_state_df: Regime labels (date | regime_label).
        n_splits:        Number of walk-forward folds.

    Returns:
        Dict with keys:
            fold_results   -- list of per-fold dicts
            mean_improvement -- average (wt_ic - eq_ic) across folds
            wt_sharpe      -- Sharpe of the improvement series
    """
    _log(f"[START] walk-forward validation | n_splits={n_splits}")

    if "date" not in panel_df.columns:
        if panel_df.index.name == "date":
            panel_df = panel_df.reset_index()
        else:
            raise ValueError("panel_df must have a 'date' column or index.")

    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel_df = panel_df.sort_values("date").reset_index(drop=True)

    factor_cols = [c for c in panel_df.columns if c.startswith("ic_")]
    if not factor_cols:
        _warn("No IC columns found in panel_df -- returning empty validation result.")
        return {"fold_results": [], "mean_improvement": 0.0, "wt_sharpe": np.nan}

    dates = panel_df["date"].values
    n = len(dates)
    if n < n_splits * 2:
        _warn(f"Too few rows ({n}) for {n_splits} splits -- reducing n_splits to 2.")
        n_splits = 2

    fold_size = n // n_splits
    fold_results = []
    improvements = []

    for fold in range(n_splits):
        train_end_idx = (fold + 1) * fold_size
        test_end_idx = min((fold + 2) * fold_size, n)

        if train_end_idx >= n:
            break

        train_df = panel_df.iloc[:train_end_idx]
        test_df = panel_df.iloc[train_end_idx:test_end_idx]

        if len(test_df) == 0:
            continue

        # Train weights
        try:
            weights = train_regime_weights(
                train_df,
                regime_state_df,
                factor_cols,
                method="shrinkage",
                shrinkage_to_equal=0.3,
            )
        except Exception as exc:
            _warn(f"Fold {fold}: training failed ({exc!r}) -- skipping.")
            continue

        # Merge test IC with regime labels
        reg_df = regime_state_df.copy()
        reg_df["date"] = pd.to_datetime(reg_df["date"])
        test_merged = test_df.merge(reg_df[["date", "regime_label"]], on="date", how="left")
        test_merged["regime_label"] = test_merged["regime_label"].fillna("sideways")

        # Compute weighted vs equal IC per row
        wt_ics, eq_ics = [], []
        eq_w = 1.0 / len(factor_cols)

        for _, row in test_merged.iterrows():
            regime = str(row["regime_label"])
            w_map = weights.get(regime, _equal_weights(factor_cols))
            ic_vals = {c: row[c] for c in factor_cols if not np.isnan(row[c])}
            if not ic_vals:
                continue
            wt_ic = sum(w_map.get(c, eq_w) * v for c, v in ic_vals.items())
            eq_ic = sum(eq_w * v for v in ic_vals.values())
            wt_ics.append(wt_ic)
            eq_ics.append(eq_ic)

        if not wt_ics:
            continue

        mean_wt = float(np.mean(wt_ics))
        mean_eq = float(np.mean(eq_ics))
        improvement = mean_wt - mean_eq
        improvements.append(improvement)

        fold_results.append({
            "fold": fold,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "wt_avg_ic": round(mean_wt, 6),
            "eq_avg_ic": round(mean_eq, 6),
            "improvement": round(improvement, 6),
        })
        _log(
            f"[OK] Fold {fold}: wt_ic={mean_wt:.4f} | eq_ic={mean_eq:.4f} | "
            f"delta={improvement:+.4f}"
        )

    if not improvements:
        return {"fold_results": fold_results, "mean_improvement": 0.0, "wt_sharpe": np.nan}

    arr = np.array(improvements)
    mean_imp = float(arr.mean())
    wt_sharpe = float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 1e-12 else np.nan

    _log(
        f"[DONE] WF validation | mean_improvement={mean_imp:+.4f} | "
        f"wt_sharpe={wt_sharpe:.2f} | folds={len(improvements)}"
    )

    return {
        "fold_results": fold_results,
        "mean_improvement": round(mean_imp, 6),
        "wt_sharpe": round(wt_sharpe, 4) if not np.isnan(wt_sharpe) else None,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_ic_timeseries(ic_dir: Path) -> pd.DataFrame | None:
    """Load and concatenate all ic_timeseries_*.parquet files in ic_dir."""
    parquet_files = sorted(ic_dir.glob("ic_timeseries_*.parquet"))
    if not parquet_files:
        _warn(f"No ic_timeseries_*.parquet files found in {ic_dir}")
        return None

    frames = []
    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
            _log(f"Loaded {fp.name}: {len(df)} rows, {len(df.columns)-1} factors")
            frames.append(df)
        except Exception as exc:
            _warn(f"Failed to load {fp}: {exc!r}")

    if not frames:
        return None

    # Concatenate; on date conflicts keep last (longest horizon typically wins)
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values("date").reset_index(drop=True)

    _log(f"Combined IC timeseries: {len(combined)} rows")
    return combined


def _build_meta(
    method: str,
    shrinkage_lambda: float,
    regime_weights: dict[str, dict[str, float]],
    per_regime_stats: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build the _meta block for the output JSON."""
    n_days: dict[str, int] = {}
    if per_regime_stats:
        for regime, stats_df in per_regime_stats.items():
            if "n_obs" in stats_df.columns:
                n_days[regime] = int(stats_df["n_obs"].max())

    return {
        "method": method,
        "shrinkage_lambda": shrinkage_lambda,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_days": n_days,
        "regimes_trained": sorted(regime_weights.keys()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train regime-conditional factor weights from IC timeseries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ic-dir",
        type=Path,
        default=Path("output/factor_analysis/ic_timeseries"),
        help="Directory containing ic_timeseries_*.parquet files.",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=None,
        help="Path to price/returns panel parquet (used to generate regime labels via HMM). "
             "Required unless --regime-labels-path is given.",
    )
    parser.add_argument(
        "--regime-labels-path",
        type=Path,
        default=None,
        help="Pre-computed regime labels parquet (date | regime_label). "
             "If provided, --panel-path is not needed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("configs/factor_weights_by_regime.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--method",
        choices=["ic_ir_weighted", "optimization", "shrinkage"],
        default="shrinkage",
        help="Weight-training method.",
    )
    parser.add_argument(
        "--shrinkage-lambda",
        type=float,
        default=0.3,
        help="Shrinkage towards equal weights (0=pure data, 1=equal). Default 0.3.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="SPY",
        help="Symbol to use as HMM regime signal.",
    )
    parser.add_argument(
        "--n-regimes",
        type=int,
        default=3,
        help="Number of HMM regime states.",
    )
    parser.add_argument(
        "--min-days-per-regime",
        type=int,
        default=60,
        help="Minimum observations per regime; below this equal weights are used.",
    )
    parser.add_argument(
        "--run-wf-validation",
        action="store_true",
        default=False,
        help="Also run walk-forward validation and print summary.",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=5,
        help="Number of walk-forward folds.",
    )
    args = parser.parse_args(argv)

    _log("=" * 60)
    _log("train_regime_weights.py -- START")
    _log(f"ic_dir          : {args.ic_dir}")
    _log(f"output_path     : {args.output_path}")
    _log(f"method          : {args.method}")
    _log(f"shrinkage_lambda: {args.shrinkage_lambda}")
    _log("=" * 60)

    # ------------------------------------------------------------------ 1. IC
    ic_df = _load_ic_timeseries(args.ic_dir)
    if ic_df is None:
        _warn("No IC data available -- cannot train. Exiting.")
        sys.exit(1)

    factor_cols = [c for c in ic_df.columns if c.startswith("ic_")]
    _log(f"Factor columns in IC data: {len(factor_cols)} -> {factor_cols[:8]}{'...' if len(factor_cols) > 8 else ''}")

    # ---------------------------------------------------------- 2. Regime labels
    if args.regime_labels_path and args.regime_labels_path.exists():
        _log(f"Loading pre-computed regime labels from {args.regime_labels_path}")
        regime_df = pd.read_parquet(args.regime_labels_path)
        regime_df["date"] = pd.to_datetime(regime_df["date"])
    elif args.panel_path and args.panel_path.exists():
        _log(f"Loading panel from {args.panel_path} to generate HMM regime labels")
        raw_panel = pd.read_parquet(args.panel_path)
        # Pivot from long (timestamp, symbol, close) to wide (date x symbol) for HMM
        prices_df = raw_panel.pivot_table(
            index="timestamp", columns="symbol", values="close", aggfunc="first"
        )
        regime_df = generate_regime_labels(
            prices_df,
            method="hmm",
            n_regimes=args.n_regimes,
            benchmark_symbol=args.benchmark_symbol,
        )
    else:
        _warn(
            "Neither --regime-labels-path nor a valid --panel-path supplied. "
            "Assigning 'sideways' to all dates as fallback."
        )
        if "date" in ic_df.columns:
            regime_df = _fallback_labels(pd.to_datetime(ic_df["date"]))
        else:
            regime_df = _fallback_labels(pd.RangeIndex(len(ic_df)))

    # ---------------------------------------------------------- 3. Train weights
    regime_weights = train_regime_weights(
        ic_timeseries_df=ic_df,
        regime_state_df=regime_df,
        factor_cols=factor_cols,
        method=args.method,
        shrinkage_to_equal=args.shrinkage_lambda,
        min_days_per_regime=args.min_days_per_regime,
    )

    # ---- build per-regime stats for meta block ----------------------------
    per_regime_stats = compute_per_regime_ic(ic_df, regime_df, factor_cols)
    meta = _build_meta(args.method, args.shrinkage_lambda, regime_weights, per_regime_stats)

    # ---------------------------------------------------------- 4. Save JSON
    output: dict[str, Any] = {}
    for regime in sorted(regime_weights.keys()):
        output[regime] = regime_weights[regime]
    output["_meta"] = meta

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    _log(f"[OK] Weights written to {args.output_path}")
    _log(f"Regimes: {sorted(regime_weights.keys())}")

    # ---------------------------------------------------------- 5. WF validation
    if args.run_wf_validation:
        _log("Running walk-forward validation ...")
        wf_result = validate_regime_weights_wf(
            panel_df=ic_df,
            regime_state_df=regime_df,
            n_splits=args.wf_splits,
        )
        _log(
            f"WF result: mean_improvement={wf_result['mean_improvement']:+.4f} | "
            f"wt_sharpe={wf_result['wt_sharpe']}"
        )
        for fr in wf_result["fold_results"]:
            _log(
                f"  fold={fr['fold']} | train={fr['train_rows']} | test={fr['test_rows']} | "
                f"wt_ic={fr['wt_avg_ic']:.4f} | eq_ic={fr['eq_avg_ic']:.4f} | delta={fr['improvement']:+.4f}"
            )

    _log("train_regime_weights.py -- DONE")


if __name__ == "__main__":
    main()
