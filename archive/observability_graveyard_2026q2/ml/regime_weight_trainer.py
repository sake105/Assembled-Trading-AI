"""Regime-conditional factor weight training -- library module.

This module provides pure functions for computing per-regime IC statistics
and training regime-conditional factor weights from an IC timeseries.

Public API
----------
compute_per_regime_ic(ic_timeseries_df, regime_state_df, factor_cols)
    -> dict[str, DataFrame]

train_regime_weights(ic_timeseries_df, regime_state_df, factor_cols, ...)
    -> dict[str, dict[str, float]]

validate_regime_weights_wf(panel_df, regime_state_df, n_splits=5)
    -> dict

Intended use: import from other modules or call via scripts/training/train_regime_weights.py.
Do NOT run this file directly (it is a library, not a script).

Methods available
-----------------
ic_ir_weighted
    w_f = max(0, IC_IR_f) / sum(max(0, IC_IR_f))
optimization
    Maximise w'@mean_ic s.t. sum(w)=1, 0<=w<=0.5 via scipy SLSQP.
shrinkage  [recommended]
    w_final = (1-lambda)*w_data + lambda*w_equal

Guardrails
----------
* No single factor weight > 50 %.
* Minimum min_days_per_regime obs per regime; else fall back to equal weights.
* Weights always sum to 1.0.
* Factor cols missing from IC timeseries are silently skipped.

Log prefix: [REGIME-TRAIN]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_1samp as _ttest_1samp
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
_TAG = "[REGIME-TRAIN]"


def _log(msg: str) -> None:
    logger.info("%s %s", _TAG, msg)


def _warn(msg: str) -> None:
    logger.warning("%s %s", _TAG, msg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KNOWN_REGIMES: list[str] = ["bull", "bear", "sideways"]

MAX_SINGLE_WEIGHT: float = 0.50
MIN_SINGLE_WEIGHT: float = 0.00

_MIN_OBS_DEFAULT: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _equal_weights(factor_cols: list[str]) -> dict[str, float]:
    """Return equal positive weights summing to 1."""
    n = len(factor_cols)
    if n == 0:
        return {}
    w = round(1.0 / n, 8)
    return {c: w for c in factor_cols}


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    """Clip to [MIN, MAX] and normalise to sum = 1.  Fallback: equal weights."""
    clipped = {
        k: max(MIN_SINGLE_WEIGHT, min(MAX_SINGLE_WEIGHT, v))
        for k, v in raw.items()
    }
    total = sum(clipped.values())
    if total < 1e-12:
        return _equal_weights(list(raw.keys()))
    return {k: round(v / total, 8) for k, v in clipped.items()}


def _ic_ir_weighted(
    stats_df: pd.DataFrame,
    factor_cols: list[str],
) -> dict[str, float]:
    """Compute IC-IR weighted factor weights for a single regime."""
    raw: dict[str, float] = {}
    for col in factor_cols:
        if col in stats_df.index:
            ic_ir = stats_df.loc[col, "ic_ir"]
            raw[col] = float(ic_ir) if (not np.isnan(ic_ir) and ic_ir > 0) else 0.0
        else:
            raw[col] = 0.0
    return _normalise(raw)


def _optimisation_weights(
    stats_df: pd.DataFrame,
    factor_cols: list[str],
) -> dict[str, float]:
    """Scipy SLSQP: maximise w'@mean_ic s.t. sum(w)=1, 0<=w<=0.5."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        _warn("scipy not available -- falling back to IC-IR weighted method.")
        return _ic_ir_weighted(stats_df, factor_cols)

    n = len(factor_cols)
    if n == 0:
        return {}

    mean_ics = np.array(
        [
            float(stats_df.loc[c, "mean_ic"]) if c in stats_df.index else 0.0
            for c in factor_cols
        ],
        dtype=float,
    )
    mean_ics = np.nan_to_num(mean_ics, nan=0.0)

    w0 = np.full(n, 1.0 / n)

    def _neg_ic(w: np.ndarray) -> float:
        return -float(w @ mean_ics)

    def _neg_ic_grad(w: np.ndarray) -> np.ndarray:
        return -mean_ics

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(MIN_SINGLE_WEIGHT, MAX_SINGLE_WEIGHT)] * n

    res = minimize(
        _neg_ic,
        w0,
        jac=_neg_ic_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if res.success:
        w_opt = np.clip(res.x, MIN_SINGLE_WEIGHT, MAX_SINGLE_WEIGHT)
        w_opt = w_opt / w_opt.sum()
        return {c: round(float(w), 8) for c, w in zip(factor_cols, w_opt)}

    _warn(f"SLSQP did not converge ({res.message}) -- falling back to IC-IR method.")
    return _ic_ir_weighted(stats_df, factor_cols)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_per_regime_ic(
    ic_timeseries_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    factor_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Join IC timeseries with regime labels; compute per-regime IC statistics.

    For each regime returns a DataFrame indexed by factor with columns:
        mean_ic, ic_ir, hit_ratio, t_stat, n_obs

    Parameters
    ----------
    ic_timeseries_df:
        DataFrame with 'date' column (or DatetimeIndex named 'date') and
        IC columns such as 'ic_momentum', 'ic_quality', etc.
    regime_state_df:
        DataFrame with 'date' and 'regime_label' columns.
    factor_cols:
        List of IC column names to include.

    Returns
    -------
    Dict mapping regime name -> summary DataFrame.
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

    available = [c for c in factor_cols if c in ic_df.columns]
    missing = set(factor_cols) - set(available)
    if missing:
        _warn(
            f"Skipping {len(missing)} factor cols not in IC timeseries: "
            f"{sorted(missing)[:10]}"
        )
    if not available:
        _warn("No valid factor columns -- returning empty dict.")
        return {}

    merged = ic_df[["date"] + available].merge(
        reg_df[["date", "regime_label"]], on="date", how="inner"
    )
    _log(
        f"Merged rows: {len(merged)} across "
        f"{merged['regime_label'].nunique()} regimes"
    )

    results: dict[str, pd.DataFrame] = {}

    for regime, group in merged.groupby("regime_label"):
        rows = []
        for col in available:
            series = group[col].dropna()
            n = len(series)
            if n < 2:
                rows.append(
                    {
                        "factor": col,
                        "mean_ic": np.nan,
                        "ic_ir": np.nan,
                        "hit_ratio": np.nan,
                        "t_stat": np.nan,
                        "n_obs": n,
                    }
                )
                continue

            mean_ic = float(series.mean())
            std_ic = float(series.std(ddof=1))
            ic_ir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
            hit_ratio = float((series > 0).mean())

            if _SCIPY_AVAILABLE:
                t_stat_val, _ = _ttest_1samp(series, popmean=0.0)
                t_stat_val = float(t_stat_val)
            else:
                # Manual t-statistic
                t_stat_val = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-12 else 0.0

            rows.append(
                {
                    "factor": col,
                    "mean_ic": round(mean_ic, 6),
                    "ic_ir": round(ic_ir, 6),
                    "hit_ratio": round(hit_ratio, 4),
                    "t_stat": round(t_stat_val, 4),
                    "n_obs": n,
                }
            )

        df_out = pd.DataFrame(rows).set_index("factor")
        results[str(regime)] = df_out
        _log(
            f"[OK] Regime '{regime}': {len(group)} obs, "
            f"{len(available)} factors computed"
        )

    return results


def train_regime_weights(
    ic_timeseries_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    factor_cols: list[str],
    method: str = "shrinkage",
    shrinkage_to_equal: float = 0.3,
    min_days_per_regime: int = _MIN_OBS_DEFAULT,
) -> dict[str, dict[str, float]]:
    """Train regime-conditional factor weights from IC data.

    Parameters
    ----------
    ic_timeseries_df:
        IC timeseries (date | ic_<factor>...).
    regime_state_df:
        Regime labels (date | regime_label).
    factor_cols:
        IC column names to use.
    method:
        'ic_ir_weighted' | 'optimization' | 'shrinkage'  (default: shrinkage).
    shrinkage_to_equal:
        Lambda for shrinkage blend.  0 = pure data weights, 1 = equal weights.
    min_days_per_regime:
        Minimum observations a regime must have; regimes below threshold get
        equal weights instead.

    Returns
    -------
    Dict: regime_name -> {factor_col: weight, ...}
    Each weight dict sums to 1.0 and no single weight exceeds 0.50.
    """
    _log(
        f"[START] train_regime_weights | "
        f"method={method} | lambda={shrinkage_to_equal} | "
        f"min_days={min_days_per_regime}"
    )

    per_regime_stats = compute_per_regime_ic(
        ic_timeseries_df, regime_state_df, factor_cols
    )

    for r in KNOWN_REGIMES:
        if r not in per_regime_stats:
            _warn(f"Regime '{r}' has no IC data -- will assign equal weights.")

    # Resolve the set of factor columns that were actually computed
    computed_cols: list[str] = (
        list(next(iter(per_regime_stats.values())).index)
        if per_regime_stats
        else factor_cols
    )

    target_regimes = sorted(set(KNOWN_REGIMES) | set(per_regime_stats.keys()))
    regime_weights: dict[str, dict[str, float]] = {}

    for regime in target_regimes:
        eq = _equal_weights(computed_cols)

        if regime not in per_regime_stats:
            regime_weights[regime] = eq
            continue

        stats_df = per_regime_stats[regime]
        n_obs = (
            int(stats_df["n_obs"].max())
            if "n_obs" in stats_df.columns
            else 0
        )

        if n_obs < min_days_per_regime:
            _warn(
                f"Regime '{regime}' has {n_obs} obs "
                f"(< min {min_days_per_regime}) -- using equal weights."
            )
            regime_weights[regime] = eq
            continue

        # ---- data-driven weights -----------------------------------------
        if method == "ic_ir_weighted":
            w_data = _ic_ir_weighted(stats_df, computed_cols)

        elif method == "optimization":
            w_data = _optimisation_weights(stats_df, computed_cols)

        elif method == "shrinkage":
            w_raw = _ic_ir_weighted(stats_df, computed_cols)
            lam = float(shrinkage_to_equal)
            w_blend = {
                c: (1.0 - lam) * w_raw.get(c, 0.0) + lam * eq.get(c, 0.0)
                for c in computed_cols
            }
            w_data = _normalise(w_blend)

        else:
            _warn(f"Unknown method '{method}' -- falling back to shrinkage.")
            w_raw = _ic_ir_weighted(stats_df, computed_cols)
            lam = float(shrinkage_to_equal)
            w_blend = {
                c: (1.0 - lam) * w_raw.get(c, 0.0) + lam * eq.get(c, 0.0)
                for c in computed_cols
            }
            w_data = _normalise(w_blend)

        # ---- guardrail: verify no single weight > 50 % -------------------
        max_w = max(w_data.values(), default=0.0)
        if max_w > MAX_SINGLE_WEIGHT + 1e-9:
            _warn(
                f"Regime '{regime}': max weight {max_w:.3f} exceeds "
                f"{MAX_SINGLE_WEIGHT} -- clamping via _normalise."
            )
            w_data = _normalise(w_data)

        # ---- guardrail: verify sum = 1.0 ---------------------------------
        total = sum(w_data.values())
        if abs(total - 1.0) > 1e-6:
            _warn(
                f"Regime '{regime}': weight sum {total:.8f} != 1.0 -- re-normalising."
            )
            w_data = {k: v / total for k, v in w_data.items()}

        regime_weights[regime] = w_data
        top3 = sorted(w_data.items(), key=lambda x: -x[1])[:3]
        _log(
            f"[OK] Regime '{regime}': {n_obs} obs | "
            f"top-3 weights = {top3}"
        )

    return regime_weights


def validate_regime_weights_wf(
    panel_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Walk-forward validation of regime weights vs an equal-weight baseline.

    Splits the IC timeseries into n_splits chronological folds.  For each fold:
    - Train on all data up to the fold boundary.
    - Evaluate on the next fold window.
    - Compare weighted-average IC vs equal-weight IC.

    Parameters
    ----------
    panel_df:
        IC timeseries DataFrame (date | ic_<factor>...).
    regime_state_df:
        Regime labels (date | regime_label).
    n_splits:
        Number of walk-forward folds.

    Returns
    -------
    Dict with keys:
        fold_results        -- list of per-fold result dicts
        mean_improvement    -- avg (wt_ic - eq_ic) across folds
        wt_sharpe           -- Sharpe of the fold improvement series
    """
    _log(f"[START] validate_regime_weights_wf | n_splits={n_splits}")

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
        _warn("No IC columns found -- returning empty validation result.")
        return {"fold_results": [], "mean_improvement": 0.0, "wt_sharpe": None}

    n = len(panel_df)
    if n < n_splits * 2:
        _warn(f"Too few rows ({n}) for {n_splits} splits -- clamping to 2.")
        n_splits = 2

    fold_size = n // n_splits
    fold_results: list[dict[str, Any]] = []
    improvements: list[float] = []
    eq_w = 1.0 / len(factor_cols)

    for fold in range(n_splits):
        train_end_idx = (fold + 1) * fold_size
        test_end_idx = min((fold + 2) * fold_size, n)

        if train_end_idx >= n:
            break

        train_df = panel_df.iloc[:train_end_idx]
        test_df = panel_df.iloc[train_end_idx:test_end_idx]

        if len(test_df) == 0:
            continue

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
        test_merged = test_df.merge(
            reg_df[["date", "regime_label"]], on="date", how="left"
        )
        test_merged["regime_label"] = test_merged["regime_label"].fillna("sideways")

        wt_ics: list[float] = []
        eq_ics: list[float] = []

        for _, row in test_merged.iterrows():
            regime = str(row["regime_label"])
            w_map = weights.get(regime, _equal_weights(factor_cols))
            ic_vals = {c: row[c] for c in factor_cols if not np.isnan(row[c])}
            if not ic_vals:
                continue
            wt_ic = sum(w_map.get(c, eq_w) * v for c, v in ic_vals.items())
            eq_ic_val = sum(eq_w * v for v in ic_vals.values())
            wt_ics.append(wt_ic)
            eq_ics.append(eq_ic_val)

        if not wt_ics:
            continue

        mean_wt = float(np.mean(wt_ics))
        mean_eq = float(np.mean(eq_ics))
        improvement = mean_wt - mean_eq
        improvements.append(improvement)

        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "wt_avg_ic": round(mean_wt, 6),
                "eq_avg_ic": round(mean_eq, 6),
                "improvement": round(improvement, 6),
            }
        )
        _log(
            f"[OK] Fold {fold}: wt_ic={mean_wt:.4f} | "
            f"eq_ic={mean_eq:.4f} | delta={improvement:+.4f}"
        )

    if not improvements:
        return {"fold_results": fold_results, "mean_improvement": 0.0, "wt_sharpe": None}

    arr = np.array(improvements)
    mean_imp = float(arr.mean())
    wt_sharpe: float | None
    if arr.std(ddof=1) > 1e-12:
        wt_sharpe = round(float(arr.mean() / arr.std(ddof=1)), 4)
    else:
        wt_sharpe = None

    _log(
        f"[DONE] WF validation | "
        f"mean_improvement={mean_imp:+.4f} | "
        f"wt_sharpe={wt_sharpe} | "
        f"folds={len(improvements)}"
    )

    return {
        "fold_results": fold_results,
        "mean_improvement": round(mean_imp, 6),
        "wt_sharpe": wt_sharpe,
    }
