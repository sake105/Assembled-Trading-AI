"""Time-series attribution analysis.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §6.
Provides:
  - Rolling IC aggregation per dimension over attribution history
  - Dead-feature detection (IC ≈ 0 sustained over 90 days)
  - Attribution distribution-shift detection (KS test)
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from assembled_core.attribution.schemas import CompositeAttribution

# ---------------------------------------------------------------------------
# IC aggregation
# ---------------------------------------------------------------------------

def attributions_to_df(attrs: list[CompositeAttribution]) -> pd.DataFrame:
    """Convert a list of CompositeAttribution to a flat DataFrame.

    Columns: timestamp, ticker, composite_score, regime, model_version,
    + contrib_<dim> columns.
    """
    rows = []
    for a in attrs:
        row: dict[str, Any] = {
            "timestamp": a.timestamp,
            "ticker": a.ticker,
            "composite_score": a.composite_score,
            "regime": a.regime,
            "model_version": a.model_version,
        }
        for dim, val in a.dimension_contributions.items():
            row[f"contrib_{dim}"] = val
        for dim, val in a.dimension_raw_scores.items():
            row[f"raw_{dim}"] = val
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def rolling_dimension_ic(
    attrs_df: pd.DataFrame,
    forward_returns: pd.Series,
    window_days: int = 30,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute rolling IC (Information Coefficient) for each dimension.

    Joins attrs_df on date with forward_returns (date → return for that ticker
    on next day).  IC is computed per rolling window.

    Args:
        attrs_df: output of attributions_to_df()
        forward_returns: Series indexed by (date, ticker) with 1-day forward
            returns.  Can also be a flat Series if attrs_df is single-ticker.
        window_days: rolling window in calendar days
        method: 'spearman' or 'pearson'

    Returns:
        DataFrame[window_end, dim_name] → IC value
    """
    contrib_cols = [c for c in attrs_df.columns if c.startswith("contrib_")]
    if not contrib_cols:
        return pd.DataFrame()

    if "timestamp" not in attrs_df.columns:
        return pd.DataFrame()

    df = attrs_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    # Align forward returns
    if isinstance(forward_returns.index, pd.MultiIndex):
        df = df.set_index(["date", "ticker"]).join(
            forward_returns.rename("fwd"), how="left"
        ).reset_index()
    else:
        fwd_df = forward_returns.rename("fwd").reset_index()
        fwd_df.columns = ["date", "fwd"]
        df = df.merge(fwd_df, on="date", how="left")

    df = df.dropna(subset=["fwd"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("date")
    dates = df["date"].unique()

    records = []
    for end_date in dates:
        start_date = end_date - timedelta(days=window_days)
        window_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if len(window_df) < 10:
            continue
        row: dict[str, Any] = {"window_end": end_date}
        fwd_values = window_df["fwd"].values
        for col in contrib_cols:
            dim = col.replace("contrib_", "")
            x = window_df[col].values
            row[dim] = _ic(x, fwd_values, method)
        records.append(row)

    return pd.DataFrame(records).set_index("window_end") if records else pd.DataFrame()


def _ic(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return float("nan")
    if method == "spearman":
        x = _rank(x)
        y = _rank(y)
    return float(_pearson(x, y))


def _rank(arr: np.ndarray) -> np.ndarray:
    temp = arr.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(arr) + 1)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Dead-feature detection
# ---------------------------------------------------------------------------

_DEAD_FEATURE_IC_THRESHOLD = 0.02
_DEAD_FEATURE_MIN_WINDOWS = 20


def detect_dead_features(
    rolling_ic_df: pd.DataFrame,
    ic_threshold: float = _DEAD_FEATURE_IC_THRESHOLD,
    min_windows: int = _DEAD_FEATURE_MIN_WINDOWS,
) -> dict[str, dict[str, Any]]:
    """Identify dimensions whose mean |IC| fell below threshold.

    A feature is 'dead' if, over the last `min_windows` IC observations,
    the mean absolute IC is below `ic_threshold`.

    Returns:
        dict[dim_name] → {"mean_abs_ic": float, "is_dead": bool,
                          "n_windows": int}
    """
    if rolling_ic_df.empty:
        return {}

    result: dict[str, dict[str, Any]] = {}
    recent = rolling_ic_df.tail(min_windows)

    for col in recent.columns:
        vals = recent[col].dropna()
        if len(vals) == 0:
            result[col] = {"mean_abs_ic": float("nan"), "is_dead": True,
                           "n_windows": 0}
            continue
        mean_abs = float(vals.abs().mean())
        result[col] = {
            "mean_abs_ic": mean_abs,
            "is_dead": mean_abs < ic_threshold,
            "n_windows": len(vals),
        }
    return result


def dead_feature_report(dead_features: dict[str, dict[str, Any]]) -> str:
    """Human-readable dead-feature summary."""
    lines = ["Dead Feature Report"]
    lines.append("-" * 40)
    dead = [k for k, v in dead_features.items() if v.get("is_dead")]
    alive = [k for k, v in dead_features.items() if not v.get("is_dead")]

    if dead:
        lines.append(f"DEAD ({len(dead)}):")
        for d in sorted(dead):
            ic = dead_features[d]["mean_abs_ic"]
            n = dead_features[d]["n_windows"]
            lines.append(f"  {d:30s}  mean|IC|={ic:.4f}  n={n}")
    else:
        lines.append("No dead features detected.")

    if alive:
        lines.append(f"ALIVE ({len(alive)}):")
        for a in sorted(alive):
            ic = dead_features[a]["mean_abs_ic"]
            lines.append(f"  {a:30s}  mean|IC|={ic:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attribution distribution shift (KS test)
# ---------------------------------------------------------------------------

def detect_attribution_drift(
    recent_attrs: list[CompositeAttribution],
    baseline_attrs: list[CompositeAttribution],
    threshold_p: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """KS test per dimension between recent and baseline attribution distributions.

    Returns:
        dict[dim_name] → {"ks_statistic": float, "p_value": float,
                          "is_drift": bool}
    """
    if not recent_attrs or not baseline_attrs:
        return {}

    def contribs_for(attrs: list[CompositeAttribution]) -> dict[str, list[float]]:
        d: dict[str, list[float]] = {}
        for a in attrs:
            for dim, val in a.dimension_contributions.items():
                d.setdefault(dim, []).append(val)
        return d

    recent_d = contribs_for(recent_attrs)
    baseline_d = contribs_for(baseline_attrs)
    dims = set(recent_d) | set(baseline_d)

    try:
        from scipy import stats as _stats

        def ks_test(a: list[float], b: list[float]) -> tuple[float, float]:
            stat, p = _stats.ks_2samp(a, b)
            return float(stat), float(p)

    except ImportError:
        def ks_test(a: list[float], b: list[float]) -> tuple[float, float]:
            # Approximate: compare means (weak proxy without scipy)
            ma, mb = float(np.mean(a)), float(np.mean(b))
            diff = abs(ma - mb) / (np.std(a + b, ddof=1) + 1e-9)
            p_approx = float(np.exp(-2 * diff ** 2 * min(len(a), len(b)) / (len(a) + len(b))))
            return diff, min(p_approx, 1.0)

    result: dict[str, dict[str, Any]] = {}
    for dim in sorted(dims):
        r = recent_d.get(dim, [])
        b = baseline_d.get(dim, [])
        if len(r) < 5 or len(b) < 5:
            result[dim] = {"ks_statistic": float("nan"), "p_value": float("nan"),
                           "is_drift": False}
            continue
        stat, p = ks_test(r, b)
        result[dim] = {"ks_statistic": stat, "p_value": p,
                       "is_drift": p < threshold_p}

    return result
