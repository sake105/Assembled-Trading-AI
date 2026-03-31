"""Parameter stability checks for risk controls.

Evaluates whether risk parameters (vol, turnover, drawdown) are stable
across rolling windows and regime windows. Used in backtests and reporting
to flag instability before live deployment.

Key checks:
  - Rolling vol stability: std-of-vols across windows / mean vol
  - Turnover stability: coefficient of variation of turnover series
  - Drawdown consistency: max drawdown does not vary wildly window to window
  - Combined stability report

All functions are stateless with no side effects.

M6-T09: implement parameter stability checks.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Rolling vol stability
# ---------------------------------------------------------------------------


def compute_rolling_vol_estimates(
    equity_curve: pd.Series,
    window_sizes: list[int],
    annualize_factor: float = 252.0,
    min_observations: int = 5,
) -> dict[int, float]:
    """Compute realized vol estimate for each window size.

    Args:
        equity_curve: Series of equity values (not returns).
        window_sizes: List of lookback window sizes in bars.
        annualize_factor: Annualization multiplier (default 252 for daily).
        min_observations: Minimum observations required per window.

    Returns:
        window_size -> annualized vol dict.
        Windows with insufficient data get float('nan').
    """
    if equity_curve is None or not isinstance(equity_curve, pd.Series) or len(equity_curve) < 2:
        return {w: float("nan") for w in window_sizes}

    returns = equity_curve.pct_change().dropna()
    result: dict[int, float] = {}

    for w in window_sizes:
        tail = returns.tail(w)
        if len(tail) < min_observations:
            result[w] = float("nan")
            continue
        std = float(tail.std(ddof=1))
        result[w] = std * math.sqrt(annualize_factor)

    return result


def check_vol_stability(
    equity_curve: pd.Series,
    window_sizes: list[int] | None = None,
    stability_threshold: float = 0.30,
    annualize_factor: float = 252.0,
    min_observations: int = 5,
) -> dict[str, Any]:
    """Check whether realized vol is stable across multiple window sizes.

    Stability is measured as the coefficient of variation (CV = std/mean) of
    the vol estimates across window sizes. CV < stability_threshold = stable.

    Args:
        equity_curve: Series of equity values.
        window_sizes: Lookback windows to compare (default [10, 20, 40, 60]).
        stability_threshold: Max CV to call stable (default 0.30 = 30%).
        annualize_factor: Annualization multiplier.
        min_observations: Min observations per window.

    Returns:
        Dict with:
            ``vol_by_window``: window_size -> realized vol
            ``mean_vol``: mean of valid vol estimates
            ``cv``: coefficient of variation (std/mean) of estimates
            ``is_stable``: True if cv <= stability_threshold
            ``valid_windows``: number of windows with valid estimates
            ``status``: "ok", "insufficient_data", or "all_nan"
    """
    if window_sizes is None:
        window_sizes = [10, 20, 40, 60]

    vol_by_window = compute_rolling_vol_estimates(
        equity_curve, window_sizes, annualize_factor, min_observations
    )

    valid_vols = [v for v in vol_by_window.values() if not math.isnan(v)]

    if len(valid_vols) == 0:
        return {
            "vol_by_window": vol_by_window,
            "mean_vol": float("nan"),
            "cv": float("nan"),
            "is_stable": False,
            "valid_windows": 0,
            "status": "all_nan",
        }

    if len(valid_vols) < 2:
        mean_vol = float(np.mean(valid_vols))
        return {
            "vol_by_window": vol_by_window,
            "mean_vol": mean_vol,
            "cv": float("nan"),
            "is_stable": True,  # Can't measure instability with one point
            "valid_windows": len(valid_vols),
            "status": "insufficient_data",
        }

    mean_vol = float(np.mean(valid_vols))
    std_vol = float(np.std(valid_vols, ddof=1))
    cv = std_vol / mean_vol if mean_vol > 0.0 else float("nan")

    is_stable = (not math.isnan(cv)) and (cv <= stability_threshold)

    return {
        "vol_by_window": vol_by_window,
        "mean_vol": mean_vol,
        "cv": cv,
        "is_stable": is_stable,
        "valid_windows": len(valid_vols),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Turnover stability
# ---------------------------------------------------------------------------


def check_turnover_stability(
    turnover_series: pd.Series,
    stability_threshold: float = 0.50,
    min_observations: int = 5,
) -> dict[str, Any]:
    """Check whether turnover is stable (low coefficient of variation).

    Args:
        turnover_series: Series of period turnover values (0-1 fractions).
        stability_threshold: Max CV to call stable (default 0.50 = 50%).
        min_observations: Minimum non-NaN observations required.

    Returns:
        Dict with:
            ``mean_turnover``: mean turnover
            ``std_turnover``: standard deviation of turnover
            ``cv``: coefficient of variation (std/mean)
            ``max_turnover``: maximum observed turnover
            ``is_stable``: True if cv <= stability_threshold
            ``n_observations``: number of valid observations
            ``status``: "ok", "insufficient_data", or "empty"
    """
    if turnover_series is None or not isinstance(turnover_series, pd.Series):
        return {
            "mean_turnover": float("nan"),
            "std_turnover": float("nan"),
            "cv": float("nan"),
            "max_turnover": float("nan"),
            "is_stable": False,
            "n_observations": 0,
            "status": "empty",
        }

    clean = turnover_series.dropna()

    if len(clean) == 0:
        return {
            "mean_turnover": float("nan"),
            "std_turnover": float("nan"),
            "cv": float("nan"),
            "max_turnover": float("nan"),
            "is_stable": False,
            "n_observations": 0,
            "status": "empty",
        }

    if len(clean) < min_observations:
        return {
            "mean_turnover": float(clean.mean()),
            "std_turnover": float(clean.std(ddof=1)) if len(clean) > 1 else float("nan"),
            "cv": float("nan"),
            "max_turnover": float(clean.max()),
            "is_stable": False,
            "n_observations": len(clean),
            "status": "insufficient_data",
        }

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    cv = std / mean if mean > 0.0 else float("nan")
    is_stable = (not math.isnan(cv)) and (cv <= stability_threshold)

    return {
        "mean_turnover": mean,
        "std_turnover": std,
        "cv": cv,
        "max_turnover": float(clean.max()),
        "is_stable": is_stable,
        "n_observations": len(clean),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Drawdown stability
# ---------------------------------------------------------------------------


def compute_rolling_max_drawdown(
    equity_curve: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling maximum drawdown over a fixed window.

    Args:
        equity_curve: Series of equity values.
        window: Rolling window size in bars.

    Returns:
        Series of rolling max drawdown values (negative floats, e.g. -0.15 = 15% DD).
        Empty Series if input is insufficient.
    """
    if equity_curve is None or not isinstance(equity_curve, pd.Series) or len(equity_curve) < window:
        return pd.Series(dtype=float)

    def _max_dd(sub: pd.Series) -> float:
        peak = sub.expanding().max()
        dd = (sub - peak) / peak
        return float(dd.min())

    return equity_curve.rolling(window).apply(_max_dd, raw=False)


def check_drawdown_stability(
    equity_curve: pd.Series,
    window: int = 40,
    stability_threshold: float = 0.50,
    min_observations: int = 10,
) -> dict[str, Any]:
    """Check whether max drawdown is consistent across rolling windows.

    Measures CV of rolling max-drawdown series. Low CV = consistent drawdowns.

    Args:
        equity_curve: Series of equity values.
        window: Rolling window size for drawdown calculation.
        stability_threshold: Max CV to call stable (default 0.50).
        min_observations: Min non-NaN rolling drawdown values required.

    Returns:
        Dict with:
            ``mean_max_dd``: mean of rolling max drawdowns
            ``cv``: coefficient of variation
            ``worst_dd``: single worst drawdown observed
            ``is_stable``: True if cv <= stability_threshold
            ``n_observations``: valid rolling observations
            ``status``: "ok", "insufficient_data", or "empty"
    """
    rolling_dd = compute_rolling_max_drawdown(equity_curve, window)
    clean = rolling_dd.dropna()

    if len(clean) == 0:
        return {
            "mean_max_dd": float("nan"),
            "cv": float("nan"),
            "worst_dd": float("nan"),
            "is_stable": False,
            "n_observations": 0,
            "status": "empty",
        }

    if len(clean) < min_observations:
        return {
            "mean_max_dd": float(clean.mean()),
            "cv": float("nan"),
            "worst_dd": float(clean.min()),
            "is_stable": False,
            "n_observations": len(clean),
            "status": "insufficient_data",
        }

    # Use abs values to compute CV
    abs_dd = clean.abs()
    mean_dd = float(abs_dd.mean())
    std_dd = float(abs_dd.std(ddof=1))
    cv = std_dd / mean_dd if mean_dd > 0.0 else float("nan")
    is_stable = (not math.isnan(cv)) and (cv <= stability_threshold)

    return {
        "mean_max_dd": float(clean.mean()),  # negative
        "cv": cv,
        "worst_dd": float(clean.min()),  # most negative
        "is_stable": is_stable,
        "n_observations": len(clean),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Combined stability report
# ---------------------------------------------------------------------------


def compute_stability_report(
    equity_curve: pd.Series,
    turnover_series: pd.Series | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all stability checks and produce a combined report.

    Args:
        equity_curve: Series of equity values.
        turnover_series: Optional turnover series for turnover stability check.
        policy: Optional policy dict. Reads from ``param_stability`` section:
            - vol_window_sizes (list[int], default [10, 20, 40, 60])
            - vol_stability_threshold (float, default 0.30)
            - turnover_stability_threshold (float, default 0.50)
            - drawdown_window (int, default 40)
            - drawdown_stability_threshold (float, default 0.50)
            - annualize_factor (float, default 252.0)

    Returns:
        Dict with:
            ``vol_stability``: result of check_vol_stability
            ``turnover_stability``: result of check_turnover_stability (or None)
            ``drawdown_stability``: result of check_drawdown_stability
            ``all_stable``: True if all available checks pass
            ``checks_passed``: count of stable checks
            ``checks_total``: count of checks with valid data
    """
    ps = ((policy or {}).get("param_stability") or {})
    vol_windows = ps.get("vol_window_sizes", [10, 20, 40, 60])
    vol_thresh = float(ps.get("vol_stability_threshold", 0.30) or 0.30)
    to_thresh = float(ps.get("turnover_stability_threshold", 0.50) or 0.50)
    dd_window = int(ps.get("drawdown_window", 40) or 40)
    dd_thresh = float(ps.get("drawdown_stability_threshold", 0.50) or 0.50)
    af = float(ps.get("annualize_factor", 252.0) or 252.0)

    vol_result = check_vol_stability(
        equity_curve,
        window_sizes=vol_windows,
        stability_threshold=vol_thresh,
        annualize_factor=af,
    )

    dd_result = check_drawdown_stability(
        equity_curve,
        window=dd_window,
        stability_threshold=dd_thresh,
    )

    to_result = None
    if turnover_series is not None:
        to_result = check_turnover_stability(
            turnover_series,
            stability_threshold=to_thresh,
        )

    # Count checks
    checks: list[bool] = []
    if vol_result["status"] == "ok":
        checks.append(vol_result["is_stable"])
    if dd_result["status"] == "ok":
        checks.append(dd_result["is_stable"])
    if to_result is not None and to_result["status"] == "ok":
        checks.append(to_result["is_stable"])

    checks_total = len(checks)
    checks_passed = sum(checks)
    all_stable = checks_total > 0 and checks_passed == checks_total

    return {
        "vol_stability": vol_result,
        "turnover_stability": to_result,
        "drawdown_stability": dd_result,
        "all_stable": all_stable,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }


__all__ = [
    "compute_rolling_vol_estimates",
    "check_vol_stability",
    "check_turnover_stability",
    "compute_rolling_max_drawdown",
    "check_drawdown_stability",
    "compute_stability_report",
]
