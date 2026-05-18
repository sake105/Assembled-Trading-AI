"""Out-of-Regime test (audit C2-019).

Splits an equity-curve CSV into regime sub-periods (Bull / Bear / Sideways)
based on a simple trailing-return-sign rule and computes per-regime
performance metrics. A strategy with a real *edge* should produce positive
Sharpe in ALL regimes; if Sharpe collapses or flips sign in one regime, the
strategy is regime-dependent and the headline metrics are inflated by the
favourable regime.

Regime classification (no external data dependency):

- ``rolling_window``-day trailing return on the equity curve itself
  classifies each day:
    * trailing_return > +threshold   → "bull"
    * trailing_return < -threshold   → "bear"
    * else                            → "sideways"

The default ``rolling_window=120`` (~6 months) and ``threshold=0.05`` (5%)
are the conservative defaults; tune via CLI args.

Note: classifying regime from the strategy's OWN equity is a self-referential
heuristic. The "right" classification uses external benchmark (e.g. SPY
trailing return) — that's the C2-018 Out-of-Universe test, which needs
external data. C2-019 deliberately uses the strategy's equity so it stays
in-repo and runs offline. Document this honestly in the report.

Usage::

    python scripts/forensic/out_of_regime_test.py
    python scripts/forensic/out_of_regime_test.py \\
        --input output/equity_curve_baseline.csv \\
        --rolling-window 120 --threshold 0.05

Output: JSON + Markdown under ``output/qa/out_of_regime_<run_id>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def classify_regimes(
    equity: np.ndarray,
    rolling_window: int = 120,
    threshold: float = 0.05,
) -> np.ndarray:
    """Per-day regime labels: 'bull' / 'bear' / 'sideways' / 'warmup'.

    The first ``rolling_window`` days are labeled 'warmup' because the
    trailing-return signal is not yet defined.
    """
    n = len(equity)
    if n < rolling_window + 1:
        return np.full(n, "warmup", dtype=object)
    labels = np.full(n, "warmup", dtype=object)
    # Trailing return: equity[t] / equity[t-window] - 1
    for t in range(rolling_window, n):
        ret = equity[t] / equity[t - rolling_window] - 1.0
        if ret > threshold:
            labels[t] = "bull"
        elif ret < -threshold:
            labels[t] = "bear"
        else:
            labels[t] = "sideways"
    return labels


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------


def _annualised_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std <= 0:
        return float("nan")
    return mean / std * float(np.sqrt(periods_per_year))


def _max_drawdown(equity_subset: np.ndarray) -> float:
    if len(equity_subset) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity_subset)
    drawdowns = equity_subset / running_max - 1.0
    return float(drawdowns.min())


def per_regime_metrics(
    equity: np.ndarray,
    returns: np.ndarray,
    labels: np.ndarray,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Compute Sharpe / MDD / count for each regime.

    Note: returns array is len(equity)-1 (pct_change). Labels are len(equity).
    We align by skipping the first label (no return available for day 0).
    """
    if len(equity) != len(labels):
        raise ValueError(f"len(equity)={len(equity)} != len(labels)={len(labels)}")
    if len(returns) != len(equity) - 1:
        raise ValueError(
            f"len(returns)={len(returns)} expected len(equity)-1={len(equity) - 1}"
        )
    # Align: return[i] corresponds to the transition from equity[i] to
    # equity[i+1]. Use the label of equity[i+1] (target day).
    aligned_labels = labels[1:]
    out: dict[str, Any] = {}
    for regime in ("bull", "bear", "sideways", "warmup"):
        mask = aligned_labels == regime
        n = int(mask.sum())
        if n == 0:
            out[regime] = {"n_days": 0}
            continue
        regime_returns = returns[mask]
        # Reconstruct local equity for MDD: cumulative product of returns
        # in the regime period (NOT the original equity, since regime days
        # may be non-contiguous).
        regime_equity = np.cumprod(1.0 + regime_returns)
        out[regime] = {
            "n_days": n,
            "sharpe": _annualised_sharpe(regime_returns, periods_per_year),
            "mean_daily_return": float(regime_returns.mean()),
            "std_daily_return": float(regime_returns.std(ddof=1)) if n > 1 else 0.0,
            "max_drawdown": _max_drawdown(regime_equity),
            "total_return": float(regime_equity[-1] - 1.0),
        }
    return out


# ---------------------------------------------------------------------------
# Edge consistency check
# ---------------------------------------------------------------------------


def check_edge_consistency(
    per_regime: dict[str, Any],
    sharpe_min_in_each_regime: float = 0.5,
) -> dict[str, Any]:
    """Audit C2-019 verdict: does Sharpe survive in ALL regimes?

    Returns dict with `verdict`, `regime_sharpes`, `regimes_below_threshold`.
    """
    sharpes: dict[str, float] = {}
    failing: list[str] = []
    for regime, stats in per_regime.items():
        if regime == "warmup" or stats.get("n_days", 0) < 20:
            continue
        s = stats.get("sharpe")
        if s is None or not np.isfinite(s):
            continue
        sharpes[regime] = float(s)
        if s < sharpe_min_in_each_regime:
            failing.append(regime)
    if not sharpes:
        verdict = "insufficient_data"
    elif not failing:
        verdict = "robust"
    else:
        verdict = f"regime_dependent: {','.join(failing)}"
    return {
        "verdict": verdict,
        "regime_sharpes": sharpes,
        "regimes_below_threshold": failing,
        "sharpe_min_threshold": sharpe_min_in_each_regime,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_out_of_regime_test(
    equity_curve_path: Path,
    rolling_window: int = 120,
    threshold: float = 0.05,
    periods_per_year: int = 252,
    sharpe_min: float = 0.5,
) -> dict[str, Any]:
    """Full test: load equity, classify regimes, compute per-regime metrics."""
    if not equity_curve_path.exists():
        raise FileNotFoundError(f"equity curve not found: {equity_curve_path}")
    df = pd.read_csv(equity_curve_path)
    if "equity" not in df.columns:
        raise ValueError(f"missing 'equity' column in {equity_curve_path}")
    equity = df["equity"].to_numpy(dtype=float)
    if "daily_return" in df.columns:
        returns = df["daily_return"].dropna().to_numpy(dtype=float)
        # Align: pct_change drops first row, equity has all rows
        if len(returns) == len(equity):
            returns = returns[1:]
    else:
        returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)
    labels = classify_regimes(equity, rolling_window, threshold)
    per_regime = per_regime_metrics(equity, returns, labels, periods_per_year)
    consistency = check_edge_consistency(per_regime, sharpe_min)
    return {
        "input_path": str(equity_curve_path),
        "n_periods": int(len(equity)),
        "params": {
            "rolling_window": int(rolling_window),
            "threshold": float(threshold),
            "periods_per_year": int(periods_per_year),
            "sharpe_min_threshold": float(sharpe_min),
        },
        "per_regime": per_regime,
        "consistency": consistency,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Out-of-Regime Test (C2-019)",
        "",
        f"**Input:** `{report['input_path']}`",
        f"**Periods:** {report['n_periods']}",
        f"**Rolling window:** {report['params']['rolling_window']} days",
        f"**Bull/Bear threshold:** ±{report['params']['threshold']:.0%}",
        f"**Sharpe-min per regime:** {report['params']['sharpe_min_threshold']}",
        "",
        "## Per-Regime Metrics",
        "",
        "| Regime | N Days | Sharpe | Mean Daily Return | MDD | Total Return |",
        "|--------|-------:|-------:|-----------------:|----:|-------------:|",
    ]
    for regime in ("bull", "bear", "sideways", "warmup"):
        stats = report["per_regime"].get(regime, {})
        n = stats.get("n_days", 0)
        if n == 0:
            lines.append(f"| {regime} | 0 | — | — | — | — |")
            continue
        sharpe = stats.get("sharpe", float("nan"))
        mean_ret = stats.get("mean_daily_return", float("nan"))
        mdd = stats.get("max_drawdown", float("nan"))
        total = stats.get("total_return", float("nan"))
        lines.append(
            f"| {regime} | {n} | {sharpe:.3f} | {mean_ret:.5f} | "
            f"{mdd:.2%} | {total:.2%} |"
        )
    lines.append("")
    lines.append("## Edge Consistency Verdict")
    cons = report["consistency"]
    lines.append(f"**Verdict:** `{cons['verdict']}`")
    lines.append("")
    if cons["regimes_below_threshold"]:
        lines.append("**Regimes below Sharpe threshold:**")
        for r in cons["regimes_below_threshold"]:
            sharpe = cons["regime_sharpes"].get(r, float("nan"))
            lines.append(f"- {r}: Sharpe = {sharpe:.3f}")
    else:
        lines.append("Sharpe ≥ threshold in all regimes with sufficient data.")
    lines.append("")
    lines.append("## Honesty Disclosure")
    lines.append(
        "Regime is classified from the strategy's OWN equity-curve trailing "
        "return. This is a self-referential heuristic — the strategy may "
        "have *defined* the regime by its own behaviour. The 'true' "
        "Out-of-Regime test uses an EXTERNAL benchmark (e.g. SPY) which "
        "requires data not in this repo (audit C2-018 Out-of-Universe scope)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/equity_curve_baseline.csv"),
        help="Path to equity-curve CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/qa"),
        help="Output directory for JSON + Markdown report",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=120,
        help="Trailing-return window for regime classification (default 120d ≈ 6m)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Bull/Bear threshold on trailing return (default 0.05 = 5%%)",
    )
    parser.add_argument(
        "--sharpe-min",
        type=float,
        default=0.5,
        help="Minimum per-regime Sharpe for 'robust' verdict (default 0.5)",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualisation factor (default 252)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_out_of_regime_test(
        equity_curve_path=args.input,
        rolling_window=args.rolling_window,
        threshold=args.threshold,
        periods_per_year=args.periods_per_year,
        sharpe_min=args.sharpe_min,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.input.stem
    json_path = args.out / f"out_of_regime_{run_id}.json"
    md_path = args.out / f"out_of_regime_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[out_of_regime] JSON: %s", json_path)
    logger.info("[out_of_regime] Markdown: %s", md_path)
    cons = report["consistency"]
    logger.info(
        "[out_of_regime] verdict=%s regime_sharpes=%s",
        cons["verdict"],
        {k: round(v, 3) for k, v in cons["regime_sharpes"].items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
