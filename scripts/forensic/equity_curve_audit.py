"""Equity-curve forensic audit (C3-030).

Reads ``output/equity_curve_baseline.csv`` (or a path supplied via
``--input``) and runs a batch of statistical sanity checks against the
existing qa/ modules:

- Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)
- Probabilistic Sharpe Ratio (Bailey-Lopez de Prado 2012)
- Bootstrap CI on Sharpe / CAGR / MaxDD via shuffle_trades (block_size=5)
- Returns-Distribution: skewness, kurtosis, Jarque-Bera normality
- Ljung-Box autocorrelation test (lags 1, 5, 10, 20)
- Drawdown duration distribution

Output: machine-readable JSON + human-readable Markdown report under
``output/qa/equity_curve_audit_<run_id>.{json,md}``.

Usage::

    .venv/Scripts/python.exe scripts/forensic/equity_curve_audit.py
    .venv/Scripts/python.exe scripts/forensic/equity_curve_audit.py \\
        --input output/equity_curve_baseline.csv --out output/qa/

The script is read-only on the input file. PIT-safe (no future-looking
calls). Intended for nightly CI gate against baseline drift.
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

# F-S2-HOL-1 helper-extract: shared metric helpers live in _helpers.py
# since 2026-05-18 to avoid duplicating across 4 forensic scripts.
from scripts.forensic._helpers import annualised_sharpe as _annualised_sharpe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers (wrap existing qa/ functions or compute locally)
# ---------------------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Returns (max_dd_pct, duration_days, peak_index).

    Local extended variant of ``_helpers.max_drawdown`` that additionally
    returns the trough/peak indices for duration computation. The single-
    value max_dd matches ``_helpers.max_drawdown(equity)``.
    """
    if len(equity) < 2:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    trough_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[trough_idx])
    # Duration: days from peak to trough
    peak_idx = int(np.argmax(equity[: trough_idx + 1])) if trough_idx > 0 else 0
    duration = trough_idx - peak_idx
    return max_dd, duration, peak_idx


def _cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return float("nan")
    total = equity[-1] / equity[0]
    years = len(equity) / periods_per_year
    if total <= 0 or years <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def _drawdown_duration_distribution(equity: np.ndarray) -> dict[str, float]:
    """Distribution of drawdown durations (days underwater).

    F-senior-1: An episode counts contiguous days where ``equity < running_max``
    (strict <). A day at the new high (equity == running_max) ENDS an episode
    — even if equity drops on the next day, that's a new episode starting from
    the higher peak. This matches the standard "days under water" definition
    where touching the prior peak resets the underwater counter.
    """
    if len(equity) < 2:
        return {}
    running_max = np.maximum.accumulate(equity)
    underwater = equity < running_max
    durations: list[int] = []
    cur = 0
    for u in underwater:
        if u:
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
            cur = 0
    if cur > 0:
        durations.append(cur)
    if not durations:
        return {"count": 0}
    arr = np.array(durations, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _ljung_box_pvalues(returns: np.ndarray, lags: list[int]) -> dict[str, float]:
    """Ljung-Box test for serial autocorrelation at multiple lags."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore[import]

        result = acorr_ljungbox(returns, lags=lags, return_df=True)
        return {
            f"lag_{lag}": float(result["lb_pvalue"].iloc[i])
            for i, lag in enumerate(lags)
        }
    except Exception as exc:
        logger.warning("Ljung-Box skipped: %s", exc)
        return {}


def _jarque_bera(returns: np.ndarray) -> dict[str, float]:
    """Jarque-Bera normality test."""
    try:
        from scipy import stats  # type: ignore[import]

        jb_stat, p_value = stats.jarque_bera(returns)
        return {"stat": float(jb_stat), "p_value": float(p_value)}
    except Exception as exc:
        logger.warning("Jarque-Bera skipped: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit_equity_curve(
    equity_curve_path: Path,
    periods_per_year: int = 252,
    n_bootstrap: int = 500,
    n_strategies_tried: int = 10,
) -> dict[str, Any]:
    """Run the full forensic audit on an equity curve CSV.

    Args:
        equity_curve_path: Path to CSV with at least columns ``equity`` and
            (optionally) ``daily_return``. If ``daily_return`` is absent it
            is derived via pct_change of ``equity``.
        periods_per_year: 252 daily / 12 monthly / 52 weekly.
        n_bootstrap: Number of bootstrap iterations for CI estimates.
        n_strategies_tried: DSR adjustment — number of strategy variants
            tested during research (default 10, conservative).

    Returns:
        Dict with all statistics. Same content as the JSON report.
    """
    if not equity_curve_path.exists():
        raise FileNotFoundError(f"equity curve not found: {equity_curve_path}")
    df = pd.read_csv(equity_curve_path)
    if "equity" not in df.columns:
        raise ValueError(
            f"missing 'equity' column in {equity_curve_path}. "
            f"Got columns: {list(df.columns)}"
        )
    # F-senior-7: explicit dtype + positivity check on equity column.
    # to_numpy(dtype=float) silently coerces strings like "$1,000.00" to NaN
    # without warning. Validate after conversion to surface bad inputs early.
    try:
        equity = df["equity"].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"'equity' column in {equity_curve_path} cannot be converted to "
            f"float (got dtype {df['equity'].dtype}): {exc}"
        ) from exc
    if np.all(np.isnan(equity)):
        raise ValueError(
            f"'equity' column in {equity_curve_path} is all-NaN after float "
            f"conversion (original dtype: {df['equity'].dtype}). Check for "
            "string formatting like '$1,000.00' or empty cells."
        )
    if len(equity) > 0 and not (equity[0] > 0):
        raise ValueError(
            f"'equity' column starts non-positive ({equity[0]}), expected "
            "initial capital > 0. Drawdown/CAGR math is undefined for "
            "non-positive starting equity."
        )
    if "daily_return" in df.columns:
        returns = df["daily_return"].dropna().to_numpy(dtype=float)
    else:
        returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)

    n_obs = int(len(returns))
    sharpe = _annualised_sharpe(returns, periods_per_year)
    cagr_val = _cagr(equity, periods_per_year)
    max_dd, dd_duration, _ = _max_drawdown(equity)
    skew_val = float(pd.Series(returns).skew())
    kurt_val = float(pd.Series(returns).kurtosis() + 3.0)  # raw, not excess

    # Deflated / Probabilistic Sharpe
    psr = float("nan")
    dsr = float("nan")
    min_trl = float("nan")
    try:
        from src.assembled_core.qa.metrics import probabilistic_sharpe_ratio

        psr = probabilistic_sharpe_ratio(
            sharpe_observed=sharpe,
            n_obs=n_obs,
            sharpe_benchmark=0.0,
            skew=skew_val,
            kurtosis=kurt_val,
        )
    except (ImportError, ValueError, TypeError) as exc:
        # F-senior-3: narrow except — only swallow expected errors from
        # missing module / API mismatch / bad numeric input. Lets real bugs
        # in PSR propagate so they're not hidden by the soft-fail layer.
        logger.warning("PSR skipped: %s", exc)
    try:
        from src.assembled_core.qa.metrics import deflated_sharpe_ratio_from_returns

        # n_tests = multiple-testing adjustment for false-discovery
        scale = "daily" if periods_per_year == 252 else "monthly"
        dsr = float(
            deflated_sharpe_ratio_from_returns(
                pd.Series(returns),
                n_tests=int(n_strategies_tried),
                scale=scale,
                skew=skew_val,
                kurtosis=kurt_val,
            )
        )
        # min-TRL = how many periods needed to beat sharpe_benchmark with PSR ≥ 0.95
        # F-senior-2: standard Bailey-Lopez de Prado MinTRL formula uses
        # EXCESS kurtosis (kurt_excess = kurt_raw - 3), not raw kurtosis-1.
        # Variance factor of the Sharpe estimator (Bailey-LdP 2012, eq.4):
        #   V[SR] = (1 - skew·SR + ((kurt_excess - 1)/4)·SR²) / (T - 1)
        # MinTRL solves V[SR]·T = (z_α/SR)² → T = (z_α/SR)² · V_factor(T-1).
        # Heuristic approximation drops the T-1 term and uses z_α=1.645 (one-sided 95%).
        if np.isfinite(dsr) and sharpe > 0:
            kurt_excess = kurt_val - 3.0
            var_factor = max(
                1.0 - skew_val * sharpe + ((kurt_excess - 1.0) / 4.0) * sharpe**2,
                1e-6,
            )
            min_trl = float((1.645 / sharpe) ** 2 * var_factor)
    except (ImportError, ValueError, TypeError) as exc:
        # F-senior-3: narrow except — same rationale as PSR above.
        logger.warning("DSR skipped: %s", exc)

    # Bootstrap CI via shuffle_trades (block bootstrap, daily autocorrelation)
    bootstrap_ci: dict[str, Any] = {}
    try:
        from src.assembled_core.risk.monte_carlo import shuffle_trades

        result = shuffle_trades(
            returns,
            n_iterations=n_bootstrap,
            seed=42,
            annualization_factor=periods_per_year,
            block_size=5,
        )
        lo_s, hi_s = result.confidence_interval("sharpe", lo=0.025, hi=0.975)
        lo_mdd, hi_mdd = result.confidence_interval("max_drawdown", lo=0.025, hi=0.975)
        lo_ret, hi_ret = result.confidence_interval("total_return", lo=0.025, hi=0.975)
        bootstrap_ci = {
            "n_iterations": int(n_bootstrap),
            "block_size": 5,
            "sharpe": {
                "point": sharpe,
                "ci_lo_95": float(lo_s),
                "ci_hi_95": float(hi_s),
                "p_value_vs_zero": float((result.sharpe_distribution <= 0).mean()),
            },
            "max_drawdown": {
                "point": float(max_dd),
                "ci_lo_95": float(lo_mdd),
                "ci_hi_95": float(hi_mdd),
            },
            "total_return": {
                "ci_lo_95": float(lo_ret),
                "ci_hi_95": float(hi_ret),
            },
        }
    except (ImportError, ValueError, TypeError) as exc:
        # F-senior-3 + F-senior-4: narrow except + explicit sentinel.
        # shuffle_trades validates r<=-1.0 (ValueError), n_iterations>=1
        # (ValueError), etc. — those are expected soft-fails. Real bugs
        # in the bootstrap path propagate via other exception types.
        logger.warning("Bootstrap CI skipped: %s", exc)
        bootstrap_ci = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "skipped": True,
        }

    return {
        "input_path": str(equity_curve_path),
        "n_periods": n_obs,
        "years": round(n_obs / periods_per_year, 3),
        "sharpe": sharpe,
        "cagr": cagr_val,
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_days": dd_duration,
        "returns_distribution": {
            "mean": float(returns.mean()),
            "std": float(returns.std(ddof=1)),
            "skewness": skew_val,
            "kurtosis_raw": kurt_val,
            "kurtosis_excess": kurt_val - 3.0,
            "jarque_bera": _jarque_bera(returns),
        },
        "autocorrelation": {
            "ljung_box_p_values": _ljung_box_pvalues(returns, [1, 5, 10, 20]),
        },
        "drawdown_duration_distribution": _drawdown_duration_distribution(equity),
        "probabilistic_sharpe_ratio": psr,
        "deflated_sharpe_ratio": dsr,
        "min_track_record_length_periods": min_trl,
        "bootstrap_ci": bootstrap_ci,
        "audit_params": {
            "periods_per_year": periods_per_year,
            "n_bootstrap": n_bootstrap,
            "n_strategies_tried": n_strategies_tried,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the audit dict as a human-readable Markdown report."""
    lines = [
        "# Equity-Curve Forensic Audit (C3-030)",
        "",
        f"**Input:** `{report['input_path']}`",
        f"**Periods:** {report['n_periods']} (~{report['years']} years)",
        "",
        "## Headline Metrics",
        f"- **Sharpe:** {report['sharpe']:.4f}",
        f"- **CAGR:** {report['cagr']:.2%}",
        f"- **Max Drawdown:** {report['max_drawdown_pct']:.2%}"
        f" (duration: {report['max_drawdown_duration_days']} days)",
        "",
        "## Bias-Adjusted Sharpe",
        f"- **PSR (vs 0):** {report['probabilistic_sharpe_ratio']:.4f}",
        f"- **DSR (n_strategies={report['audit_params']['n_strategies_tried']}):** "
        f"{report['deflated_sharpe_ratio']:.4f}",
        f"- **Min Track Record Length:** "
        f"{report['min_track_record_length_periods']:.0f} periods",
        "",
        "## Returns Distribution",
        f"- **Mean:** {report['returns_distribution']['mean']:.6f}",
        f"- **Std:** {report['returns_distribution']['std']:.6f}",
        f"- **Skewness:** {report['returns_distribution']['skewness']:.4f}",
        f"- **Excess Kurtosis:** {report['returns_distribution']['kurtosis_excess']:.4f}",
    ]
    jb = report["returns_distribution"].get("jarque_bera", {})
    if jb:
        lines.append(
            f"- **Jarque-Bera p-value:** {jb.get('p_value', float('nan')):.4f} "
            f"(p<0.05 ⇒ reject normality)"
        )
    lines.append("")
    lines.append("## Autocorrelation (Ljung-Box p-values)")
    for lag, p in report["autocorrelation"]["ljung_box_p_values"].items():
        flag = " ⚠️" if p < 0.05 else ""
        lines.append(f"- **{lag}:** {p:.4f}{flag}")
    lines.append("")
    lines.append("## Drawdown Duration Distribution (days)")
    dd = report["drawdown_duration_distribution"]
    if dd.get("count", 0) > 0:
        lines.append(f"- **Episodes:** {dd['count']}")
        lines.append(f"- **Mean:** {dd['mean']:.1f}")
        lines.append(
            f"- **P50 / P90 / P99 / Max:** "
            f"{dd['p50']:.0f} / {dd['p90']:.0f} / "
            f"{dd['p99']:.0f} / {dd['max']:.0f}"
        )
    lines.append("")
    bs = report.get("bootstrap_ci", {})
    if bs:
        lines.append(
            f"## Bootstrap 95% CIs ({bs.get('n_iterations', 0)} paths, "
            f"block_size={bs.get('block_size', 1)})"
        )
        for metric_name in ("sharpe", "max_drawdown", "total_return"):
            m = bs.get(metric_name, {})
            if m:
                lines.append(
                    f"- **{metric_name}:** "
                    f"[{m.get('ci_lo_95', float('nan')):.4f}, "
                    f"{m.get('ci_hi_95', float('nan')):.4f}]"
                )
        sharpe_block = bs.get("sharpe", {})
        if "p_value_vs_zero" in sharpe_block:
            lines.append(f"- **P(Sharpe ≤ 0):** {sharpe_block['p_value_vs_zero']:.4f}")
        lines.append("")
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
        help="Output directory for audit JSON + Markdown",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualisation factor (default 252 = daily)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap iterations for CI",
    )
    parser.add_argument(
        "--n-strategies-tried",
        type=int,
        default=10,
        help="DSR multiple-testing adjustment (default 10, conservative)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = audit_equity_curve(
        equity_curve_path=args.input,
        periods_per_year=args.periods_per_year,
        n_bootstrap=args.n_bootstrap,
        n_strategies_tried=args.n_strategies_tried,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.input.stem
    json_path = args.out / f"equity_curve_audit_{run_id}.json"
    md_path = args.out / f"equity_curve_audit_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[equity_audit] JSON: %s", json_path)
    logger.info("[equity_audit] Markdown: %s", md_path)
    logger.info(
        "[equity_audit] Sharpe=%.3f CAGR=%.2f%% MDD=%.2f%% PSR=%.3f DSR=%.3f",
        report["sharpe"],
        report["cagr"] * 100,
        report["max_drawdown_pct"] * 100,
        report["probabilistic_sharpe_ratio"],
        report["deflated_sharpe_ratio"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
