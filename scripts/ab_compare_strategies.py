"""Backlog Item 136 — Generic A/B Strategy Comparison with Statistical Significance.

Loads two backtest result directories OR two equity-curve CSV files, computes
standard metrics for each, runs a Sharpe-Ratio difference test (Lo 2002 /
Jobson-Korkie approximation), and outputs a Markdown + JSON comparison report.

Usage — directory mode (original):
    python scripts/ab_compare_strategies.py \\
        --strategy-a output/accounting_report_backtest_<hash_A> \\
        --strategy-b output/accounting_report_backtest_<hash_B> \\
        --label-a "Baseline" --label-b "With HRP" \\
        --output output/qa/ab_comparison_YYYYMMDD.json

    # Compare policy.yaml variants:
    python scripts/ab_compare_strategies.py \\
        --strategy-a output/bt_no_leverage --strategy-b output/bt_leverage \\
        --label-a "No Leverage (1.0x)" --label-b "Leverage (1.2x)"

Usage — CSV equity-curve mode (Backlog Item 136 extension):
    python scripts/ab_compare_strategies.py \\
        --equity-a output/equity_curve_baseline.csv \\
        --equity-b output/equity_curve_hrp.csv \\
        --name-a "Baseline" --name-b "With HRP" \\
        --output-json output/qa/ab_comparison.json

    Expected CSV format: date column + one numeric column (equity / portfolio_value
    / cumulative_returns / value / net_liq / return / ret).

Exit codes (CSV mode):
    0  A wins or tie (Sharpe_A >= Sharpe_B or not significant)
    1  B wins (Sharpe_B > Sharpe_A and significant)
    2  Insufficient data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ─── Sharpe ratio significance test (Lo 2002 approximation) ──────────────────


def _sharpe_z_test(returns_a: list[float], returns_b: list[float]) -> dict:
    """Jobson-Korkie / Lo (2002) Sharpe-difference z-test.

    H0: SR_A == SR_B (no difference).
    Returns p-value and z-statistic.
    """
    import statistics

    n = min(len(returns_a), len(returns_b))
    if n < 10:
        return {
            "z_stat": None,
            "p_value": None,
            "significant_95": None,
            "note": f"Too few observations (n={n})",
        }

    ra = returns_a[:n]
    rb = returns_b[:n]

    mu_a = statistics.mean(ra)
    mu_b = statistics.mean(rb)
    sigma_a = statistics.stdev(ra) or 1e-9
    sigma_b = statistics.stdev(rb) or 1e-9
    sr_a = mu_a / sigma_a
    sr_b = mu_b / sigma_b

    # Correlation between A and B returns
    try:
        cov = sum((ra[i] - mu_a) * (rb[i] - mu_b) for i in range(n)) / (n - 1)
        rho = cov / (sigma_a * sigma_b)
        rho = max(-1.0, min(1.0, rho))
    except ZeroDivisionError:
        rho = 0.0

    # Variance of (SR_A - SR_B) per Lo 2002
    var_diff = (1 / n) * (
        2 - 2 * rho + 0.5 * sr_a**2 + 0.5 * sr_b**2 - rho * sr_a * sr_b
    )
    if var_diff <= 0:
        return {
            "z_stat": None,
            "p_value": None,
            "significant_95": None,
            "note": "Variance non-positive",
        }

    z = (sr_a - sr_b) / math.sqrt(var_diff)

    # Two-tailed p-value via normal CDF approximation (Abramowitz & Stegun)
    def _norm_cdf(x: float) -> float:
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (
            0.319381530
            + t
            * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2) * poly
        return cdf if x >= 0 else 1 - cdf

    p_two_tailed = 2 * (1 - _norm_cdf(abs(z)))

    return {
        "sr_a": round(sr_a * math.sqrt(252), 4),  # annualised
        "sr_b": round(sr_b * math.sqrt(252), 4),
        "z_stat": round(z, 4),
        "p_value": round(p_two_tailed, 4),
        "significant_95": p_two_tailed < 0.05,
        "significant_99": p_two_tailed < 0.01,
        "interpretation": (
            "Difference is statistically significant (p<0.05)"
            if p_two_tailed < 0.05
            else "No statistically significant difference"
        ),
    }


# ─── Metric extraction ────────────────────────────────────────────────────────


def _load_metrics(result_dir: Path) -> dict:
    """Extract performance metrics from a backtest output directory."""
    metrics: dict = {"dir": str(result_dir), "found": False}

    # Try summary.json / performance.json first
    for fname in [
        "summary.json",
        "performance.json",
        "backtest_summary.json",
        "metrics.json",
        "report.json",
    ]:
        p = result_dir / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            metrics.update(data)
            metrics["found"] = True
            metrics["source"] = fname
            break

    # Try to load equity curve for returns
    daily_returns: list[float] = []
    try:
        import pandas as pd

        for fname in ["equity_curve.parquet", "equity.parquet", "equity_curve.csv"]:
            p = result_dir / fname
            if p.exists():
                if fname.endswith(".parquet"):
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p, index_col=0, parse_dates=True)
                eq_col = next(
                    (
                        c
                        for c in [
                            "equity",
                            "portfolio_value",
                            "cumulative_returns",
                            "value",
                            "net_liq",
                        ]
                        if c in df.columns
                    ),
                    None,
                )
                if eq_col:
                    daily_returns = df[eq_col].pct_change().dropna().tolist()
                    metrics["daily_returns"] = daily_returns
                    metrics["found"] = True
                    break
    except ImportError:
        pass

    return metrics


def _compute_metrics(m: dict) -> dict:
    """Compute standard metrics if not already present."""
    rets = m.get("daily_returns", [])
    computed: dict = {}

    if rets:
        import statistics

        n = len(rets)
        mu = statistics.mean(rets)
        sigma = statistics.stdev(rets) if n > 1 else 1e-9

        cagr = (1 + sum(rets)) ** (252 / n) - 1 if n > 0 else 0.0
        sharpe = (mu / sigma) * math.sqrt(252) if sigma > 0 else 0.0

        # Max drawdown
        cum = [1.0]
        for r in rets:
            cum.append(cum[-1] * (1 + r))
        peak = cum[0]
        mdd = 0.0
        for v in cum:
            if v > peak:
                peak = v
            mdd = min(mdd, (v - peak) / peak)

        computed = {
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 4),
            "mdd_pct": round(mdd * 100, 2),
            "n_days": n,
        }

    # Merge with existing metrics (prefer existing)
    for k, v in computed.items():
        m.setdefault(k, v)

    return m


def format_markdown(
    label_a: str, label_b: str, ma: dict, mb: dict, sig_test: dict
) -> str:
    """Generate a human-readable Markdown comparison report."""
    lines = [
        "# A/B Strategy Comparison",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Performance Summary",
        "",
        f"| Metric | {label_a} | {label_b} | Delta (B-A) |",
        "|--------|-----------|-----------|-------------|",
    ]

    metrics_to_show = [
        ("CAGR (%)", "cagr_pct"),
        ("Sharpe", "sharpe"),
        ("MDD (%)", "mdd_pct"),
        ("Trades", "n_trades"),
        ("Days", "n_days"),
    ]
    for label, key in metrics_to_show:
        va = ma.get(key, "—")
        vb = mb.get(key, "—")
        if isinstance(va, float) and isinstance(vb, float):
            delta = round(vb - va, 2)
            sign = "+" if delta > 0 else ""
            lines.append(f"| {label} | {va} | {vb} | {sign}{delta} |")
        else:
            lines.append(f"| {label} | {va} | {vb} | — |")

    lines += [
        "",
        "## Statistical Significance (Sharpe-Ratio Difference Test)",
        "",
        f"- z-statistic: {sig_test.get('z_stat', '—')}",
        f"- p-value: {sig_test.get('p_value', '—')}",
        f"- Significant at 95%: {sig_test.get('significant_95', '—')}",
        f"- **{sig_test.get('interpretation', 'N/A')}**",
        "",
        f"> Note: {sig_test.get('note', '')}",
        "",
        "## Paths",
        f"- A: `{ma.get('dir', '?')}`",
        f"- B: `{mb.get('dir', '?')}`",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="A/B Strategy Comparison")
    parser.add_argument("--strategy-a", required=True, help="Path to backtest dir A")
    parser.add_argument("--strategy-b", required=True, help="Path to backtest dir B")
    parser.add_argument("--label-a", default="Strategy A")
    parser.add_argument("--label-b", default="Strategy B")
    parser.add_argument("--output-dir", default="output/qa")
    args = parser.parse_args(argv)

    dir_a = ROOT / args.strategy_a
    dir_b = ROOT / args.strategy_b

    logger.info("[AB] Loading metrics for %s", args.label_a)
    ma = _compute_metrics(_load_metrics(dir_a))

    logger.info("[AB] Loading metrics for %s", args.label_b)
    mb = _compute_metrics(_load_metrics(dir_b))

    rets_a = ma.get("daily_returns", [])
    rets_b = mb.get("daily_returns", [])

    logger.info(
        "[AB] Running Sharpe significance test (n_a=%d, n_b=%d)",
        len(rets_a),
        len(rets_b),
    )
    sig = _sharpe_z_test(rets_a, rets_b)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "metrics_a": {k: v for k, v in ma.items() if k != "daily_returns"},
        "metrics_b": {k: v for k, v in mb.items() if k != "daily_returns"},
        "significance_test": sig,
    }

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path = out_dir / f"ab_comparison_{ts}.json"
    md_path = out_dir / f"ab_comparison_{ts}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_content = format_markdown(args.label_a, args.label_b, ma, mb, sig)
    md_path.write_text(md_content, encoding="utf-8")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"A/B COMPARISON: {args.label_a} vs {args.label_b}")
    print(f"{'=' * 60}")
    for k in ["cagr_pct", "sharpe", "mdd_pct"]:
        va, vb = ma.get(k, "?"), mb.get(k, "?")
        print(f"  {k:12s}: A={va}  B={vb}")
    print(f"\nSharpe test: {sig.get('interpretation', 'N/A')}")
    print(
        f"  p={sig.get('p_value', '?')}, significant_95={sig.get('significant_95', '?')}"
    )
    print(f"\nReports: {json_path.name}, {md_path.name}")
    print(f"{'=' * 60}\n")

    return 0


# ─── CSV equity-curve mode (Backlog Item 136 extension) ──────────────────────


def _load_equity_csv(path: Path) -> list[float]:
    """Load an equity-curve CSV and return daily returns.

    Accepts two formats:
    1. Column named 'return' or 'ret' — raw daily returns, used directly.
    2. Any other numeric column (equity, portfolio_value, …) — pct_change applied.

    The first column may be dates (used as index, ignored for returns).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required for CSV loading") from exc

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    ret_cols = {"return", "ret", "daily_return", "daily_ret"}
    eq_cols = {
        "equity",
        "portfolio_value",
        "cumulative_returns",
        "value",
        "net_liq",
        "nav",
        "total_value",
    }

    # Prefer explicit return columns
    for c in ret_cols:
        if c in df.columns:
            return df[c].dropna().tolist()

    # Fall back to equity-like columns
    for c in eq_cols:
        if c in df.columns:
            return df[c].pct_change().dropna().tolist()

    # Last resort: first numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[0]].pct_change().dropna().tolist()

    raise ValueError(f"No usable numeric column found in {path}")


def _metrics_from_returns(rets: list[float], label: str) -> dict:
    """Compute CAGR, Sharpe, MDD, Win Rate, Total Return from daily returns."""
    if not rets:
        return {"label": label, "error": "empty returns"}

    import statistics as _st

    n = len(rets)
    mu = _st.mean(rets)
    sigma = _st.stdev(rets) if n > 1 else 1e-9
    sharpe = (mu / sigma) * math.sqrt(252) if sigma > 0 else 0.0

    total = 1.0
    for r in rets:
        total *= 1 + r
    cagr = total ** (252 / n) - 1

    # MDD
    cum = [1.0]
    for r in rets:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    mdd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd

    win_rate = sum(1 for r in rets if r > 0) / n if n > 0 else 0.0

    return {
        "label": label,
        "n_days": n,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "total_return_pct": round((total - 1) * 100, 2),
    }


def _print_comparison_table(ma: dict, mb: dict, sig: dict) -> None:
    name_a = ma.get("label", "A")
    name_b = mb.get("label", "B")
    keys = [
        ("CAGR (%)", "cagr_pct"),
        ("Sharpe", "sharpe"),
        ("MDD (%)", "mdd_pct"),
        ("Win Rate (%)", "win_rate_pct"),
        ("Total Return (%)", "total_return_pct"),
        ("N Days", "n_days"),
    ]
    col_w = max(len(name_a), len(name_b), 12)
    header = f"{'Metric':<22} {name_a:>{col_w}}  {name_b:>{col_w}}  {'Delta (B-A)':>12}"
    print("\n" + "=" * len(header))
    print(f"A/B COMPARISON: {name_a} vs {name_b}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for lbl, key in keys:
        va = ma.get(key)
        vb = mb.get(key)
        if va is not None and vb is not None:
            try:
                delta = round(float(vb) - float(va), 2)
                sign = "+" if delta > 0 else ""
                print(
                    f"{lbl:<22} {va:>{col_w}}  {vb:>{col_w}}  {sign + str(delta):>12}"
                )
            except (TypeError, ValueError):
                print(f"{lbl:<22} {str(va):>{col_w}}  {str(vb):>{col_w}}  {'—':>12}")
        else:
            print(f"{lbl:<22} {'—':>{col_w}}  {'—':>{col_w}}  {'—':>12}")
    print("=" * len(header))
    print(f"\nSharpe significance test: {sig.get('interpretation', 'N/A')}")
    print(
        f"  z={sig.get('z_stat', '?')}  p={sig.get('p_value', '?')}  sig_95={sig.get('significant_95', '?')}"
    )


def main_csv(argv: list[str] | None = None) -> int:
    """Entry point for CSV equity-curve comparison mode."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="A/B Strategy Comparison (equity-curve CSV mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--equity-a", required=True, help="CSV equity curve for strategy A"
    )
    parser.add_argument(
        "--equity-b", required=True, help="CSV equity curve for strategy B"
    )
    parser.add_argument("--name-a", default="Strategy A", help="Label for A")
    parser.add_argument("--name-b", default="Strategy B", help="Label for B")
    parser.add_argument(
        "--output-json", default=None, help="Optional path to save JSON result"
    )
    args = parser.parse_args(argv)

    path_a = Path(args.equity_a)
    path_b = Path(args.equity_b)

    try:
        rets_a = _load_equity_csv(path_a)
        rets_b = _load_equity_csv(path_b)
    except Exception as exc:
        logger.error("[AB] Failed to load equity CSV: %s", exc)
        return 2

    if len(rets_a) < 5 or len(rets_b) < 5:
        logger.error(
            "[AB] Insufficient data: A=%d days, B=%d days (need ≥5 each)",
            len(rets_a),
            len(rets_b),
        )
        return 2

    ma = _metrics_from_returns(rets_a, args.name_a)
    mb = _metrics_from_returns(rets_b, args.name_b)
    sig = _sharpe_z_test(rets_a, rets_b)

    _print_comparison_table(ma, mb, sig)

    # Winner recommendation
    sharpe_a = ma.get("sharpe", 0.0) or 0.0
    sharpe_b = mb.get("sharpe", 0.0) or 0.0
    significant = sig.get("significant_95", False)
    if significant and sharpe_b > sharpe_a:
        winner = args.name_b
        exit_code = 1
    else:
        winner = (
            args.name_a if sharpe_a >= sharpe_b else f"{args.name_a} (not significant)"
        )
        exit_code = 0

    print(
        f"\nRECOMMENDATION: {winner} wins"
        if exit_code == 1
        else f"\nRECOMMENDATION: {winner} (A wins or tie)"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_a": args.name_a,
        "label_b": args.name_b,
        "metrics_a": ma,
        "metrics_b": mb,
        "significance_test": sig,
        "winner": winner,
        "exit_code": exit_code,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("[AB] JSON saved to %s", out)

    return exit_code


if __name__ == "__main__":
    # Auto-detect mode: if --equity-a is in argv use CSV mode, else directory mode
    import sys as _sys

    if "--equity-a" in (_sys.argv[1:] or []):
        _sys.exit(main_csv())
    else:
        _sys.exit(main())
