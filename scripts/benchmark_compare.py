"""Backlog Item 30 — Benchmark Comparison Report.

Compares a strategy backtest against standard benchmarks:
  - SPY (US equity beta)
  - 60/40 (SPY + AGG blend)
  - Risk-Parity proxy (equal-vol weighted)

Outputs per-benchmark: CAGR, Sharpe, MDD, Alpha (beta-adjusted), Information Ratio.

Usage:
    python scripts/benchmark_compare.py \\
        --equity output/accounting_report_backtest_<hash>/equity_curve.parquet \\
        --start 2023-01-01 --end 2026-04-30 \\
        --output output/qa/benchmark_compare_YYYYMMDD.md

    # From a backtest summary JSON:
    python scripts/benchmark_compare.py \\
        --equity output/accounting_report_backtest_<hash>/equity_curve.parquet
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

# ─── Metric helpers ──────────────────────────────────────────────────────────


def _metrics_from_returns(rets: list[float], label: str) -> dict:
    """Compute CAGR, Sharpe, MDD from a daily return series."""
    import statistics

    n = len(rets)
    if n < 5:
        return {"label": label, "n_days": n, "error": "too few observations"}

    mu = statistics.mean(rets)
    sigma = statistics.stdev(rets) if n > 1 else 1e-9
    ann_factor = 252

    cagr = (1 + sum(rets)) ** (ann_factor / n) - 1 if n > 0 else 0.0
    sharpe = (mu / sigma) * math.sqrt(ann_factor) if sigma > 0 else 0.0

    cum = [1.0]
    for r in rets:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    mdd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        mdd = min(mdd, dd)

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
        "n_days": n,
        "ann_vol_pct": round(sigma * math.sqrt(ann_factor) * 100, 2),
    }


def _alpha_beta(strategy_rets: list[float], benchmark_rets: list[float]) -> dict:
    """Compute CAPM alpha, beta, and Information Ratio vs benchmark."""
    import statistics

    n = min(len(strategy_rets), len(benchmark_rets))
    if n < 10:
        return {"alpha_ann_pct": None, "beta": None, "ir": None}

    s = strategy_rets[:n]
    b = benchmark_rets[:n]

    mu_s = statistics.mean(s)
    mu_b = statistics.mean(b)
    sigma_b = statistics.stdev(b) if n > 1 else 1e-9

    cov = sum((s[i] - mu_s) * (b[i] - mu_b) for i in range(n)) / (n - 1)
    beta = cov / (sigma_b**2) if sigma_b > 0 else 0.0

    # Daily alpha (Jensen's alpha)
    rf = 0.0  # simplified: risk-free rate ≈ 0
    alpha_daily = mu_s - (rf + beta * (mu_b - rf))
    alpha_ann = alpha_daily * 252

    # Information Ratio: active return / tracking error
    active = [s[i] - b[i] for i in range(n)]
    tracking_error = statistics.stdev(active) * math.sqrt(252) if n > 1 else 1e-9
    active_ann = statistics.mean(active) * 252
    ir = active_ann / tracking_error if tracking_error > 0 else 0.0

    return {
        "alpha_ann_pct": round(alpha_ann * 100, 2),
        "beta": round(beta, 3),
        "ir": round(ir, 3),
    }


# ─── Benchmark loading ───────────────────────────────────────────────────────


def _load_benchmark_returns(
    ticker: str,
    start: str | None,
    end: str | None,
    n_days: int,
) -> list[float]:
    """Try to load benchmark returns from yfinance or generate synthetic data."""
    try:
        import yfinance as yf

        kwargs: dict = {}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        if not start and not end:
            kwargs["period"] = f"{max(n_days, 252)}d"
        hist = yf.download(ticker, progress=False, auto_adjust=True, **kwargs)
        if hist.empty:
            raise ValueError(f"empty data for {ticker}")
        close_col = "Close" if "Close" in hist.columns else hist.columns[0]
        rets = hist[close_col].pct_change().dropna().tolist()
        logger.info("[benchmark] loaded %d %s returns from yfinance", len(rets), ticker)
        return rets
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[benchmark] yfinance failed for %s: %s — using empty", ticker, exc
        )
        return []


def _build_60_40_returns(spy_rets: list[float], agg_rets: list[float]) -> list[float]:
    n = min(len(spy_rets), len(agg_rets))
    return [0.6 * spy_rets[i] + 0.4 * agg_rets[i] for i in range(n)]


def _build_risk_parity_returns(
    spy_rets: list[float], agg_rets: list[float]
) -> list[float]:
    """Equal-vol weighted SPY+AGG as risk-parity proxy."""
    import statistics

    if not spy_rets or not agg_rets:
        return []
    n = min(len(spy_rets), len(agg_rets))
    vol_spy = statistics.stdev(spy_rets[:n]) or 1e-9
    vol_agg = statistics.stdev(agg_rets[:n]) or 1e-9
    inv_spy = 1.0 / vol_spy
    inv_agg = 1.0 / vol_agg
    total = inv_spy + inv_agg
    w_spy = inv_spy / total
    w_agg = inv_agg / total
    return [w_spy * spy_rets[i] + w_agg * agg_rets[i] for i in range(n)]


# ─── Strategy returns loading ─────────────────────────────────────────────────


def _load_strategy_returns(equity_path: Path) -> list[float]:
    """Load daily returns from an equity curve file."""
    try:
        import pandas as pd

        if equity_path.suffix == ".parquet":
            df = pd.read_parquet(equity_path)
        elif equity_path.suffix == ".csv":
            df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"unsupported format: {equity_path.suffix}")

        eq_col = next(
            (
                c
                for c in [
                    "equity",
                    "portfolio_value",
                    "value",
                    "net_liq",
                    "cumulative_returns",
                    "close",
                ]
                if c in df.columns
            ),
            None,
        )
        if eq_col:
            return df[eq_col].pct_change().dropna().tolist()

        # Try index as series
        if hasattr(df, "squeeze"):
            s = df.squeeze()
            if s.ndim == 1:
                return s.pct_change().dropna().tolist()

    except Exception as exc:  # noqa: BLE001
        logger.warning("[strategy] failed to load equity curve: %s", exc)

    return []


# ─── Report formatting ────────────────────────────────────────────────────────


def _format_report(strategy_metrics: dict, benchmarks: list[dict]) -> str:
    lines = [
        "# Benchmark Comparison Report",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Performance Summary",
        "",
        "| Metric | Strategy |" + "".join(f" {b['label']} |" for b in benchmarks),
        "|--------|----------|" + "".join("-----------:|" for _ in benchmarks),
    ]

    for key, name in [
        ("cagr_pct", "CAGR (%)"),
        ("sharpe", "Sharpe"),
        ("mdd_pct", "MDD (%)"),
        ("ann_vol_pct", "Ann Vol (%)"),
        ("n_days", "Days"),
    ]:
        row = f"| {name} | {strategy_metrics.get(key, '—')} |"
        for b in benchmarks:
            row += f" {b.get(key, '—')} |"
        lines.append(row)

    lines += ["", "## Alpha & Information Ratio vs Benchmarks", ""]
    lines.append("| Benchmark | Alpha (ann %) | Beta | IR |")
    lines.append("|-----------|--------------|------|-----|")
    for b in benchmarks:
        alpha = b.get("alpha_ann_pct", "—")
        beta = b.get("beta", "—")
        ir = b.get("ir", "—")
        lines.append(f"| {b['label']} | {alpha} | {beta} | {ir} |")

    lines += [
        "",
        "## Notes",
        "- Alpha = Jensen's alpha (annualised), CAPM with RF=0.",
        "- IR = (active return ann.) / (tracking error ann.).",
        "- 60/40 = 60% SPY + 40% AGG.",
        "- Risk-Parity = equal-vol weighted SPY + AGG.",
        "- Benchmark returns sourced from yfinance (adj. close).",
        "- Survivorship bias may be present in strategy returns.",
    ]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark Comparison Report")
    parser.add_argument(
        "--equity", required=True, help="Path to equity_curve.parquet or .csv"
    )
    parser.add_argument(
        "--start", help="Start date YYYY-MM-DD (for benchmark download)"
    )
    parser.add_argument("--end", help="End date YYYY-MM-DD (for benchmark download)")
    parser.add_argument("--output-dir", default="output/qa")
    args = parser.parse_args(argv)

    equity_path = ROOT / args.equity

    logger.info("[benchmark] Loading strategy returns from %s", equity_path)
    strategy_rets = _load_strategy_returns(equity_path)
    if not strategy_rets:
        logger.error("[benchmark] No strategy returns found — aborting")
        return 1

    n_days = len(strategy_rets)
    strategy_metrics = _metrics_from_returns(strategy_rets, "Strategy")
    logger.info(
        "[benchmark] Strategy: CAGR=%.2f%% Sharpe=%.3f MDD=%.2f%% n=%d",
        strategy_metrics.get("cagr_pct", 0),
        strategy_metrics.get("sharpe", 0),
        strategy_metrics.get("mdd_pct", 0),
        n_days,
    )

    # Load benchmark data
    logger.info("[benchmark] Downloading benchmark data (SPY, AGG)...")
    spy_rets = _load_benchmark_returns("SPY", args.start, args.end, n_days)
    agg_rets = _load_benchmark_returns("AGG", args.start, args.end, n_days)

    # Align lengths to strategy
    n = min(len(strategy_rets), len(spy_rets)) if spy_rets else 0

    benchmarks: list[dict] = []

    # SPY
    if spy_rets:
        spy_aligned = spy_rets[-n:] if len(spy_rets) >= n else spy_rets
        strat_aligned = strategy_rets[-n:]
        spy_m = _metrics_from_returns(spy_aligned, "SPY")
        spy_m.update(_alpha_beta(strat_aligned, spy_aligned))
        benchmarks.append(spy_m)

    # 60/40
    if spy_rets and agg_rets:
        n_both = min(len(spy_rets), len(agg_rets))
        rets_6040 = _build_60_40_returns(spy_rets[-n_both:], agg_rets[-n_both:])
        n_align = min(len(strategy_rets), len(rets_6040))
        m_6040 = _metrics_from_returns(rets_6040[-n_align:], "60/40")
        m_6040.update(_alpha_beta(strategy_rets[-n_align:], rets_6040[-n_align:]))
        benchmarks.append(m_6040)

    # Risk-Parity
    if spy_rets and agg_rets:
        rp_rets = _build_risk_parity_returns(spy_rets[-n_both:], agg_rets[-n_both:])
        n_align_rp = min(len(strategy_rets), len(rp_rets))
        m_rp = _metrics_from_returns(rp_rets[-n_align_rp:], "Risk-Parity")
        m_rp.update(_alpha_beta(strategy_rets[-n_align_rp:], rp_rets[-n_align_rp:]))
        benchmarks.append(m_rp)

    # Output
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    md_path = out_dir / f"benchmark_compare_{ts}.md"
    json_path = out_dir / f"benchmark_compare_{ts}.json"

    md_content = _format_report(strategy_metrics, benchmarks)
    md_path.write_text(md_content, encoding="utf-8")

    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_metrics,
        "benchmarks": benchmarks,
    }
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'BENCHMARK COMPARISON':^65}")
    print(f"{'='*65}")
    print(f"{'Metric':<18} {'Strategy':>10}", end="")
    for b in benchmarks:
        print(f" {b['label']:>12}", end="")
    print()
    print("-" * 65)
    for key, name in [
        ("cagr_pct", "CAGR (%)"),
        ("sharpe", "Sharpe"),
        ("mdd_pct", "MDD (%)"),
        ("ann_vol_pct", "Ann Vol (%)"),
    ]:
        print(f"{name:<18} {strategy_metrics.get(key, '?'):>10}", end="")
        for b in benchmarks:
            print(f" {b.get(key, '—'):>12}", end="")
        print()
    print(f"{'='*65}")
    print("\nAlpha vs benchmarks:")
    for b in benchmarks:
        alpha = b.get("alpha_ann_pct")
        ir = b.get("ir")
        print(f"  vs {b['label']:<14}: alpha={alpha}% ann, IR={ir}")
    print(f"\nReports: {md_path.name}, {json_path.name}")
    print(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
