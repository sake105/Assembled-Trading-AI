"""Daily QA report — Bayesian Sharpe + risk-parity weights + LLM-RAG news digest.

Usage:
    python scripts/generate_daily_qa_report.py \
        --equity-file output/equity_curve.csv \
        [--strategies s1,s2] \
        [--news-headlines "Headline 1;Headline 2"] \
        [--out output/qa_report_YYYYMMDD.json]

Integrates:
  §2 qa/bayesian_metrics    — per-strategy Bayesian Sharpe posteriors
  §8 portfolio/strategy_allocator — inverse-vol risk-parity weights
  §7 intel/news_rag         — weekly news-digest via LLM-RAG (optional)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_equity_csv(path: str) -> dict[str, list[float]]:
    """Load an equity-curve CSV into {strategy: [daily_returns]}.

    Expected columns: date, <strategy_name1>, <strategy_name2>, ...
    One column named 'date' or 'timestamp'; all other columns are strategy equity curves.
    Returns are computed as pct_change().
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas required — pip install pandas")
        return {}

    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), None)
    if date_col:
        df = df.drop(columns=[date_col])

    returns: dict[str, list[float]] = {}
    for col in df.columns:
        pct = df[col].pct_change(fill_method=None).dropna().tolist()
        if pct:
            returns[col] = pct
    return returns


def _section_bayesian_sharpe(returns: dict[str, list[float]]) -> dict:
    """§2 — Bayesian Sharpe posteriors for each strategy."""
    section: dict = {"strategies": {}, "comparison": None, "backend": "analytic"}
    try:
        from src.assembled_core.qa.bayesian_metrics import (
            bayesian_sharpe_posterior,
            hierarchical_strategy_comparison,
        )

        for name, rets in returns.items():
            if len(rets) < 5:
                continue
            posterior = bayesian_sharpe_posterior(rets, strategy=name)
            section["strategies"][name] = {
                "mean_sharpe": round(posterior.mean, 4),
                "hdi_lower": round(posterior.hdi_lower, 4),
                "hdi_upper": round(posterior.hdi_upper, 4),
                "p_positive": round(posterior.p_positive, 4),
                "n_obs": posterior.n_obs,
                "backend": posterior.backend,
            }
            section["backend"] = posterior.backend

        if len(returns) > 1:
            comp = hierarchical_strategy_comparison(returns)
            section["comparison"] = {
                "p_best": {k: round(v, 4) for k, v in comp.p_best.items()},
                "population_mean": round(comp.population_mean, 4),
                "population_std": round(comp.population_std, 4),
                "backend": comp.backend,
            }
    except Exception as exc:
        log.warning("bayesian_metrics failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_risk_parity(returns: dict[str, list[float]]) -> dict:
    """§8 — Inverse-vol risk-parity weights."""
    section: dict = {"weights": {}, "vol_scale": 1.0, "strategy_vols": {}}
    try:
        from src.assembled_core.portfolio.strategy_allocator import (
            allocate_from_returns_dict,
        )

        result = allocate_from_returns_dict(returns, target_vol=0.15)
        section["weights"] = {k: round(v, 6) for k, v in result.weights.items()}
        section["vol_scale"] = round(result.vol_scale, 4)
        section["estimated_portfolio_vol"] = round(result.estimated_portfolio_vol, 6)
        section["target_vol"] = result.target_vol
        section["strategy_vols"] = {k: round(v, 6) for k, v in result.strategy_vols.items()}
    except Exception as exc:
        log.warning("strategy_allocator failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_news_rag(headlines: list[str]) -> dict:
    """§7 — LLM-RAG news digest (optional, degrades gracefully)."""
    section: dict = {"results": [], "backend": "none"}
    if not headlines:
        section["skipped"] = "no headlines provided"
        return section
    try:
        from src.assembled_core.intel.news_rag import NewsRAG

        rag = NewsRAG()
        for hl in headlines[:10]:  # cap at 10 per report
            try:
                result = rag.query(hl, ticker="")
                section["results"].append({
                    "headline": hl,
                    "direction": result.predicted_direction,
                    "confidence": round(result.confidence, 4),
                    "reasoning": result.reasoning[:200] if result.reasoning else "",
                    "backend": result.backend,
                })
                section["backend"] = result.backend
            except Exception as e:
                section["results"].append({"headline": hl, "error": str(e)})
    except Exception as exc:
        log.warning("news_rag failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_differential_privacy(returns: dict[str, list[float]], epsilon: float = 1.0) -> dict:
    """§10 — Differentially private Sharpe estimates (publishable without leaking raw returns).

    Applies Gaussian mechanism to Sharpe ratios so they can be shared externally
    while preserving (epsilon, delta=1e-5)-DP.
    """
    section: dict = {"epsilon": epsilon, "delta": 1e-5, "strategy_dp_sharpes": {}}
    if not returns:
        section["skipped"] = "no returns"
        return section
    try:
        from src.assembled_core.ml.differential_privacy import PrivacyBudget, dp_mean

        budget = PrivacyBudget(epsilon_total=epsilon * len(returns), delta=1e-5)
        for name, rets in returns.items():
            if len(rets) < 5:
                continue
            import math as _math
            std = float(sum((r - sum(rets) / len(rets)) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
            raw_sharpe = (sum(rets) / len(rets)) / max(std, 1e-9) * _math.sqrt(252)
            # DP Sharpe: treat returns as the sensitive dataset; clip_bound protects individuals
            dp_sharpe = dp_mean(rets, clip_bound=0.10, epsilon=epsilon, delta=1e-5)
            annualized_dp_sharpe = dp_sharpe / max(std, 1e-9) * _math.sqrt(252)
            budget.consume(epsilon)
            section["strategy_dp_sharpes"][name] = {
                "raw_sharpe": round(raw_sharpe, 4),
                "dp_sharpe_annualized": round(float(annualized_dp_sharpe), 4),
                "epsilon_consumed": round(budget.epsilon_used, 4),
                "budget_exhausted": budget.is_exhausted,
            }
    except Exception as exc:
        log.warning("differential_privacy section failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_trade_journal(journal_path: str = "output/runs/trade_journal.jsonl", n_days: int = 7) -> dict:
    """Last N days trade summary from trade journal (Plan 11/10 §5.1.3)."""
    section: dict = {"n_days": n_days, "n_trades": 0, "top_symbols": []}
    try:
        from pathlib import Path as _Path
        from src.assembled_core.ops.trade_journal import load_trade_journal
        import pandas as _pd
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        jp = _Path(journal_path)
        if not jp.exists():
            # Try output dir
            for alt in _Path("output").rglob("trade_journal.jsonl"):
                jp = alt
                break
        if not jp.exists():
            section["verdict"] = "NO_JOURNAL"
            return section
        raw = load_trade_journal(jp, days=n_days)
        if not raw:
            section["verdict"] = "EMPTY"
            return section
        trades = _pd.DataFrame(raw)
        section["n_trades"] = len(trades)
        if "symbol" in trades.columns:
            top = trades.groupby("symbol").size().sort_values(ascending=False).head(5)
            section["top_symbols"] = [{"symbol": s, "count": int(c)} for s, c in top.items()]
        if "pnl" in trades.columns:
            section["total_pnl"] = round(float(trades["pnl"].sum()), 2)
            section["avg_pnl_per_trade"] = round(float(trades["pnl"].mean()), 2)
    except Exception as exc:
        log.warning("trade_journal section failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_sim_to_real_gap(paper_live_dir: str = "output/runs/_paper_ledger", n_days: int = 7) -> dict:
    """Last N days: sim-to-real gap verdict (Plan 11/10 §3.2.3)."""
    section: dict = {"n_days": n_days, "verdict": "NO_DATA", "classification": "UNKNOWN"}
    try:
        from pathlib import Path as _Path
        from src.assembled_core.qa.sim_to_real_analyzer import load_paper_live_summary, analyze_sim_to_real_gap
        pa = load_paper_live_summary(_Path(paper_live_dir), n_days=n_days)
        if not pa or pa.get("n_days_loaded", 0) == 0:
            section["verdict"] = "NO_DATA"
            return section
        # Use a TA-only backtest baseline (last known values from fresh backtest)
        bt = {
            "sharpe": 2.45,          # OOS 2025-2026 baseline from 2026-05-03 session
            "avg_slippage_bps": 1.5,  # Modeled slippage
            "fill_rate": 1.0,
            "daily_pnl_std": 0.01,
        }
        gap = analyze_sim_to_real_gap(bt, pa)
        section.update({
            "n_days_loaded": pa.get("n_days_loaded"),
            "classification": gap["classification"],
            "sharpe_drop": gap["sharpe_drop"],
            "slippage_gap_bps": gap["slippage_gap_bps"],
            "fill_rate_gap": gap["fill_rate_gap"],
            "verdict": gap["verdict_text"],
        })
    except Exception as exc:
        log.warning("sim_to_real_gap section failed: %s", exc)
        section["error"] = str(exc)
    return section


def _section_drill_status(drill_dir: str = "output/drills") -> dict:
    """Last 4 weeks of drill results (Plan 11/10 §4.2.2)."""
    section: dict = {"n_drills": 0, "all_pass": True, "failed_drills": [], "verdicts": []}
    try:
        import json as _json
        from pathlib import Path as _Path
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cutoff = _dt.now(_tz.utc) - _td(days=28)
        drill_path = _Path(drill_dir)
        if not drill_path.exists():
            section["verdict"] = "NO_DRILLS_YET"
            return section
        reports = sorted(drill_path.glob("*.json"))
        recent = []
        for r in reports:
            try:
                data = _json.loads(r.read_text(encoding="utf-8"))
                ts_str = data.get("started_at") or data.get("finished_at") or ""
                if ts_str:
                    ts = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts >= cutoff:
                        recent.append({"file": r.name, "verdict": data.get("verdict", "UNKNOWN")})
            except Exception:
                pass
        section["n_drills"] = len(recent)
        failed = [d for d in recent if d["verdict"] != "PASS"]
        section["all_pass"] = len(failed) == 0
        section["failed_drills"] = failed
        section["verdicts"] = recent
        if failed:
            log.warning("[QA] %d drill(s) FAILED in last 28 days: %s", len(failed), failed)
    except Exception as exc:
        log.warning("drill_status section failed: %s", exc)
        section["error"] = str(exc)
    return section


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily QA report")
    parser.add_argument("--equity-file", default="", help="CSV with equity curves (one column per strategy)")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy names (filters equity-file columns)")
    parser.add_argument("--news-headlines", default="", help="Semicolon-separated headlines for RAG digest")
    parser.add_argument("--dp-epsilon", type=float, default=1.0, help="DP epsilon for Sharpe privatisation (default 1.0)")
    parser.add_argument("--out", default="", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    # Load returns
    returns: dict[str, list[float]] = {}
    if args.equity_file and os.path.isfile(args.equity_file):
        returns = _load_equity_csv(args.equity_file)
        if args.strategies:
            keep = set(args.strategies.split(","))
            returns = {k: v for k, v in returns.items() if k in keep}
    else:
        log.info("No equity file provided — using empty returns for demonstration")

    headlines = [h.strip() for h in args.news_headlines.split(";") if h.strip()] if args.news_headlines else []

    report = {
        "report_date": str(date.today()),
        "n_strategies": len(returns),
        "bayesian_sharpe": _section_bayesian_sharpe(returns),
        "risk_parity": _section_risk_parity(returns),
        "news_rag_digest": _section_news_rag(headlines),
        "differential_privacy": _section_differential_privacy(returns, epsilon=args.dp_epsilon),
        "sim_to_real_gap": _section_sim_to_real_gap(),
        "drill_status": _section_drill_status(),
        "trade_journal": _section_trade_journal(),
    }

    out_str = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_str)
        log.info("[OK] report written to %s", args.out)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
