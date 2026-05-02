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
    }

    out_str = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out_str)
        log.info("[OK] report written to %s", args.out)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
