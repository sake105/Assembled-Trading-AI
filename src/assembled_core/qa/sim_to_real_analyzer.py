"""Sim-to-real gap analyzer (Plan 11/10 §3.2).

Quantifies the gap between backtest (modeled) and paper-live (real broker).
Compares: slippage, fill rate, Sharpe, daily P&L variance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_sim_to_real_gap(
    backtest_period_results: dict,
    paper_live_period_results: dict,
) -> dict:
    """Compute sim-to-real gap metrics.

    Args:
        backtest_period_results: Dict with keys: sharpe, cagr, avg_slippage_bps,
            fill_rate, n_trades, daily_pnl_std.
        paper_live_period_results: Same schema as backtest_period_results.

    Returns:
        Dict: sharpe_drop, slippage_gap_bps, fill_rate_gap, classification,
            details, verdict_text.
    """
    bt = backtest_period_results
    pa = paper_live_period_results

    bt_sharpe = float(bt.get("sharpe", 0) or 0)
    pa_sharpe = float(pa.get("sharpe", 0) or 0)
    sharpe_drop = pa_sharpe - bt_sharpe

    bt_slip = float(bt.get("avg_slippage_bps", 0) or 0)
    pa_slip = float(pa.get("avg_slippage_bps", 0) or 0)
    slip_gap_bps = pa_slip - bt_slip

    bt_fill = float(bt.get("fill_rate", 1.0) or 1.0)
    pa_fill = float(pa.get("fill_rate", 1.0) or 1.0)
    fill_rate_gap = pa_fill - bt_fill

    bt_vol = float(bt.get("daily_pnl_std", 0) or 0)
    pa_vol = float(pa.get("daily_pnl_std", 0) or 0)
    vol_ratio = (pa_vol / bt_vol) if bt_vol > 0 else None

    classification = _classify_gap(sharpe_drop, slip_gap_bps)

    result = {
        "sharpe_backtest": bt_sharpe,
        "sharpe_paper": pa_sharpe,
        "sharpe_drop": round(sharpe_drop, 4),
        "avg_slippage_bps_backtest": bt_slip,
        "avg_slippage_bps_paper": pa_slip,
        "slippage_gap_bps": round(slip_gap_bps, 2),
        "fill_rate_backtest": bt_fill,
        "fill_rate_paper": pa_fill,
        "fill_rate_gap": round(fill_rate_gap, 4),
        "daily_pnl_vol_ratio": round(vol_ratio, 4) if vol_ratio is not None else None,
        "classification": classification,
        "verdict_text": _verdict_text(classification, sharpe_drop, slip_gap_bps),
    }

    logger.info(
        "[sim_to_real] Sharpe: bt=%.2f pa=%.2f drop=%.2f | slip gap=%.1f bps | %s",
        bt_sharpe, pa_sharpe, sharpe_drop, slip_gap_bps, classification,
    )
    return result


def _classify_gap(sharpe_drop: float, slip_gap_bps: float) -> str:
    if sharpe_drop > -0.3 and slip_gap_bps < 5:
        return "EXCELLENT"
    elif sharpe_drop > -0.7 and slip_gap_bps < 15:
        return "ACCEPTABLE"
    elif sharpe_drop > -1.5:
        return "CONCERNING"
    else:
        return "ALARMING"


def _verdict_text(classification: str, sharpe_drop: float, slip_gap_bps: float) -> str:
    if classification == "EXCELLENT":
        return "Backtest realistic — ready to scale"
    elif classification == "ACCEPTABLE":
        return "Typical sim-to-real gap — ready for live with caution"
    elif classification == "CONCERNING":
        return (
            f"Significant gap (Sharpe drop {sharpe_drop:+.2f}, "
            f"slippage +{slip_gap_bps:.0f} bps) — investigate before live"
        )
    else:
        return (
            f"Critical gap (Sharpe drop {sharpe_drop:+.2f}, "
            f"slippage +{slip_gap_bps:.0f} bps) — do NOT proceed to live"
        )


def load_paper_live_summary(paper_run_dir: Path, n_days: int = 7) -> dict:
    """Load and aggregate last n_days of paper-live daily summaries.

    Looks for files matching *_summary.json in paper_run_dir.
    Returns aggregated metrics dict.
    """
    if not paper_run_dir.exists():
        logger.warning("[sim_to_real] paper run dir not found: %s", paper_run_dir)
        return {}

    files = sorted(paper_run_dir.rglob("*_summary.json"))[-n_days:]
    if not files:
        logger.warning("[sim_to_real] no summary files in %s", paper_run_dir)
        return {}

    sharpes, slippages, fill_rates, pnl_stds = [], [], [], []
    n_trades_total = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "sharpe" in data:
                sharpes.append(float(data["sharpe"]))
            if "avg_slippage_bps" in data:
                slippages.append(float(data["avg_slippage_bps"]))
            if "fill_rate" in data:
                fill_rates.append(float(data["fill_rate"]))
            if "daily_pnl_std" in data:
                pnl_stds.append(float(data["daily_pnl_std"]))
            if "n_trades" in data:
                n_trades_total += int(data.get("n_trades", 0))
        except Exception as exc:
            logger.debug("[sim_to_real] skip %s: %s", f.name, exc)

    return {
        "sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
        "avg_slippage_bps": sum(slippages) / len(slippages) if slippages else 0.0,
        "fill_rate": sum(fill_rates) / len(fill_rates) if fill_rates else 1.0,
        "daily_pnl_std": sum(pnl_stds) / len(pnl_stds) if pnl_stds else 0.0,
        "n_trades": n_trades_total,
        "n_days_loaded": len(files),
    }
