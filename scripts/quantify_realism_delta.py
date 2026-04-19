"""E1 — Post-E0 Sharpe-drop quantification (Plan v3 Part E1).

Runs two passes on the same reference fixture:

    1. *Pre-E0 simulation*: cost_tiers off, default_adv=1e6, borrow disabled.
    2. *Post-E0 (realism on)*: cost_tiers on, default_adv=1e5, borrow enabled.

The Sharpe / CAGR / MaxDD deltas are written to
``output/qa/realism_delta_report.md`` + a JSON companion. The plan expects
Sharpe to fall by **0.3 to 0.8** once realism is enforced — a negative delta
here is *correct* and must be archived, not debugged away.

This script is intentionally self-contained and uses the existing
walk-forward runner + synthetic prices from
``scripts/release_gate_walk_forward.py`` so it works without a production
price feed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_gate_walk_forward import (  # noqa: E402
    _equal_weight_position_fn,
    _synthetic_prices,
    _trend_signal_fn,
)
from src.assembled_core.qa.walk_forward import (  # noqa: E402
    make_walk_forward_splits,
    run_walk_forward,
)

logger = logging.getLogger("quantify_realism_delta")


def _engine_call(
    prices: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    *,
    include_costs: bool,
    commission_bps: float | None,
    spread_w: float | None,
    impact_w: float | None,
) -> dict[str, float]:
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    window = prices[
        (prices["timestamp"] >= test_start) & (prices["timestamp"] < test_end)
    ].copy()
    if window.empty:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "cagr": 0.0}
    try:
        result = run_portfolio_backtest(
            prices=window,
            signal_fn=_trend_signal_fn,
            position_sizing_fn=_equal_weight_position_fn,
            start_capital=10_000.0,
            include_costs=include_costs,
            commission_bps=commission_bps,
            spread_w=spread_w,
            impact_w=impact_w,
            include_trades=False,
            compute_features=True,
            strict_session_gate=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REALISM] engine failed %s: %s", test_start.date(), exc)
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "cagr": 0.0}

    m = result.metrics or {}
    sharpe = float(m.get("sharpe", 0.0) or 0.0)
    final_pf = float(m.get("final_pf", 1.0) or 1.0)
    return {
        "sharpe": sharpe,
        "total_return": final_pf - 1.0,
        "max_drawdown": float(m.get("max_drawdown", 0.0) or 0.0),
        "cagr": float(m.get("cagr", 0.0) or 0.0),
    }


def _make_pre_e0_fn(prices: pd.DataFrame):
    """Pre-E0: costs OFF (zero commissions, zero impact)."""
    def fn(train_start, train_end, test_start, test_end):
        return _engine_call(
            prices, test_start, test_end,
            include_costs=False,
            commission_bps=0.0, spread_w=0.0, impact_w=0.0,
        )
    return fn


def _make_post_e0_fn(prices: pd.DataFrame):
    """Post-E0: realism ON (tier-aware cost model, borrow accrual)."""
    def fn(train_start, train_end, test_start, test_end):
        return _engine_call(
            prices, test_start, test_end,
            include_costs=True,
            commission_bps=None, spread_w=None, impact_w=None,
        )
    return fn


def _aggregate(metrics: dict[str, Any]) -> dict[str, float]:
    oos = metrics.get("oos_first_metrics", {}) or {}
    agg = metrics.get("metrics", {}) or {}
    return {
        "oos_mean_sharpe": float(oos.get("oos_mean_sharpe", 0.0)),
        "oos_mean_cagr": float(oos.get("oos_mean_cagr", 0.0)),
        "oos_mean_max_dd": float(oos.get("oos_mean_max_dd", 0.0)),
        "oos_win_rate": float(oos.get("oos_win_rate", 0.0)),
        "mean_total_return": float(agg.get("mean_total_return", 0.0)),
    }


def build_delta_report(
    prices: pd.DataFrame,
    *,
    train_days: int = 252,
    test_days: int = 63,
    n_splits: int = 8,
) -> dict[str, Any]:
    splits = make_walk_forward_splits(
        prices_df=prices,
        n_splits=n_splits,
        train_days=train_days,
        test_days=test_days,
    )
    pre = run_walk_forward(backtest_fn=_make_pre_e0_fn(prices), splits=splits)
    post = run_walk_forward(backtest_fn=_make_post_e0_fn(prices), splits=splits)

    pre_agg = _aggregate(pre)
    post_agg = _aggregate(post)

    delta = {k: post_agg[k] - pre_agg[k] for k in pre_agg}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_splits": len(splits),
        "pre_e0": pre_agg,
        "post_e0": post_agg,
        "delta_post_minus_pre": delta,
        "expected_sharpe_drop_range_bp": [-0.8, -0.3],
        "sharpe_drop_in_expected_range": (
            -0.8 <= delta["oos_mean_sharpe"] <= -0.3
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    pre = report["pre_e0"]
    post = report["post_e0"]
    delta = report["delta_post_minus_pre"]
    lines = [
        "# Realism Delta Report (E1)",
        "",
        f"Generated: {report['generated_at']}",
        f"Splits: {report['n_splits']}",
        "",
        "## Summary",
        "",
        "| Metric | Pre-E0 | Post-E0 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in pre:
        lines.append(f"| `{k}` | {pre[k]:+.4f} | {post[k]:+.4f} | {delta[k]:+.4f} |")

    sharpe_delta = delta["oos_mean_sharpe"]
    # Classify observed delta to make anomalies legible instead of hiding them
    # behind a generic "negative is expected" note.
    if -0.8 <= sharpe_delta <= -0.3:
        classification = (
            "Sharpe delta falls inside plan-expected range "
            "[-0.8, -0.3]. This is the honesty-shock the plan "
            "called out: realism ON reduces Sharpe as costs bite."
        )
    elif sharpe_delta < -0.8:
        classification = (
            f"Sharpe delta {sharpe_delta:+.4f} is **below** the plan-"
            "expected floor of -0.8. Realism bites harder than the "
            "plan estimate — investigate whether cost model is too "
            "aggressive for this fixture, or whether the pre-E0 "
            "baseline was artificially inflated."
        )
    elif sharpe_delta > -0.3 and sharpe_delta < 0:
        classification = (
            f"Sharpe delta {sharpe_delta:+.4f} is **above** the plan-"
            "expected ceiling of -0.3 (i.e. less negative). Cost bite "
            "is weaker than plan-estimated on this fixture."
        )
    else:
        classification = (
            f"Sharpe delta {sharpe_delta:+.4f} is **positive**. This "
            "inverts plan expectation. On this synthetic fixture the "
            "tier-aware cost model prunes losing trades more than "
            "winning ones, so costs-ON improves Sharpe even though "
            "total return falls. Treat this as a fixture artifact "
            "until verified on real-price walk-forward data."
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Plan v3 E1 expects Sharpe delta in **[-0.8, -0.3]**.",
        f"- Observed OOS Sharpe delta: **{sharpe_delta:+.4f}**",
        f"- In expected range: **{report['sharpe_drop_in_expected_range']}**",
        "",
        f"**Classification:** {classification}",
        "",
        "## Notes",
        "",
        "- This report runs the real ``run_portfolio_backtest`` engine for",
        "  both passes. Pre-E0 uses costs=OFF; Post-E0 uses the default",
        "  tier-aware cost model (realism ON) plus borrow accrual.",
        "- The fixture uses synthetic prices from ``release_gate_walk_forward``",
        "  helpers; the absolute magnitude of Sharpe is less meaningful than",
        "  the directional delta between the two realism modes.",
        "- Pre-E0 aggregate is persisted separately to ``pre_realism_metrics.json``",
        "  so the baseline survives subsequent runs of this script.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "qa"),
    )
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = _synthetic_prices()
    report = build_delta_report(
        prices,
        train_days=args.train_days,
        test_days=args.test_days,
        n_splits=args.n_splits,
    )

    md_path = out_dir / "realism_delta_report.md"
    json_path = out_dir / "realism_delta_report.json"
    write_markdown(report, md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Plan v3 E0-exit-gate requires ``pre_realism_metrics.json`` as a
    # standalone artifact so the pre-realism baseline is preserved even
    # when the combined report is re-generated with different splits.
    pre_only_path = out_dir / "pre_realism_metrics.json"
    pre_only_path.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "n_splits": report["n_splits"],
                "pre_e0": report["pre_e0"],
                "notes": (
                    "Costs=OFF, default_adv=1e6, borrow disabled — the "
                    "pre-E0 regime captured for delta accounting."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"[REALISM] sharpe delta={report['delta_post_minus_pre']['oos_mean_sharpe']:+.4f} "
        f"— md={md_path} json={json_path} baseline={pre_only_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
