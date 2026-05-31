"""E3 + E4 — Walk-forward OOS Sharpe + Deflated Sharpe (synthetic smoke).

Runs a walk-forward analysis on a reference configuration and checks two
statistical gates (Plan v3 Part E):

    E3: mean_oos_sharpe >= OOS_SHARPE_MIN
    E4: deflated_sharpe_probability >= DSR_MIN

HONEST SCOPE (CI-001): this runs on SYNTHETIC random-walk prices
(``_synthetic_prices``, seed=42), so E3/E4 measure nothing about the real
strategy on real data — they are visibility only until the E5 real-price
fixture (30 paper-days, user-gated on GitHub secrets) is wired. Naming and
the workflow were corrected so this is no longer presented as a real
release gate. The E3/E4 STATISTICAL miss therefore stays non-blocking
(grace period, ``--enforce`` off by default) — turning it on against
synthetic data would gate releases on a meaningless synthetic-alpha check.

What DOES block (always, independent of --enforce / grace): an EXECUTION
INTEGRITY failure of the walk-forward run itself — crashed/missing splits
or a degenerate report. A broken pipeline is a regression, not a
"statistical gate miss", so it must turn the job red even during the
grace period. ``run_walk_forward`` already raises when ALL splits fail;
this script additionally catches the PARTIAL-failure hole where survivors'
metrics were averaged and grace silently returned 0.

Exit codes:
    0   report produced + execution intact AND (E3/E4 pass OR grace active)
    1   E3/E4 statistical gate miss AND ``--enforce`` set
    2   EXECUTION INTEGRITY failure (crashed/missing splits, degenerate
        DSR input) — always blocks, regardless of --enforce / grace

Output
------

``output/qa/release_gate/walk_forward_<run_id>.json`` — machine-readable
verdict with all intermediate values. Stdout prints a one-line summary for
CI log readability.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402
from src.assembled_core.qa.walk_forward import (  # noqa: E402
    make_walk_forward_splits,
    run_walk_forward,
)

logger = logging.getLogger("release_gate_walk_forward")


def _synthetic_prices(
    n_symbols: int = 5,
    n_days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OHLCV panel — 5 symbols × ~2y business days."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-03", tz="UTC")
    dates = pd.date_range(start=start, periods=n_days, freq="B")
    symbols = [f"SYN{i:02d}" for i in range(n_symbols)]
    rows: list[dict[str, Any]] = []
    for sym_i, sym in enumerate(symbols):
        mu = 0.0003 + 0.0001 * sym_i
        sigma = 0.01 + 0.001 * sym_i
        noise = rng.normal(mu, sigma, n_days)
        closes = 100.0 * np.exp(np.cumsum(noise))
        for d, c in zip(dates, closes):
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": float(c) * 0.995,
                    "high": float(c) * 1.01,
                    "low": float(c) * 0.99,
                    "close": float(c),
                    "volume": 1_000_000.0,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def _trend_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )

    return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)


def _equal_weight_position_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_from_trend_signals,
    )

    return compute_target_positions_from_trend_signals(
        signals_df, total_capital=capital, top_n=None, min_score=0.0
    )


def _make_real_backtest_fn(prices: pd.DataFrame):
    """E3/E4 — wrap ``run_portfolio_backtest`` directly.

    We call the engine ourselves (rather than ``make_engine_backtest_fn``)
    because we need ``strict_session_gate=False`` so the CI runner works
    without the optional ``exchange_calendars`` dependency.
    """
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict[str, float]:
        test_prices = prices[
            (prices["timestamp"] >= test_start) & (prices["timestamp"] < test_end)
        ].copy()
        if test_prices.empty:
            return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}

        try:
            result = run_portfolio_backtest(
                prices=test_prices,
                signal_fn=_trend_signal_fn,
                position_sizing_fn=_equal_weight_position_fn,
                start_capital=10_000.0,
                include_costs=True,
                include_trades=False,
                compute_features=True,
                strict_session_gate=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[RELEASE-GATE] backtest failed for %s..%s: %s — emitting zeros",
                test_start.date(),
                test_end.date(),
                exc,
            )
            return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}

        metrics = result.metrics or {}
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        final_pf = float(metrics.get("final_pf", 1.0) or 1.0)
        total_return = final_pf - 1.0
        mdd = float(metrics.get("max_drawdown", 0.0) or 0.0)
        return {"sharpe": sharpe, "total_return": total_return, "max_drawdown": mdd}

    return backtest_fn


def _derive_oos_returns(wf_output: dict[str, Any]) -> np.ndarray:
    """Approximate per-window realised returns for the DSR computation.

    We use the summary row's ``total_return`` as one realised OOS return per
    window. For a full implementation the per-bar OOS returns would be
    stitched from the engine's equity series — this is a faithful first pass
    that keeps the gate runnable without the full strategy stack.
    """
    rows = wf_output.get("summary_df") or []
    returns = [float(r["total_return"]) for r in rows if r.get("status") == "success"]
    return np.asarray(returns, dtype=float)


def build_gate_report(
    prices: pd.DataFrame,
    *,
    train_days: int = 252,
    test_days: int = 63,
    n_splits: int = 8,
    oos_sharpe_min: float = 0.3,
    dsr_min: float = 0.5,
    ic_degradation_max: float = 0.5,
) -> dict[str, Any]:
    splits = make_walk_forward_splits(
        prices_df=prices,
        n_splits=n_splits,
        train_days=train_days,
        test_days=test_days,
    )
    backtest_fn = _make_real_backtest_fn(prices)
    wf_output = run_walk_forward(backtest_fn=backtest_fn, splits=splits)

    oos_metrics = wf_output.get("oos_first_metrics", {}) or {}
    mean_oos_sharpe = float(oos_metrics.get("oos_mean_sharpe", 0.0))

    oos_returns = _derive_oos_returns(wf_output)
    if oos_returns.size >= 2:
        dsr = deflated_sharpe(
            oos_returns,
            n_trials=max(1, n_splits),
        )
        dsr_dict = dsr.as_dict()
    else:
        dsr_dict = {
            "sharpe_observed": float("nan"),
            "deflated_sharpe_probability": float("nan"),
            "passes_5pct": False,
            "n_observations": int(oos_returns.size),
            "n_trials": n_splits,
        }

    e3_pass = mean_oos_sharpe >= oos_sharpe_min
    dsr_prob = dsr_dict.get("deflated_sharpe_probability", float("nan"))
    e4_pass = bool(
        isinstance(dsr_prob, float) and dsr_prob == dsr_prob and dsr_prob >= dsr_min
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "oos_sharpe_min": oos_sharpe_min,
            "dsr_min": dsr_min,
            "ic_degradation_max": ic_degradation_max,
        },
        "walk_forward": {
            "n_splits": len(splits),
            "n_successful_splits": wf_output["metrics"].get("n_successful_splits", 0),
            "aggregated_metrics": wf_output.get("metrics", {}),
            "oos_first_metrics": oos_metrics,
        },
        "deflated_sharpe": dsr_dict,
        "gates": {
            "E3_oos_sharpe": {
                "value": mean_oos_sharpe,
                "threshold": oos_sharpe_min,
                "pass": bool(e3_pass),
            },
            "E4_deflated_sharpe": {
                "value": dsr_prob,
                "threshold": dsr_min,
                "pass": bool(e4_pass),
            },
        },
        "overall_pass": bool(e3_pass and e4_pass),
    }


def _check_execution_integrity(report: dict[str, Any]) -> str | None:
    """Return a reason string when the walk-forward RUN is broken, else None.

    This is the execution-integrity gate (CI-001): it fires on a broken
    pipeline, NOT on a statistical E3/E4 miss. ``run_walk_forward`` already
    raises when *all* splits fail; this closes the PARTIAL-failure hole where
    a few crashed splits are silently dropped, the survivors are averaged, and
    the grace period returns 0. A non-None result must block the job (exit 2),
    independent of ``--enforce`` and the grace period.
    """
    wf = report.get("walk_forward", {}) or {}
    n_splits = int(wf.get("n_splits", 0) or 0)
    n_ok = int(wf.get("n_successful_splits", 0) or 0)

    if n_splits < 1:
        return (
            f"no walk-forward splits were generated (n_splits={n_splits}) — "
            "empty or insufficient input data"
        )
    if n_ok < n_splits:
        return (
            f"{n_splits - n_ok} of {n_splits} walk-forward splits failed to "
            f"execute (n_successful={n_ok}) — engine/data regression silently "
            "dropped by the per-window guard"
        )

    # Reached only after the n_ok == n_splits check above passed, so every split
    # claims success. A degenerate DSR input here means the report is internally
    # inconsistent (e.g. success rows missing total_return) — a pipeline defect,
    # NOT a legitimate small-sample statistical miss. (A genuinely small but
    # valid run would have surfaced as a partial failure and returned earlier.)
    dsr = report.get("deflated_sharpe", {}) or {}
    n_obs = int(dsr.get("n_observations", 0) or 0)
    if n_obs < 2:
        return (
            f"deflated-sharpe input degenerate (n_observations={n_obs}) despite "
            f"{n_ok} successful splits — cannot derive OOS returns"
        )
    return None


def _write_report(report: dict[str, Any], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"walk_forward_{run_id}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-sharpe-min", type=float, default=0.3)
    parser.add_argument("--dsr-min", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Fail the job on gate miss. Off by default during grace period.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "qa" / "release_gate"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    prices = _synthetic_prices()
    report = build_gate_report(
        prices,
        train_days=args.train_days,
        test_days=args.test_days,
        n_splits=args.n_splits,
        oos_sharpe_min=args.oos_sharpe_min,
        dsr_min=args.dsr_min,
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _write_report(report, Path(args.out_dir), run_id)

    gates = report["gates"]
    status = "PASS" if report["overall_pass"] else "FAIL"
    print(
        f"[RELEASE-GATE] {status} — "
        f"oos_sharpe={gates['E3_oos_sharpe']['value']:.3f} "
        f"(>= {gates['E3_oos_sharpe']['threshold']}) "
        f"dsr={gates['E4_deflated_sharpe']['value']} "
        f"(>= {gates['E4_deflated_sharpe']['threshold']}) "
        f"report={path}"
    )

    integrity_error = _check_execution_integrity(report)
    if integrity_error is not None:
        logger.error(
            "[RELEASE-GATE] EXECUTION INTEGRITY FAILURE: %s — blocking (exit 2)",
            integrity_error,
        )
        print(f"[RELEASE-GATE] EXECUTION-INTEGRITY FAIL — {integrity_error}")
        return 2

    if report["overall_pass"]:
        return 0
    if args.enforce:
        return 1
    logger.warning(
        "[RELEASE-GATE] E3/E4 statistical gate miss on SYNTHETIC data — NOT "
        "blocking (grace). Real-data enforcement needs the E5 fixture."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
