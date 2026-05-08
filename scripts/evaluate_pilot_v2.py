"""scripts/evaluate_pilot_v2.py — Backlog Item 129/148: Pilot GO/NO-GO evaluation.

Reads pilot_v2_manifest.json hard-stop criteria, loads the equity curve, and
produces a verdict at the current milestone (day-7, day-14, day-30, or manual).

Usage:
    python scripts/evaluate_pilot_v2.py
    python scripts/evaluate_pilot_v2.py --milestone day-14
    python scripts/evaluate_pilot_v2.py --equity-file output/pilot/equity_curve.csv

Output: JSON + human-readable verdict printed to stdout.
         Exit code 0 = CONTINUE / GO, 1 = HALT / NO-GO, 2 = data-insufficient.
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

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = ROOT / "output" / "pilot" / "pilot_v2_manifest.json"
DEFAULT_EQUITY_PATHS = [
    ROOT / "output" / "pilot" / "equity_curve.csv",
    ROOT / "output" / "pilot" / "equity_curve.parquet",
    ROOT / "output" / "reports" / "equity_curve.csv",
]
TRADING_DAYS_PER_YEAR = 252


# ─── Equity curve loading ─────────────────────────────────────────────────────


def _load_equity(path: Path) -> list[dict]:
    """Load equity curve as list of {date, equity} dicts sorted ascending."""
    if not path.exists():
        return []
    if path.suffix == ".parquet":
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            if "equity" not in df.columns and "portfolio_value" in df.columns:
                df = df.rename(columns={"portfolio_value": "equity"})
            if "date" not in df.columns and df.index.name in ("date", "timestamp"):
                df = df.reset_index().rename(columns={df.index.name: "date"})
            return df[["date", "equity"]].to_dict("records")
        except Exception as exc:
            log.warning("parquet load failed: %s", exc)
            return []
    else:
        import csv

        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    eq_val = float(row.get("equity") or row.get("portfolio_value") or 0)
                    rows.append(
                        {
                            "date": row.get("date") or row.get("timestamp"),
                            "equity": eq_val,
                        }
                    )
                except (ValueError, KeyError):
                    pass
        return rows


# ─── Metrics computation ──────────────────────────────────────────────────────


def _compute_metrics(equity_rows: list[dict], seed_capital: float) -> dict:
    if len(equity_rows) < 2:
        return {"n_days": len(equity_rows), "insufficient_data": True}

    equities = [r["equity"] for r in equity_rows]
    n = len(equities)
    final = equities[-1]
    initial = seed_capital or equities[0]

    # CAGR
    years = n / TRADING_DAYS_PER_YEAR
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 and initial > 0 else 0.0

    # Daily returns
    rets = [
        (equities[i] - equities[i - 1]) / max(equities[i - 1], 1e-10)
        for i in range(1, n)
    ]
    if not rets:
        return {"n_days": n, "insufficient_data": True}

    mean_ret = sum(rets) / len(rets)
    variance = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
    std_ret = math.sqrt(variance) if variance > 0 else 1e-10

    sharpe = (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Max Drawdown
    peak = equities[0]
    mdd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)

    # Win rate (positive return days)
    wins = sum(1 for r in rets if r > 0)
    win_rate_pct = 100.0 * wins / len(rets)

    # Consecutive loss days (current streak)
    consec_losses = 0
    max_consec_losses = 0
    for r in rets:
        if r < 0:
            consec_losses += 1
            max_consec_losses = max(max_consec_losses, consec_losses)
        else:
            consec_losses = 0

    # Daily PnL swing
    pnl_swings_pct = [abs(r) * 100 for r in rets]
    max_daily_swing_pct = max(pnl_swings_pct) if pnl_swings_pct else 0.0

    return {
        "n_days": n,
        "final_equity": final,
        "seed_capital": seed_capital,
        "total_return_pct": 100.0 * (final - initial) / initial if initial > 0 else 0.0,
        "cagr_pct": 100.0 * cagr,
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(100.0 * mdd, 3),
        "win_rate_pct": round(win_rate_pct, 1),
        "max_consec_losses": max_consec_losses,
        "max_daily_swing_pct": round(max_daily_swing_pct, 3),
        "insufficient_data": False,
    }


# ─── Verdict logic ────────────────────────────────────────────────────────────


def _evaluate(metrics: dict, manifest: dict, milestone: str) -> dict:
    hs = manifest.get("hard_stop_criteria", {})
    gn = manifest.get("go_nogo_criteria", {})
    pc = manifest.get("pause_criteria", {})

    verdict = "CONTINUE"
    flags: list[str] = []
    hard_stops_triggered: list[str] = []
    pause_triggered = False

    if metrics.get("insufficient_data"):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "flags": ["Not enough trading days to evaluate"],
            "metrics": metrics,
        }

    n = metrics.get("n_days", 0)
    mdd = metrics.get("max_drawdown_pct", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    consec_losses = metrics.get("max_consec_losses", 0)
    max_swing = metrics.get("max_daily_swing_pct", 0.0)
    win_rate = metrics.get("win_rate_pct", 0.0)

    # Hard stops (trigger immediate halt)
    if mdd >= hs.get("max_drawdown_pct", 8.0):
        hard_stops_triggered.append(
            f"MDD {mdd:.1f}% >= hard-stop {hs['max_drawdown_pct']}%"
        )
    if consec_losses >= hs.get("max_consecutive_loss_days", 7):
        hard_stops_triggered.append(
            f"Consecutive losses {consec_losses} >= {hs['max_consecutive_loss_days']}"
        )
    if max_swing >= hs.get("max_daily_pnl_swing_pct", 5.0):
        hard_stops_triggered.append(
            f"Daily PnL swing {max_swing:.1f}% >= {hs['max_daily_pnl_swing_pct']}%"
        )
    if n >= 14 and sharpe < hs.get("min_sharpe_after_14d", 0.5):
        hard_stops_triggered.append(
            f"Day-14 Sharpe {sharpe:.3f} < {hs['min_sharpe_after_14d']}"
        )

    if hard_stops_triggered:
        return {
            "verdict": "HALT",
            "action": hs.get("kill_switch_action", "halt_trading_send_alert"),
            "hard_stops_triggered": hard_stops_triggered,
            "flags": hard_stops_triggered,
            "metrics": metrics,
        }

    # Soft pause check
    if mdd >= pc.get("drawdown_soft_pct", 5.0):
        pause_triggered = True
        exposure_factor = pc.get("exposure_reduction_factor", 0.5)
        flags.append(
            f"Soft pause: MDD {mdd:.1f}% >= {pc['drawdown_soft_pct']}% — "
            f"reduce exposure to {100 * exposure_factor:.0f}%"
        )

    # Milestone-specific evaluation
    if milestone == "day-30" or n >= 30:
        if sharpe < gn.get("min_sharpe_30d", 0.8):
            flags.append(f"30d Sharpe {sharpe:.3f} < target {gn['min_sharpe_30d']}")
            verdict = "NO_GO_RECONFIGURE"
        if mdd > gn.get("max_mdd_30d_pct", 8.0):
            flags.append(f"30d MDD {mdd:.1f}% > {gn['max_mdd_30d_pct']}%")
            verdict = "NO_GO_RECONFIGURE"
        if win_rate < gn.get("min_win_rate_pct", 48.0):
            flags.append(f"Win rate {win_rate:.1f}% < {gn['min_win_rate_pct']}%")

        if not flags:
            verdict = "GO_LIVE_SMALL"
        elif verdict == "NO_GO_RECONFIGURE":
            verdict = "NO_GO_RECONFIGURE"

    elif milestone == "day-14" or (10 <= n < 30):
        if sharpe < 0.5:
            flags.append(f"Day-14 Sharpe {sharpe:.3f} < 0.5 — approaching hard-stop")
        verdict = "CONTINUE" if not flags else "EXTEND_OR_PAUSE"

    else:
        verdict = "CONTINUE"

    if pause_triggered and verdict == "CONTINUE":
        verdict = "PAUSE_REDUCE_EXPOSURE"

    return {
        "verdict": verdict,
        "flags": flags,
        "milestone": milestone,
        "metrics": metrics,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Pilot v2 GO/NO-GO status")
    ap.add_argument(
        "--milestone",
        choices=["day-7", "day-14", "day-30", "manual"],
        default="manual",
        help="Evaluation milestone (default: manual)",
    )
    ap.add_argument(
        "--equity-file",
        type=Path,
        default=None,
        help="Path to equity curve CSV or parquet",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to pilot_v2_manifest.json",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save evaluation result to JSON file",
    )
    args = ap.parse_args()

    # Load manifest
    manifest_path = args.manifest
    if not manifest_path.exists():
        log.error("[evaluate] manifest not found: %s", manifest_path)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_capital = float(manifest.get("seed_capital", 100_000))

    # Load equity
    equity_path = args.equity_file
    if equity_path is None:
        for p in DEFAULT_EQUITY_PATHS:
            if p.exists():
                equity_path = p
                break
    if equity_path is None or not equity_path.exists():
        log.warning("[evaluate] No equity curve found — using seed capital as baseline")
        equity_rows = []
    else:
        equity_rows = _load_equity(equity_path)
        log.info(
            "[evaluate] Loaded %d equity rows from %s", len(equity_rows), equity_path
        )

    # Compute metrics
    metrics = _compute_metrics(equity_rows, seed_capital)

    # Determine milestone
    milestone = args.milestone
    if milestone == "manual":
        n = metrics.get("n_days", 0)
        if n >= 30:
            milestone = "day-30"
        elif n >= 14:
            milestone = "day-14"
        elif n >= 7:
            milestone = "day-7"
        else:
            milestone = "early"

    # Evaluate
    result = _evaluate(metrics, manifest, milestone)
    result["evaluated_at"] = datetime.now(tz=timezone.utc).isoformat()
    result["manifest_version"] = manifest.get("pilot_version", "v2")

    # Print
    print("\n" + "=" * 60)
    print(f"  PILOT v2 EVALUATION — {milestone.upper()}")
    print("=" * 60)
    verdict = result["verdict"]
    if verdict in ("HALT", "NO_GO_RECONFIGURE"):
        print(f"  VERDICT: *** {verdict} ***")
    else:
        print(f"  VERDICT: {verdict}")
    print()
    m = result.get("metrics", {})
    if not m.get("insufficient_data"):
        print(f"  Trading days : {m.get('n_days', '?')}")
        print(f"  Sharpe       : {m.get('sharpe', '?'):.3f}")
        print(f"  CAGR         : {m.get('cagr_pct', 0):.1f}%")
        print(f"  Max DD       : {m.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Win rate     : {m.get('win_rate_pct', 0):.1f}%")
        print(f"  Max consec L : {m.get('max_consec_losses', 0)}")
        print(f"  Max daily sw : {m.get('max_daily_swing_pct', 0):.2f}%")
    flags = result.get("flags", [])
    if flags:
        print("\n  FLAGS:")
        for f in flags:
            print(f"    • {f}")
    hs_t = result.get("hard_stops_triggered", [])
    if hs_t:
        print("\n  HARD STOPS TRIGGERED:")
        for h in hs_t:
            print(f"    *** {h}")
    print("=" * 60 + "\n")

    # Save JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        log.info("[evaluate] Saved result to %s", args.output_json)

    # Exit codes
    if verdict in ("HALT",):
        return 1
    if verdict in ("INSUFFICIENT_DATA",):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
