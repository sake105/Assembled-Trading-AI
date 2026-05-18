"""Promotion-Gate-Check (audit C2-074).

Verifies an Account-R live-run against the Two-Account-Setup promotion
checklist (``docs/TWO_ACCOUNT_SETUP.md``) and emits a verdict on whether
the strategy is ready to promote to Account T.

This is a CHECK script — it does NOT execute the promotion. The operator
runs the check, reads the report, and decides explicitly.

Pflicht-Kriterien (all must pass for ``promotion_verdict = ready``):

1. ≥ 90 days of R-track-record on the equity curve
2. Average Sharpe over last 30/60/90 days all ≥ 1.0
3. Maximum drawdown < 20% over R-run
4. Hold-Out-Leakage verdict ≠ ``negative_sharpe`` and ≠ ``undefined``
5. Survivorship-Bias verdict ≤ ``medium``
6. Out-of-Regime verdict = ``robust`` OR Bear sample present
7. Fill-Modell verdict ≠ ``high``
8. Equity-Curve DSR > 1.0
9. (Manual) Operator confirms kill-switch + pre-trade gates active (CLI flag)

Usage::

    python scripts/ops/check_promotion_gate.py
    python scripts/ops/check_promotion_gate.py \\
        --equity-curve output/equity_curve_baseline.csv \\
        --kill-switch-confirmed --pre-trade-gates-confirmed

Output: JSON + Markdown under ``output/ops/promotion_gate.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-criterion checks
# ---------------------------------------------------------------------------


def check_track_record_length(equity: pd.Series, min_days: int = 90) -> dict[str, Any]:
    n_days = int(len(equity))
    return {
        "name": "track_record_length",
        "pass": bool(n_days >= min_days),
        "actual": n_days,
        "threshold": min_days,
        "message": f"{n_days} days observed, need ≥{min_days}",
    }


def check_rolling_sharpe(
    returns: np.ndarray, periods_per_year: int = 252
) -> dict[str, Any]:
    """Check that 30/60/90-day rolling-Sharpe-windows are all ≥ 1.0."""
    if len(returns) < 30:
        return {
            "name": "rolling_sharpe_consistency",
            "pass": False,
            "actual": None,
            "threshold": 1.0,
            "message": f"need ≥30 returns, got {len(returns)}",
        }
    results = {}
    for window in (30, 60, 90):
        if len(returns) < window:
            results[f"sharpe_{window}d"] = None
            continue
        recent = returns[-window:]
        mean = float(recent.mean())
        std = float(recent.std(ddof=1))
        s = mean / std * float(np.sqrt(periods_per_year)) if std > 0 else 0.0
        results[f"sharpe_{window}d"] = round(s, 4)
    valid = [v for v in results.values() if v is not None]
    pass_all = len(valid) > 0 and all(s >= 1.0 for s in valid)
    return {
        "name": "rolling_sharpe_consistency",
        "pass": pass_all,
        "actual": results,
        "threshold": 1.0,
        "message": f"30/60/90d Sharpes: {results}",
    }


def check_max_drawdown(
    equity: np.ndarray, max_dd_threshold: float = -0.20
) -> dict[str, Any]:
    if len(equity) < 2:
        return {
            "name": "max_drawdown",
            "pass": False,
            "actual": 0.0,
            "threshold": max_dd_threshold,
            "message": "no equity series",
        }
    rm = np.maximum.accumulate(equity)
    dd = float((equity / rm - 1.0).min())
    return {
        "name": "max_drawdown",
        "pass": bool(dd >= max_dd_threshold),
        "actual": dd,
        "threshold": max_dd_threshold,
        "message": f"observed MDD {dd:.2%}, threshold {max_dd_threshold:.2%}",
    }


def check_forensic_audit(
    script_name: str,
    expected_verdict: list[str],
    equity_path: Path,
) -> dict[str, Any]:
    """Run a forensic-audit script and check verdict against expected set.

    Returns pass=True if the script's emitted verdict is in expected_verdict.
    Pass=False if script fails or verdict not in set.
    """
    # F-senior-1: registry-style dispatch. Each script either takes the
    # default `--input <equity>` signature OR has no equity arg (works on
    # configs/static data). Add to NO_EQUITY_ARG when a new forensic script
    # follows that pattern.
    NO_EQUITY_ARG = {"survivorship_bias_check", "fill_model_audit"}
    script_path = f"scripts/forensic/{script_name}.py"
    if script_name in NO_EQUITY_ARG:
        cmd = [sys.executable, script_path]
    else:
        cmd = [sys.executable, script_path, "--input", str(equity_path)]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        out = proc.stderr or proc.stdout  # logging goes to stderr
        verdict = "unknown"
        for line in out.splitlines():
            if "verdict=" in line:
                # parse "verdict=<value>" (strip trailing args like flags=N)
                idx = line.find("verdict=")
                tail = line[idx + len("verdict=") :]
                verdict = tail.split()[0].strip()
                break
        return {
            "name": script_name,
            "pass": verdict in expected_verdict,
            "actual": verdict,
            "threshold": expected_verdict,
            "message": f"verdict={verdict}, expected one of {expected_verdict}",
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "name": script_name,
            "pass": False,
            "actual": "error",
            "threshold": expected_verdict,
            "message": f"forensic script failed: {exc}",
        }


def check_dsr_threshold(
    equity: np.ndarray, dsr_threshold: float = 1.0
) -> dict[str, Any]:
    """Compute simplified DSR-like signal: Sharpe / sqrt(1 + sharpe²/4) ≥ threshold.

    This is a fast proxy; full DSR via qa/metrics.deflated_sharpe_ratio_from_returns
    requires the equity_curve_audit.py output JSON. Here we use a lightweight
    Sharpe-confidence proxy.
    """
    if len(equity) < 30:
        return {
            "name": "dsr_proxy",
            "pass": False,
            "actual": None,
            "threshold": dsr_threshold,
            "message": "insufficient data",
        }
    returns = np.diff(equity) / equity[:-1]
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std <= 0:
        return {
            "name": "dsr_proxy",
            "pass": False,
            "actual": None,
            "threshold": dsr_threshold,
            "message": "zero variance",
        }
    sharpe = mean / std * float(np.sqrt(252))
    # Proxy: deflated by sqrt(1 + sharpe²/4) factor (rough Bonferroni-style)
    dsr_proxy = float(sharpe / float(np.sqrt(1.0 + sharpe**2 / 4.0)))
    return {
        "name": "dsr_proxy",
        "pass": bool(dsr_proxy >= dsr_threshold),
        "actual": round(dsr_proxy, 4),
        "threshold": dsr_threshold,
        "message": f"Sharpe {sharpe:.3f}, DSR-proxy {dsr_proxy:.3f}",
    }


def check_operator_flag(name: str, confirmed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(confirmed),
        "actual": confirmed,
        "threshold": True,
        "message": (
            f"operator-confirmed: {confirmed} (CLI flag --{name.replace('_', '-')})"
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_promotion_gate(
    equity_curve_path: Path,
    kill_switch_confirmed: bool = False,
    pre_trade_gates_confirmed: bool = False,
    run_forensic: bool = True,
) -> dict[str, Any]:
    if not equity_curve_path.exists():
        return {
            "error": f"equity curve not found: {equity_curve_path}",
            "promotion_verdict": "blocked",
        }
    df = pd.read_csv(equity_curve_path)
    if "equity" not in df.columns:
        return {
            "error": f"missing 'equity' column in {equity_curve_path}",
            "promotion_verdict": "blocked",
        }
    equity = df["equity"].to_numpy(dtype=float)
    if "daily_return" in df.columns:
        returns = df["daily_return"].dropna().to_numpy(dtype=float)
    else:
        returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)

    checks: list[dict[str, Any]] = []
    checks.append(check_track_record_length(df["equity"], min_days=90))
    checks.append(check_rolling_sharpe(returns))
    checks.append(check_max_drawdown(equity, max_dd_threshold=-0.20))
    checks.append(check_dsr_threshold(equity, dsr_threshold=1.0))
    checks.append(check_operator_flag("kill_switch_confirmed", kill_switch_confirmed))
    checks.append(
        check_operator_flag("pre_trade_gates_confirmed", pre_trade_gates_confirmed)
    )

    if run_forensic:
        # Hold-Out-Leakage: pass if not negative_sharpe or undefined
        checks.append(
            check_forensic_audit(
                "hold_out_leakage_test",
                expected_verdict=[
                    "hold_out_edge_significant",
                    "hold_out_edge_weak",
                    "hold_out_edge_indistinguishable_from_random",
                ],
                equity_path=equity_curve_path,
            )
        )
        # Survivorship: pass if not high
        checks.append(
            check_forensic_audit(
                "survivorship_bias_check",
                expected_verdict=["low", "medium"],
                equity_path=equity_curve_path,
            )
        )
        # Out-of-Regime: pass if robust
        checks.append(
            check_forensic_audit(
                "out_of_regime_test",
                expected_verdict=["robust"],
                equity_path=equity_curve_path,
            )
        )
        # Fill-Modell: pass if not high
        checks.append(
            check_forensic_audit(
                "fill_model_audit",
                expected_verdict=["low", "medium"],
                equity_path=equity_curve_path,
            )
        )

    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)
    if n_pass == n_total:
        verdict = "ready"
    elif n_pass >= n_total - 2:
        verdict = "blocked_minor"
    else:
        verdict = "blocked_major"

    failing = [c["name"] for c in checks if not c["pass"]]

    return {
        "input_equity_curve": str(equity_curve_path),
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "failing_checks": failing,
        "promotion_verdict": verdict,
        "limitations": (
            "Promotion-Gate is a CHECK, not an enforcer. The operator must "
            "still review the report and decide explicitly. Some criteria "
            "(operator-confirmation flags) cannot be verified by the script "
            "alone."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Promotion-Gate-Check (C2-074)",
        "",
        f"**Equity-Curve:** `{report.get('input_equity_curve', 'n/a')}`",
        "",
        f"## Verdict: `{report.get('promotion_verdict')}` "
        f"({report.get('n_pass', 0)}/{report.get('n_total', 0)} checks pass)",
        "",
    ]
    if report.get("failing_checks"):
        lines.append("### Failing Checks")
        for name in report["failing_checks"]:
            lines.append(f"- `{name}`")
        lines.append("")
    lines.append("## Per-Check Detail")
    lines.append("")
    lines.append("| Check | Pass | Actual | Threshold | Message |")
    lines.append("|------:|:----:|-------:|----------:|:--------|")
    for c in report.get("checks", []):
        actual = c["actual"]
        if isinstance(actual, dict):
            actual = str(actual)
        elif isinstance(actual, float):
            actual = f"{actual:.4f}"
        status = "✅" if c["pass"] else "❌"
        lines.append(
            f"| {c['name']} | {status} | {actual} | {c['threshold']} | {c['message']} |"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append(report.get("limitations", ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equity-curve",
        type=Path,
        default=Path("output/equity_curve_baseline.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("output/ops"))
    parser.add_argument(
        "--kill-switch-confirmed",
        action="store_true",
        help="Operator confirms kill-switch is wired and tested.",
    )
    parser.add_argument(
        "--pre-trade-gates-confirmed",
        action="store_true",
        help="Operator confirms pre-trade gates active.",
    )
    parser.add_argument(
        "--skip-forensic",
        action="store_true",
        help="Skip the forensic-audit script subprocess calls (faster smoke).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_promotion_gate(
        equity_curve_path=args.equity_curve,
        kill_switch_confirmed=args.kill_switch_confirmed,
        pre_trade_gates_confirmed=args.pre_trade_gates_confirmed,
        run_forensic=not args.skip_forensic,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "promotion_gate.json"
    md_path = args.out / "promotion_gate.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[promotion_gate] JSON: %s", json_path)
    logger.info("[promotion_gate] Markdown: %s", md_path)
    logger.info(
        "[promotion_gate] verdict=%s pass=%d/%d",
        report.get("promotion_verdict"),
        report.get("n_pass", 0),
        report.get("n_total", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
