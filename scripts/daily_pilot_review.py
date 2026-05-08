"""Daily Pilot Review - generates a structured markdown summary for paper pilot monitoring.

Run after US market close (22:30 CET / 16:30 ET):
    python scripts/daily_pilot_review.py

Output: prints to stdout + saves to output/pilot/daily_review_YYYY-MM-DD.md
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_experience_log() -> list[dict]:
    path = ROOT / "output" / "experience" / "experience_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_pilot_manifest() -> dict:
    path = ROOT / "output" / "pilot" / "pilot_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_trades(n_days: int = 7) -> list[dict]:
    path = ROOT / "output" / "reports" / "trades_1d.csv"
    if not path.exists():
        return []
    import csv

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    cutoff = datetime.now(timezone.utc).date().toordinal() - n_days
    recent = []
    for r in rows:
        try:
            d_str = r.get("date") or r.get("timestamp", "")[:10]
            d_ord = date.fromisoformat(d_str).toordinal()
            if d_ord >= cutoff:
                recent.append(r)
        except (ValueError, TypeError):
            pass
    return recent


def _sharpe_rolling(equity_series: list[float], window: int = 7) -> float | None:
    if len(equity_series) < window + 1:
        return None
    rets = [
        (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
        for i in range(len(equity_series) - window, len(equity_series))
    ]
    if not rets:
        return None
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return (mean_r / std * math.sqrt(252)) if std > 0 else None


def main() -> None:
    today = datetime.now(timezone.utc).date()
    rows = _load_experience_log()
    manifest = _load_pilot_manifest()
    hard_stop = manifest.get("hard_stop_criteria", {})
    success = manifest.get("success_criteria", {}).get("success", {})

    # Filter pilot rows (broker mode, after pilot start)
    pilot_start_str = manifest.get("started_at", "")[:10]
    pilot_rows = [
        r
        for r in rows
        if r.get("execution_mode") == "broker"
        and r.get("cycle_date", "") >= pilot_start_str
    ]

    if not pilot_rows:
        print("No pilot data found.")
        return

    # Sort by timestamp
    pilot_rows.sort(key=lambda r: r.get("timestamp_utc", ""))

    # Equity series
    equities = [r.get("broker_equity", 0.0) for r in pilot_rows]
    start_equity = equities[0] if equities else 100_000.0
    current_equity = equities[-1] if equities else start_equity
    peak_equity = max(equities) if equities else start_equity

    total_return_pct = (current_equity - start_equity) / start_equity * 100
    current_dd_pct = (
        (current_equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    )

    days_elapsed = len(set(r.get("cycle_date", "") for r in pilot_rows))
    sharpe_7d = _sharpe_rolling(equities, window=min(7, len(equities) - 1))

    # Reconcile status counts
    ok_count = sum(1 for r in pilot_rows if r.get("reconcile_status") == "OK")
    fail_count = sum(1 for r in pilot_rows if r.get("reconcile_status") != "OK")

    # Crash days
    crashed = [r for r in pilot_rows if r.get("exit_code", 0) != 0]

    # Recent trades
    recent_trades = _load_trades(n_days=7)
    n_trades_7d = len(recent_trades)

    # Hard-stop check
    max_dd_limit = hard_stop.get("max_drawdown_pct", -8.0)
    min_sharpe_14d = hard_stop.get("min_sharpe_after_14d", 0.5)
    hard_stop_triggered = current_dd_pct <= max_dd_limit
    sharpe_warn = (
        days_elapsed >= 14 and sharpe_7d is not None and sharpe_7d < min_sharpe_14d
    )

    lines: list[str] = [
        f"# Pilot Daily Review - {today}",
        "",
        f"**Pilot day:** {days_elapsed}  |  **Start equity:** ${start_equity:,.2f}",
        "",
        "## Equity Summary",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Current equity | **${current_equity:,.2f}** |",
        f"| Total return | **{total_return_pct:+.2f}%** |",
        f"| Peak equity | ${peak_equity:,.2f} |",
        f"| Current drawdown | {current_dd_pct:.2f}% |",
        f"| Rolling 7d Sharpe | {f'{sharpe_7d:.2f}' if sharpe_7d is not None else 'N/A (insufficient data)'} |",
        "",
        "## Operations",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Reconcile OK | {ok_count} |",
        f"| Reconcile FAIL | {fail_count} |",
        f"| Crash days | {len(crashed)} |",
        f"| Trades (last 7d) | {n_trades_7d} |",
        "",
        "## Hard-Stop Status",
    ]

    if hard_stop_triggered:
        lines += [
            f"[!!! HARD STOP TRIGGERED] drawdown {current_dd_pct:.2f}% <= limit {max_dd_limit:.1f}%",
            f"Action required: {hard_stop.get('kill_switch_action', 'halt_trading')}",
        ]
    elif sharpe_warn:
        lines += [
            f"[WARN] Rolling 7d Sharpe {sharpe_7d:.2f} < {min_sharpe_14d} after day {days_elapsed}",
        ]
    else:
        dd_room = current_dd_pct - max_dd_limit
        lines += [
            f"[OK] Drawdown {current_dd_pct:.2f}% (room: {dd_room:.1f}pp to hard stop)",
        ]

    lines += [
        "",
        "## Success Criteria Progress",
        "| Criterion | Target | Current | Status |",
        "|-----------|--------|---------|--------|",
        f"| CAGR | >={success.get('min_cagr_pct', 20)}% | (needs full period) | - |",
        f"| Sharpe | >={success.get('min_sharpe', 1.5)} | {f'{sharpe_7d:.2f}' if sharpe_7d else '--'} | {'PASS' if sharpe_7d and sharpe_7d >= success.get('min_sharpe', 1.5) else '--'} |",
        f"| MDD | >={success.get('max_mdd_pct', -10)}% | {current_dd_pct:.2f}% | {'PASS' if current_dd_pct > success.get('max_mdd_pct', -10) else 'FAIL'} |",
        "",
        "---",
        f"*Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*",
    ]

    report = "\n".join(lines)
    print(report)

    out_dir = ROOT / "output" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_review_{today}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
