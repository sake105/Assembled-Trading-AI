"""Why did the system trade X? (Plan 11/10 §5.1.2)

Reads trade_journal/*.jsonl files and pretty-prints full reasoning for a trade.

Usage:
    python scripts/explain_trade.py --symbol AAPL --date 2026-05-03
    python scripts/explain_trade.py --order-id alpaca_xyz789
    python scripts/explain_trade.py --symbol AAPL  # last trade for symbol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_journal(journal_dir: Path) -> list[dict]:
    entries = []
    for f in sorted(journal_dir.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _match(
    entry: dict, symbol: str | None, date: str | None, order_id: str | None
) -> bool:
    if order_id and entry.get("order_id") != order_id:
        return False
    if symbol and entry.get("symbol", "").upper() != symbol.upper():
        return False
    if date and not str(entry.get("timestamp", "")).startswith(date):
        return False
    return True


def _render(entry: dict) -> str:
    lines = ["=== TRADE EXPLANATION ==="]
    ts = entry.get("timestamp", "unknown")
    sym = entry.get("symbol", "?")
    side = entry.get("side", "?").upper()
    qty = entry.get("qty", "?")
    price = entry.get("target_price", "?")
    lines.append(f"{ts}  {side} {qty} {sym} @ {price}")
    lines.append("")

    comps = entry.get("signal_components", {})
    if comps:
        lines.append("Signal components:")
        for name, val in comps.items():
            lines.append(
                f"  {name:<20} {val:+.4f}"
                if isinstance(val, float)
                else f"  {name:<20} {val}"
            )
        score = entry.get("signal_score")
        if score is not None:
            lines.append(f"  {'COMPOSITE':<20} {score:+.4f}")
        lines.append("")

    regime = entry.get("regime_detected")
    if regime:
        lines.append(f"Regime: {regime}")

    mults = entry.get("exposure_multipliers", {})
    if mults:
        lines.append("")
        lines.append("Exposure stack:")
        for k, v in mults.items():
            status = f"{v:.4f}" if v is not None else "DISABLED"
            lines.append(f"  {k:<30} {status}")

    gates = entry.get("risk_gates_passed", [])
    if gates:
        lines.append("")
        lines.append("Risk gates passed:")
        for g in gates:
            lines.append(f"  ✓ {g}")

    rejected = entry.get("rejected_alternatives", {})
    if rejected:
        lines.append("")
        lines.append("Rejected signals:")
        for sym2, reason in rejected.items():
            lines.append(f"  ✗ {sym2}: {reason}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain why the system made a trade")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--date", default=None, help="ISO date prefix, e.g. 2026-05-03")
    parser.add_argument("--order-id", default=None)
    parser.add_argument("--journal-dir", default="output/trade_journal")
    parser.add_argument(
        "--last", type=int, default=1, help="Show last N matching trades"
    )
    args = parser.parse_args()

    if not (args.symbol or args.order_id):
        parser.error("Provide --symbol and/or --order-id")

    journal_dir = Path(args.journal_dir)
    if not journal_dir.exists():
        print(f"Journal directory not found: {journal_dir}", file=sys.stderr)
        print(
            "Trade journal is written when the system runs in paper/live mode.",
            file=sys.stderr,
        )
        return 1

    entries = _load_journal(journal_dir)
    matches = [e for e in entries if _match(e, args.symbol, args.date, args.order_id)]

    if not matches:
        print(
            f"No trades found for symbol={args.symbol} date={args.date} order_id={args.order_id}"
        )
        return 1

    for entry in matches[-args.last :]:
        print(_render(entry))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
