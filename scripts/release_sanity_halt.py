"""Release a sanity-halted order (Plan 11/10 §5.2.3).

A sanity halt removes an order from submission when SanityChecker flags it
as high-severity. This script records a manual override in the trade journal
so the halt decision is auditable.

Usage:
    python scripts/release_sanity_halt.py \\
        --symbol AAPL --date 2026-05-04 \\
        --reason "Legitimate scaled entry, confirmed manually"

    python scripts/release_sanity_halt.py \\
        --journal output/runs/trade_journal.jsonl \\
        --symbol AAPL --date 2026-05-04 \\
        --reason "Confirmed: position in-universe, size correct"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _find_recent_halts(journal_path: Path, symbol: str, date: str) -> list[dict]:
    if not journal_path.exists():
        return []
    halts = []
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("event") == "sanity_halt"
                and entry.get("symbol", "").upper() == symbol.upper()
                and (not date or entry.get("date", "") == date)
            ):
                halts.append(entry)
    return halts


def _write_override(
    journal_path: Path, symbol: str, date: str, reason: str, actor: str
) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "sanity_halt_override",
        "symbol": symbol.upper(),
        "date": date,
        "reason": reason,
        "actor": actor,
        "override_ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("[release_sanity_halt] override written: %s %s — %s", symbol, date, reason)


def main() -> int:
    parser = argparse.ArgumentParser(description="Release a sanity-halted order")
    parser.add_argument("--symbol", required=True, help="Ticker symbol of halted order")
    parser.add_argument(
        "--date", default="", help="YYYY-MM-DD of halt (default: today)"
    )
    parser.add_argument(
        "--reason", required=True, help="Human-readable override reason"
    )
    parser.add_argument("--actor", default="manual", help="Who is releasing the halt")
    parser.add_argument(
        "--journal",
        default="output/runs/trade_journal.jsonl",
        help="Path to trade journal file",
    )
    args = parser.parse_args()

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal_path = Path(args.journal)

    # Try to find the halted entry for confirmation
    halts = _find_recent_halts(journal_path, args.symbol, date)
    if halts:
        log.info(
            "[release_sanity_halt] found %d halt(s) for %s on %s",
            len(halts),
            args.symbol,
            date,
        )
        for h in halts:
            flags = h.get("flags", [])
            log.info("  halt flags: %s", [f.get("rule") for f in flags])
    else:
        log.warning(
            "[release_sanity_halt] no halt record found for %s on %s in %s — writing override anyway",
            args.symbol,
            date,
            journal_path,
        )

    # Write override record
    _write_override(journal_path, args.symbol, date, args.reason, args.actor)
    log.info(
        "[release_sanity_halt] DONE — override recorded. Order can be re-submitted manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
