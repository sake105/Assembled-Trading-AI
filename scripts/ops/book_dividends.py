"""Book broker dividend payouts into the paper ledger (W15, GESAMTBEWERTUNG).

Structural gap (accounting deep-audit 2026-07-21): the paper ledger NEVER
booked dividends while Alpaca credits real cash payouts (TLT distributes
monthly) — a slow, silent ledger<broker cash drift absorbed by the $100
reconcile threshold until it surfaces as an "unexplained" halt contribution.

Design: source of truth is the BROKER (Alpaca account activities, type DIV)
— booking exactly what the broker credited makes ledger and broker converge
to the cent, unlike any market-data-derived dividend estimate. alpaca-py's
TradingClient exposes no activities endpoint (that lives in BrokerClient),
so this script calls the documented REST endpoint directly with the same
paper credentials.

Idempotency: every booked activity id is appended to
``output/ops/dividends_booked.jsonl``; already-booked ids are skipped, so
the script can run every cycle (wired best-effort into run_live_paper) or
manually, any number of times.

Failure policy: best-effort. An API failure logs WARNING and exits 0 — the
reconcile halt remains the backstop; this script only removes the known
drift source, it is not itself a safety control.

Known crash window (Stage-2 F-senior-5, accepted trade-off): the ledger
save happens BEFORE the booked-log append. A process kill between the two
double-credits on the next run (single TLT payout ~25 USD, below the $100
reconcile gate). The inverse ordering would instead degrade to
never-booked on a crash — permanently growing drift. Chosen direction:
rare double-credit (bounded, reconcile-visible on accumulation) over
silent permanent drift.

Usage:
    python scripts/ops/book_dividends.py            # book new DIV activities
    python scripts/ops/book_dividends.py --dry-run  # show without writing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

LEDGER_PATH = ROOT / "output" / "runs" / "_paper_ledger" / "ledger_state.json"
BOOKED_LOG = ROOT / "output" / "ops" / "dividends_booked.jsonl"
PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def fetch_dividend_activities(after_iso: str | None = None) -> list[dict]:
    """Fetch DIV activities from the Alpaca account-activities REST endpoint.

    Returns a list of activity dicts (id, symbol, net_amount, date, ...).
    Raises on HTTP/credential errors — callers decide the failure policy.
    """
    import requests

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_API_SECRET not set")
    base = os.environ.get("ALPACA_BASE_URL", PAPER_BASE_URL).rstrip("/")
    params: dict[str, str] = {"page_size": "100"}
    if after_iso:
        params["after"] = after_iso
    resp = requests.get(
        f"{base}/v2/account/activities/DIV",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _load_booked_ids(booked_log: Path) -> set[str]:
    if not booked_log.exists():
        return set()
    ids: set[str] = set()
    for line in booked_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(str(json.loads(line).get("activity_id")))
        except json.JSONDecodeError:
            continue
    return ids


def book_pending_dividends(
    *,
    ledger_path: Path = LEDGER_PATH,
    booked_log: Path = BOOKED_LOG,
    fetch=fetch_dividend_activities,
    dry_run: bool = False,
) -> int:
    """Book unbooked broker DIV activities into the ledger cash.

    Returns the number of newly booked activities. Never raises on API
    failure (logs WARNING, returns 0); ledger write errors DO raise.
    """
    from src.assembled_core.ops.paper_ledger import (
        load_ledger_state,
        save_ledger_state,
    )

    try:
        activities = fetch()
    except Exception as exc:  # best-effort by policy (see module docstring)
        logger.warning("[dividends] activity fetch failed (%s) — skipping", exc)
        return 0

    booked_ids = _load_booked_ids(booked_log)
    new = [
        a
        for a in activities
        if str(a.get("id")) not in booked_ids and a.get("net_amount") not in (None, "")
    ]
    if not new:
        logger.info("[dividends] no unbooked DIV activities")
        return 0

    if not ledger_path.exists():
        logger.warning("[dividends] ledger not found at %s — skipping", ledger_path)
        return 0

    state = load_ledger_state(ledger_path)
    total = 0.0
    entries = []
    for a in new:
        # Stage-2 F-senior-3: guard the parse — one malformed net_amount
        # (e.g. "N/A") must skip that activity, not crash the whole booking.
        try:
            amount = float(a.get("net_amount", 0.0))
        except (TypeError, ValueError):
            logger.warning(
                "[dividends] unparseable net_amount %r for activity %s (%s) — "
                "skipped, NOT marked booked (retried next run)",
                a.get("net_amount"),
                a.get("id"),
                a.get("symbol"),
            )
            continue
        total += amount
        entries.append(
            {
                "activity_id": str(a.get("id")),
                "symbol": a.get("symbol"),
                "net_amount": amount,
                "activity_date": a.get("date"),
                "booked_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            "[dividends] %s %s: %+.2f USD (%s)",
            "DRY-RUN would book" if dry_run else "booking",
            a.get("symbol"),
            amount,
            a.get("date"),
        )

    if dry_run:
        logger.info(
            "[dividends] DRY-RUN: %d activities, %+.2f USD total", len(new), total
        )
        return len(new)

    # Booked-log parent FIRST (Stage-1 fix: was the module constant, not the
    # parameter — a custom booked_log with missing parent would fail AFTER
    # the ledger write, leaving the activity unlogged -> double-booked on
    # retry). Creating the dir before the ledger save keeps the failure
    # window to the append itself.
    booked_log.parent.mkdir(parents=True, exist_ok=True)
    state["cash"] = float(state.get("cash", 0.0)) + total
    save_ledger_state(state, ledger_path)
    with booked_log.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    logger.info(
        "[dividends] booked %d DIV activities, %+.2f USD -> ledger cash %.2f",
        len(new),
        total,
        state["cash"],
    )
    return len(new)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    book_pending_dividends(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
