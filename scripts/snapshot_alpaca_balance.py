"""A4 — Alpaca EOD balance snapshot.

Captures Alpaca paper-account state after each paper-trading cycle and
writes it to ``output/ops/alpaca_eod_<date>.json``. Serves as the second
independent stall-detector (the first being the local heartbeat monitor).

If a local state file exists, a cash-delta vs. Alpaca is computed and a
warning is emitted when the delta exceeds $50 — this is a cheap early
signal for reconciliation drift. It does NOT raise / gate; the
reconcile-halt policy (E0.4) is the authoritative halting mechanism.

Usage:
    python scripts/snapshot_alpaca_balance.py

Exits 0 on success or non-fatal snapshot failure. Exits non-zero only if
the caller requires strict mode (not enabled by default — snapshot is
always best-effort so it never blocks the paper cycle).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("snapshot_alpaca_balance")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

OUTPUT_DIR = Path("output/ops")
LOCAL_STATE_PATH = Path("output/paper_state/paper_state.json")
CASH_DELTA_WARN_USD = 50.0


def _load_local_cash() -> float | None:
    if not LOCAL_STATE_PATH.exists():
        return None
    try:
        data = json.loads(LOCAL_STATE_PATH.read_text(encoding="utf-8"))
        return float(data.get("cash", 0.0))
    except Exception as exc:
        logger.warning("[SNAPSHOT] could not read local state: %s", exc)
        return None


def _fetch_alpaca_account() -> dict | None:
    try:
        from src.assembled_core.execution.broker_adapter import AlpacaAdapter
    except Exception as exc:
        logger.error("[SNAPSHOT] AlpacaAdapter import failed: %s", exc)
        return None
    try:
        adapter = AlpacaAdapter()
        account = adapter.get_account()
        positions = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in adapter.get_positions()
        ]
        return {"account": account, "positions": positions}
    except Exception as exc:
        logger.error("[SNAPSHOT] Alpaca fetch failed: %s", exc)
        return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = OUTPUT_DIR / f"alpaca_eod_{today}.json"

    alpaca = _fetch_alpaca_account()
    if alpaca is None:
        logger.warning("[SNAPSHOT] skipping — Alpaca unreachable")
        return 0

    local_cash = _load_local_cash()
    alpaca_cash = float(alpaca["account"].get("cash", 0.0))
    cash_delta = None
    delta_alert = False
    if local_cash is not None:
        cash_delta = alpaca_cash - local_cash
        if abs(cash_delta) > CASH_DELTA_WARN_USD:
            delta_alert = True
            logger.warning(
                "[SNAPSHOT] cash delta %.2f USD exceeds threshold %.2f — review recommended",
                cash_delta,
                CASH_DELTA_WARN_USD,
            )

    payload = {
        "snapshot_date": today,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpaca": alpaca,
        "local_cash": local_cash,
        "cash_delta_usd": cash_delta,
        "cash_delta_alert": delta_alert,
    }

    try:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(target)
        logger.info("[SNAPSHOT] wrote %s", target)
    except Exception as exc:
        logger.error("[SNAPSHOT] write failed: %s", exc)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
