"""Position Sync — Reconcile ledger state against broker positions.

Provides:
- sync_positions_from_broker: Compare ledger vs Alpaca positions
- rebuild_ledger_from_broker: Emergency rebuild from broker state
- get_broker_equity: Fetch real equity for pre-trade checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.assembled_core.execution.broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a position sync operation."""

    ok: bool = True
    ledger_cash: float = 0.0
    broker_cash: float = 0.0
    cash_diff: float = 0.0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    missing_in_ledger: list[str] = field(default_factory=list)
    missing_in_broker: list[str] = field(default_factory=list)
    broker_equity: float = 0.0
    message: str = ""


def sync_positions_from_broker(
    adapter: BrokerAdapter,
    ledger_state: dict[str, Any],
) -> SyncResult:
    """Compare ledger positions against broker positions.

    Args:
        adapter: Broker adapter instance.
        ledger_state: Current ledger state dict (cash, positions).

    Returns:
        SyncResult with comparison details.
    """
    from src.assembled_core.accounting.reconciliation import (
        reconcile_ledger_vs_broker,
    )

    result = SyncResult()

    # Fetch broker state
    try:
        broker_positions = adapter.get_positions()
        broker_account = adapter.get_account()
    except Exception as exc:
        logger.error("[position_sync] failed to fetch broker state: %s", exc)
        result.ok = False
        result.message = f"Broker fetch failed: {exc}"
        return result

    # Convert broker positions to DataFrame
    broker_rows = [
        {"symbol": p.symbol, "qty": p.qty}
        for p in broker_positions
    ]
    broker_df = (
        pd.DataFrame(broker_rows, columns=["symbol", "qty"])
        if broker_rows
        else pd.DataFrame(columns=["symbol", "qty"])
    )

    broker_cash = float(broker_account.get("cash", 0))
    result.broker_cash = broker_cash
    result.broker_equity = float(broker_account.get("equity", 0))

    # Convert ledger positions to DataFrame
    positions = ledger_state.get("positions") or {}
    ledger_rows = [
        {"symbol": sym, "qty": float(p.get("qty", 0))}
        for sym, p in positions.items()
        if float(p.get("qty", 0)) != 0
    ]
    ledger_df = (
        pd.DataFrame(ledger_rows, columns=["symbol", "qty"])
        if ledger_rows
        else pd.DataFrame(columns=["symbol", "qty"])
    )
    ledger_cash = float(ledger_state.get("cash", 0))
    result.ledger_cash = ledger_cash

    # Run reconciliation
    recon = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger_df,
        ledger_cash=ledger_cash,
        broker_positions_df=broker_df,
        broker_cash=broker_cash,
        cash_tol=0.01,  # 1 cent tolerance for real broker
        qty_tol=0.001,  # fractional share tolerance
    )

    result.ok = recon.get("ok", False)
    result.cash_diff = recon.get("cash_diff", 0.0)
    result.missing_in_ledger = recon.get("missing_in_ledger", [])
    result.missing_in_broker = recon.get("missing_in_broker", [])
    result.message = recon.get("message", "")

    # Extract position mismatches
    diffs_df = recon.get("position_diffs_df")
    if diffs_df is not None and not diffs_df.empty:
        result.mismatches = diffs_df.to_dict("records")

    if result.ok:
        logger.info("[position_sync] reconciliation OK — ledger matches broker")
    else:
        logger.warning(
            "[position_sync] MISMATCH — cash_diff=%.2f, %d position diffs, "
            "%d missing_in_ledger, %d missing_in_broker",
            result.cash_diff,
            len(result.mismatches),
            len(result.missing_in_ledger),
            len(result.missing_in_broker),
        )

    return result


def rebuild_ledger_from_broker(
    adapter: BrokerAdapter,
    start_capital: float = 10000.0,
) -> dict[str, Any]:
    """EMERGENCY: Rebuild ledger state entirely from broker positions.

    WARNING: This loses all historical equity curve data.
    Only use when ledger is corrupted and no backup is recoverable.

    Args:
        adapter: Broker adapter instance.
        start_capital: Fallback capital if broker account fetch fails.

    Returns:
        Fresh ledger state dict built from broker data.
    """
    from src.assembled_core.ops.paper_ledger import SCHEMA_VERSION

    logger.critical(
        "[position_sync] EMERGENCY REBUILD — rebuilding ledger from broker positions"
    )

    try:
        broker_positions = adapter.get_positions()
        broker_account = adapter.get_account()
    except Exception as exc:
        logger.error("[position_sync] rebuild failed — cannot fetch broker: %s", exc)
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_utc": None,
            "cash": start_capital,
            "positions": {},
            "equity_curve": [],
        }

    positions: dict[str, dict[str, float]] = {}
    for p in broker_positions:
        if p.qty != 0:
            positions[p.symbol] = {
                "qty": p.qty,
                "avg_price": p.avg_entry_price,
            }

    cash = float(broker_account.get("cash", start_capital))

    logger.info(
        "[position_sync] rebuilt ledger: cash=%.2f, %d positions",
        cash,
        len(positions),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": None,
        "cash": cash,
        "positions": positions,
        "equity_curve": [],  # Historical data is lost
    }


def get_broker_equity(adapter: BrokerAdapter) -> float | None:
    """Fetch real equity from broker for pre-trade checks.

    Args:
        adapter: Broker adapter instance.

    Returns:
        Current account equity as float, or None on error.
        Callers MUST handle None explicitly (do not use as 0.0).
    """
    try:
        account = adapter.get_account()
        equity = float(account.get("equity", 0))
        logger.debug("[position_sync] broker equity: %.2f", equity)
        return equity
    except Exception as exc:
        logger.error("[position_sync] failed to fetch broker equity: %s", exc)
        return None


__all__ = [
    "SyncResult",
    "sync_positions_from_broker",
    "rebuild_ledger_from_broker",
    "get_broker_equity",
]
