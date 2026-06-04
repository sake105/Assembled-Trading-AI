"""Broker Execution Bridge — Submit orders to broker, poll fills, convert to ledger format.

This module sits between the trading cycle output (orders DataFrame) and the
AlpacaAdapter. It handles:
- Kill switch check at the broker boundary (defense-in-depth)
- Per-order submission with retry logic
- Fill polling with timeout
- Conversion of BrokerOrder fills to ledger-compatible format
- Intent store integration for crash recovery
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from src.assembled_core.execution.broker_adapter import BrokerAdapter, BrokerOrder

logger = logging.getLogger(__name__)


@dataclass
class BrokerExecutionResult:
    """Result of a broker execution cycle."""

    submitted: list[BrokerOrder] = field(default_factory=list)
    filled: list[BrokerOrder] = field(default_factory=list)
    rejected: list[BrokerOrder] = field(default_factory=list)
    timed_out: list[BrokerOrder] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    fills_for_ledger: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False
    execution_time_s: float = 0.0


# Terminal order statuses (no further state changes expected)
_TERMINAL_STATUSES = frozenset({"filled", "cancelled", "rejected", "expired"})


def submit_orders_to_broker(
    adapter: BrokerAdapter,
    orders_df: pd.DataFrame,
    *,
    dry_run: bool = False,
    intent_store_path: str | None = None,
) -> tuple[list[BrokerOrder | None], dict[str, str]]:
    """Submit each order in orders_df to the broker.

    Safety:
    - Checks kill switch before any submission
    - Records intent before API call (crash recovery)
    - Try/except per order (one failure doesn't block others)
    - Respects rate limiting via api_resilience

    Args:
        adapter: Broker adapter instance.
        orders_df: DataFrame with columns [symbol, side, qty].
        dry_run: If True, log orders but don't actually submit.
        intent_store_path: Path to intent store (None = default).

    Returns:
        Tuple of (list of BrokerOrder or None, dict mapping order_id to intent_key).
    """
    from src.assembled_core.execution.kill_switch import is_kill_switch_engaged

    if is_kill_switch_engaged():
        logger.critical(
            "[broker_execution] KILL SWITCH ENGAGED — blocking all %d orders",
            len(orders_df),
        )
        return [None] * len(orders_df), {}

    if orders_df.empty:
        logger.info("[broker_execution] no orders to submit")
        return [], {}

    from src.assembled_core.execution.api_resilience import (
        get_alpaca_rate_limiter,
        retry_with_backoff,
    )
    from src.assembled_core.execution.intent_store import (
        record_order_complete,
        record_order_submit,
    )

    rate_limiter = get_alpaca_rate_limiter()
    results: list[BrokerOrder | None] = []
    # Map order_id -> intent key for later completion tracking
    intent_keys: dict[str, str] = {}

    for idx, row in orders_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        side = str(row.get("side", "")).strip().lower()
        qty = float(row.get("qty", 0))

        if not symbol or side not in ("buy", "sell") or qty <= 0:
            logger.warning(
                "[broker_execution] skipping invalid order: symbol=%r side=%r qty=%s",
                symbol,
                side,
                qty,
            )
            results.append(None)
            continue

        if dry_run:
            logger.info(
                "[broker_execution] DRY RUN: would submit %s %s qty=%.2f",
                side.upper(),
                symbol,
                qty,
            )
            results.append(
                BrokerOrder(
                    order_id=f"dry_run_{symbol}_{side}",
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="market",
                    status="dry_run",
                )
            )
            continue

        # Record intent BEFORE API call (crash recovery)
        intent_key: str | None = None
        try:
            intent_record = record_order_submit(
                symbol, side, qty, store_path=intent_store_path
            )
            intent_key = intent_record.get("idempotency_key")
        except Exception as exc:
            logger.warning("[broker_execution] intent store write failed: %s", exc)

        try:
            broker_order = retry_with_backoff(
                adapter.submit_market_order,
                symbol,
                qty,
                side,
                rate_limiter=rate_limiter,
                operation_name=f"submit_{side}_{symbol}",
            )
            results.append(broker_order)
            if intent_key:
                intent_keys[broker_order.order_id] = intent_key
            logger.info(
                "[broker_execution] submitted %s %s qty=%.2f -> order_id=%s status=%s",
                side.upper(),
                symbol,
                qty,
                broker_order.order_id,
                broker_order.status,
            )
        except Exception as exc:
            logger.error(
                "[broker_execution] FAILED to submit %s %s qty=%.2f: %s",
                side.upper(),
                symbol,
                qty,
                exc,
            )
            # Record completion for failed submit so crash recovery
            # doesn't treat this as a "lost" order
            if intent_key:
                try:
                    record_order_complete(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        filled_qty=0.0,
                        filled_price=None,
                        status="submit_failed",
                        intent_key=intent_key,
                        store_path=intent_store_path,
                    )
                except Exception as _exc:
                    logger.debug(
                        "[broker_execution] intent completion write failed: %s", _exc
                    )
            results.append(None)

    return results, intent_keys


def poll_order_fills(
    adapter: BrokerAdapter,
    broker_orders: list[BrokerOrder | None],
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 2.0,
) -> list[BrokerOrder]:
    """Poll broker for order fill status until all orders reach a terminal state.

    Args:
        adapter: Broker adapter instance.
        broker_orders: List of submitted BrokerOrder objects (None entries skipped).
        timeout_s: Maximum time to wait for all fills.
        poll_interval_s: Interval between poll cycles.

    Returns:
        List of updated BrokerOrder objects with final statuses.
    """
    from src.assembled_core.execution.api_resilience import (
        get_alpaca_rate_limiter,
        retry_with_backoff,
    )

    rate_limiter = get_alpaca_rate_limiter()
    pending = {
        o.order_id: o
        for o in broker_orders
        if o is not None
        and o.status not in _TERMINAL_STATUSES
        and o.status != "dry_run"
    }

    if not pending:
        return [o for o in broker_orders if o is not None]

    completed: dict[str, BrokerOrder] = {}
    deadline = time.monotonic() + timeout_s
    logger.info(
        "[broker_execution] polling %d orders for fills (timeout %.0fs)",
        len(pending),
        timeout_s,
    )

    while pending and time.monotonic() < deadline:
        time.sleep(poll_interval_s)

        for order_id in list(pending.keys()):
            try:
                updated = retry_with_backoff(
                    adapter.get_order_status,
                    order_id,
                    rate_limiter=rate_limiter,
                    operation_name=f"poll_{order_id[:8]}",
                )
                pending[order_id] = updated
                if updated.status in _TERMINAL_STATUSES:
                    logger.info(
                        "[broker_execution] order %s -> %s (filled_qty=%.2f price=%s)",
                        order_id[:8],
                        updated.status,
                        updated.filled_qty,
                        updated.filled_avg_price,
                    )
                    completed[order_id] = updated
                    del pending[order_id]
            except Exception as exc:
                logger.warning(
                    "[broker_execution] poll failed for %s: %s", order_id[:8], exc
                )

    if pending:
        logger.warning(
            "[broker_execution] %d orders still pending after %.0fs timeout: %s",
            len(pending),
            timeout_s,
            [oid[:8] for oid in pending],
        )

    # Merge: completed orders were stored before deletion from pending
    final_map: dict[str, BrokerOrder] = {}
    for o in broker_orders:
        if o is not None:
            final_map[o.order_id] = o  # Start with original

    # Overwrite with any updated versions (still pending at timeout)
    for oid, updated in pending.items():
        final_map[oid] = updated

    # Overwrite with completed versions tracked during polling
    for oid, updated in completed.items():
        final_map[oid] = updated

    return list(final_map.values())


def convert_broker_fills_to_ledger_format(
    broker_orders: list[BrokerOrder],
    *,
    intent_keys: dict[str, str] | None = None,
    intent_store_path: str | None = None,
) -> list[dict[str, Any]]:
    """Convert filled BrokerOrders to the dict format expected by apply_fills_to_ledger().

    CRITICAL conversions:
    - BrokerOrder.side is lowercase ("buy"/"sell") -> must be UPPERCASE for ledger
    - Uses filled_qty (not requested qty) to handle partial fills correctly
    - Uses filled_avg_price (actual fill price from broker)

    Args:
        broker_orders: List of BrokerOrder objects after polling.
        intent_store_path: Path to intent store for completion recording.

    Returns:
        List of dicts with keys: symbol, side, qty, price.
    """
    from src.assembled_core.execution.intent_store import record_order_complete

    fills: list[dict[str, Any]] = []

    for order in broker_orders:
        if order.status == "dry_run":
            continue

        # Record completion in intent store (paired with submit via intent_key)
        _intent_key = (intent_keys or {}).get(order.order_id)
        try:
            record_order_complete(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                filled_qty=order.filled_qty,
                filled_price=order.filled_avg_price,
                status=order.status,
                intent_key=_intent_key,
                store_path=intent_store_path,
            )
        except Exception as exc:
            logger.warning("[broker_execution] intent completion write failed: %s", exc)

        if order.status != "filled":
            logger.info(
                "[broker_execution] order %s status=%s — not converting to fill",
                order.symbol,
                order.status,
            )
            continue

        if order.filled_qty <= 0:
            logger.warning(
                "[broker_execution] order %s filled but filled_qty=0 — skipping",
                order.symbol,
            )
            continue

        fill_price = order.filled_avg_price
        if fill_price is None or fill_price <= 0:
            logger.warning(
                "[broker_execution] order %s filled but no valid fill price — skipping",
                order.symbol,
            )
            continue

        fills.append(
            {
                "symbol": order.symbol,
                "side": order.side.upper(),  # CRITICAL: ledger expects uppercase
                "qty": order.filled_qty,  # Use filled_qty for partial fill safety
                "price": fill_price,  # Actual broker fill price
            }
        )

    return fills


def execute_via_broker(
    adapter: BrokerAdapter,
    orders_df: pd.DataFrame,
    *,
    dry_run: bool = False,
    timeout_s: float = 120.0,
    poll_interval_s: float = 2.0,
    intent_store_path: str | None = None,
) -> BrokerExecutionResult:
    """High-level orchestrator: submit orders -> poll fills -> convert to ledger format.

    This is the main entry point for broker-based execution.

    Args:
        adapter: Broker adapter instance (e.g. AlpacaAdapter).
        orders_df: DataFrame with columns [symbol, side, qty].
        dry_run: If True, log orders but don't submit to broker.
        timeout_s: Maximum time to wait for fills.
        poll_interval_s: Interval between fill polls.
        intent_store_path: Path to intent store for crash recovery.

    Returns:
        BrokerExecutionResult with all submitted, filled, rejected, timed_out orders
        and fills_for_ledger ready for apply_fills_to_ledger().
    """
    t0 = time.monotonic()
    result = BrokerExecutionResult(dry_run=dry_run)

    # Pre-trade sanity checks (§5.2) — runs in paper/live mode only
    if not dry_run and not orders_df.empty:
        try:
            from src.assembled_core.qa.sanity_checks import SanityChecker
            from src.assembled_core.ops.alerting import AlertManager

            _checker = SanityChecker()
            _alert = AlertManager()
            _halted_syms: list[str] = []
            for _, _row in orders_df.iterrows():
                _order = {
                    "symbol": _row.get("symbol", ""),
                    "side": _row.get("side", ""),
                    "qty": float(_row.get("qty", 0) or 0),
                }
                _result = _checker.check_order(_order)
                if _result.get("halt_recommendation"):
                    _sym = str(_order["symbol"])
                    _halted_syms.append(_sym)
                    _flags_str = "; ".join(f["rule"] for f in _result.get("flags", []))
                    _alert.fire(
                        "sanity_check_halt",
                        {
                            "symbol": _sym,
                            "flags": _flags_str or "unknown",
                        },
                    )
                    logger.warning(
                        "[broker_execution] SANITY HALT: %s %s — %s",
                        _order["side"],
                        _sym,
                        _flags_str,
                    )
            if _halted_syms:
                orders_df = orders_df[
                    ~orders_df["symbol"].astype(str).isin(_halted_syms)
                ].reset_index(drop=True)
                logger.warning(
                    "[broker_execution] %d order(s) removed by sanity checks: %s",
                    len(_halted_syms),
                    _halted_syms,
                )
        except Exception as _se:
            logger.debug("[broker_execution] sanity check skipped: %s", _se)

    # Step 1: Submit
    logger.info(
        "[broker_execution] starting %s execution for %d orders",
        "DRY RUN" if dry_run else "LIVE",
        len(orders_df),
    )
    submitted_raw, intent_keys = submit_orders_to_broker(
        adapter,
        orders_df,
        dry_run=dry_run,
        intent_store_path=intent_store_path,
    )

    submitted = [o for o in submitted_raw if o is not None]
    failed_count = sum(1 for o in submitted_raw if o is None)
    result.submitted = submitted

    if failed_count > 0:
        result.errors.append({"phase": "submit", "failed_count": failed_count})
        logger.warning(
            "[broker_execution] %d/%d orders failed to submit",
            failed_count,
            len(orders_df),
        )

    if dry_run:
        result.execution_time_s = time.monotonic() - t0
        logger.info("[broker_execution] DRY RUN complete — no fills")
        return result

    # Step 2: Poll for fills
    if submitted:
        final_orders = poll_order_fills(
            adapter,
            cast("list[BrokerOrder | None]", submitted),
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
    else:
        final_orders = []

    # Categorize results
    for order in final_orders:
        if order.status == "filled":
            result.filled.append(order)
        elif order.status in ("cancelled", "rejected", "expired"):
            result.rejected.append(order)
        else:
            result.timed_out.append(order)

    # Step 3: Convert fills to ledger format
    result.fills_for_ledger = convert_broker_fills_to_ledger_format(
        final_orders,
        intent_keys=intent_keys,
        intent_store_path=intent_store_path,
    )

    result.execution_time_s = time.monotonic() - t0
    logger.info(
        "[broker_execution] complete in %.1fs: %d filled, %d rejected, %d timed_out, %d ledger fills",
        result.execution_time_s,
        len(result.filled),
        len(result.rejected),
        len(result.timed_out),
        len(result.fills_for_ledger),
    )

    # Fire fill_rate_low alert if fill rate is below threshold
    if submitted and not dry_run:
        try:
            fill_rate = len(result.filled) / len(submitted)
            if fill_rate < 0.5:
                from src.assembled_core.ops.alerting import AlertManager

                AlertManager().fire("fill_rate_low", {"daily_fill_rate": fill_rate})
        except Exception as _fe:
            logger.debug("[broker_execution] fill rate alert skipped: %s", _fe)

    return result


__all__ = [
    "BrokerExecutionResult",
    "submit_orders_to_broker",
    "poll_order_fills",
    "convert_broker_fills_to_ledger_format",
    "execute_via_broker",
]
