"""Idempotent order submission helpers.

From 33_EXECUTION_ORDERMANAGEMENT.md §33.2.

Prevents duplicate orders when a worker crashes after submit() but before
the DB-write.  client_order_id is deterministic from signal+intent, so
a retry will hit Alpaca's duplicate-rejection and we can recover the
existing order instead of placing a second one.
"""

from __future__ import annotations

import hashlib


def compute_intent_hash(
    symbol: str,
    side: str,
    qty: float,
    order_type: str,
    limit_price: float | None = None,
) -> str:
    """SHA-256 of the intent-defining fields.

    Args:
        symbol: Ticker symbol.
        side: 'buy' or 'sell' (or 'sell_short', 'buy_to_cover').
        qty: Intended quantity (absolute, positive).
        order_type: 'market', 'limit', 'stop', 'stop_limit'.
        limit_price: Limit price if applicable.

    Returns:
        64-char hex digest.
    """
    canon = f"{symbol}|{side}|{abs(qty):.6f}|{order_type}|{limit_price if limit_price is not None else '-'}"
    return hashlib.sha256(canon.encode()).hexdigest()


def build_client_order_id(
    signal_id: str,
    intent_hash: str,
    attempt: int = 0,
) -> str:
    """Build a deterministic Alpaca client_order_id (≤ 48 chars).

    Same signal_id + intent_hash → same client_order_id on every attempt.
    Set attempt > 0 when you explicitly want a fresh order (not a retry).

    Args:
        signal_id: UUID-string identifying the originating signal row.
        intent_hash: Output of ``compute_intent_hash()``.
        attempt: Retry counter (0 = first attempt).

    Returns:
        String starting with 'ata-', max 48 chars total.

    Note:
        Alpaca enforces a 48-character limit on client_order_id.
        We take only the first 20 hex chars of the SHA-256 digest,
        giving 2^80 ≈ 1.2 × 10^24 collision space — sufficient.
    """
    raw = f"{signal_id}:{intent_hash}:{attempt}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return f"ata-{h}"  # 4 + 20 = 24 chars, well within 48-char limit


def is_duplicate_error(error_message: str) -> bool:
    """Return True if a broker error string indicates a duplicate client_order_id."""
    msg = error_message.lower()
    return "duplicate" in msg and "client_order_id" in msg


__all__ = ["compute_intent_hash", "build_client_order_id", "is_duplicate_error"]
