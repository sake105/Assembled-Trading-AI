"""Idempotent order submission helpers.

From 33_EXECUTION_ORDERMANAGEMENT.md §33.2.

Prevents duplicate orders when a worker crashes after submit() but before
the DB-write.  client_order_id is deterministic from signal+intent, so
a retry will hit Alpaca's duplicate-rejection and we can recover the
existing order instead of placing a second one.
"""

from __future__ import annotations

import hashlib
import re

# Standalone word "order" (not a substring of "border"/"reorder"/"recorder"/
# "ordering"/"disorder"/"order book"), OR the exact field token "client_order_id".
# Word-boundary match keeps the order/id guard from false-firing on look-alikes
# while still recognising every real broker duplicate signature.
_ORDER_REF_RE = re.compile(r"\border\b|client_order_id")


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
    """Return True if a broker error string indicates a duplicate/idempotency rejection.

    Pure string classifier (the only input is the broker's error text). Used by the
    broker_adapter duplicate-order recovery path (B-exec-1): when this returns True,
    the adapter tries to *adopt* the pre-existing order by its deterministic
    ``client_order_id`` instead of placing a second one.

    SAFETY / COMPOSITION NOTE: this composes with the FAIL-SAFE recovery in
    ``broker_adapter`` — if the order is not actually found by ``client_order_id``,
    the ORIGINAL submit error is re-raised and no order is fabricated. A false
    *positive* here therefore degrades to "re-raise the original error" (no second
    order, no silent swallow), while a false *negative* misclassifies a genuine
    duplicate as a hard failure (the original B-exec-2 bug). So we deliberately match
    the real broker duplicate/idempotency signatures, but we must NEVER match a
    clearly generic/transient failure (timeout, 503, connection reset, rate limit,
    insufficient buying power) — those are not duplicates and must propagate.

    Accepted duplicate signatures (case-insensitive). Real Alpaca rejections often do
    NOT contain both the word "duplicate" AND the literal token "client_order_id"
    (the original over-strict check), so we recognise the genuine variants:

      1. Both tokens present ("duplicate" + "client_order_id") — the original,
         most explicit form. Kept for back-compat.
      2. The bare word "duplicate" combined with an order/id reference
         ("client_order_id" or "order") — Alpaca/HTTP 403/422 bodies that say e.g.
         "duplicate order" without spelling out the field name.
      3. "... already exists" guarded by an order/id reference
         ("order already exists", "client_order_id ... already exists", or a
         "422 ... already exists" body) — the broker rejected the resubmit because
         that deterministic id is already on file. The order/id guard prevents
         matching an unrelated "already exists" from some other API surface.
      4. "potential wash trade" — Alpaca returns this specific 422 when the SAME
         ``client_order_id``/intent is re-submitted against an open same-side
         resting order; for our deterministic-coid retry it is a duplicate signal.
    """
    msg = error_message.lower()

    # An order/id reference shared by several patterns below. Requiring this guard
    # keeps "already exists" / bare "duplicate" from matching unrelated surfaces.
    # Word-boundary match on the standalone word "order" (or the exact token
    # "client_order_id") so look-alike substrings — "border", "reorder",
    # "recorder", "ordering", "disorder" — do NOT count as an order reference.
    has_order_ref = bool(_ORDER_REF_RE.search(msg))

    # (1)+(2): explicit duplicate of a client_order_id, or a "duplicate ... order".
    if "duplicate" in msg and has_order_ref:
        return True

    # (3): broker rejected the resubmit because the (deterministic) id already exists.
    # Guarded by an order/id reference so a generic "already exists" elsewhere is not
    # swept in. Covers "order already exists", "client_order_id ... already exists",
    # and a "422 ... already exists" body.
    if "already exists" in msg and has_order_ref:
        return True

    # (4): Alpaca's 422 for re-submitting the same intent/coid against a resting
    # same-side order — a duplicate signal for our deterministic-coid retry path.
    if "potential wash trade" in msg:
        return True

    return False


__all__ = ["compute_intent_hash", "build_client_order_id", "is_duplicate_error"]
