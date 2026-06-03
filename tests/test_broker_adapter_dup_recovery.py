"""B-exec-1: duplicate-order recovery actually adopts the existing order.

Diagnostik §execution MAJOR (a): both duplicate branches in submit_market_order
and submit_limit_order used to ``raise`` unconditionally, so the idempotency
design (deterministic client_order_id -> broker duplicate-rejection -> adopt the
existing order) never recovered — a crash/retry only re-propagated the exception.

Fix: on a duplicate error, fetch the existing order by client_order_id (alpaca-py
``get_order_by_client_id`` or legacy ``get_order_by_client_order_id``) and return
it normalized. If the recovery fetch is impossible/fails, the ORIGINAL submit
error is re-raised (fail-safe — never fabricate or swallow).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]

# is_duplicate_error requires both "duplicate" and "client_order_id" in the msg.
_DUP_MSG = "400 Bad Request: duplicate client_order_id ata-abc already exists"
_NON_DUP_MSG = "503 Service Unavailable: broker temporarily down"


@pytest.fixture
def adapter():
    from src.assembled_core.execution.broker_adapter import AlpacaAdapter, BrokerOrder

    a = AlpacaAdapter(
        api_key="test-key",  # pragma: allowlist secret
        api_secret="test-secret",  # pragma: allowlist secret
        force_paper=True,
        enforce_market_hours=False,
    )
    a._validate_market_hours = lambda: None
    a._check_cycle_limits = lambda *args, **kw: None
    a._estimate_price = lambda sym: 100.0
    # Normalize echoes the raw object's id so we can prove WHICH order was returned.
    a._normalize_order = lambda raw: BrokerOrder(
        order_id=getattr(raw, "id", "stub"),
        symbol="TEST",
        side="buy",
        qty=1.0,
        order_type="market",
        status="accepted",
    )
    return a


def _api_with_dup(getter_name: str):
    """MagicMock broker API: submit_order raises a dup error; <getter_name>
    returns the pre-existing order. spec= ensures the OTHER getter is absent."""
    existing = MagicMock(id="existing-order-id")
    api = MagicMock(spec=["submit_order", getter_name])
    api.submit_order.side_effect = RuntimeError(_DUP_MSG)
    getattr(api, getter_name).return_value = existing
    return api, existing


# --------------------------------------------------------------------------- #
# Recovery success — both SDK variants, both order types
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "getter", ["get_order_by_client_id", "get_order_by_client_order_id"]
)
def test_market_dup_adopts_existing_order(adapter, getter):
    api, existing = _api_with_dup(getter)
    adapter._get_api = lambda: api

    result = adapter.submit_market_order("AAPL", 10.0, "buy")

    # Returned the EXISTING order (normalized), not an exception.
    assert result.order_id == "existing-order-id"
    # submit_order tried exactly once (no second live order placed).
    assert api.submit_order.call_count == 1
    # Recovery fetched by the deterministic coid.
    called_coid = getattr(api, getter).call_args[0][0]
    assert called_coid.startswith("ata-")


@pytest.mark.parametrize(
    "getter", ["get_order_by_client_id", "get_order_by_client_order_id"]
)
def test_limit_dup_adopts_existing_order(adapter, getter):
    api, existing = _api_with_dup(getter)
    adapter._get_api = lambda: api

    result = adapter.submit_limit_order("AAPL", 10.0, "buy", limit_price=99.0)

    assert result.order_id == "existing-order-id"
    assert api.submit_order.call_count == 1
    assert getattr(api, getter).call_count == 1


# --------------------------------------------------------------------------- #
# Stop-order symmetry (Diagnostik §execution): submit_stop_order had NO
# dup-recovery (asymmetric vs market/limit). It uses a deterministic
# client_order_id and CAN hit the broker dup rejection, so it must apply the
# SAME adopt-existing / fail-safe-reraise pattern.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "getter", ["get_order_by_client_id", "get_order_by_client_order_id"]
)
def test_stop_dup_adopts_existing_order(adapter, getter):
    api, existing = _api_with_dup(getter)
    adapter._get_api = lambda: api

    result = adapter.submit_stop_order("AAPL", 10.0, "sell", stop_price=95.0)

    # Adopted the EXISTING order, not a second live order, not an exception.
    assert result.order_id == "existing-order-id"
    assert api.submit_order.call_count == 1
    # Recovery fetched by the deterministic coid.
    called_coid = getattr(api, getter).call_args[0][0]
    assert called_coid.startswith("ata-")


def test_stop_dup_recovery_fetch_fails_reraises_original(adapter):
    """Stop fail-safe: recovery lookup raises -> ORIGINAL submit error
    propagates (chained), never the lookup error, never a fabricated order."""
    api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    api.submit_order.side_effect = RuntimeError(_DUP_MSG)
    api.get_order_by_client_id.side_effect = ConnectionError("lookup boom")
    adapter._get_api = lambda: api

    with pytest.raises(RuntimeError, match="duplicate client_order_id") as exc_info:
        adapter.submit_stop_order("AAPL", 10.0, "sell", stop_price=95.0)
    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_stop_non_duplicate_error_propagates_without_recovery(adapter):
    api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    api.submit_order.side_effect = RuntimeError(_NON_DUP_MSG)
    adapter._get_api = lambda: api

    with pytest.raises(RuntimeError, match="Service Unavailable"):
        adapter.submit_stop_order("AAPL", 10.0, "sell", stop_price=95.0)
    api.get_order_by_client_id.assert_not_called()


def test_stop_adopted_order_does_not_increment_cycle_counters(adapter):
    """An adopted (recovered) stop order must not consume this cycle's budget."""
    api, _existing = _api_with_dup("get_order_by_client_id")
    adapter._get_api = lambda: api

    before = adapter._cycle_order_count
    adapter.submit_stop_order("AAPL", 10.0, "sell", stop_price=95.0)
    assert adapter._cycle_order_count == before
    assert adapter._cycle_notional_total == 0.0


# --------------------------------------------------------------------------- #
# Fail-safe: recovery impossible / failing -> original error propagates
# --------------------------------------------------------------------------- #


def test_dup_but_no_lookup_method_reraises_original(adapter):
    """Broker API exposes neither getter -> original submit error propagates."""
    api = MagicMock(spec=["submit_order"])
    api.submit_order.side_effect = RuntimeError(_DUP_MSG)
    adapter._get_api = lambda: api

    with pytest.raises(RuntimeError, match="duplicate client_order_id"):
        adapter.submit_market_order("AAPL", 10.0, "buy")


def test_dup_but_recovery_fetch_fails_reraises_original(adapter):
    """Recovery lookup itself raises -> ORIGINAL submit error propagates
    (chained), never the lookup error and never a fabricated order."""
    api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    api.submit_order.side_effect = RuntimeError(_DUP_MSG)
    api.get_order_by_client_id.side_effect = ConnectionError("lookup boom")
    adapter._get_api = lambda: api

    with pytest.raises(RuntimeError, match="duplicate client_order_id") as exc_info:
        adapter.submit_market_order("AAPL", 10.0, "buy")
    # The lookup error is chained as __cause__, not surfaced as the primary error.
    assert isinstance(exc_info.value.__cause__, ConnectionError)


# --------------------------------------------------------------------------- #
# Non-duplicate errors are NOT recovered — they propagate untouched
# --------------------------------------------------------------------------- #


def test_non_duplicate_error_propagates_without_recovery(adapter):
    api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    api.submit_order.side_effect = RuntimeError(_NON_DUP_MSG)
    adapter._get_api = lambda: api

    with pytest.raises(RuntimeError, match="Service Unavailable"):
        adapter.submit_market_order("AAPL", 10.0, "buy")
    # No recovery attempt for a non-duplicate failure.
    api.get_order_by_client_id.assert_not_called()


def test_successful_submit_is_unchanged(adapter):
    """Happy path: no exception -> the submitted order is returned, no recovery."""
    api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    api.submit_order.return_value = MagicMock(id="fresh-order-id")
    adapter._get_api = lambda: api

    result = adapter.submit_market_order("AAPL", 10.0, "buy")

    assert result.order_id == "fresh-order-id"
    api.get_order_by_client_id.assert_not_called()
    # A freshly-submitted order DOES consume a cycle slot.
    assert adapter._cycle_order_count == 1


# --------------------------------------------------------------------------- #
# Adopted (recovered) orders must NOT consume this cycle's fresh-order budget
# --------------------------------------------------------------------------- #


def test_adopted_order_does_not_increment_cycle_counters(adapter):
    """An order placed on a PRIOR attempt (recovered) must not count against this
    cycle's order/notional budget (it was already placed, possibly in another
    process). Over-counting it would shrink the soft cap incorrectly."""
    api, _existing = _api_with_dup("get_order_by_client_id")
    adapter._get_api = lambda: api

    before = adapter._cycle_order_count
    adapter.submit_market_order("AAPL", 10.0, "buy")
    assert adapter._cycle_order_count == before
    assert adapter._cycle_notional_total == 0.0


def test_fresh_then_dup_counts_only_the_fresh(adapter):
    """Two calls: a fresh submit increments the counter; a following duplicate
    that is adopted does not. Net cycle_order_count == 1."""
    fresh_api = MagicMock(spec=["submit_order", "get_order_by_client_id"])
    fresh_api.submit_order.return_value = MagicMock(id="fresh")
    adapter._get_api = lambda: fresh_api
    adapter.submit_market_order("AAPL", 10.0, "buy")

    dup_api, _ = _api_with_dup("get_order_by_client_id")
    adapter._get_api = lambda: dup_api
    adapter.submit_market_order("MSFT", 5.0, "buy")

    assert adapter._cycle_order_count == 1
