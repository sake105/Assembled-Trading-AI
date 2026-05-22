"""Regression test for F-A3-4: broker_adapter passes client_order_id to Alpaca SDK.

R3 audit (F-A3-4): submit_market/limit/stop_order built Alpaca SDK requests
WITHOUT client_order_id. Network retry created duplicate broker orders with
fresh UUIDs → position doubling.

R4 fix (f3a8b5d): _auto_client_order_id() helper using existing idempotency
module. All 3 submit methods now pass deterministic coid to SDK request +
legacy fallback.

R6 test backfill (F-C4-N-5).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]


@pytest.fixture
def adapter():
    """Build an AlpacaAdapter in test mode with mocked API."""
    from src.assembled_core.execution.broker_adapter import AlpacaAdapter

    adapter = AlpacaAdapter(
        api_key="test-key",  # pragma: allowlist secret
        api_secret="test-secret",  # pragma: allowlist secret
        force_paper=True,
        enforce_market_hours=False,
    )

    # Disable market-hour validation for tests
    adapter._validate_market_hours = lambda: None
    # Disable cycle limit checks
    adapter._check_cycle_limits = lambda *a, **kw: None
    # Estimate price returns a fixed value
    adapter._estimate_price = lambda sym: 100.0
    # Normalize_order returns a stub BrokerOrder
    from src.assembled_core.execution.broker_adapter import BrokerOrder

    adapter._normalize_order = lambda raw: BrokerOrder(
        order_id="stub-id",
        symbol="TEST",
        side="buy",
        qty=1.0,
        order_type="market",
        status="pending",
    )
    return adapter


def test_auto_client_order_id_is_deterministic_F_A3_4(adapter):
    """Same intent (symbol, side, qty, type) on same UTC day → same coid."""
    coid1 = adapter._auto_client_order_id("AAPL", "buy", 100.0, "market", None)
    coid2 = adapter._auto_client_order_id("AAPL", "buy", 100.0, "market", None)
    assert coid1 == coid2
    # Format: ata-{20-hex} = 24 chars (< Alpaca 48-char limit)
    assert coid1.startswith("ata-")
    assert len(coid1) == 24


def test_auto_client_order_id_differs_by_intent_F_A3_4(adapter):
    """Different intent → different coid."""
    coid_buy = adapter._auto_client_order_id("AAPL", "buy", 100.0, "market", None)
    coid_sell = adapter._auto_client_order_id("AAPL", "sell", 100.0, "market", None)
    coid_qty = adapter._auto_client_order_id("AAPL", "buy", 200.0, "market", None)
    coid_sym = adapter._auto_client_order_id("MSFT", "buy", 100.0, "market", None)

    # All distinct
    assert len({coid_buy, coid_sell, coid_qty, coid_sym}) == 4


def test_submit_market_order_passes_client_order_id_to_sdk_F_A3_4(adapter):
    """The SDK MarketOrderRequest must receive client_order_id."""
    mock_api = MagicMock()
    mock_api.submit_order.return_value = MagicMock(id="mock-order-id")
    adapter._get_api = lambda: mock_api

    captured_request = []

    def _capture(order_data):
        captured_request.append(order_data)
        return mock_api.submit_order.return_value

    mock_api.submit_order.side_effect = _capture

    adapter.submit_market_order("AAPL", 100.0, "buy")

    assert mock_api.submit_order.called
    assert len(captured_request) == 1
    req = captured_request[0]
    # MarketOrderRequest must have client_order_id set
    assert hasattr(req, "client_order_id"), (
        "F-A3-4 regression: MarketOrderRequest must have client_order_id field"
    )
    coid = getattr(req, "client_order_id", None)
    assert coid is not None
    assert coid.startswith("ata-")


def test_submit_market_order_explicit_client_order_id_F_A3_4(adapter):
    """Caller-provided client_order_id overrides auto-generation."""
    mock_api = MagicMock()
    captured_request = []

    def _capture(order_data):
        captured_request.append(order_data)
        return MagicMock(id="mock")

    mock_api.submit_order.side_effect = _capture
    adapter._get_api = lambda: mock_api

    explicit_coid = "my-explicit-id-123"
    adapter.submit_market_order("AAPL", 100.0, "buy", client_order_id=explicit_coid)

    assert captured_request[0].client_order_id == explicit_coid


def test_same_day_retry_yields_same_coid_F_A3_4(adapter):
    """Network retry: same intent within the same UTC day → identical coid →
    Alpaca duplicate-rejection → no double position.
    """
    # Simulate retry by calling twice in a row
    coid_a = adapter._auto_client_order_id("AAPL", "buy", 100.0, "market", None)
    coid_b = adapter._auto_client_order_id("AAPL", "buy", 100.0, "market", None)
    assert coid_a == coid_b, (
        "F-A3-4 idempotency: same-day same-intent must produce identical coid "
        "so Alpaca rejects the retry"
    )
