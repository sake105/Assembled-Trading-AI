"""A2: idempotency.py wired into paper_trading_engine + broker_adapter."""

from __future__ import annotations

import pytest


def _make_fake_scheduler(n_slices: int, price: float):
    """Return a scheduler-like object whose schedule() returns n_slices SlicedOrder-like items."""

    class _Slice:
        def __init__(self, qty, p):
            self.quantity = qty
            self.price = p

    class _FakeScheduler:
        def schedule(self, *_, **__):
            return [_Slice(1.0, price) for _ in range(n_slices)]

    return _FakeScheduler()


@pytest.mark.fast
def test_build_client_order_id_deterministic_by_intent():
    """build_client_order_id with same intent always produces the same ID (A2 core property)."""
    from src.assembled_core.execution.idempotency import (
        compute_intent_hash,
        build_client_order_id,
    )

    base_hash = compute_intent_hash("AAPL", "buy", 10.0, "twap", 150.0)
    expected_ids = [
        build_client_order_id(f"twap_slice_{i + 1}_of_3", base_hash, 0)
        for i in range(3)
    ]
    # All IDs use the idempotency prefix
    for oid in expected_ids:
        assert oid.startswith("ata-"), f"order_id {oid!r} must use idempotency prefix"
    # All 3 slice IDs are distinct (different slice labels)
    assert len(set(expected_ids)) == 3
    # Calling again gives same IDs (deterministic)
    ids2 = [
        build_client_order_id(f"twap_slice_{i + 1}_of_3", base_hash, 0)
        for i in range(3)
    ]
    assert expected_ids == ids2, "Deterministic: same intent always produces same IDs"


@pytest.mark.fast
def test_different_intents_produce_different_ids():
    """Different symbol/side/qty must produce different intent hashes."""
    from src.assembled_core.execution.idempotency import (
        compute_intent_hash,
        build_client_order_id,
    )

    hash_aapl = compute_intent_hash("AAPL", "buy", 10.0, "twap", 150.0)
    hash_msft = compute_intent_hash("MSFT", "buy", 10.0, "twap", 150.0)
    assert hash_aapl != hash_msft

    id_aapl = build_client_order_id("slice_1_of_2", hash_aapl)
    id_msft = build_client_order_id("slice_1_of_2", hash_msft)
    assert id_aapl != id_msft, "Different symbols must produce different order IDs"


@pytest.mark.fast
def test_no_uuid4_in_paper_trading_engine_source():
    """paper_trading_engine.py must not call uuid.uuid4() directly."""
    import inspect
    import src.assembled_core.execution.paper_trading_engine as mod

    src_text = inspect.getsource(mod)
    assert "uuid.uuid4()" not in src_text, (
        "paper_trading_engine must use idempotency module, not uuid.uuid4()"
    )


@pytest.mark.fast
def test_is_duplicate_error_wired_in_broker_adapter():
    """broker_adapter.py must import is_duplicate_error (A2 wiring check)."""
    import inspect
    import src.assembled_core.execution.broker_adapter as mod

    src_text = inspect.getsource(mod)
    assert "is_duplicate_error" in src_text, (
        "broker_adapter must use is_duplicate_error from idempotency module"
    )
