"""Tests for M20.1: Extended order types in broker_adapter.py."""

from __future__ import annotations

import pytest

from src.assembled_core.execution.broker_adapter import (
    AlpacaAdapter,
    BrokerAdapter,
    BrokerOrder,
)


@pytest.mark.phase12
class TestLimitOrderValidation:
    def test_submit_limit_order_validates_qty(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="qty must be positive"):
            adapter.submit_limit_order("AAPL", qty=-1.0, side="buy", limit_price=150.0)

    def test_submit_limit_order_validates_price(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="limit_price must be positive"):
            adapter.submit_limit_order("AAPL", qty=10.0, side="buy", limit_price=-5.0)

    def test_submit_limit_order_validates_side(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="side must be"):
            adapter.submit_limit_order("AAPL", qty=10.0, side="hold", limit_price=150.0)


@pytest.mark.phase12
class TestStopOrderValidation:
    def test_submit_stop_order_validates_qty(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="qty must be positive"):
            adapter.submit_stop_order("AAPL", qty=0, side="sell", stop_price=140.0)

    def test_submit_stop_order_validates_price(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="stop_price must be positive"):
            adapter.submit_stop_order("AAPL", qty=10.0, side="sell", stop_price=-1.0)

    def test_submit_stop_order_validates_side(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="side must be"):
            adapter.submit_stop_order("AAPL", qty=10.0, side="short", stop_price=140.0)


@pytest.mark.phase12
class TestMOCLOCDefaultImplementation:
    """MOC/LOC have default implementations in the ABC that delegate to market/limit."""

    def test_broker_adapter_has_moc_method(self):
        assert hasattr(BrokerAdapter, "submit_moc_order")

    def test_broker_adapter_has_loc_method(self):
        assert hasattr(BrokerAdapter, "submit_loc_order")

    def test_alpaca_inherits_moc(self):
        assert hasattr(AlpacaAdapter, "submit_moc_order")
        assert hasattr(AlpacaAdapter, "submit_loc_order")


@pytest.mark.phase12
class TestBrokerOrderTypeField:
    def test_order_type_accepts_new_types(self):
        for otype in ["market", "limit", "stop", "stop_limit", "moc", "loc"]:
            o = BrokerOrder(
                order_id="id1",
                symbol="AAPL",
                side="buy",
                qty=10.0,
                order_type=otype,
                status="pending",
            )
            assert o.order_type == otype


@pytest.mark.phase12
class TestAbstractMethodsComplete:
    def test_limit_is_abstract(self):
        """submit_limit_order should be abstract on BrokerAdapter."""
        assert "submit_limit_order" in BrokerAdapter.__abstractmethods__

    def test_stop_is_abstract(self):
        """submit_stop_order should be abstract on BrokerAdapter."""
        assert "submit_stop_order" in BrokerAdapter.__abstractmethods__

    def test_moc_is_not_abstract(self):
        """MOC has default implementation, should not be abstract."""
        assert "submit_moc_order" not in BrokerAdapter.__abstractmethods__

    def test_loc_is_not_abstract(self):
        """LOC has default implementation, should not be abstract."""
        assert "submit_loc_order" not in BrokerAdapter.__abstractmethods__
