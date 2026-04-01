"""Tests for M12: Broker Adapter — interface and Alpaca adapter."""

from __future__ import annotations

import pytest

from src.assembled_core.execution.broker_adapter import (
    AlpacaAdapter,
    BrokerAdapter,
    BrokerOrder,
    BrokerPosition,
    _safe_float,
    create_adapter_from_env,
)


@pytest.mark.phase12
@pytest.mark.phase13
class TestBrokerDataclasses:
    def test_broker_order_defaults(self):
        o = BrokerOrder(
            order_id="id1",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_type="market",
            status="pending",
        )
        assert o.filled_qty == 0.0
        assert o.filled_avg_price is None
        assert isinstance(o.raw, dict)

    def test_broker_position_fields(self):
        p = BrokerPosition(
            symbol="SPY",
            qty=100.0,
            avg_entry_price=450.0,
            market_value=45100.0,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=0.002,
        )
        assert p.symbol == "SPY"
        assert p.unrealized_pnl == 100.0


@pytest.mark.phase12
@pytest.mark.phase13
class TestAlpacaAdapterInit:
    def test_default_is_paper(self):
        adapter = AlpacaAdapter(api_key="fake", api_secret="fake")
        assert adapter.is_paper is True

    def test_paper_url_sets_is_paper(self):
        adapter = AlpacaAdapter(
            api_key="k",
            api_secret="s",
            base_url="https://paper-api.alpaca.markets",
        )
        assert adapter.is_paper is True

    def test_non_paper_url_raises_with_force_paper(self):
        with pytest.raises(ValueError, match="paper"):
            AlpacaAdapter(
                api_key="k",
                api_secret="s",
                base_url="https://api.alpaca.markets",
                force_paper=True,
            )

    def test_non_paper_url_blocked_without_allow_live_env(self):
        # force_paper=False alone is no longer sufficient — ALPACA_ALLOW_LIVE=true also required
        import os
        os.environ.pop("ALPACA_ALLOW_LIVE", None)
        with pytest.raises(ValueError, match="ALPACA_ALLOW_LIVE"):
            AlpacaAdapter(
                api_key="k",
                api_secret="s",
                base_url="https://api.alpaca.markets",
                force_paper=False,
            )

    def test_non_paper_url_ok_with_allow_live_env(self):
        # With both force_paper=False AND ALPACA_ALLOW_LIVE=true, live URL is accepted
        import os
        os.environ["ALPACA_ALLOW_LIVE"] = "true"
        try:
            adapter = AlpacaAdapter(
                api_key="k",
                api_secret="s",
                base_url="https://api.alpaca.markets",
                force_paper=False,
            )
            assert adapter.is_paper is False
        finally:
            os.environ.pop("ALPACA_ALLOW_LIVE", None)

    def test_health_check_fails_gracefully_no_keys(self):
        adapter = AlpacaAdapter(api_key="", api_secret="")
        health = adapter.health_check()
        assert health["ok"] is False
        assert "message" in health

    def test_submit_market_order_validates_qty(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="qty must be positive"):
            adapter.submit_market_order("AAPL", qty=-1.0, side="buy")

    def test_submit_market_order_validates_side(self):
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="side must be"):
            adapter.submit_market_order("AAPL", qty=10.0, side="hold")

    def test_get_api_raises_without_keys(self):
        adapter = AlpacaAdapter(api_key="", api_secret="")
        with pytest.raises(RuntimeError, match="API key and secret required"):
            adapter._get_api()


@pytest.mark.phase12
@pytest.mark.phase13
class TestAlpacaAdapterAbstract:
    def test_is_subclass_of_broker_adapter(self):
        assert issubclass(AlpacaAdapter, BrokerAdapter)

    def test_implements_all_abstract_methods(self):
        # If AlpacaAdapter can be instantiated, all abstracts are implemented
        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        assert adapter is not None


@pytest.mark.phase12
@pytest.mark.phase13
class TestCreateAdapterFromEnv:
    def test_alpaca_type_creates_alpaca_adapter(self):
        adapter = create_adapter_from_env("alpaca")
        assert isinstance(adapter, AlpacaAdapter)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter_type"):
            create_adapter_from_env("unknown_broker")


@pytest.mark.phase12
@pytest.mark.phase13
class TestSafeFloat:
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_float_str(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_int(self):
        assert _safe_float(5) == 5.0

    def test_bad_string_returns_none(self):
        assert _safe_float("not_a_number") is None


@pytest.mark.phase12
@pytest.mark.phase13
class TestNormalizeOrder:
    def test_normalize_simple_object(self):
        class FakeOrder:
            id = "abc123"
            symbol = "SPY"
            side = "buy"
            qty = "10"
            order_type = "market"
            type = "market"
            status = "pending"
            filled_qty = "0"
            filled_avg_price = None
            submitted_at = "2024-01-01T09:30:00Z"
            filled_at = None

        adapter = AlpacaAdapter(api_key="k", api_secret="s")
        order = adapter._normalize_order(FakeOrder())
        assert order.order_id == "abc123"
        assert order.symbol == "SPY"
        assert order.qty == 10.0
        assert order.filled_avg_price is None
