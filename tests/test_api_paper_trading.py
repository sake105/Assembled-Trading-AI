"""Tests for paper trading API endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app

pytestmark = pytest.mark.phase10


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_paper_orders_submit_and_list(client: TestClient):
    """Test submitting orders and listing them."""
    # Reset first
    response = client.post("/api/v1/paper/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # Submit 2 orders
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        },
        {
            "symbol": "GOOGL",
            "side": "SELL",
            "quantity": 5.0,
            "price": 2500.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 2
    
    # Check first order
    order1 = data[0]
    assert "order_id" in order1
    assert order1["symbol"] == "AAPL"
    assert order1["side"] == "BUY"
    assert order1["quantity"] == 10.0
    assert order1["price"] == 150.0
    assert order1["status"] == "FILLED"
    assert order1["reason"] is None
    
    # Check second order
    order2 = data[1]
    assert order2["symbol"] == "GOOGL"
    assert order2["side"] == "SELL"
    assert order2["quantity"] == 5.0
    assert order2["status"] == "FILLED"
    
    # List orders
    response = client.get("/api/v1/paper/orders")
    assert response.status_code == 200
    orders_list = response.json()
    
    assert isinstance(orders_list, list)
    assert len(orders_list) >= 2  # At least the 2 we just submitted
    
    # Check that all orders in list have status FILLED
    for order in orders_list[:2]:  # Check first 2 (most recent)
        assert order["status"] == "FILLED"


def test_paper_orders_submit_single(client: TestClient):
    """Test submitting a single order (not wrapped in list)."""
    # Reset first
    client.post("/api/v1/paper/reset")
    
    # Submit single order (not in list)
    order_data = {
        "symbol": "MSFT",
        "side": "BUY",
        "quantity": 20.0,
        "price": 300.0
    }
    
    response = client.post("/api/v1/paper/orders", json=order_data)
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["symbol"] == "MSFT"
    assert data[0]["status"] == "FILLED"


def test_paper_positions_aggregate_correctly(client: TestClient):
    """Test that positions aggregate correctly from orders."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit BUY 10 AAPL
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        }]
    )
    assert response.status_code == 200
    
    # Submit SELL 4 AAPL
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "SELL",
            "quantity": 4.0,
            "price": 151.0
        }]
    )
    assert response.status_code == 200
    
    # Get positions
    response = client.get("/api/v1/paper/positions")
    assert response.status_code == 200
    positions = response.json()
    
    assert isinstance(positions, list)
    
    # Find AAPL position
    aapl_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
    assert aapl_position is not None
    assert aapl_position["quantity"] == 6.0  # 10 - 4 = 6


def test_paper_reset_clears_state(client: TestClient):
    """Test that reset clears all orders and positions."""
    # Submit some orders
    response = client.post(
        "/api/v1/paper/orders",
        json=[
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0
            },
            {
                "symbol": "GOOGL",
                "side": "BUY",
                "quantity": 5.0,
                "price": 2500.0
            }
        ]
    )
    assert response.status_code == 200
    
    # Verify orders exist
    response = client.get("/api/v1/paper/orders")
    assert response.status_code == 200
    orders_before = response.json()
    assert len(orders_before) >= 2
    
    # Verify positions exist
    response = client.get("/api/v1/paper/positions")
    assert response.status_code == 200
    positions_before = response.json()
    assert len(positions_before) >= 2
    
    # Reset
    response = client.post("/api/v1/paper/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # Verify orders are cleared
    response = client.get("/api/v1/paper/orders")
    assert response.status_code == 200
    orders_after = response.json()
    assert len(orders_after) == 0
    
    # Verify positions are cleared
    response = client.get("/api/v1/paper/positions")
    assert response.status_code == 200
    positions_after = response.json()
    assert len(positions_after) == 0


def test_paper_orders_list_with_limit(client: TestClient):
    """Test listing orders with limit parameter."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit 5 orders
    orders_data = [
        {
            "symbol": f"STOCK{i}",
            "side": "BUY",
            "quantity": 1.0,
            "price": 100.0
        }
        for i in range(5)
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # List with limit=2
    response = client.get("/api/v1/paper/orders?limit=2")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) == 2
    
    # List with limit=10 (should return all 5)
    response = client.get("/api/v1/paper/orders?limit=10")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) == 5


def test_paper_order_invalid_quantity(client: TestClient):
    """Test that orders with invalid quantity are rejected by Pydantic validation."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit order with zero quantity - should be rejected by Pydantic (422)
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 0.0,  # Invalid: must be > 0
            "price": 150.0
        }]
    )
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation failed)
    
    # Submit order with negative quantity - should also be rejected
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": -10.0,  # Invalid: must be > 0
            "price": 150.0
        }]
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_paper_order_client_order_id(client: TestClient):
    """Test that client_order_id is preserved."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit order with client_order_id
    client_order_id = "my-custom-order-123"
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0,
            "client_order_id": client_order_id
        }]
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["client_order_id"] == client_order_id
    assert data[0]["status"] == "FILLED"


def test_paper_orders_respect_kill_switch(client: TestClient, monkeypatch):
    """Test that paper orders respect kill switch."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Set kill switch via environment variable
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH", "1")
    
    # Submit orders - all should be rejected
    response = client.post(
        "/api/v1/paper/orders",
        json=[
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0
            },
            {
                "symbol": "GOOGL",
                "side": "BUY",
                "quantity": 5.0,
                "price": 2500.0
            }
        ]
    )
    assert response.status_code == 200
    data = response.json()
    
    assert len(data) == 2
    # All orders should be rejected
    for order in data:
        assert order["status"] == "REJECTED"
        assert "KILL_SWITCH" in order["reason"]
    
    # Verify positions are not updated (no orders filled)
    response = client.get("/api/v1/paper/positions")
    assert response.status_code == 200
    positions = response.json()
    assert len(positions) == 0  # No positions should exist
    
    # Clear kill switch
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    
    # Now orders should go through
    response = client.post(
        "/api/v1/paper/orders",
        json=[{
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        }]
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "FILLED"


def test_paper_orders_respect_pre_trade_limits(client: TestClient, monkeypatch):
    """Test that paper orders respect pre-trade limits."""
    import os
    from src.assembled_core.api.routers import paper_trading
    from src.assembled_core.execution.pre_trade_checks import PreTradeConfig
    
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Temporarily set a PreTradeConfig with small max_notional_per_symbol
    original_config = paper_trading._DEFAULT_PRE_TRADE_CONFIG
    paper_trading._DEFAULT_PRE_TRADE_CONFIG = PreTradeConfig(
        max_notional_per_symbol=1000.0  # Small limit
    )
    
    try:
        # Submit a large order (should be rejected)
        response = client.post(
            "/api/v1/paper/orders",
            json=[{
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100.0,  # Large quantity
                "price": 150.0  # Notional = 15000 > 1000 limit
            }]
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "REJECTED"
        assert "PRE_TRADE_CHECK_FAILED" in data[0]["reason"]
        assert "max_notional" in data[0]["reason"].lower()
        
        # Verify position is not updated
        response = client.get("/api/v1/paper/positions")
        assert response.status_code == 200
        positions = response.json()
        # AAPL should not be in positions (order was rejected)
        aapl_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
        assert aapl_position is None
        
        # Submit a small order (should be filled)
        response = client.post(
            "/api/v1/paper/orders",
            json=[{
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 5.0,  # Small quantity
                "price": 150.0  # Notional = 750 < 1000 limit
            }]
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "FILLED"
        
        # Verify position is updated
        response = client.get("/api/v1/paper/positions")
        assert response.status_code == 200
        positions = response.json()
        aapl_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
        assert aapl_position is not None
        assert aapl_position["quantity"] == 5.0
    
    finally:
        # Restore original config
        paper_trading._DEFAULT_PRE_TRADE_CONFIG = original_config


def test_paper_positions_not_updated_for_rejected_orders(client: TestClient, monkeypatch):
    """Test that rejected orders do not update positions."""
    import os
    from src.assembled_core.api.routers import paper_trading
    from src.assembled_core.execution.pre_trade_checks import PreTradeConfig
    
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Set a PreTradeConfig with small max_notional_per_symbol
    original_config = paper_trading._DEFAULT_PRE_TRADE_CONFIG
    paper_trading._DEFAULT_PRE_TRADE_CONFIG = PreTradeConfig(
        max_notional_per_symbol=1000.0
    )
    
    try:
        # Submit a valid order first (should be filled)
        response = client.post(
            "/api/v1/paper/orders",
            json=[{
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 5.0,
                "price": 150.0  # Notional = 750 < 1000
            }]
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["status"] == "FILLED"
        
        # Verify position exists
        response = client.get("/api/v1/paper/positions")
        positions = response.json()
        aapl_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
        assert aapl_position is not None
        initial_quantity = aapl_position["quantity"]
        
        # Submit a rejected order (too large)
        response = client.post(
            "/api/v1/paper/orders",
            json=[{
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100.0,
                "price": 150.0  # Notional = 15000 > 1000
            }]
        )
        assert response.status_code == 200
        data = response.json()
        assert data[0]["status"] == "REJECTED"
        
        # Verify position is unchanged
        response = client.get("/api/v1/paper/positions")
        positions = response.json()
        aapl_position = next((p for p in positions if p["symbol"] == "AAPL"), None)
        assert aapl_position is not None
        assert aapl_position["quantity"] == initial_quantity  # Unchanged
    
    finally:
        # Restore original config
        paper_trading._DEFAULT_PRE_TRADE_CONFIG = original_config

