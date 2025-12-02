"""Tests for OMS (Order Management System) API endpoints."""
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


def test_oms_blotter_reflects_paper_orders(client: TestClient):
    """Test that OMS blotter reflects orders from paper trading."""
    # Reset first
    response = client.post("/api/v1/paper/reset")
    assert response.status_code == 200
    
    # Submit 3 orders
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
        },
        {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 20.0,
            "price": 300.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Get blotter
    response = client.get("/api/v1/oms/blotter")
    assert response.status_code == 200
    blotter = response.json()
    
    assert isinstance(blotter, list)
    assert len(blotter) >= 3  # At least the 3 orders we just submitted
    
    # Check that all orders have required fields
    for order in blotter[:3]:  # Check first 3 (most recent)
        assert "order_id" in order
        assert "symbol" in order
        assert "side" in order
        assert "quantity" in order
        assert "status" in order
        assert "created_at" in order
        
        # Status should be FILLED or REJECTED (not NEW)
        assert order["status"] in ["FILLED", "REJECTED"]
        
        # Route should be "PAPER"
        assert order.get("route") == "PAPER"


def test_oms_blotter_filter_by_symbol_and_status(client: TestClient):
    """Test blotter filtering by symbol and status."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit orders: use kill switch to create rejected orders
    # First, submit some orders that will be filled
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 5.0,
            "price": 150.0
        },
        {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 10.0,
            "price": 300.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Now submit orders with kill switch active to get rejected orders
    import os
    os.environ["ASSEMBLED_KILL_SWITCH"] = "1"
    
    try:
        rejected_orders_data = [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 150.0
            }
        ]
        
        response = client.post("/api/v1/paper/orders", json=rejected_orders_data)
        assert response.status_code == 200
        rejected_responses = response.json()
        
        # At least one should be rejected
        assert any(order["status"] == "REJECTED" for order in rejected_responses)
    
    finally:
        # Clear kill switch
        os.environ.pop("ASSEMBLED_KILL_SWITCH", None)
    
    # Filter by symbol=AAPL and status=FILLED
    response = client.get("/api/v1/oms/blotter?symbol=AAPL&status=FILLED")
    assert response.status_code == 200
    filtered = response.json()
    
    assert isinstance(filtered, list)
    assert len(filtered) >= 1  # At least one filled AAPL order
    
    # All should be AAPL and FILLED
    for order in filtered:
        assert order["symbol"] == "AAPL"
        assert order["status"] == "FILLED"
    
    # Filter by symbol=AAPL and status=REJECTED
    response = client.get("/api/v1/oms/blotter?symbol=AAPL&status=REJECTED")
    assert response.status_code == 200
    rejected = response.json()
    
    assert isinstance(rejected, list)
    # May have rejected orders if kill switch was active
    for order in rejected:
        assert order["symbol"] == "AAPL"
        assert order["status"] == "REJECTED"
    
    # Filter by symbol=MSFT (should get filled orders)
    response = client.get("/api/v1/oms/blotter?symbol=MSFT")
    assert response.status_code == 200
    msft_orders = response.json()
    
    assert isinstance(msft_orders, list)
    assert len(msft_orders) >= 1  # At least one MSFT order
    
    for order in msft_orders:
        assert order["symbol"] == "MSFT"


def test_oms_blotter_filter_by_route(client: TestClient):
    """Test blotter filtering by route."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit orders
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Filter by route=PAPER
    response = client.get("/api/v1/oms/blotter?route=PAPER")
    assert response.status_code == 200
    filtered = response.json()
    
    assert isinstance(filtered, list)
    assert len(filtered) >= 1
    
    # All should have route=PAPER
    for order in filtered:
        assert order.get("route") == "PAPER"


def test_oms_blotter_limit(client: TestClient):
    """Test blotter limit parameter."""
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
    
    # Get blotter with limit=2
    response = client.get("/api/v1/oms/blotter?limit=2")
    assert response.status_code == 200
    blotter = response.json()
    
    assert isinstance(blotter, list)
    assert len(blotter) == 2


def test_oms_executions_match_filled_orders(client: TestClient):
    """Test that executions match filled orders."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit 2 orders (both should be filled)
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
    submitted_orders = response.json()
    
    # All should be filled
    filled_order_ids = {order["order_id"] for order in submitted_orders if order["status"] == "FILLED"}
    
    # Get executions
    response = client.get("/api/v1/oms/executions")
    assert response.status_code == 200
    executions = response.json()
    
    assert isinstance(executions, list)
    assert len(executions) >= len(filled_order_ids)
    
    # Check that all filled orders have corresponding executions
    execution_order_ids = {exec["order_id"] for exec in executions}
    assert filled_order_ids.issubset(execution_order_ids)
    
    # Check execution structure
    for exec in executions:
        assert "exec_id" in exec
        assert exec["exec_id"].startswith("EXEC-")
        assert "order_id" in exec
        assert "symbol" in exec
        assert "side" in exec
        assert "quantity" in exec
        assert "timestamp" in exec
        assert exec.get("route") == "PAPER"


def test_oms_executions_filter_by_symbol(client: TestClient):
    """Test executions filtering by symbol."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit orders for different symbols
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        },
        {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 20.0,
            "price": 300.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Filter executions by symbol=AAPL
    response = client.get("/api/v1/oms/executions?symbol=AAPL")
    assert response.status_code == 200
    filtered = response.json()
    
    assert isinstance(filtered, list)
    assert len(filtered) >= 1
    
    # All should be AAPL
    for exec in filtered:
        assert exec["symbol"] == "AAPL"


def test_oms_executions_filter_by_route(client: TestClient):
    """Test executions filtering by route."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit orders
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Filter by route=PAPER
    response = client.get("/api/v1/oms/executions?route=PAPER")
    assert response.status_code == 200
    filtered = response.json()
    
    assert isinstance(filtered, list)
    assert len(filtered) >= 1
    
    # All should have route=PAPER
    for exec in filtered:
        assert exec.get("route") == "PAPER"


def test_oms_executions_limit(client: TestClient):
    """Test executions limit parameter."""
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
    
    # Get executions with limit=2
    response = client.get("/api/v1/oms/executions?limit=2")
    assert response.status_code == 200
    executions = response.json()
    
    assert isinstance(executions, list)
    assert len(executions) == 2


def test_oms_routes_lists_paper_route(client: TestClient):
    """Test that routes endpoint lists the paper route."""
    # Get routes
    response = client.get("/api/v1/oms/routes")
    assert response.status_code == 200
    routes = response.json()
    
    assert isinstance(routes, list)
    assert len(routes) >= 1
    
    # Find PAPER route
    paper_route = next((r for r in routes if r["route_id"] == "PAPER"), None)
    assert paper_route is not None
    assert paper_route["route_id"] == "PAPER"
    assert paper_route["description"] == "Internal paper trading route"
    assert paper_route["is_default"] is True
    
    # Check route structure
    for route in routes:
        assert "route_id" in route
        assert "description" in route
        assert "is_default" in route


def test_oms_blotter_filter_by_route_alt(client: TestClient):
    """Test blotter filtering by alternative route."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit orders with different routes
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0,
            "route": "PAPER"
        },
        {
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 5.0,
            "price": 300.0,
            "route": "PAPER"
        },
        {
            "symbol": "GOOGL",
            "side": "SELL",
            "quantity": 3.0,
            "price": 2500.0,
            "route": "PAPER_ALT"  # Different route
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Filter by route=PAPER_ALT
    response = client.get("/api/v1/oms/blotter?route=PAPER_ALT")
    assert response.status_code == 200
    filtered = response.json()
    
    assert isinstance(filtered, list)
    assert len(filtered) == 1  # Exactly one order with PAPER_ALT route
    
    # Check that all have route=PAPER_ALT
    for order in filtered:
        assert order.get("route") == "PAPER_ALT"
        assert order["symbol"] == "GOOGL"


def test_oms_blotter_includes_source_field(client: TestClient):
    """Test that blotter includes source field from orders."""
    # Reset
    client.post("/api/v1/paper/reset")
    
    # Submit order with source
    orders_data = [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10.0,
            "price": 150.0,
            "source": "MANUAL_TEST"
        }
    ]
    
    response = client.post("/api/v1/paper/orders", json=orders_data)
    assert response.status_code == 200
    
    # Get blotter
    response = client.get("/api/v1/oms/blotter")
    assert response.status_code == 200
    blotter = response.json()
    
    assert isinstance(blotter, list)
    assert len(blotter) >= 1
    
    # First entry (newest) should be our order
    first_order = blotter[0]
    assert first_order.get("source") == "MANUAL_TEST"
    assert first_order["symbol"] == "AAPL"

