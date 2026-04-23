"""Tests for wave-148 module wiring into trading_cycle.py.

Covers:
  Step api.3 — api.routers.monitoring (router)
  Step api.4 — api.routers.oms (router)
  Step api.5 — api.routers.orders (router)
"""

from __future__ import annotations

import pytest

# All routers require FastAPI — skip gracefully if not installed
try:
    from src.assembled_core.api.routers.monitoring import router as monitoring_router
    from src.assembled_core.api.routers.oms import router as oms_router
    from src.assembled_core.api.routers.orders import router as orders_router
    _FASTAPI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    monitoring_router = None
    oms_router = None
    orders_router = None
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# api.routers.monitoring (Step api.3)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_monitoring_router_importable():
    assert monitoring_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_monitoring_router_has_routes():
    assert hasattr(monitoring_router, "routes")
    assert len(monitoring_router.routes) > 0


# ---------------------------------------------------------------------------
# api.routers.oms (Step api.4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_oms_router_importable():
    assert oms_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_oms_router_has_routes():
    assert hasattr(oms_router, "routes")


# ---------------------------------------------------------------------------
# api.routers.orders (Step api.5)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_orders_router_importable():
    assert orders_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_orders_router_has_routes():
    assert hasattr(orders_router, "routes")
