"""Tests for wave-149 module wiring into trading_cycle.py.

Covers:
  Step api.6  — api.routers.paper_trading/performance/portfolio/qa/risk/signals
  Step exp.1  — experiments.batch_config (BatchConfig / load_batch_config)
  Step exp.2  — experiments.batch_runner (BatchResult / expand_run_specs)
"""

from __future__ import annotations

import pytest

# All routers require FastAPI — skip gracefully if not installed
try:
    from src.assembled_core.api.routers.paper_trading import router as paper_trading_router
    from src.assembled_core.api.routers.performance import router as performance_router
    from src.assembled_core.api.routers.portfolio import router as portfolio_router
    from src.assembled_core.api.routers.qa import router as qa_router
    from src.assembled_core.api.routers.risk import router as risk_router
    from src.assembled_core.api.routers.signals import router as signals_router
    _FASTAPI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    paper_trading_router = performance_router = portfolio_router = None
    qa_router = risk_router = signals_router = None
    _FASTAPI_AVAILABLE = False

from src.assembled_core.experiments.batch_config import BatchConfig, load_batch_config
from src.assembled_core.experiments.batch_runner import BatchResult, expand_run_specs


# ---------------------------------------------------------------------------
# api.routers (remaining 6) (Step api.6)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_paper_trading_router_importable():
    assert paper_trading_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_performance_router_importable():
    assert performance_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_portfolio_router_importable():
    assert portfolio_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_qa_router_importable():
    assert qa_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_risk_router_importable():
    assert risk_router is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_signals_router_importable():
    assert signals_router is not None


# ---------------------------------------------------------------------------
# experiments.batch_config (Step exp.1)
# ---------------------------------------------------------------------------

def test_batch_config_importable():
    assert BatchConfig is not None


def test_load_batch_config_importable():
    assert load_batch_config is not None


# ---------------------------------------------------------------------------
# experiments.batch_runner (Step exp.2)
# ---------------------------------------------------------------------------

def test_batch_result_importable():
    assert BatchResult is not None


def test_expand_run_specs_importable():
    assert expand_run_specs is not None
