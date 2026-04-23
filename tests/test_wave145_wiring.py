"""Tests for wave-145 module wiring into trading_cycle.py.

Covers:
  Step api.1 — api.app (create_app)
  Step api.2 — api.models (SignalType / Signal)
  Step cfg.1 — config package (OUTPUT_DIR / SUPPORTED_FREQS)
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS

# FastAPI may not be installed in this environment
try:
    from src.assembled_core.api.app import create_app
    from src.assembled_core.api.models import SignalType, Signal, OrderSide
    _FASTAPI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    create_app = None  # type: ignore[assignment]
    SignalType = None  # type: ignore[assignment]
    Signal = None  # type: ignore[assignment]
    OrderSide = None  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# api.app (Step api.1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_create_app_importable():
    assert create_app is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_create_app_returns_fastapi():
    app = create_app()
    assert app is not None
    assert hasattr(app, "routes")


# ---------------------------------------------------------------------------
# api.models (Step api.2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_signal_type_importable():
    assert SignalType is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_signal_type_has_members():
    assert len(SignalType) > 0


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_order_side_importable():
    assert OrderSide is not None


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
def test_signal_importable():
    assert Signal is not None


# ---------------------------------------------------------------------------
# config package (Step cfg.1)
# ---------------------------------------------------------------------------

def test_output_dir_importable():
    assert OUTPUT_DIR is not None
    assert isinstance(OUTPUT_DIR, Path)


def test_supported_freqs_importable():
    assert SUPPORTED_FREQS is not None
    assert len(SUPPORTED_FREQS) > 0
    assert "1d" in SUPPORTED_FREQS
