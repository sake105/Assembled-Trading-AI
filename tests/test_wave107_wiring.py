"""Tests for wave-107 module wiring into trading_cycle.py.

Covers:
  Step 7.50 — accounting.attribution (compute_cost_attribution)
  Step 7.51 — accounting.currency (FXConverter)
  Step 7.52 — accounting.ledger (events_from_orders)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.accounting.attribution import compute_cost_attribution
from src.assembled_core.accounting.currency import FXConverter
from src.assembled_core.accounting.ledger import events_from_orders


# ---------------------------------------------------------------------------
# accounting.attribution (Step 7.50)
# ---------------------------------------------------------------------------

def test_compute_cost_attribution_importable():
    assert compute_cost_attribution is not None


def test_compute_cost_attribution_empty_df():
    result = compute_cost_attribution(pd.DataFrame())
    assert isinstance(result, dict)


def test_compute_cost_attribution_has_per_symbol():
    result = compute_cost_attribution(pd.DataFrame())
    assert "per_symbol" in result


def test_compute_cost_attribution_has_total():
    result = compute_cost_attribution(pd.DataFrame())
    assert "total" in result


# ---------------------------------------------------------------------------
# accounting.currency (Step 7.51)
# ---------------------------------------------------------------------------

def test_fx_converter_creates():
    fx = FXConverter()
    assert isinstance(fx, FXConverter)


def test_fx_converter_has_rates():
    fx = FXConverter()
    assert isinstance(fx.rates, dict)
    assert len(fx.rates) > 0


def test_fx_converter_usd_to_usd():
    fx = FXConverter()
    result = fx.to_usd(100.0, "USD")
    assert result == pytest.approx(100.0)


def test_fx_converter_unknown_currency_raises():
    fx = FXConverter()
    with pytest.raises(ValueError):
        fx.to_usd(100.0, "XYZ_UNKNOWN")


# ---------------------------------------------------------------------------
# accounting.ledger (Step 7.52)
# ---------------------------------------------------------------------------

def test_events_from_orders_importable():
    assert events_from_orders is not None


def test_events_from_orders_empty_df():
    result = events_from_orders(pd.DataFrame(), run_id="test_run")
    assert isinstance(result, pd.DataFrame)


def test_events_from_orders_missing_cols_raises():
    with pytest.raises((ValueError, KeyError)):
        events_from_orders(pd.DataFrame({"symbol": ["AAPL"]}), run_id="test")
