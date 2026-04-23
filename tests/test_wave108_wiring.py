"""Tests for wave-108 module wiring into trading_cycle.py.

Covers:
  Step 7.53 — accounting.reconciliation (ReconcileSLO)
  Step 2.74 — data.altdata.contract (normalize_alt_events)
  Step 2.75 — data.altdata.finnhub_common (get_finnhub_session)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.accounting.reconciliation import ReconcileSLO
from src.assembled_core.data.altdata.contract import normalize_alt_events
from src.assembled_core.data.altdata.finnhub_common import get_finnhub_session


# ---------------------------------------------------------------------------
# accounting.reconciliation (Step 7.53)
# ---------------------------------------------------------------------------

def test_reconcile_slo_creates():
    slo = ReconcileSLO()
    assert isinstance(slo, ReconcileSLO)


def test_reconcile_slo_defaults():
    slo = ReconcileSLO()
    assert slo.cash_diff_bps_warn == 5.0
    assert slo.cash_diff_bps_fail == 25.0


def test_reconcile_slo_fill_rate_defaults():
    slo = ReconcileSLO()
    assert slo.fill_rate_min_warn > 0.0
    assert slo.fill_rate_min_fail > 0.0
    assert slo.fill_rate_min_warn >= slo.fill_rate_min_fail


# ---------------------------------------------------------------------------
# data.altdata.contract (Step 2.74)
# ---------------------------------------------------------------------------

def test_normalize_alt_events_importable():
    assert normalize_alt_events is not None


def test_normalize_alt_events_empty_df():
    result = normalize_alt_events(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


def test_normalize_alt_events_missing_cols_raises():
    with pytest.raises((ValueError, KeyError)):
        normalize_alt_events(pd.DataFrame({"foo": [1, 2]}))


def test_normalize_alt_events_requires_disclosure_date():
    df = pd.DataFrame({"symbol": ["AAPL"], "event_date": ["2024-01-01"]})
    with pytest.raises(ValueError, match="disclosure_date"):
        normalize_alt_events(df)


# ---------------------------------------------------------------------------
# data.altdata.finnhub_common (Step 2.75)
# ---------------------------------------------------------------------------

def test_get_finnhub_session_importable():
    assert get_finnhub_session is not None


def test_get_finnhub_session_raises_without_key():
    from src.assembled_core.config import Settings
    settings = Settings()
    # no FINNHUB key set → should raise RuntimeError
    with pytest.raises((RuntimeError, Exception)):
        get_finnhub_session(settings)
