"""Tests for wave-106 module wiring into trading_cycle.py.

Covers:
  Step 2.70 — features.earnings_insider_wrapper (compute_earnings_insider_factors)
  Step 2.71 — features.feature_flag_audit (audit_feature_flags / FEATURE_FLAGS)
  Step 2.72 — features.news_macro_wrapper (compute_news_macro_factors)
  Step 2.73 — features.shipping_features (add_shipping_features)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.features.earnings_insider_wrapper import compute_earnings_insider_factors
from src.assembled_core.features.feature_flag_audit import audit_feature_flags, FEATURE_FLAGS
from src.assembled_core.features.news_macro_wrapper import compute_news_macro_factors
from src.assembled_core.features.shipping_features import add_shipping_features


# ---------------------------------------------------------------------------
# earnings_insider_wrapper (Step 2.70)
# ---------------------------------------------------------------------------

def test_compute_earnings_insider_factors_importable():
    assert compute_earnings_insider_factors is not None


def test_compute_earnings_insider_factors_empty_universe():
    result = compute_earnings_insider_factors(
        as_of_date=pd.Timestamp("2024-06-01"),
        symbols=[],
        earnings_df=pd.DataFrame(columns=["symbol", "filing_date", "eps_actual", "eps_estimate"]),
        insider_df=pd.DataFrame(columns=["symbol", "filing_date", "transaction_type", "value_usd"]),
    )
    assert isinstance(result, pd.DataFrame)


def test_compute_earnings_insider_factors_returns_dataframe():
    result = compute_earnings_insider_factors(
        as_of_date=pd.Timestamp("2024-06-01"),
        symbols=["AAPL"],
        earnings_df=pd.DataFrame(columns=["symbol", "filing_date", "eps_actual", "eps_estimate"]),
        insider_df=pd.DataFrame(columns=["symbol", "filing_date", "transaction_type", "value_usd"]),
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# feature_flag_audit (Step 2.71)
# ---------------------------------------------------------------------------

def test_feature_flags_is_dict():
    assert isinstance(FEATURE_FLAGS, dict)


def test_feature_flags_has_ta_features():
    assert "ta_features" in FEATURE_FLAGS


def test_audit_feature_flags_empty_policy():
    result = audit_feature_flags({})
    assert isinstance(result, dict)


def test_audit_feature_flags_has_enabled_key():
    result = audit_feature_flags({})
    assert "enabled" in result


def test_audit_feature_flags_has_missing_key():
    result = audit_feature_flags({})
    assert "missing" in result


def test_audit_feature_flags_n_total():
    result = audit_feature_flags({})
    assert result.get("n_total", 0) == len(FEATURE_FLAGS)


# ---------------------------------------------------------------------------
# news_macro_wrapper (Step 2.72)
# ---------------------------------------------------------------------------

def test_compute_news_macro_factors_importable():
    assert compute_news_macro_factors is not None


def test_compute_news_macro_factors_raises_on_invalid_date():
    with pytest.raises((ValueError, TypeError)):
        compute_news_macro_factors(
            as_of_date="2024-06-01",  # not a pd.Timestamp
            symbols=["AAPL"],
            news_df=pd.DataFrame(),
            macro_df=pd.DataFrame(),
        )


# ---------------------------------------------------------------------------
# shipping_features (Step 2.73)
# ---------------------------------------------------------------------------

def test_add_shipping_features_importable():
    assert add_shipping_features is not None


def test_add_shipping_features_raises_on_missing_columns():
    with pytest.raises((KeyError, ValueError)):
        add_shipping_features(pd.DataFrame(), pd.DataFrame())
