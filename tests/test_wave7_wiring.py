"""Tests for wave-7 module wiring into trading_cycle.py.

Covers:
  Step 2.1 — qa.point_in_time_checks (check_features_pit_safe)
  Step 2.5 — features.behavioral_features (compute_behavioral_composite)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.point_in_time_checks import check_features_pit_safe
from src.assembled_core.features.behavioral_features import (
    capital_gains_overhang,
    anchoring_52w_high,
    abnormal_volume,
    compute_behavioral_composite,
)


# ---------------------------------------------------------------------------
# point_in_time_checks (Step 2.1)
# ---------------------------------------------------------------------------

def _make_feature_df(n_days: int = 10, tz: str = "UTC") -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days, tz=tz),
        "symbol": ["A"] * n_days,
        "factor": [float(i) for i in range(n_days)],
    })


def test_pit_check_all_past_returns_true():
    df = _make_feature_df(5)
    as_of = pd.Timestamp("2024-01-10", tz="UTC")
    assert check_features_pit_safe(df, as_of) is True


def test_pit_check_future_rows_returns_false():
    df = _make_feature_df(10)
    as_of = pd.Timestamp("2024-01-03", tz="UTC")
    assert check_features_pit_safe(df, as_of) is False


def test_pit_check_none_as_of_passes():
    df = _make_feature_df(10)
    assert check_features_pit_safe(df, None) is True


def test_pit_check_missing_timestamp_col_passes():
    df = pd.DataFrame({"symbol": ["A", "B"], "factor": [1.0, 2.0]})
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    assert check_features_pit_safe(df, as_of, timestamp_col="timestamp") is True


def test_pit_check_strict_raises_on_violation():
    from src.assembled_core.qa.point_in_time_checks import PointInTimeViolationError
    df = _make_feature_df(10)
    as_of = pd.Timestamp("2024-01-03", tz="UTC")
    with pytest.raises(PointInTimeViolationError):
        check_features_pit_safe(df, as_of, strict=True)


def test_pit_check_exact_boundary_passes():
    df = _make_feature_df(5)
    # as_of exactly equals the last timestamp → should pass
    as_of = pd.Timestamp("2024-01-05", tz="UTC")
    assert check_features_pit_safe(df, as_of) is True


# ---------------------------------------------------------------------------
# behavioral_features (Step 2.5)
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 80, seed: int = 0) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
    volumes = pd.Series(np.abs(rng.normal(1e6, 2e5, n)))
    returns = prices.pct_change().fillna(0)
    return prices, volumes, returns


def test_capital_gains_overhang_returns_series():
    prices, volumes, _ = _make_price_series()
    cgo = capital_gains_overhang(prices, volumes)
    assert isinstance(cgo, pd.Series)
    assert len(cgo) == len(prices)


def test_anchoring_52w_high_returns_series():
    prices, _, _ = _make_price_series()
    anchor = anchoring_52w_high(prices)
    assert isinstance(anchor, pd.Series)


def test_abnormal_volume_returns_series():
    _, volumes, _ = _make_price_series()
    abn = abnormal_volume(volumes)
    assert isinstance(abn, pd.Series)


def test_behavioral_composite_shape():
    prices, volumes, returns = _make_price_series(80)
    composite = compute_behavioral_composite(prices, volumes, returns)
    assert isinstance(composite, pd.Series)
    assert len(composite) == len(prices)


def test_behavioral_composite_finite_at_end():
    prices, volumes, returns = _make_price_series(100)
    composite = compute_behavioral_composite(prices, volumes, returns)
    last = composite.iloc[-1]
    assert pd.notna(last)


def test_behavioral_composite_custom_weights():
    prices, volumes, returns = _make_price_series(100)
    w1 = compute_behavioral_composite(prices, volumes, returns)
    w2 = compute_behavioral_composite(prices, volumes, returns, weights={"cgo": 1.0, "anchor_52w": 0.0, "abn_vol": 0.0, "max_effect": 0.0, "abn_turnover": 0.0})
    # Different weights should (generically) produce different scores
    assert isinstance(w2, pd.Series)
