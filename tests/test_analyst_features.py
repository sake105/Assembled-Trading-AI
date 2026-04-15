"""Tests for analyst revision momentum features."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.analyst_features import (
    compute_eps_revision_score,
    compute_target_price_change,
    build_analyst_features,
    get_analyst_feature_names,
)


def _synthetic_estimates(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = rng.choice(["AAPL", "MSFT", "GOOG"], n)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "symbol": symbols,
        "date": rng.choice(dates, n),
        "eps_estimate": rng.normal(5.0, 1.0, n),
        "revision_direction": rng.choice(["up", "down", "unchanged"], n),
    })


@pytest.mark.phase12
class TestEpsRevisionScore:
    def test_basic(self):
        est = _synthetic_estimates()
        result = compute_eps_revision_score(
            est, "AAPL", pd.Timestamp("2024-02-15"),
        )
        assert "eps_revision_1m" in result
        assert "revision_breadth" in result
        assert "estimate_dispersion" in result

    def test_empty_data(self):
        result = compute_eps_revision_score(
            pd.DataFrame(), "AAPL", pd.Timestamp("2024-01-01"),
        )
        assert result["eps_revision_1m"] == 0.0

    def test_no_matching_symbol(self):
        est = _synthetic_estimates()
        result = compute_eps_revision_score(
            est, "XYZ", pd.Timestamp("2024-02-15"),
        )
        assert result["eps_revision_1m"] == 0.0


@pytest.mark.phase12
class TestTargetPriceChange:
    def test_increase(self):
        assert compute_target_price_change(110.0, 100.0) == pytest.approx(0.10)

    def test_decrease(self):
        assert compute_target_price_change(90.0, 100.0) == pytest.approx(-0.10)

    def test_zero_previous(self):
        assert compute_target_price_change(100.0, 0.0) == 0.0


@pytest.mark.phase12
class TestBuildAnalystFeatures:
    def test_basic(self):
        est = _synthetic_estimates()
        result = build_analyst_features(
            est, ["AAPL", "MSFT"], pd.Timestamp("2024-02-15"),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_empty(self):
        result = build_analyst_features(pd.DataFrame(), [], pd.Timestamp("2024-01-01"))
        assert len(result) == 0

    def test_feature_names(self):
        names = get_analyst_feature_names()
        assert len(names) == 4
        assert "eps_revision_1m" in names
