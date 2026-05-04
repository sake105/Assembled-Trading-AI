"""Tests for congress trading features module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.congress_features import (
    add_congress_features,
    compute_congress_net_buy_score,
)


def _synthetic_trades(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = rng.choice(["AAPL", "MSFT", "GOOG", "AMZN"], n)
    dates = pd.date_range("2023-06-01", periods=180, freq="D")
    trade_dates = rng.choice(dates, n)
    types = rng.choice(["purchase", "sale"], n)
    amounts = rng.choice([1000, 15000, 50000, 100000], n)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "transaction_date": trade_dates,
            "type": types,
            "amount": amounts,
            "representative": [f"Rep_{i % 10}" for i in range(n)],
        }
    )


@pytest.mark.phase12
class TestComputeCongressNetBuyScore:
    def test_basic(self):
        trades = _synthetic_trades()
        result = compute_congress_net_buy_score(trades)
        assert isinstance(result, (pd.DataFrame, pd.Series, dict))

    def test_empty_trades(self):
        result = compute_congress_net_buy_score(pd.DataFrame())
        assert isinstance(result, (pd.DataFrame, pd.Series, dict))


@pytest.mark.phase12
class TestAddCongressFeatures:
    def test_basic_v2(self):
        trades = _synthetic_trades()
        panel = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
                "timestamp": list(pd.bdate_range("2023-12-01", periods=10)) * 2,
                "close": np.random.default_rng(1).normal(150, 10, 20),
            }
        )
        result = add_congress_features(panel, trades)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(panel)

    def test_empty_trades_v2(self):
        panel = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "timestamp": pd.bdate_range("2024-01-01", periods=5),
                "close": [150.0] * 5,
            }
        )
        result = add_congress_features(panel, pd.DataFrame())
        assert len(result) == 5

    def test_no_matching_symbols(self):
        trades = _synthetic_trades()
        panel = pd.DataFrame(
            {
                "symbol": ["XYZ"] * 5,
                "timestamp": pd.bdate_range("2024-01-01", periods=5),
                "close": [50.0] * 5,
            }
        )
        result = add_congress_features(panel, trades)
        assert len(result) == 5
