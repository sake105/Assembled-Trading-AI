"""Tests for Barra-style factor risk model."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.risk.factor_risk_model")
from src.assembled_core.risk.factor_risk_model import (
    FactorRiskModel,
    check_factor_exposure_limits,
)


def _synthetic_factor_data(n_assets: int = 30, n_days: int = 200, seed: int = 42):
    """Create synthetic panel returns and factor exposures."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM_{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2022-01-01", periods=n_days)

    # Factor exposures — panel format with symbol/timestamp/factor columns
    factor_col_names = ["beta_market", "log_market_cap", "book_to_market",
                        "momentum_12m_excl_1m", "roe", "rv_20"]
    exp_rows = []
    for ts in dates:
        for sym in symbols:
            row = {"symbol": sym, "timestamp": ts}
            for fc in factor_col_names:
                row[fc] = rng.normal(0, 1)
            exp_rows.append(row)
    exposures = pd.DataFrame(exp_rows)

    # Returns — panel format with symbol/timestamp/return columns
    ret_rows = []
    for ts in dates:
        for sym in symbols:
            ret_rows.append({
                "symbol": sym, "timestamp": ts,
                "return": rng.normal(0, 0.02),
            })
    returns = pd.DataFrame(ret_rows)

    return returns, exposures, symbols


@pytest.mark.phase12
class TestFactorRiskModel:
    def test_fit(self):
        returns, exposures, symbols = _synthetic_factor_data()
        model = FactorRiskModel()
        model.fit(exposures, returns)
        assert model._is_fitted

    def test_predict_portfolio_vol(self):
        returns, exposures, symbols = _synthetic_factor_data()
        model = FactorRiskModel()
        model.fit(exposures, returns)
        weights = pd.Series(1.0 / len(symbols), index=symbols)
        vol = model.predict_portfolio_vol(weights)
        assert isinstance(vol, float)
        assert vol > 0

    def test_predict_factor_contributions(self):
        returns, exposures, symbols = _synthetic_factor_data()
        model = FactorRiskModel()
        model.fit(exposures, returns)
        weights = pd.Series(1.0 / len(symbols), index=symbols)
        contrib = model.predict_factor_contributions(weights)
        assert isinstance(contrib, pd.DataFrame)
        assert len(contrib) > 0

    def test_not_fitted_raises(self):
        model = FactorRiskModel()
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        with pytest.raises(RuntimeError):
            model.predict_portfolio_vol(weights)


@pytest.mark.phase12
class TestCheckFactorExposureLimits:
    def test_within_limits(self):
        symbols = ["A", "B", "C"]
        weights = pd.Series([0.4, 0.3, 0.3], index=symbols)
        exposures = pd.DataFrame(
            [[0.5, -0.2], [0.3, 0.1], [0.2, 0.4]],
            index=symbols, columns=["f1", "f2"],
        )
        result = check_factor_exposure_limits(weights, exposures, max_factor_exposure=1.0)
        assert isinstance(result, list)
        assert len(result) == 0  # no violations

    def test_exceeds_limits(self):
        symbols = ["A", "B"]
        weights = pd.Series([0.8, 0.2], index=symbols)
        exposures = pd.DataFrame(
            [[2.0, 0.1], [0.5, 0.1]],
            index=symbols, columns=["f1", "f2"],
        )
        result = check_factor_exposure_limits(weights, exposures, max_factor_exposure=0.5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("f1" in v["factor"] for v in result)
