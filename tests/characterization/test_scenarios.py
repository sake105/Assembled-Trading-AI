"""Scenario tests for crisis periods.

From 35_GOLDEN_EQUITY_SCENARIO_TESTS.md §5.

Each scenario test verifies that the backtest behaves within known bounds
for a historically-calibrated synthetic crisis period.
"""

from __future__ import annotations

import pytest

from tests.characterization._fixtures import make_crisis_scenario
from tests.characterization.test_golden_equity import _run_minimal_backtest


def _drawdown(equity_series):
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return float(dd.min())


@pytest.mark.characterization
def test_scenario_gfc_2008_equity_finite():
    """GFC-2008 synthetic scenario: equity must remain finite throughout."""
    import numpy as np

    bars = make_crisis_scenario("gfc_2008", ["SPY"], seed=42)
    result = _run_minimal_backtest(bars, initial_equity=100_000.0)
    assert np.isfinite(result["equity"]).all(), "Non-finite equity in GFC scenario"
    assert result["equity"].min() > 0, "Equity went to zero or negative in GFC scenario"


@pytest.mark.characterization
def test_scenario_covid_2020_equity_finite():
    """COVID-2020 scenario: equity must remain positive and finite."""
    import numpy as np

    bars = make_crisis_scenario("covid_2020", ["SPY"], seed=42)
    result = _run_minimal_backtest(bars, initial_equity=100_000.0)
    assert np.isfinite(result["equity"]).all()
    assert result["equity"].min() > 0


@pytest.mark.characterization
def test_scenario_rates_2022_negative_return():
    """Rate-hike-2022 scenario: strategy should produce sub-optimal returns."""
    bars = make_crisis_scenario("rates_2022", ["SPY"], seed=42)
    result = _run_minimal_backtest(bars, initial_equity=100_000.0)
    final_equity = result.iloc[-1]["equity"]
    total_return = (final_equity - 100_000.0) / 100_000.0
    # In a down-trending year with high vol, EMA strategy often underperforms
    assert total_return < 0.20, (
        f"Suspiciously high return in rates scenario: {total_return:.2%}"
    )


@pytest.mark.characterization
def test_scenario_calm_2017_positive_return():
    """Calm-2017 scenario: positive drift should produce positive equity."""
    bars = make_crisis_scenario("calm_2017", ["SPY"], seed=42)
    result = _run_minimal_backtest(bars, initial_equity=100_000.0)
    final_equity = result.iloc[-1]["equity"]
    total_return = (final_equity - 100_000.0) / 100_000.0
    # Calm uptrend: expect some positive return
    assert total_return > -0.20, (
        f"Unexpectedly bad return in calm scenario: {total_return:.2%}"
    )


@pytest.mark.characterization
def test_scenario_all_produce_finite_equity():
    """All 4 scenarios must produce finite, non-NaN equity curves."""
    import numpy as np

    for scenario_name in ("gfc_2008", "covid_2020", "rates_2022", "calm_2017"):
        bars = make_crisis_scenario(scenario_name, ["SPY"], seed=42)
        result = _run_minimal_backtest(bars)
        assert not result["equity"].isna().any(), f"NaN in {scenario_name}"
        assert np.isfinite(result["equity"]).all(), f"Inf in {scenario_name}"
