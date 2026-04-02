"""Tests for enhanced stress scenarios in src.assembled_core.qa.scenario_engine."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.scenario_engine import (
    Scenario,
    apply_scenario_to_prices,
    run_crisis_scenarios,
    compare_crisis_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHOCK_DATE = datetime(2024, 3, 15, tzinfo=timezone.utc)
SHOCK_END = datetime(2024, 3, 25, tzinfo=timezone.utc)


def make_prices(symbols: list[str], n_days: int = 30) -> pd.DataFrame:
    """Create a simple price DataFrame with constant prices for testing."""
    rng = np.random.default_rng(7)
    rows = []
    for sym in symbols:
        base = 100.0 + rng.uniform(0, 50)
        for i in range(n_days):
            ts = pd.Timestamp("2024-03-01", tz="UTC") + pd.Timedelta(days=i)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "close": base * (1 + rng.normal(0, 0.005)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# oil_spike
# ---------------------------------------------------------------------------


def test_oil_spike_raises_energy():
    """Energy symbols should have higher prices after oil_spike."""
    energy_syms = ["XLE", "XOM", "CVX", "USO"]
    other_syms = ["AAPL", "MSFT"]
    prices = make_prices(energy_syms + other_syms, n_days=30)

    scenario = Scenario(
        name="oil_spike_test",
        shock_type="oil_spike",
        shock_magnitude=0.15,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    # For each energy symbol: shocked price in window > baseline price in window
    for sym in energy_syms:
        orig = prices[
            (prices["symbol"] == sym)
            & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (prices["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        new = shocked[
            (shocked["symbol"] == sym)
            & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (shocked["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        assert new > orig, f"{sym}: expected shock to raise price, got orig={orig:.2f} new={new:.2f}"


def test_oil_spike_drags_non_energy():
    """Non-energy symbols should have lower prices (negative drag) after oil_spike."""
    energy_syms = ["XLE", "XOM"]
    other_syms = ["AAPL", "MSFT"]
    prices = make_prices(energy_syms + other_syms, n_days=30)

    scenario = Scenario(
        name="oil_spike_drag_test",
        shock_type="oil_spike",
        shock_magnitude=0.15,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    for sym in other_syms:
        orig = prices[
            (prices["symbol"] == sym)
            & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (prices["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        new = shocked[
            (shocked["symbol"] == sym)
            & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (shocked["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        assert new < orig, f"{sym}: expected drag to lower price, got orig={orig:.2f} new={new:.2f}"


def test_oil_spike_custom_affected_symbols():
    """Custom affected_symbols should override defaults."""
    prices = make_prices(["AAA", "BBB", "CCC"], n_days=30)
    scenario = Scenario(
        name="oil_custom",
        shock_type="oil_spike",
        shock_magnitude=0.20,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
        affected_symbols=["AAA"],
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    # AAA should go up
    orig_aaa = prices[
        (prices["symbol"] == "AAA") & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
    ]["close"].mean()
    new_aaa = shocked[
        (shocked["symbol"] == "AAA") & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
    ]["close"].mean()
    assert new_aaa > orig_aaa


# ---------------------------------------------------------------------------
# gold_flight
# ---------------------------------------------------------------------------


def test_gold_flight_raises_gold():
    """Gold symbols should receive a positive shock in gold_flight scenario."""
    gold_syms = ["GLD", "IAU", "GC"]
    other_syms = ["SPY", "QQQ"]
    prices = make_prices(gold_syms + other_syms, n_days=30)

    scenario = Scenario(
        name="gold_flight_test",
        shock_type="gold_flight",
        shock_magnitude=0.10,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    for sym in gold_syms:
        orig = prices[
            (prices["symbol"] == sym)
            & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (prices["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        new = shocked[
            (shocked["symbol"] == sym)
            & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (shocked["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        assert new > orig, f"{sym}: expected gold to rise, got orig={orig:.2f} new={new:.2f}"


# ---------------------------------------------------------------------------
# defense_surge
# ---------------------------------------------------------------------------


def test_defense_surge_raises_defense():
    """Defense symbols should receive a positive shock in defense_surge."""
    defense_syms = ["LMT", "RTX"]
    other_syms = ["AAPL"]
    prices = make_prices(defense_syms + other_syms, n_days=30)

    scenario = Scenario(
        name="defense_surge_test",
        shock_type="defense_surge",
        shock_magnitude=0.08,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    for sym in defense_syms:
        orig = prices[
            (prices["symbol"] == sym)
            & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (prices["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        new = shocked[
            (shocked["symbol"] == sym)
            & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
            & (shocked["timestamp"] <= pd.Timestamp(SHOCK_END))
        ]["close"].mean()
        assert new > orig, f"{sym}: expected defense to rise"


def test_defense_surge_neutral_for_others():
    """Non-defense symbols should be unchanged in defense_surge (multiplier=0)."""
    prices = make_prices(["LMT", "AAPL"], n_days=30)
    scenario = Scenario(
        name="defense_neutral_test",
        shock_type="defense_surge",
        shock_magnitude=0.08,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    orig_aapl = prices[prices["symbol"] == "AAPL"]["close"].to_numpy()
    new_aapl = shocked[shocked["symbol"] == "AAPL"]["close"].to_numpy()
    np.testing.assert_allclose(orig_aapl, new_aapl, rtol=1e-9)


# ---------------------------------------------------------------------------
# geopolitical_shock
# ---------------------------------------------------------------------------


def test_geopolitical_shock_energy_up():
    """Energy symbols should be positive under geopolitical_shock."""
    prices = make_prices(["XLE", "XOM", "GLD", "AAPL"], n_days=30)
    scenario = Scenario(
        name="geo_shock_test",
        shock_type="geopolitical_shock",
        shock_magnitude=0.10,
        shock_start=SHOCK_DATE,
        shock_end=SHOCK_END,
    )
    shocked = apply_scenario_to_prices(prices, scenario)

    for sym in ["XLE", "XOM"]:
        orig = prices[
            (prices["symbol"] == sym)
            & (prices["timestamp"] >= pd.Timestamp(SHOCK_DATE))
        ]["close"].mean()
        new = shocked[
            (shocked["symbol"] == sym)
            & (shocked["timestamp"] >= pd.Timestamp(SHOCK_DATE))
        ]["close"].mean()
        assert new > orig, f"Expected {sym} to rise in geopolitical shock"


# ---------------------------------------------------------------------------
# run_crisis_scenarios
# ---------------------------------------------------------------------------


def test_geopolitical_shock_returns_multiple():
    """run_crisis_scenarios for geopolitical_escalation should return multiple scenarios."""
    prices = make_prices(["XLE", "GLD", "LMT", "AAPL", "SPY"], n_days=30)
    results = run_crisis_scenarios(prices, "geopolitical_escalation", SHOCK_DATE)
    assert isinstance(results, dict)
    assert len(results) >= 3, f"Expected >= 3 scenarios, got {len(results)}"


def test_run_crisis_scenarios_all_types():
    """All four crisis types should run without error."""
    prices = make_prices(["XLE", "GLD", "LMT", "AAPL", "SPY"], n_days=30)
    for crisis in ["geopolitical_escalation", "energy_shock", "cyber_attack", "financial_stress"]:
        results = run_crisis_scenarios(prices, crisis, SHOCK_DATE)
        assert isinstance(results, dict)
        assert len(results) > 0, f"{crisis} returned empty results"


def test_run_crisis_scenarios_unknown_raises():
    """Unknown crisis_type should raise ValueError."""
    prices = make_prices(["AAPL"], n_days=10)
    with pytest.raises(ValueError, match="Unknown crisis_type"):
        run_crisis_scenarios(prices, "alien_invasion", SHOCK_DATE)


def test_run_crisis_scenarios_returns_dataframes():
    """Each value in the returned dict should be a DataFrame."""
    prices = make_prices(["XLE", "GLD", "AAPL"], n_days=30)
    results = run_crisis_scenarios(prices, "energy_shock", SHOCK_DATE)
    for name, df in results.items():
        assert isinstance(df, pd.DataFrame), f"{name} result is not a DataFrame"
        assert "close" in df.columns


# ---------------------------------------------------------------------------
# compare_crisis_scenarios
# ---------------------------------------------------------------------------


def test_compare_crisis_scenarios_shape():
    """compare_crisis_scenarios should return DataFrame with expected columns."""
    prices = make_prices(["XLE", "GLD", "AAPL", "SPY"], n_days=40)
    shocked_scenarios = run_crisis_scenarios(prices, "financial_stress", SHOCK_DATE)

    baseline_pivot = prices.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    )
    baseline_equity = baseline_pivot.mean(axis=1)

    comparison = compare_crisis_scenarios(baseline_equity, shocked_scenarios, prices)
    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison.columns) == {"scenario_name", "total_return", "max_drawdown", "sharpe"}
    assert len(comparison) == len(shocked_scenarios)


def test_compare_crisis_scenarios_scenario_names():
    """scenario_name column should match the keys of the shocked_scenarios dict."""
    prices = make_prices(["XLE", "GLD", "AAPL"], n_days=40)
    shocked = run_crisis_scenarios(prices, "energy_shock", SHOCK_DATE)

    baseline_pivot = prices.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    )
    baseline_equity = baseline_pivot.mean(axis=1)

    comparison = compare_crisis_scenarios(baseline_equity, shocked, prices)
    returned_names = set(comparison["scenario_name"].tolist())
    expected_names = set(shocked.keys())
    assert returned_names == expected_names


def test_compare_crisis_scenarios_numeric_columns():
    """total_return, max_drawdown, sharpe should be numeric."""
    prices = make_prices(["XLE", "AAPL", "SPY"], n_days=40)
    shocked = run_crisis_scenarios(prices, "cyber_attack", SHOCK_DATE)

    baseline_pivot = prices.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    )
    baseline_equity = baseline_pivot.mean(axis=1)

    comparison = compare_crisis_scenarios(baseline_equity, shocked, prices)
    for col in ["total_return", "max_drawdown", "sharpe"]:
        assert pd.api.types.is_numeric_dtype(comparison[col]), f"{col} is not numeric"
