"""Tests for deterministic sensitivity suite (RB3).

These tests verify that sensitivity variants are generated deterministically:
- Same input -> same variant order
- Deterministic variant generation
- Cost and slippage multipliers work correctly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import pytest; pytest.importorskip('src.assembled_core.qa.robustness')
from src.assembled_core.qa.robustness import (
    export_sensitivity_results,
    run_sensitivity_suite,
)


def test_run_sensitivity_suite_deterministic():
    """Test that same input produces identical variant order."""

    # Define simple backtest function
    def backtest_fn(config):
        commission = config.get("commission_bps", 0.0)
        spread = config.get("spread_w", 0.0)
        impact = config.get("impact_w", 0.0)
        # Simulate: higher costs -> lower sharpe
        sharpe = 1.5 - (commission + spread + impact) * 0.1
        return {
            "sharpe": sharpe,
            "cagr": 0.10,
            "max_drawdown": -0.05,
        }

    base_config = {
        "commission_bps": 1.0,
        "spread_w": 0.5,
        "impact_w": 1.0,
    }

    # Run suite twice
    results1 = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[-2, 0, 2],
        deterministic=True,
    )

    results2 = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[-2, 0, 2],
        deterministic=True,
    )

    # Assert identical results
    pd.testing.assert_frame_equal(results1, results2)

    # Verify variant names
    variant_names = results1["variant_name"].tolist()
    assert "baseline" in variant_names
    assert "costs_x2" in variant_names
    assert "slippage_x2" in variant_names
    assert "alt_delay_-2" in variant_names
    assert "alt_delay_+2" in variant_names
    # alt_delay_0 should be skipped (covered by baseline)

    # Verify deterministic ordering (sorted by variant_name)
    assert variant_names == sorted(variant_names)


def test_run_sensitivity_suite_costs_x2():
    """Test that costs_x2 variant doubles all cost parameters."""

    def backtest_fn(config):
        return {
            "sharpe": 1.0,
            "commission_bps": config.get("commission_bps", 0.0),
            "spread_w": config.get("spread_w", 0.0),
            "impact_w": config.get("impact_w", 0.0),
        }

    base_config = {
        "commission_bps": 1.0,
        "spread_w": 0.5,
        "impact_w": 1.0,
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[],  # No delay variants for this test
        deterministic=True,
    )

    # Find costs_x2 variant
    costs_x2_row = results[results["variant_name"] == "costs_x2"].iloc[0]

    # Verify costs are doubled
    assert costs_x2_row["commission_bps"] == 2.0
    assert costs_x2_row["spread_w"] == 1.0
    assert costs_x2_row["impact_w"] == 2.0


def test_run_sensitivity_suite_slippage_x2():
    """Test that slippage_x2 variant only doubles impact_w."""

    def backtest_fn(config):
        return {
            "sharpe": 1.0,
            "commission_bps": config.get("commission_bps", 0.0),
            "spread_w": config.get("spread_w", 0.0),
            "impact_w": config.get("impact_w", 0.0),
        }

    base_config = {
        "commission_bps": 1.0,
        "spread_w": 0.5,
        "impact_w": 1.0,
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[],  # No delay variants
        deterministic=True,
    )

    # Find slippage_x2 variant
    slippage_x2_row = results[results["variant_name"] == "slippage_x2"].iloc[0]

    # Verify only impact_w is doubled
    assert slippage_x2_row["commission_bps"] == 1.0  # Unchanged
    assert slippage_x2_row["spread_w"] == 0.5  # Unchanged
    assert slippage_x2_row["impact_w"] == 2.0  # Doubled


def test_run_sensitivity_suite_alt_delay():
    """Test that alt_delay variants apply delay to events_df."""

    def backtest_fn(config):
        events_df = config.get("events_df")
        if events_df is not None and not events_df.empty:
            # Check if delay was applied (by checking disclosure_date)
            # For this test, we'll just return a metric
            return {"sharpe": 1.0, "delay_applied": True}
        return {"sharpe": 1.0, "delay_applied": False}

    # Create synthetic events
    events_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "event_date": pd.to_datetime(["2020-01-01"], utc=True),
            "disclosure_date": pd.to_datetime(["2020-01-05"], utc=True),
        }
    )

    base_config: dict[str, Any] = {
        "events_df": events_df,
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[-2, 2],
        deterministic=True,
    )

    # Verify alt_delay variants exist
    assert "alt_delay_-2" in results["variant_name"].values
    assert "alt_delay_+2" in results["variant_name"].values

    # Verify delay=0 is skipped (covered by baseline)
    assert "alt_delay_0" not in results["variant_name"].values


def test_run_sensitivity_suite_negative_delay_warning():
    """Test that negative delay_days generates warning."""

    def backtest_fn(config):
        return {"sharpe": 1.0}

    base_config: dict[str, Any] = {
        "events_df": pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "event_date": pd.to_datetime(["2020-01-01"], utc=True),
                "disclosure_date": pd.to_datetime(["2020-01-05"], utc=True),
            }
        ),
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[-2],
        deterministic=True,
    )

    # Find alt_delay_-2 variant
    delay_row = results[results["variant_name"] == "alt_delay_-2"].iloc[0]

    # Verify warning is present
    assert "warnings" in delay_row
    assert delay_row["warnings"] is not None
    assert "leakage" in str(delay_row["warnings"]).lower()


def test_run_sensitivity_suite_handles_failures():
    """Test that failed variants are handled gracefully."""
    call_count = [0]

    def backtest_fn(config):
        call_count[0] += 1
        if call_count[0] == 2:  # Fail second variant
            raise ValueError("Simulated failure")
        return {"sharpe": 1.0}

    base_config: dict[str, Any] = {
        "commission_bps": 1.0,
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[],
        deterministic=True,
    )

    # Should have multiple variants
    assert len(results) > 1

    # One variant should have error
    assert "error" in results.columns
    assert results["error"].notna().sum() == 1


def test_run_sensitivity_suite_empty_delay_list():
    """Test that empty delay_days_list only runs cost variants."""

    def backtest_fn(config):
        return {"sharpe": 1.0}

    base_config: dict[str, Any] = {
        "commission_bps": 1.0,
    }

    results = run_sensitivity_suite(
        backtest_fn=backtest_fn,
        base_config=base_config,
        delay_days_list=[],
        deterministic=True,
    )

    # Should have baseline + costs_x2 + slippage_x2 = 3 variants
    assert len(results) == 3
    assert "baseline" in results["variant_name"].values
    assert "costs_x2" in results["variant_name"].values
    assert "slippage_x2" in results["variant_name"].values

    # No alt_delay variants
    assert not results["variant_name"].str.startswith("alt_delay").any()


def test_export_sensitivity_results_smoke(tmp_path: Path):
    """Test that export_sensitivity_results produces stable CSV file."""
    # Create synthetic results
    results_df = pd.DataFrame(
        {
            "variant_name": ["baseline", "costs_x2", "slippage_x2"],
            "sharpe": [1.5, 1.2, 1.3],
            "cagr": [0.15, 0.12, 0.13],
            "warnings": [None, None, None],
        }
    )

    # Export
    csv_path = export_sensitivity_results(
        results_df=results_df,
        output_dir=tmp_path,
        run_id="test_run",
    )

    # Verify file exists
    assert csv_path.exists()

    # Verify CSV is readable
    results_read = pd.read_csv(csv_path)
    assert len(results_read) == 3
    assert "variant_name" in results_read.columns
    assert "sharpe" in results_read.columns

    # Verify deterministic ordering (sorted by variant_name)
    assert list(results_read["variant_name"]) == sorted(results_read["variant_name"])
