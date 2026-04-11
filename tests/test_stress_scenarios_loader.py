"""Tests for stress scenarios YAML loader (Sprint 3 / C6)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.scenario_engine import (
    apply_scenario_to_prices,
    load_scenarios_from_yaml,
)

# yaml is an optional dep in some environments; skip if missing.
pytest.importorskip("yaml")

CONFIG_PATH = str(ROOT / "configs" / "stress_scenarios.yaml")


def test_config_file_exists_and_loads() -> None:
    scenarios = load_scenarios_from_yaml(CONFIG_PATH)
    assert len(scenarios) == 6
    names = [s.name for s in scenarios]
    assert "2008_lehman" in names
    assert "2020_covid" in names
    assert "2022_bonds" in names


def test_scenarios_have_valid_shock_types() -> None:
    scenarios = load_scenarios_from_yaml(CONFIG_PATH)
    valid = {"equity_crash", "vol_spike"}  # only those used in this config
    for s in scenarios:
        assert s.shock_type in valid
        assert s.shock_magnitude != 0


def test_missing_file_returns_empty_list() -> None:
    out = load_scenarios_from_yaml("does/not/exist.yaml")
    assert out == []


def test_loaded_scenario_can_be_applied_to_prices() -> None:
    """End-to-end: load → apply → shocked prices are lower for equity_crash."""
    scenarios = load_scenarios_from_yaml(CONFIG_PATH)
    covid = next(s for s in scenarios if s.name == "2020_covid")

    # Synthetic price series that straddles the covid window
    dates = pd.date_range("2020-01-01", "2020-04-30", freq="D", tz="UTC")
    prices = pd.DataFrame(
        [
            {"timestamp": d, "symbol": "SPY", "close": 300.0}
            for d in dates
        ]
    )

    shocked = apply_scenario_to_prices(prices, covid)
    # Within the crash window the shocked close should be well below 300
    in_window = shocked[
        (shocked["timestamp"] >= pd.Timestamp("2020-02-19", tz="UTC"))
        & (shocked["timestamp"] <= pd.Timestamp("2020-03-23", tz="UTC"))
    ]
    assert not in_window.empty
    # -34% crash → mean in-window must be at most ~0.66 * 300 = 198
    assert in_window["close"].mean() < 300.0 * 0.7


def test_invalid_rows_are_skipped(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
scenarios:
  - name: "good"
    shock_type: "equity_crash"
    shock_magnitude: -0.1
  - name: "unknown_type"
    shock_type: "nuclear_war"
    shock_magnitude: -1.0
  - name: "missing_mag"
    shock_type: "equity_crash"
""",
        encoding="utf-8",
    )
    out = load_scenarios_from_yaml(str(bad_yaml))
    assert len(out) == 1
    assert out[0].name == "good"


def test_dates_are_timezone_aware() -> None:
    scenarios = load_scenarios_from_yaml(CONFIG_PATH)
    for s in scenarios:
        if s.shock_start is not None:
            assert isinstance(s.shock_start, datetime)
            assert s.shock_start.tzinfo is not None
            assert s.shock_start.tzinfo.utcoffset(s.shock_start) == timezone.utc.utcoffset(s.shock_start)
