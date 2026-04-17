"""R4 wiring — verify qa.signal_decay is consumed by the offline CI script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.phase_realism]

from scripts.compute_signal_decay_profile import (  # noqa: E402
    DEFAULT_FACTOR_COLS,
    _synthetic_factor_panel,
    build_report,
)
from src.assembled_core.qa import signal_decay as decay_mod  # noqa: E402


def test_signal_decay_module_has_expected_api() -> None:
    assert hasattr(decay_mod, "SignalDecayProfile")
    assert hasattr(decay_mod, "analyze_all_signals")
    assert hasattr(decay_mod, "compute_ic_series")


def test_build_report_contains_expected_factor_keys() -> None:
    panel = _synthetic_factor_panel(n_days=30, n_symbols=10)
    report = build_report(panel, DEFAULT_FACTOR_COLS, universe="unit-test")
    assert report["universe"] == "unit-test"
    assert set(report["factors"].keys()) == set(DEFAULT_FACTOR_COLS)
    for entry in report["factors"].values():
        assert "ic_mean" in entry
        assert "is_stale" in entry


def test_report_is_json_serialisable_and_matches_gate_contract(tmp_path: Path) -> None:
    panel = _synthetic_factor_panel(n_days=30, n_symbols=10)
    report = build_report(panel, DEFAULT_FACTOR_COLS)
    out = tmp_path / "latest.json"
    out.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    # Gate contract: strategies/signal_decay_gate.py reads 'factors' section
    from src.assembled_core.strategies.signal_decay_gate import compute_multipliers

    multipliers = compute_multipliers(DEFAULT_FACTOR_COLS, report_path=out)
    assert set(multipliers.keys()) == set(DEFAULT_FACTOR_COLS)
    for m in multipliers.values():
        assert m in (0.0, 1.0)
