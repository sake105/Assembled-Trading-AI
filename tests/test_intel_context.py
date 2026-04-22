"""Unit tests for Part B ctx wiring helper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.assembled_core.paper.intel_context import (
    MIN_SHOCK_SEVERITY,
    TOPIC_TO_SHOCKS,
    active_shocks_from_triggers,
    populate_ctx_from_artifacts,
)


def test_empty_triggers_returns_empty():
    assert active_shocks_from_triggers([]) == []


def test_below_severity_filtered():
    items = [{"topic_id": "energy_crisis", "severity": 1}]
    assert active_shocks_from_triggers(items) == []


def test_single_high_severity_shock_mapped():
    items = [{"topic_id": "energy_crisis", "severity": 2}]
    result = active_shocks_from_triggers(items)
    assert "oil_supply_risk" in result
    assert "energy_price_spike" in result


def test_unknown_topic_ignored():
    items = [
        {"topic_id": "weather_report", "severity": 3},
        {"topic_id": "energy_crisis", "severity": 2},
    ]
    result = active_shocks_from_triggers(items)
    assert result == sorted({"oil_supply_risk", "energy_price_spike"})


def test_duplicates_deduplicated():
    items = [
        {"topic_id": "geopolitical_conflict", "severity": 3},
        {"topic_id": "market_crash", "severity": 3},
    ]
    result = active_shocks_from_triggers(items)
    # both map to global_risk_off — deduplicated
    assert result.count("global_risk_off") == 1
    assert "defense_demand_surge" in result


def test_missing_severity_skipped():
    items = [{"topic_id": "energy_crisis"}]  # no severity
    assert active_shocks_from_triggers(items) == []


def test_malformed_severity_skipped():
    items = [{"topic_id": "energy_crisis", "severity": "high"}]
    assert active_shocks_from_triggers(items) == []


def test_custom_min_severity():
    items = [{"topic_id": "central_bank", "severity": 1}]
    # default cutoff (=2) would filter
    assert active_shocks_from_triggers(items) == []
    # lowered cutoff passes through
    assert "rate_shock" in active_shocks_from_triggers(items, min_severity=1)


def test_all_curated_topics_map_to_known_shocks():
    # Sanity: every value in TOPIC_TO_SHOCKS must be a SHOCK_BENEFICIARY_MAP key
    from src.assembled_core.signals.intel_signal_adapter import SHOCK_BENEFICIARY_MAP

    known = set(SHOCK_BENEFICIARY_MAP.keys())
    for topic, shocks in TOPIC_TO_SHOCKS.items():
        unknown = [s for s in shocks if s not in known]
        assert not unknown, f"{topic} maps to unknown shocks: {unknown}"


def test_populate_ctx_no_artifact(tmp_path: Path):
    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    # No attribute set when artifact missing — trading_cycle uses getattr default
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_empty_items(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"items": []}), encoding="utf-8")

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_active_shocks(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({
            "items": [
                {"topic_id": "energy_crisis", "severity": 2},
                {"topic_id": "weather_report", "severity": 3},  # unknown, dropped
                {"topic_id": "nuclear_risk", "severity": 3},
            ]
        }),
        encoding="utf-8",
    )

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path)

    assert hasattr(ctx, "intel_active_shocks")
    shocks = set(ctx.intel_active_shocks)
    assert "oil_supply_risk" in shocks
    assert "energy_price_spike" in shocks
    assert "nuclear_escalation_risk" in shocks


def test_populate_ctx_corrupt_artifact(tmp_path: Path):
    artifact = tmp_path / "output" / "intel" / "news" / "triggers_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{not valid json", encoding="utf-8")

    ctx = SimpleNamespace()
    # Must not raise; warning is logged
    populate_ctx_from_artifacts(ctx, tmp_path)
    assert not hasattr(ctx, "intel_active_shocks")


def test_populate_ctx_explicit_path(tmp_path: Path):
    custom = tmp_path / "custom_triggers.json"
    custom.write_text(
        json.dumps({"items": [{"topic_id": "market_crash", "severity": 2}]}),
        encoding="utf-8",
    )

    ctx = SimpleNamespace()
    populate_ctx_from_artifacts(ctx, tmp_path, news_triggers_path=str(custom))

    assert ctx.intel_active_shocks == ["global_risk_off"]


def test_min_shock_severity_constant():
    assert MIN_SHOCK_SEVERITY == 2
