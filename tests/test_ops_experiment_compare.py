"""Tests for OPS-7 A/B experiment runner and compare (deep_merge, compare_summaries). OPS-12: app_overrides."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.assembled_core.ops.compare import compare_summaries
from src.assembled_core.ops.experiment_runner import deep_merge_policy, run_experiment

pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_deep_merge_policy_overrides() -> None:
    """Deep-merge only overrides nested keys; other keys unchanged."""
    base = {
        "risk_state_machine": {
            "enabled": True,
            "hysteresis": {
                "activate_score": 2,
                "require_disclosures_confirm": False,
            },
        },
        "georisk_overlay": {"enabled": True},
    }
    overrides = {
        "risk_state_machine": {
            "hysteresis": {
                "require_disclosures_confirm": True,
            },
        },
    }
    merged = deep_merge_policy(base, overrides)
    assert merged["risk_state_machine"]["enabled"] is True
    assert merged["risk_state_machine"]["hysteresis"]["activate_score"] == 2
    assert (
        merged["risk_state_machine"]["hysteresis"]["require_disclosures_confirm"]
        is True
    )
    assert merged["georisk_overlay"]["enabled"] is True


def test_compare_summaries_delta(tmp_path: Path) -> None:
    """compare_summaries returns schema a, b, delta with b-a for key metrics."""
    summary_a = {
        "schema_version": "paper.summary.v1",
        "total_return": 0.05,
        "max_drawdown": -0.02,
        "avg_final_multiplier": 0.85,
        "risk_state_transitions": 3,
        "alerts_count_by_level": {"warn": 2, "critical": 0},
        "risk_state_pct": {"ACTIVE": 0.5},
        "risk_state_reason_counts": {"disclosures_confirm": 2},
    }
    summary_b = {
        "schema_version": "paper.summary.v1",
        "total_return": 0.08,
        "max_drawdown": -0.03,
        "avg_final_multiplier": 0.90,
        "risk_state_transitions": 5,
        "alerts_count_by_level": {"warn": 1, "critical": 1},
        "risk_state_pct": {"ACTIVE": 0.3},
        "risk_state_reason_counts": {"disclosures_confirm": 8},
    }
    path_a = tmp_path / "summary_a.json"
    path_b = tmp_path / "summary_b.json"
    path_a.write_text(json.dumps(summary_a), encoding="utf-8")
    path_b.write_text(json.dumps(summary_b), encoding="utf-8")

    result = compare_summaries(path_a, path_b)
    assert result["schema_version"] == "paper.compare.v1"
    assert result["a"]["total_return"] == 0.05
    assert result["b"]["total_return"] == 0.08
    assert result["delta"]["total_return"] == pytest.approx(0.03)
    assert result["delta"]["max_drawdown"] == pytest.approx(-0.01)
    assert result["delta"]["avg_final_multiplier"] == pytest.approx(0.05)
    assert result["delta"]["alerts_warn"] == -1
    assert result["delta"]["alerts_critical"] == 1
    assert result["delta"]["risk_state_transitions"] == 2
    assert result["delta"]["active_pct"] == pytest.approx(-0.2)
    assert result["delta"]["disclosures_confirm_blocks"] == 6


def test_deep_merge_app_overrides() -> None:
    """Deep-merge app_overrides into app config (same algorithm as policy); paper_runner.intel.mode etc."""
    base = {
        "paper_runner": {
            "intel": {"mode": "sim", "run_news_pipeline": False},
            "other": "unchanged",
        },
    }
    overrides = {
        "paper_runner": {
            "intel": {
                "mode": "real",
                "run_news_pipeline": True,
                "run_disclosures_pipeline": True,
            },
        },
    }
    merged = deep_merge_policy(base, overrides)
    assert merged["paper_runner"]["intel"]["mode"] == "real"
    assert merged["paper_runner"]["intel"]["run_news_pipeline"] is True
    assert merged["paper_runner"]["intel"]["run_disclosures_pipeline"] is True
    assert merged["paper_runner"]["other"] == "unchanged"


def test_run_experiment_writes_app_snapshot(tmp_path: Path) -> None:
    """run_experiment with app_overrides writes app_snapshot.yaml with merged config."""
    import pandas as pd

    app_overrides = {"paper_runner": {"intel": {"mode": "real"}}}
    policy_overrides = {}
    prices = pd.DataFrame(
        {"timestamp": ["2025-06-26T00:00:00Z"], "symbol": ["AAPL"], "close": [100.0]}
    )
    with (
        patch(
            "src.assembled_core.data.prices_ingest.load_eod_prices", return_value=prices
        ),
        patch(
            "src.assembled_core.ops.paper_runner.run_paper_daily_one",
            return_value=(0, MagicMock()),
        ),
    ):
        exp_root = run_experiment(
            name="test_app_snap",
            start_date="2025-06-26",
            end_date="2025-06-26",
            mode="paper",
            output_root=tmp_path,
            policy_overrides=policy_overrides,
            app_overrides=app_overrides,
            root=tmp_path,
        )
    app_snapshot = exp_root / "app_snapshot.yaml"
    assert app_snapshot.exists()
    with app_snapshot.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None
    assert data.get("paper_runner", {}).get("intel", {}).get("mode") == "real"


def test_cli_parse_app_overrides() -> None:
    """CLI run_paper_experiment parses --app-overrides (JSON string) and passes to run_experiment."""
    from scripts.cli import run_paper_experiment_subcommand

    captured_kwargs: dict = {}

    def capture_run_experiment(*, app_overrides=None, **kwargs: object) -> Path:
        captured_kwargs["app_overrides"] = app_overrides
        captured_kwargs["kwargs"] = kwargs
        return Path("/tmp/dummy_exp")

    args = type(
        "Args",
        (),
        {
            "name": "cli_test",
            "start": "2025-06-26",
            "end": "2025-06-27",
            "mode": "paper",
            "output_root": None,
            "overrides": "{}",
            "app_overrides": '{"paper_runner":{"intel":{"mode":"real"}}}',
        },
    )()
    with patch(
        "src.assembled_core.ops.experiment_runner.run_experiment",
        side_effect=capture_run_experiment,
    ):
        run_paper_experiment_subcommand(args)
    assert captured_kwargs.get("app_overrides") == {
        "paper_runner": {"intel": {"mode": "real"}}
    }
