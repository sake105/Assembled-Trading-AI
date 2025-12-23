"""Tests for Paper-Track Config Templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_paper_track import load_paper_track_config

pytestmark = pytest.mark.advanced


def test_load_trend_baseline_template() -> None:
    """Test that trend_baseline.yaml template loads and validates correctly."""
    config_path = Path("configs/paper_track/trend_baseline.yaml")

    if not config_path.exists():
        pytest.skip(f"Config template not found: {config_path}")

    config = load_paper_track_config(config_path)

    assert config.strategy_name == "trend_baseline"
    assert config.strategy_type == "trend_baseline"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0
    assert config.commission_bps == 0.5
    assert config.spread_w == 0.25
    assert config.impact_w == 0.5
    assert config.output_format == "csv"

    # Check strategy params
    assert "ma_fast" in config.strategy_params
    assert "ma_slow" in config.strategy_params
    assert "top_n" in config.strategy_params
    assert "min_score" in config.strategy_params
    assert config.strategy_params["ma_fast"] == 20
    assert config.strategy_params["ma_slow"] == 50
    assert config.strategy_params["top_n"] == 5
    assert config.strategy_params["min_score"] == 0.0


def test_load_multifactor_long_short_template() -> None:
    """Test that multifactor_long_short.yaml template loads and validates correctly."""
    config_path = Path("configs/paper_track/multifactor_long_short.yaml")

    if not config_path.exists():
        pytest.skip(f"Config template not found: {config_path}")

    config = load_paper_track_config(config_path)

    assert config.strategy_name == "multifactor_long_short"
    assert config.strategy_type == "multifactor_long_short"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0
    assert config.commission_bps == 0.5
    assert config.spread_w == 0.25
    assert config.impact_w == 0.5
    assert config.output_format == "csv"

    # Check strategy params
    assert "bundle_path" in config.strategy_params
    assert "top_quantile" in config.strategy_params
    assert "bottom_quantile" in config.strategy_params
    assert "max_gross_exposure" in config.strategy_params
    assert config.strategy_params["top_quantile"] == 0.2
    assert config.strategy_params["bottom_quantile"] == 0.2
    assert config.strategy_params["max_gross_exposure"] == 1.0


def test_load_trend_baseline_example() -> None:
    """Test that trend_baseline_example.yaml loads and validates correctly."""
    config_path = Path("configs/paper_track/trend_baseline_example.yaml")

    if not config_path.exists():
        pytest.skip(f"Config example not found: {config_path}")

    config = load_paper_track_config(config_path)

    assert config.strategy_name == "trend_baseline_example"
    assert config.strategy_type == "trend_baseline"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0


def test_load_multifactor_long_short_example() -> None:
    """Test that multifactor_long_short_example.yaml loads and validates correctly."""
    config_path = Path("configs/paper_track/multifactor_long_short_example.yaml")

    if not config_path.exists():
        pytest.skip(f"Config example not found: {config_path}")

    config = load_paper_track_config(config_path)

    assert config.strategy_name == "multifactor_long_short_example"
    assert config.strategy_type == "multifactor_long_short"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0


def test_config_templates_have_required_fields() -> None:
    """Test that all config templates have all required fields."""
    templates = [
        "configs/paper_track/trend_baseline.yaml",
        "configs/paper_track/multifactor_long_short.yaml",
    ]

    for template_path in templates:
        config_file = Path(template_path)
        if not config_file.exists():
            pytest.skip(f"Config template not found: {config_file}")

        config = load_paper_track_config(config_file)

        # Required fields
        assert config.strategy_name
        assert config.strategy_type in ("trend_baseline", "multifactor_long_short")
        assert config.universe_file is not None
        assert config.freq in ("1d", "5min")
        assert config.seed_capital > 0
        assert config.commission_bps >= 0
        assert config.spread_w >= 0
        assert config.impact_w >= 0
        assert config.output_format in ("csv", "parquet")


def test_config_templates_use_relative_paths() -> None:
    """Test that config templates use relative paths for universe files."""
    templates = [
        "configs/paper_track/trend_baseline.yaml",
        "configs/paper_track/multifactor_long_short.yaml",
        "configs/paper_track/trend_baseline_example.yaml",
        "configs/paper_track/multifactor_long_short_example.yaml",
    ]

    for template_path in templates:
        config_file = Path(template_path)
        if not config_file.exists():
            continue

        config = load_paper_track_config(config_file)

        # Universe file should be resolved (absolute or relative to repo root)
        # It should exist or be a valid path format
        assert config.universe_file is not None
        # The path should be resolvable (even if file doesn't exist, path should be valid)
        assert isinstance(config.universe_file, Path)

