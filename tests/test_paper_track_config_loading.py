"""Tests for Paper Track config loading.

Tests loading of example config files to ensure they are valid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_paper_track import load_paper_track_config


@pytest.fixture
def config_dir() -> Path:
    """Return path to configs directory."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "configs" / "paper_track"


def test_load_trend_baseline_example_config(config_dir: Path) -> None:
    """Test loading trend_baseline_example.yaml config."""
    config_path = config_dir / "trend_baseline_example.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_paper_track_config(config_path)

    # Validate config
    assert config.strategy_name == "trend_baseline_example"
    assert config.strategy_type == "trend_baseline"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0
    assert config.commission_bps == 0.5
    assert config.spread_w == 0.25
    assert config.impact_w == 0.5
    assert config.strategy_params.get("ma_fast") == 20
    assert config.strategy_params.get("ma_slow") == 50
    assert config.strategy_params.get("top_n") == 5
    assert config.universe_file.exists(), f"Universe file should exist: {config.universe_file}"


def test_load_multifactor_long_short_example_config(config_dir: Path) -> None:
    """Test loading multifactor_long_short_example.yaml config."""
    config_path = config_dir / "multifactor_long_short_example.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_paper_track_config(config_path)

    # Validate config
    assert config.strategy_name == "multifactor_long_short_example"
    assert config.strategy_type == "multifactor_long_short"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0
    assert config.strategy_params.get("bundle_path") is not None
    assert config.universe_file.exists(), f"Universe file should exist: {config.universe_file}"


def test_config_optional_fields(config_dir: Path) -> None:
    """Test that optional fields in config are handled correctly."""
    config_path = config_dir / "trend_baseline_example.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_paper_track_config(config_path)

    # Optional fields should have defaults
    assert config.output_root is not None  # Should be computed from strategy_name
    assert config.enable_pit_checks is True  # Default value

