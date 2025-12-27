"""Tests for Paper Track PIT env var handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    _should_enable_pit_checks,
)


@pytest.fixture
def default_config() -> PaperTrackConfig:
    """Create a default PaperTrackConfig."""
    from pathlib import Path

    return PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        universe_file=Path("watchlist.txt"),
        freq="1d",
        seed_capital=100000.0,
        enable_pit_checks=True,
    )


def test_should_enable_pit_checks_unified_env_var_true(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that PAPER_TRACK_STRICT_PIT_CHECKS=true enables PIT checks."""
    monkeypatch.setenv("PAPER_TRACK_STRICT_PIT_CHECKS", "true")
    # Clear legacy env var if set
    monkeypatch.delenv("PAPER_TRACK_STRICT_PIT", raising=False)

    result = _should_enable_pit_checks(default_config)
    assert result is True


def test_should_enable_pit_checks_unified_env_var_false(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that PAPER_TRACK_STRICT_PIT_CHECKS=false disables PIT checks."""
    monkeypatch.setenv("PAPER_TRACK_STRICT_PIT_CHECKS", "false")
    monkeypatch.delenv("PAPER_TRACK_STRICT_PIT", raising=False)

    result = _should_enable_pit_checks(default_config)
    assert result is False


def test_should_enable_pit_checks_legacy_env_var_deprecation(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that legacy PAPER_TRACK_STRICT_PIT works but shows deprecation warning."""
    monkeypatch.delenv("PAPER_TRACK_STRICT_PIT_CHECKS", raising=False)
    monkeypatch.setenv("PAPER_TRACK_STRICT_PIT", "true")

    # Should still work but show deprecation warning
    with pytest.warns(DeprecationWarning, match="Deprecated env var PAPER_TRACK_STRICT_PIT"):
        result = _should_enable_pit_checks(default_config)
        assert result is True


def test_should_enable_pit_checks_config_default(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that config value is used when no env var is set."""
    monkeypatch.delenv("PAPER_TRACK_STRICT_PIT_CHECKS", raising=False)
    monkeypatch.delenv("PAPER_TRACK_STRICT_PIT", raising=False)

    # Config has enable_pit_checks=True
    result = _should_enable_pit_checks(default_config)
    assert result is True

    # Config with enable_pit_checks=False
    config_false = PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        universe_file=Path("watchlist.txt"),
        freq="1d",
        seed_capital=100000.0,
        enable_pit_checks=False,
    )
    result = _should_enable_pit_checks(config_false)
    assert result is False


def test_should_enable_pit_checks_unified_overrides_legacy(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that unified env var takes precedence over legacy."""
    monkeypatch.setenv("PAPER_TRACK_STRICT_PIT_CHECKS", "false")
    monkeypatch.setenv("PAPER_TRACK_STRICT_PIT", "true")  # Legacy should be ignored

    result = _should_enable_pit_checks(default_config)
    assert result is False  # Unified var takes precedence


def test_should_enable_pit_checks_various_truthy_values(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that various truthy values enable PIT checks."""
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("PAPER_TRACK_STRICT_PIT_CHECKS", value)
        result = _should_enable_pit_checks(default_config)
        assert result is True, f"Expected {value} to enable PIT checks"


def test_should_enable_pit_checks_various_falsy_values(
    default_config: PaperTrackConfig, monkeypatch
) -> None:
    """Test that various falsy values disable PIT checks."""
    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("PAPER_TRACK_STRICT_PIT_CHECKS", value)
        result = _should_enable_pit_checks(default_config)
        assert result is False, f"Expected {value} to disable PIT checks"

