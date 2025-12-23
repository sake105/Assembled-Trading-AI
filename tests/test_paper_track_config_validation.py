"""Tests for Paper Track Config Validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_paper_track import load_paper_track_config, validate_config_dict

pytestmark = pytest.mark.advanced


def test_config_validation_missing_strategy_name(tmp_path: Path):
    """Test that missing strategy_name raises ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_type: trend_baseline
universe:
  file: universe.txt
trading:
  freq: 1d
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strategy_name.*Required"):
        load_paper_track_config(config_file)


def test_config_validation_invalid_strategy_type(tmp_path: Path):
    """Test that invalid strategy_type raises ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test_strategy
strategy_type: invalid_type
universe:
  file: universe.txt
trading:
  freq: 1d
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strategy_type.*Must be"):
        load_paper_track_config(config_file)


def test_config_validation_invalid_freq(tmp_path: Path):
    """Test that invalid freq raises ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test_strategy
strategy_type: trend_baseline
universe:
  file: universe.txt
trading:
  freq: invalid_freq
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trading.freq.*Must be"):
        load_paper_track_config(config_file)


def test_config_validation_negative_seed_capital(tmp_path: Path):
    """Test that negative seed_capital raises ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test_strategy
strategy_type: trend_baseline
universe:
  file: universe.txt
trading:
  freq: 1d
portfolio:
  seed_capital: -1000.0
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="portfolio.seed_capital.*Must be > 0"):
        load_paper_track_config(config_file)


def test_config_validation_invalid_output_format(tmp_path: Path):
    """Test that invalid output.format raises ValueError."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test_strategy
strategy_type: trend_baseline
universe:
  file: universe.txt
trading:
  freq: 1d
output:
  format: invalid_format
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output.format.*Must be"):
        load_paper_track_config(config_file)


def test_config_validation_valid_config(tmp_path: Path):
    """Test that a valid config loads successfully."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test_strategy
strategy_type: trend_baseline
universe:
  file: universe.txt
trading:
  freq: 1d
portfolio:
  seed_capital: 100000.0
costs:
  commission_bps: 0.5
  spread_w: 0.25
  impact_w: 0.5
output:
  format: csv
""",
        encoding="utf-8",
    )
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    config = load_paper_track_config(config_file)
    assert config.strategy_name == "test_strategy"
    assert config.strategy_type == "trend_baseline"
    assert config.freq == "1d"
    assert config.seed_capital == 100000.0
    assert config.output_format == "csv"

