"""Tests for batch runner config validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.batch_runner import (
    BatchConfig,
    RunConfig,
    TradingFreq,
    load_batch_config,
)


def test_run_config_required_fields():
    """Test that RunConfig requires all mandatory fields."""
    # Missing strategy
    with pytest.raises(Exception):  # Pydantic ValidationError
        RunConfig(
            id="test",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

    # Missing freq
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

    # Missing start_date
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            end_date="2020-12-31",
        )

    # Missing end_date
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
        )


def test_run_config_date_format():
    """Test that RunConfig validates date format."""
    # Valid date format
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    assert cfg.start_date == "2020-01-01"

    # Invalid date format (wrong separator)
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020/01/01",
            end_date="2020-12-31",
        )

    # Invalid date format (wrong length)
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-1-1",
            end_date="2020-12-31",
        )

    # Invalid date (not a real date)
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-13-01",  # Month 13 doesn't exist
            end_date="2020-12-31",
        )


def test_run_config_date_logic():
    """Test that RunConfig validates end_date >= start_date."""
    # Valid: end_date after start_date
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    assert cfg.end_date > cfg.start_date

    # Valid: end_date equals start_date
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-01-01",
    )
    assert cfg.end_date == cfg.start_date

    # Invalid: end_date before start_date
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-12-31",
            end_date="2020-01-01",
        )


def test_run_config_freq_enum():
    """Test that RunConfig validates freq enum."""
    # Valid: "1d"
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    assert cfg.freq == TradingFreq.DAILY

    # Valid: "5min"
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="5min",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    assert cfg.freq == TradingFreq.INTRADAY_5MIN

    # Invalid: wrong freq
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1h",  # Not supported
            start_date="2020-01-01",
            end_date="2020-12-31",
        )


def test_run_config_start_capital():
    """Test that RunConfig validates start_capital > 0."""
    # Valid: positive capital
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
        start_capital=100000.0,
    )
    assert cfg.start_capital == 100000.0

    # Invalid: zero capital
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
            start_capital=0.0,
        )

    # Invalid: negative capital
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
            start_capital=-1000.0,
        )


def test_run_config_universe_validation():
    """Test that RunConfig validates universe field."""
    # Valid: None
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
        universe=None,
    )
    assert cfg.universe is None

    # Valid: non-empty string
    cfg = RunConfig(
        id="test",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2020-12-31",
        universe="watchlist.txt",
    )
    assert cfg.universe == "watchlist.txt"

    # Invalid: empty string
    with pytest.raises(Exception):
        RunConfig(
            id="test",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
            universe="",  # Empty string not allowed
        )


def test_batch_config_run_ids_unique():
    """Test that BatchConfig validates run_id uniqueness."""
    runs = [
        RunConfig(
            id="run1",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
        RunConfig(
            id="run2",
            strategy="trend_baseline",
            freq="5min",
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
    ]

    # Valid: unique IDs
    cfg = BatchConfig(
        batch_name="test",
        output_root=Path("/tmp"),
        seed=42,
        runs=runs,
    )
    assert len(cfg.runs) == 2

    # Invalid: duplicate IDs
    runs_duplicate = [
        RunConfig(
            id="run1",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
        RunConfig(
            id="run1",  # Duplicate
            strategy="trend_baseline",
            freq="5min",
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
    ]

    with pytest.raises(Exception):
        BatchConfig(
            batch_name="test",
            output_root=Path("/tmp"),
            seed=42,
            runs=runs_duplicate,
        )


def test_batch_config_seed_validation():
    """Test that BatchConfig validates seed >= 0."""
    runs = [
        RunConfig(
            id="run1",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
        ),
    ]

    # Valid: non-negative seed
    cfg = BatchConfig(
        batch_name="test",
        output_root=Path("/tmp"),
        seed=42,
        runs=runs,
    )
    assert cfg.seed == 42

    # Valid: zero seed
    cfg = BatchConfig(
        batch_name="test",
        output_root=Path("/tmp"),
        seed=0,
        runs=runs,
    )
    assert cfg.seed == 0

    # Invalid: negative seed
    with pytest.raises(Exception):
        BatchConfig(
            batch_name="test",
            output_root=Path("/tmp"),
            seed=-1,
            runs=runs,
        )


def test_batch_config_runs_non_empty():
    """Test that BatchConfig requires non-empty runs list."""
    # Invalid: empty runs list
    with pytest.raises(Exception):
        BatchConfig(
            batch_name="test",
            output_root=Path("/tmp"),
            seed=42,
            runs=[],  # Empty list not allowed
        )


def test_load_batch_config_valid():
    """Test loading a valid batch config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
batch_name: test_batch
output_root: output/batch_test
seed: 42

runs:
  - id: run1
    strategy: trend_baseline
    freq: 1d
    start_date: 2020-01-01
    end_date: 2020-12-31
    start_capital: 100000.0
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        cfg = load_batch_config(config_path)
        assert cfg.batch_name == "test_batch"
        assert cfg.seed == 42
        assert len(cfg.runs) == 1
        assert cfg.runs[0].id == "run1"
        assert cfg.runs[0].strategy == "trend_baseline"
        assert cfg.runs[0].freq == TradingFreq.DAILY
    finally:
        config_path.unlink()


def test_load_batch_config_missing_batch_name():
    """Test that missing batch_name raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
output_root: output/batch_test
seed: 42

runs:
  - id: run1
    strategy: trend_baseline
    freq: 1d
    start_date: 2020-01-01
    end_date: 2020-12-31
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="batch_name"):
            load_batch_config(config_path)
    finally:
        config_path.unlink()


def test_load_batch_config_invalid_date_format():
    """Test that invalid date format raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
batch_name: test_batch
output_root: output/batch_test
seed: 42

runs:
  - id: run1
    strategy: trend_baseline
    freq: 1d
    start_date: 2020/01/01  # Wrong format
    end_date: 2020-12-31
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="date|Date"):
            load_batch_config(config_path)
    finally:
        config_path.unlink()


def test_load_batch_config_invalid_freq():
    """Test that invalid freq raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
batch_name: test_batch
output_root: output/batch_test
seed: 42

runs:
  - id: run1
    strategy: trend_baseline
    freq: 1h  # Invalid freq
    start_date: 2020-01-01
    end_date: 2020-12-31
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="freq|Freq"):
            load_batch_config(config_path)
    finally:
        config_path.unlink()


def test_load_batch_config_end_before_start():
    """Test that end_date before start_date raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
batch_name: test_batch
output_root: output/batch_test
seed: 42

runs:
  - id: run1
    strategy: trend_baseline
    freq: 1d
    start_date: 2020-12-31
    end_date: 2020-01-01  # Before start_date
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="end_date|start_date"):
            load_batch_config(config_path)
    finally:
        config_path.unlink()


def test_load_batch_config_auto_generate_run_ids():
    """Test that run IDs are auto-generated when not provided."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
batch_name: test_batch
output_root: output/batch_test
seed: 42

runs:
  - strategy: trend_baseline
    freq: 1d
    start_date: 2020-01-01
    end_date: 2020-12-31
  - strategy: trend_baseline
    freq: 5min
    start_date: 2020-01-01
    end_date: 2020-12-31
"""
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    try:
        cfg = load_batch_config(config_path)
        assert len(cfg.runs) == 2
        # Both should have auto-generated IDs
        assert cfg.runs[0].id
        assert cfg.runs[1].id
        # IDs should be unique
        assert cfg.runs[0].id != cfg.runs[1].id
    finally:
        config_path.unlink()

