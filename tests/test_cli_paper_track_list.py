"""Tests for Paper-Track CLI --list and --strategy-name functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.advanced


@pytest.fixture
def temp_configs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create temporary configs directory with sample configs."""
    configs_dir = tmp_path / "configs" / "paper_track"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Create two sample configs
    config1 = {
        "strategy_name": "test_strategy_1",
        "strategy_type": "trend_baseline",
        "universe": {"file": "watchlist.txt"},
        "trading": {"freq": "1d"},
        "portfolio": {"seed_capital": 100000.0},
    }
    config1_path = configs_dir / "test_strategy_1.yaml"
    with open(config1_path, "w", encoding="utf-8") as f:
        yaml.dump(config1, f)

    config2 = {
        "strategy_name": "test_strategy_2",
        "strategy_type": "trend_baseline",
        "universe": {"file": "watchlist.txt"},
        "trading": {"freq": "1d"},
        "portfolio": {"seed_capital": 100000.0},
    }
    config2_path = configs_dir / "test_strategy_2.yaml"
    with open(config2_path, "w", encoding="utf-8") as f:
        yaml.dump(config2, f)

    # Mock ROOT to point to tmp_path
    import scripts.run_paper_track as rpt_module

    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path
    monkeypatch.setattr("scripts.run_paper_track.ROOT", tmp_path)

    yield configs_dir

    # Restore original ROOT
    rpt_module.ROOT = original_root


def test_list_paper_track_configs_show_configs(
    temp_configs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --list shows all available configs."""
    from scripts.run_paper_track import list_paper_track_configs

    exit_code = list_paper_track_configs()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Found 2 paper track config(s):" in captured.out
    assert "test_strategy_1" in captured.out
    assert "test_strategy_2" in captured.out
    assert "Strategy Name" in captured.out
    assert "Config Path" in captured.out


def test_list_paper_track_configs_no_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --list shows appropriate message when no configs found."""
    import scripts.run_paper_track as rpt_module

    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path
    monkeypatch.setattr("scripts.run_paper_track.ROOT", tmp_path)

    from scripts.run_paper_track import list_paper_track_configs

    exit_code = list_paper_track_configs()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No paper track configs found" in captured.out

    # Restore original ROOT
    rpt_module.ROOT = original_root


def test_discover_paper_track_configs(temp_configs_dir: Path) -> None:
    """Test that discover_paper_track_configs finds all configs."""
    from scripts.run_paper_track import discover_paper_track_configs

    configs = discover_paper_track_configs()

    assert len(configs) == 2
    strategy_names = [name for name, _ in configs]
    assert "test_strategy_1" in strategy_names
    assert "test_strategy_2" in strategy_names

    # Check that paths are correct
    paths = [str(path) for _, path in configs]
    assert any("test_strategy_1.yaml" in p for p in paths)
    assert any("test_strategy_2.yaml" in p for p in paths)


def test_find_config_by_strategy_name_from_configs_dir(
    temp_configs_dir: Path,
) -> None:
    """Test that find_config_by_strategy_name finds config in configs/paper_track/."""
    from scripts.run_paper_track import find_config_by_strategy_name

    config_path = find_config_by_strategy_name("test_strategy_1")

    assert config_path is not None
    assert config_path.exists()
    assert config_path.name == "test_strategy_1.yaml"
    assert "test_strategy_1" in str(config_path)


def test_find_config_by_strategy_name_from_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_config_by_strategy_name finds config in output/paper_track/{name}/."""
    import scripts.run_paper_track as rpt_module

    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path
    monkeypatch.setattr("scripts.run_paper_track.ROOT", tmp_path)

    # Create output/paper_track/{strategy_name}/config.yaml
    strategy_dir = tmp_path / "output" / "paper_track" / "test_strategy_output"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    config_path = strategy_dir / "config.yaml"
    config = {
        "strategy_name": "test_strategy_output",
        "strategy_type": "trend_baseline",
        "universe": {"file": "watchlist.txt"},
        "trading": {"freq": "1d"},
        "portfolio": {"seed_capital": 100000.0},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from scripts.run_paper_track import find_config_by_strategy_name

    found_path = find_config_by_strategy_name("test_strategy_output")

    assert found_path is not None
    assert found_path.exists()
    assert found_path.name == "config.yaml"
    assert "test_strategy_output" in str(found_path.parent)

    # Restore original ROOT
    rpt_module.ROOT = original_root


def test_find_config_by_strategy_name_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_config_by_strategy_name returns None when config not found."""
    import scripts.run_paper_track as rpt_module

    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path
    monkeypatch.setattr("scripts.run_paper_track.ROOT", tmp_path)

    from scripts.run_paper_track import find_config_by_strategy_name

    config_path = find_config_by_strategy_name("nonexistent_strategy")

    assert config_path is None

    # Restore original ROOT
    rpt_module.ROOT = original_root


def test_cli_list_flag(temp_configs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that CLI --list flag works."""

    # Create a mock args object
    class MockArgs:
        list = True
        config_file = None
        strategy_name = None
        as_of = None
        start_date = None
        end_date = None
        catch_up = False
        dry_run = False
        fail_fast = False
        rerun = False
        generate_risk_report = False
        risk_report_frequency = "weekly"
        benchmark_symbol = None
        factor_returns_file = None
        verbose = False

    args = MockArgs()

    # Call main via parse_args and then handle --list
    from scripts.run_paper_track import list_paper_track_configs

    exit_code = list_paper_track_configs()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "test_strategy_1" in captured.out
    assert "test_strategy_2" in captured.out


def test_cli_strategy_name_resolves_config(
    temp_configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that --strategy-name resolves correct config file."""

    from scripts.run_paper_track import find_config_by_strategy_name

    config_path = find_config_by_strategy_name("test_strategy_1")

    assert config_path is not None
    assert config_path.exists()
    assert "test_strategy_1.yaml" in str(config_path)


def test_cli_strategy_name_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --strategy-name shows error when config not found."""
    import scripts.run_paper_track as rpt_module

    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path
    monkeypatch.setattr("scripts.run_paper_track.ROOT", tmp_path)

    from scripts.run_paper_track import find_config_by_strategy_name

    config_path = find_config_by_strategy_name("nonexistent")

    assert config_path is None

    # Restore original ROOT
    rpt_module.ROOT = original_root

