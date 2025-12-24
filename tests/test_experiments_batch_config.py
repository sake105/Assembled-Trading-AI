"""Unit tests for batch configuration module.

Tests cover:
- Loading valid configs (YAML/JSON)
- Loading invalid configs (error handling)
- Grid expansion (count, deterministic ordering)
- Path normalization
- Validation errors
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.assembled_core.experiments.batch_config import (
    BatchConfig,
    RunSpec,
    load_batch_config,
)


def test_load_valid_yaml_config(tmp_path: Path) -> None:
    """Test loading a valid YAML config."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch configuration
output_root: output/test
run_tag: experiment_2025
seed: 42
max_workers: 4
fail_fast: false

base_args:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  rebalance_freq: "M"
  max_gross_exposure: 1.0
  start_capital: 100000.0

runs:
  - id: run1
    bundle_path: config/bundle1.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    tags: ["test", "baseline"]
  - id: run2
    bundle_path: config/bundle2.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    tags: ["test", "ml"]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)

    assert config.batch_name == "test_batch"
    assert config.description == "Test batch configuration"
    # output_root is resolved relative to config file, so it's absolute
    assert "output" in str(config.output_root) and "test" in str(config.output_root)
    assert config.run_tag == "experiment_2025"
    assert config.seed == 42
    assert config.max_workers == 4
    assert config.fail_fast is False
    assert len(config.runs) == 2
    assert config.runs[0].id == "run1"
    assert config.runs[1].id == "run2"


def test_load_valid_json_config(tmp_path: Path) -> None:
    """Test loading a valid JSON config."""
    config_file = tmp_path / "test_batch.json"
    config_file.write_text(
        """{
  "batch_name": "test_batch",
  "description": "Test batch configuration",
  "output_root": "output/test",
  "base_args": {
    "freq": "1d",
    "data_source": "local"
  },
  "runs": [
    {
      "id": "run1",
      "bundle_path": "config/bundle1.yaml",
      "start_date": "2015-01-01",
      "end_date": "2020-12-31"
    }
  ]
}""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)

    assert config.batch_name == "test_batch"
    assert len(config.runs) == 1
    assert config.runs[0].id == "run1"


def test_load_invalid_missing_required_field(tmp_path: Path) -> None:
    """Test loading config with missing required field."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
description: Test batch
output_root: output/test
runs: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_name"):
        load_batch_config(config_file)


def test_load_invalid_empty_batch_name(tmp_path: Path) -> None:
    """Test loading config with empty batch_name."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: ""
description: Test batch
output_root: output/test
runs: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_name must not be empty"):
        load_batch_config(config_file)


def test_load_invalid_invalid_run_id(tmp_path: Path) -> None:
    """Test loading config with invalid run ID."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
runs:
  - id: "run with spaces"
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run id must contain only"):
        load_batch_config(config_file)


def test_load_invalid_date_format(tmp_path: Path) -> None:
    """Test loading config with invalid date format."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015/01/01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="date must be in YYYY-MM-DD format"):
        load_batch_config(config_file)


def test_load_invalid_no_runs_no_grid(tmp_path: Path) -> None:
    """Test loading config with neither runs nor grid."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  freq: "1d"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must specify either 'runs' or 'grid'"):
        load_batch_config(config_file)


def test_grid_expansion_count(tmp_path: Path) -> None:
    """Test grid expansion generates correct number of runs."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  bundle_path: config/bundle.yaml
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  freq: "1d"
grid:
  max_gross_exposure: [0.6, 0.8, 1.0]
  commission_bps: [0.0, 0.5]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)
    expanded_runs = config.expand_runs()

    # 3 exposures * 2 commission values = 6 combinations
    assert len(expanded_runs) == 6


def test_grid_expansion_deterministic_ordering(tmp_path: Path) -> None:
    """Test grid expansion produces deterministic ordering."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  bundle_path: config/bundle.yaml
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  freq: "1d"
grid:
  max_gross_exposure: [0.6, 1.0]
  commission_bps: [0.0, 0.5]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)
    expanded_runs1 = config.expand_runs()
    expanded_runs2 = config.expand_runs()

    # Should be identical (deterministic)
    assert len(expanded_runs1) == len(expanded_runs2)
    assert [r.id for r in expanded_runs1] == [r.id for r in expanded_runs2]

    # Should be sorted
    run_ids = [r.id for r in expanded_runs1]
    assert run_ids == sorted(run_ids)


def test_grid_expansion_run_ids(tmp_path: Path) -> None:
    """Test grid expansion generates correct run IDs."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
run_tag: experiment_2025
base_args:
  bundle_path: config/bundle.yaml
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  freq: "1d"
grid:
  max_gross_exposure: [0.6, 1.0]
  commission_bps: [0.0]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)
    expanded_runs = config.expand_runs()

    # Should have run_tag prefix
    assert all(r.id.startswith("experiment_2025_") for r in expanded_runs)

    # Should contain parameter values (sanitized: 0.6 -> 0_6)
    run_ids = [r.id for r in expanded_runs]
    assert any("max_gross_exposure_0_6" in rid for rid in run_ids)
    assert any("max_gross_exposure_1_0" in rid for rid in run_ids)
    assert any("commission_bps_0_0" in rid for rid in run_ids)


def test_grid_expansion_overrides(tmp_path: Path) -> None:
    """Test grid expansion merges overrides correctly."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  bundle_path: config/bundle.yaml
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  freq: "1d"
  max_gross_exposure: 1.0
grid:
  max_gross_exposure: [0.6, 0.8]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)
    expanded_runs = config.expand_runs()

    # Grid values should override base_args
    assert expanded_runs[0].overrides["max_gross_exposure"] == 0.6
    assert expanded_runs[1].overrides["max_gross_exposure"] == 0.8

    # base_args should still be present
    assert expanded_runs[0].overrides["freq"] == "1d"


def test_grid_expansion_missing_base_args(tmp_path: Path) -> None:
    """Test grid expansion fails if required base_args are missing."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  freq: "1d"
grid:
  max_gross_exposure: [0.6, 0.8]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Grid search requires base_args"):
        load_batch_config(config_file)


def test_grid_expansion_both_runs_and_grid(tmp_path: Path) -> None:
    """Test grid expansion fails if both runs and grid are specified."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: output/test
base_args:
  bundle_path: config/bundle.yaml
  start_date: "2015-01-01"
  end_date: "2020-12-31"
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
grid:
  max_gross_exposure: [0.6, 0.8]
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)

    with pytest.raises(ValueError, match="Cannot specify both 'runs' and 'grid'"):
        config.expand_runs()


def test_path_normalization_relative(tmp_path: Path) -> None:
    """Test that relative paths are normalized relative to config file."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    bundle_file = config_dir / "bundle.yaml"
    bundle_file.write_text("test", encoding="utf-8")

    config_file = config_dir / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
output_root: ../output/test
runs:
  - id: run1
    bundle_path: bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)

    # bundle_path should be resolved relative to config file
    assert config.runs[0].bundle_path == config_dir / "bundle.yaml"
    assert config.runs[0].bundle_path.exists()


def test_path_normalization_absolute(tmp_path: Path) -> None:
    """Test that absolute paths are preserved."""
    bundle_file = tmp_path / "bundle.yaml"
    bundle_file.write_text("test", encoding="utf-8")

    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        f"""
batch_name: test_batch
description: Test batch
output_root: {tmp_path / "output"}
runs:
  - id: run1
    bundle_path: {bundle_file}
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    config = load_batch_config(config_file)

    # Absolute paths should be preserved
    assert config.runs[0].bundle_path == bundle_file


def test_run_spec_validation() -> None:
    """Test RunSpec validation."""
    # Valid RunSpec
    run_spec = RunSpec(
        id="run1",
        bundle_path=Path("config/bundle.yaml"),
        start_date="2015-01-01",
        end_date="2020-12-31",
    )
    assert run_spec.id == "run1"

    # Invalid ID (spaces)
    with pytest.raises(ValueError, match="run id must contain only"):
        RunSpec(
            id="run 1",
            bundle_path=Path("config/bundle.yaml"),
            start_date="2015-01-01",
            end_date="2020-12-31",
        )

    # Invalid date format
    with pytest.raises(ValueError, match="date must be in YYYY-MM-DD format"):
        RunSpec(
            id="run1",
            bundle_path=Path("config/bundle.yaml"),
            start_date="2015/01/01",
            end_date="2020-12-31",
        )


def test_batch_config_validation() -> None:
    """Test BatchConfig validation."""
    # Valid BatchConfig
    config = BatchConfig(
        batch_name="test_batch",
        description="Test",
        output_root=Path("output/test"),
        runs=[
            RunSpec(
                id="run1",
                bundle_path=Path("config/bundle.yaml"),
                start_date="2015-01-01",
                end_date="2020-12-31",
            )
        ],
    )
    assert config.batch_name == "test_batch"

    # Invalid batch_name (spaces)
    with pytest.raises(ValueError, match="batch_name must contain only"):
        BatchConfig(
            batch_name="test batch",
            description="Test",
            output_root=Path("output/test"),
            runs=[],
        )

    # Invalid max_workers
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BatchConfig(
            batch_name="test_batch",
            description="Test",
            output_root=Path("output/test"),
            max_workers=0,
            runs=[],
        )

