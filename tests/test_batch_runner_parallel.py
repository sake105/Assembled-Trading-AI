"""Tests for parallel batch runner execution."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_id_deterministic(tmp_path: Path) -> None:
    """Test that run IDs are deterministic (same params -> same hash)."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, _compute_run_id_hash

    run_cfg1 = RunConfig(
        id="",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        universe="watchlist.txt",
        start_capital=100000.0,
    )

    run_cfg2 = RunConfig(
        id="",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        universe="watchlist.txt",
        start_capital=100000.0,
    )

    seed = 42

    hash1 = _compute_run_id_hash(run_cfg1, seed)
    hash2 = _compute_run_id_hash(run_cfg2, seed)

    # Same params + seed -> same hash
    assert hash1 == hash2
    assert len(hash1) == 16  # SHA256 first 16 chars

    # Different seed -> different hash
    hash3 = _compute_run_id_hash(run_cfg1, seed=43)
    assert hash1 != hash3

    # Different params -> different hash
    run_cfg3 = RunConfig(
        id="",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-02",  # Different date
        end_date="2023-12-31",
        universe="watchlist.txt",
        start_capital=100000.0,
    )
    hash4 = _compute_run_id_hash(run_cfg3, seed)
    assert hash1 != hash4


def test_load_config_generates_run_ids(tmp_path: Path) -> None:
    """Test that config loading generates deterministic run IDs when not explicitly set."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    # Create config without explicit run IDs
    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    # Run ID should be generated
    assert len(batch_cfg.runs) == 1
    assert batch_cfg.runs[0].id
    assert len(batch_cfg.runs[0].id) == 16  # SHA256 hash length

    # Loading again should generate the same ID
    batch_cfg2 = load_batch_config(config_file)
    assert batch_cfg.runs[0].id == batch_cfg2.runs[0].id


def test_load_config_preserves_explicit_run_ids(tmp_path: Path) -> None:
    """Test that explicit run IDs in config are preserved."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - id: "my_custom_run_id"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    assert batch_cfg.runs[0].id == "my_custom_run_id"


def test_parallel_run_batch_dry_run(tmp_path: Path) -> None:
    """Test parallel batch execution in dry-run mode."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_batch

    run_cfgs = [
        RunConfig(
            id=f"run_{i}",
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2023-12-31",
            start_capital=100000.0,
        )
        for i in range(3)
    ]

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=run_cfgs,
    )

    # Dry-run should not fail
    exit_code = run_batch(batch_cfg, max_workers=2, dry_run=True)
    assert exit_code == 0

    # Run directories should exist
    batch_output_root = batch_cfg.output_root / "batch"
    for run_cfg in run_cfgs:
        run_dir = batch_output_root / run_cfg.id
        assert run_dir.exists()


def test_output_path_structure(tmp_path: Path) -> None:
    """Test that output paths follow structure: output/batch/<run_id>/"""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, run_single_backtest

    run_cfg = RunConfig(
        id="test_run_123",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_output_root = tmp_path / "output" / "batch"
    batch_output_root.mkdir(parents=True, exist_ok=True)

    # Run in dry-run mode
    status, runtime_sec, exit_code, error = run_single_backtest(
        run_cfg=run_cfg,
        batch_output_root=batch_output_root,
        seed=42,
        dry_run=True,
    )

    # Check output structure
    run_dir = batch_output_root / run_cfg.id
    assert run_dir.exists()
    assert run_dir.parent == batch_output_root
    assert run_dir.name == run_cfg.id


def test_no_folder_collisions(tmp_path: Path) -> None:
    """Test that parallel runs don't create folder collisions."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_batch

    # Create multiple runs with different configs (should get different IDs)
    run_cfgs = [
        RunConfig(
            id="",  # Will be auto-generated
            strategy="trend_baseline",
            freq="1d",
            start_date="2020-01-01",
            end_date="2023-12-31",
            universe="watchlist.txt",
            start_capital=100000.0,
        ),
        RunConfig(
            id="",  # Will be auto-generated (different params -> different ID)
            strategy="trend_baseline",
            freq="1d",
            start_date="2021-01-01",  # Different date
            end_date="2023-12-31",
            universe="watchlist.txt",
            start_capital=100000.0,
        ),
    ]

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=run_cfgs,
    )

    # Generate IDs
    from scripts.batch_runner import _compute_run_id_hash
    for run_cfg in batch_cfg.runs:
        if not run_cfg.id:
            run_cfg.id = _compute_run_id_hash(run_cfg, batch_cfg.seed)

    # IDs should be different
    assert run_cfgs[0].id != run_cfgs[1].id

    # Dry-run should create separate directories
    exit_code = run_batch(batch_cfg, max_workers=2, dry_run=True)
    assert exit_code == 0

    batch_output_root = batch_cfg.output_root / "batch"
    for run_cfg in run_cfgs:
        run_dir = batch_output_root / run_cfg.id
        assert run_dir.exists()
        # Each run should have its own directory
        assert run_dir.is_dir()

