"""Test doc-style schema compatibility (name alias, params, defaults)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_doc_schema_with_name_alias(tmp_path: Path) -> None:
    """Test that 'name' can be used as alias for 'id'."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - name: "run1"  # Using 'name' instead of 'id'
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    assert batch_cfg.batch_name == "test_batch"
    assert len(batch_cfg.runs) == 1
    assert batch_cfg.runs[0].id == "run1"  # Should be accessible as 'id'


def test_doc_schema_with_params(tmp_path: Path) -> None:
    """Test that params dict is accepted and included in run config."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - id: "run1"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    params:
      ema_fast: 20
      ema_slow: 50
      verbose: true
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    assert len(batch_cfg.runs) == 1
    run_cfg = batch_cfg.runs[0]
    assert run_cfg.params == {"ema_fast": 20, "ema_slow": 50, "verbose": True}


def test_run_id_changes_with_params(tmp_path: Path) -> None:
    """Test that run_id changes when params change."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config, _compute_run_id_hash

    # Config with params
    config_file1 = tmp_path / "test_batch1.yaml"
    config_content1 = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    params:
      ema_fast: 20
"""
    config_file1.write_text(config_content1, encoding="utf-8")

    batch_cfg1 = load_batch_config(config_file1)
    run_cfg1 = batch_cfg1.runs[0]
    run_id1 = run_cfg1.id if run_cfg1.id else _compute_run_id_hash(run_cfg1, 42)

    # Config with different params
    config_file2 = tmp_path / "test_batch2.yaml"
    config_content2 = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    params:
      ema_fast: 30  # Different value
"""
    config_file2.write_text(config_content2, encoding="utf-8")

    batch_cfg2 = load_batch_config(config_file2)
    run_cfg2 = batch_cfg2.runs[0]
    run_id2 = run_cfg2.id if run_cfg2.id else _compute_run_id_hash(run_cfg2, 42)

    # Run IDs should be different
    assert run_id1 != run_id2, "Run IDs should differ when params differ"


def test_params_in_args(tmp_path: Path) -> None:
    """Test that params are mapped to CLI flags in args."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config, build_args_from_run_config

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - id: "run1"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    params:
      ema_fast: 20
      ema_slow: 50
      verbose: true
      some_list: [1, 2, 3]
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)
    run_cfg = batch_cfg.runs[0]

    output_dir = tmp_path / "output" / "run1"
    args = build_args_from_run_config(run_cfg, output_dir)

    # Check that params are mapped to CLI flags (as attributes with underscore)
    assert hasattr(args, "ema_fast"), "ema_fast should be in args"
    assert hasattr(args, "ema_slow"), "ema_slow should be in args"
    assert hasattr(args, "verbose"), "verbose should be in args"
    # Check values
    assert getattr(args, "ema_fast") == 20
    assert getattr(args, "ema_slow") == 50
    assert getattr(args, "verbose") is True


def test_batch_defaults_applied(tmp_path: Path) -> None:
    """Test that batch-level defaults are applied to runs."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

defaults:
  freq: "1d"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  universe: "watchlist.txt"
  start_capital: 100000.0

runs:
  - id: "run1"
    strategy: "trend_baseline"
    # freq, start_date, end_date, universe, start_capital from defaults
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    assert len(batch_cfg.runs) == 1
    run_cfg = batch_cfg.runs[0]
    assert run_cfg.freq == "1d"
    assert run_cfg.start_date == "2020-01-01"
    assert run_cfg.end_date == "2023-12-31"
    assert run_cfg.universe == "watchlist.txt"
    assert run_cfg.start_capital == 100000.0


def test_params_in_manifest(tmp_path: Path) -> None:
    """Test that params are included in manifest for reproducibility."""
    import sys
    sys.path.insert(0, str(ROOT))

    from datetime import datetime
    from scripts.batch_runner import load_batch_config, write_run_manifest

    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - id: "run1"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    params:
      ema_fast: 20
      ema_slow: 50
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)
    run_cfg = batch_cfg.runs[0]

    run_output_dir = tmp_path / "output" / "run1"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        run_id="run1",
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="success",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        runtime_sec=1.0,
        exit_code=0,
        seed=42,
    )

    manifest_path = run_output_dir / "run_manifest.json"
    assert manifest_path.exists(), "Manifest should be written"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "params" in manifest, "Manifest should contain params"
    assert "params" in manifest["params"], "Manifest params should contain params dict"
    assert manifest["params"]["params"] == {"ema_fast": 20, "ema_slow": 50}

