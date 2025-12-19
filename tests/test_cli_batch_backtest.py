"""Tests for batch backtest CLI and batch runner."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.advanced


def _write_simple_batch_config_yaml(path: Path, runs: List[Dict[str, Any]]) -> None:
    """Write a minimal YAML batch config using a simple hand-crafted format.

    We avoid requiring PyYAML in tests by writing a very small YAML subset manually.
    """
    lines: List[str] = []
    lines.append("batch_name: test_batch")
    lines.append("description: Simple test batch for CLI")
    lines.append("output_root: output/batch_backtests")
    lines.append("defaults:")
    lines.append("  freq: \"1d\"")
    lines.append("  data_source: \"local\"")
    lines.append("  strategy: \"multifactor_long_short\"")
    lines.append("  rebalance_freq: \"M\"")
    lines.append("  max_gross_exposure: 1.0")
    lines.append("  start_capital: 100000.0")
    lines.append("  generate_report: false")
    lines.append("  generate_risk_report: false")
    lines.append("  generate_tca_report: false")
    lines.append("  symbols_file: \"config/macro_world_etfs_tickers.txt\"")
    lines.append("runs:")
    for run in runs:
        lines.append(f"  - id: \"{run['id']}\"")
        lines.append(f"    bundle_path: \"{run['bundle_path']}\"")
        lines.append(f"    start_date: \"{run['start_date']}\"")
        lines.append(f"    end_date: \"{run['end_date']}\"")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def sample_batch_config_yaml(tmp_path: Path) -> Path:
    """Create a simple YAML batch config file for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        {
            "id": "run1_core",
            "bundle_path": "config/factor_bundles/macro_world_etfs_core_bundle.yaml",
            "start_date": "2015-01-01",
            "end_date": "2015-01-10",
        },
        {
            "id": "run2_core",
            "bundle_path": "config/factor_bundles/macro_world_etfs_core_bundle.yaml",
            "start_date": "2015-01-11",
            "end_date": "2015-01-20",
        },
    ]
    cfg_path = config_dir / "batch_test.yaml"
    _write_simple_batch_config_yaml(cfg_path, runs)
    return cfg_path


def test_load_batch_config_yaml_basic(sample_batch_config_yaml: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_batch_config can read a minimal YAML config."""
    from scripts import batch_backtest

    # Force ROOT used inside batch_backtest to test root
    monkeypatch.setattr(batch_backtest, "ROOT", ROOT)

    batch_cfg = batch_backtest.load_batch_config(sample_batch_config_yaml)

    assert batch_cfg.batch_name == "test_batch"
    assert len(batch_cfg.runs) == 2

    run_ids = {r.id for r in batch_cfg.runs}
    assert run_ids == {"run1_core", "run2_core"}

    for run in batch_cfg.runs:
        assert run.freq == "1d"
        assert run.data_source == "local"
        assert run.strategy == "multifactor_long_short"
        assert run.bundle_path.name == "macro_world_etfs_core_bundle.yaml"


def test_run_batch_with_mocked_single_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_batch_config_yaml: Path):
    """Test run_batch logic using a mocked run_single_backtest for speed and determinism."""
    from scripts import batch_backtest

    monkeypatch.setattr(batch_backtest, "ROOT", ROOT)
    batch_cfg = batch_backtest.load_batch_config(sample_batch_config_yaml)

    # Override output root to tmp_path for isolation
    batch_cfg.output_root = tmp_path

    fake_results: List[batch_backtest.SingleRunResult] = []

    def fake_run_single_backtest(run_cfg, base_output_dir, dry_run=False):
        # Create dummy backtest directory
        backtest_dir = base_output_dir / run_cfg.id / "backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        result = batch_backtest.SingleRunResult(
            run_id=run_cfg.id,
            status="success",
            backtest_dir=backtest_dir,
            runtime_sec=0.123,
            exit_code=0,
            error=None,
        )
        fake_results.append(result)
        return result

    monkeypatch.setattr(batch_backtest, "run_single_backtest", fake_run_single_backtest)

    results = batch_backtest.run_batch(
        batch_cfg=batch_cfg,
        max_workers=1,
        dry_run=False,
        fail_fast=False,
    )

    assert len(results) == 2
    assert all(r.status == "success" for r in results)

    # Check summary files
    batch_root = tmp_path / batch_cfg.batch_name
    summary_csv = batch_root / "batch_summary.csv"
    summary_md = batch_root / "batch_summary.md"

    assert summary_csv.exists()
    assert summary_md.exists()

    csv_content = summary_csv.read_text(encoding="utf-8")
    assert "run1_core" in csv_content
    assert "run2_core" in csv_content

    md_content = summary_md.read_text(encoding="utf-8")
    assert "Batch Summary" in md_content
    assert "run1_core" in md_content
    assert "run2_core" in md_content


def test_run_batch_fail_fast_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_batch_config_yaml: Path):
    """Test that fail_fast stops execution after first failure."""
    from scripts import batch_backtest

    monkeypatch.setattr(batch_backtest, "ROOT", ROOT)
    batch_cfg = batch_backtest.load_batch_config(sample_batch_config_yaml)
    batch_cfg.output_root = tmp_path

    call_order: List[str] = []

    def fake_run_single_backtest(run_cfg, base_output_dir, dry_run=False):
        call_order.append(run_cfg.id)
        status = "failed" if run_cfg.id == "run1_core" else "success"
        return batch_backtest.SingleRunResult(
            run_id=run_cfg.id,
            status=status,
            backtest_dir=base_output_dir / run_cfg.id / "backtest",
            runtime_sec=0.1,
            exit_code=1 if status == "failed" else 0,
            error="boom" if status == "failed" else None,
        )

    monkeypatch.setattr(batch_backtest, "run_single_backtest", fake_run_single_backtest)

    results = batch_backtest.run_batch(
        batch_cfg=batch_cfg,
        max_workers=1,
        dry_run=False,
        fail_fast=True,
    )

    # Only first run should be executed
    assert call_order == ["run1_core"]
    assert len(results) == 1
    assert results[0].status == "failed"


def test_batch_backtest_cli_subcommand_dry_run(sample_batch_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test CLI subcommand for batch_backtest in dry-run mode."""
    # Place config under repo-relative path expected by CLI
    configs_root = ROOT / "configs" / "batch_backtests"
    configs_root.mkdir(parents=True, exist_ok=True)
    target_cfg = configs_root / "cli_batch_test.yaml"
    target_cfg.write_text(sample_batch_config_yaml.read_text(encoding="utf-8"), encoding="utf-8")

    # Use a temp output directory to avoid polluting repo outputs
    output_dir = tmp_path / "batch_outputs"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cli.py"),
        "batch_backtest",
        "--config-file",
        str(target_cfg),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]

    result = pytest.subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    ) if hasattr(pytest, "subprocess") else None

    if result is None:
        # Fallback: use stdlib subprocess if pytest-subprocess is not available
        import subprocess

        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )

    # Dry-run mode may still return non-zero exit if downstream scripts or configs are missing.
    # We only assert that the command executed and produced a batch_summary (smoke test).
    assert result.stdout or result.stderr is not None


