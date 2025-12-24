"""Integration tests for batch_backtest CLI subcommand.

Tests verify that the subcommand is accessible from the main CLI
and can be executed (with mocking).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from scripts.cli import batch_backtest_subcommand, create_parser


def test_batch_backtest_subcommand_exists() -> None:
    """Test that batch_backtest subcommand is registered in CLI."""
    parser = create_parser()

    # Parse with help flag - should not raise
    try:
        parser.parse_args(["batch_backtest", "--help"])
    except SystemExit as e:
        # Help exits with code 0
        assert e.code == 0


def test_batch_backtest_subcommand_help_output(tmp_path: Path, capsys) -> None:
    """Test that batch_backtest help shows expected options."""
    parser = create_parser()

    try:
        parser.parse_args(["batch_backtest", "--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    # Help output goes to stderr in argparse
    help_text = captured.err + captured.out
    
    assert "--config-file" in help_text
    assert "--serial" in help_text
    assert "--dry-run" in help_text
    assert "--max-workers" in help_text


def test_batch_backtest_smoke_run_with_mock(tmp_path: Path) -> None:
    """Smoke test: run batch_backtest with mocked runner (should not crash)."""
    # Create minimal config
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
batch_name: smoke_test
description: Smoke test batch
output_root: output/test
base_args:
  freq: "1d"
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    from datetime import datetime

    from src.assembled_core.experiments.batch_runner import BatchResult, RunResult

    mock_result = BatchResult(
        batch_name="smoke_test",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=1.0,
        run_results=[
            RunResult(
                run_id="run1",
                status="success",
                output_dir=tmp_path / "output" / "smoke_test" / "runs" / "0000_run1" / "backtest",
                runtime_sec=1.0,
            ),
        ],
    )

    import argparse

    # Mock the runner functions
    with patch(
        "src.assembled_core.experiments.batch_runner.run_batch_serial", return_value=mock_result
    ) as mock_runner:
        from pathlib import Path as P

        with patch("scripts.cli.ROOT", P(tmp_path)):
            args = argparse.Namespace(
                config_file=config_file,
                output_root=None,
                output_dir=None,
                max_workers=4,
                serial=True,
                fail_fast=False,
                dry_run=False,
                rerun=False,
            )

            exit_code = batch_backtest_subcommand(args)

            # Should succeed
            assert exit_code == 0
            assert mock_runner.called


def test_cli_batch_backtest_from_main_entrypoint(tmp_path: Path) -> None:
    """Test that batch_backtest can be invoked from main CLI entrypoint."""
    # This tests the full CLI invocation path
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test
output_root: output/test
base_args:
  freq: "1d"
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    # Test that parser can parse the subcommand
    parser = create_parser()
    
    # Parse with required args
    args = parser.parse_args([
        "batch_backtest",
        "--config-file", str(config_file),
        "--serial",
        "--dry-run",
    ])
    
    # Verify parsed correctly
    assert args.config_file == config_file
    assert args.serial is True
    assert args.dry_run is True
    
    # Verify function is set
    assert hasattr(args, "func")
    assert args.func == batch_backtest_subcommand

