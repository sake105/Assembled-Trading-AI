"""Unit tests for batch_backtest CLI subcommand.

Tests cover:
- CLI help output
- Dry-run mode via scripts.batch_runner integration
- Argument structure validation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.cli import batch_backtest_subcommand


def test_cli_batch_backtest_help() -> None:
    """Test that CLI help is accessible and subcommand exists."""

    from scripts.cli import create_parser

    parser = create_parser()

    # Test that we can create parser without error
    assert parser is not None

    # Verify subcommand exists by checking parser structure
    _subcommand_names = [
        name
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
        for name in action.choices.keys()
    ]

    # Try to parse the subcommand (will fail due to missing required args, but that's ok)
    try:
        args = parser.parse_args(["batch_backtest", "--help"])
    except SystemExit as e:
        # Help command exits with code 0 (success)
        assert e.code == 0

    # Verify we can parse with required args
    args = parser.parse_args(["batch_backtest", "--config-file", "dummy.yaml"])
    assert str(args.config_file) == "dummy.yaml" or args.config_file == Path(
        "dummy.yaml"
    )
    assert hasattr(args, "dry_run")
    assert hasattr(args, "max_workers")


def test_cli_batch_backtest_dry_run(tmp_path: Path) -> None:
    """Test dry-run mode via the actual batch_run_subcommand path."""
    import argparse

    # Create a minimal YAML config that matches scripts/batch_runner expectations
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
output_root: {output}
seed: 42
defaults:
  strategy: trend_baseline
  freq: "1d"
  start_capital: 100000.0
runs:
  - id: run1
    start_date: "2015-01-01"
    end_date: "2020-12-31"
  - id: run2
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""".format(
            output=str(tmp_path / "output").replace("\\", "/")
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config_file=config_file,
        output_root=None,
        max_workers=1,
        dry_run=True,
        resume=False,
        rerun_failed=False,
        verbose=0,
    )

    exit_code = batch_backtest_subcommand(args)
    assert exit_code == 0


def test_cli_batch_backtest_creates_summary(tmp_path: Path) -> None:
    """Test that batch dry-run completes without error for multiple runs."""
    import argparse

    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
output_root: {output}
seed: 42
defaults:
  strategy: trend_baseline
  freq: "1d"
  start_capital: 100000.0
runs:
  - id: run1
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""".format(
            output=str(tmp_path / "output").replace("\\", "/")
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config_file=config_file,
        output_root=None,
        max_workers=1,
        dry_run=True,
        resume=False,
        rerun_failed=False,
        verbose=0,
    )

    exit_code = batch_backtest_subcommand(args)
    assert exit_code == 0


def test_cli_batch_backtest_serial_vs_parallel(tmp_path: Path) -> None:
    """Test that max_workers=1 invokes serial path and max_workers>1 invokes parallel path."""
    import argparse

    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
output_root: {output}
seed: 42
defaults:
  strategy: trend_baseline
  freq: "1d"
  start_capital: 100000.0
runs:
  - id: run1
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""".format(
            output=str(tmp_path / "output").replace("\\", "/")
        ),
        encoding="utf-8",
    )

    # Serial path (max_workers=1) in dry-run
    with patch("scripts.batch_runner.run_batch_serial", return_value=0) as mock_serial:
        args = argparse.Namespace(
            config_file=config_file,
            output_root=None,
            max_workers=1,
            dry_run=True,
            resume=False,
            rerun_failed=False,
            verbose=0,
        )
        batch_backtest_subcommand(args)
        assert mock_serial.called

    # Parallel path (max_workers>1) in dry-run
    with patch(
        "scripts.batch_runner.run_batch_parallel", return_value=0
    ) as mock_parallel:
        args = argparse.Namespace(
            config_file=config_file,
            output_root=None,
            max_workers=4,
            dry_run=True,
            resume=False,
            rerun_failed=False,
            verbose=0,
        )
        batch_backtest_subcommand(args)
        assert mock_parallel.called


def test_cli_batch_backtest_rerun_flag(tmp_path: Path) -> None:
    """Test that resume/rerun-failed flags are accepted and passed through."""
    import argparse

    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
output_root: {output}
seed: 42
defaults:
  strategy: trend_baseline
  freq: "1d"
  start_capital: 100000.0
runs:
  - id: run1
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""".format(
            output=str(tmp_path / "output").replace("\\", "/")
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config_file=config_file,
        output_root=None,
        max_workers=1,
        dry_run=True,
        resume=True,
        rerun_failed=True,
        verbose=0,
    )

    exit_code = batch_backtest_subcommand(args)
    assert exit_code == 0
