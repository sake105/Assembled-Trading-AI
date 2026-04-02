"""Integration tests for batch_backtest CLI subcommand.

Tests verify that the subcommand is accessible from the main CLI
and can be executed (with dry-run).
"""

from __future__ import annotations

from pathlib import Path


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
    assert "--dry-run" in help_text
    assert "--max-workers" in help_text


def test_batch_backtest_smoke_run_with_mock(tmp_path: Path) -> None:
    """Smoke test: run batch_backtest dry-run with valid config."""
    # Create minimal config matching scripts/batch_runner format
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
batch_name: smoke_test
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

    import argparse

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


def test_cli_batch_backtest_from_main_entrypoint(tmp_path: Path) -> None:
    """Test that batch_backtest can be invoked from main CLI entrypoint."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
batch_name: test_batch
output_root: {output}
seed: 42
defaults:
  strategy: trend_baseline
  freq: "1d"
runs:
  - id: run1
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""".format(
            output=str(tmp_path / "output").replace("\\", "/")
        ),
        encoding="utf-8",
    )

    # Test that parser can parse the subcommand
    parser = create_parser()

    # Parse with required args
    args = parser.parse_args(
        [
            "batch_backtest",
            "--config-file",
            str(config_file),
            "--dry-run",
        ]
    )

    # Verify parsed correctly
    assert args.config_file == config_file
    assert args.dry_run is True

    # Verify function is set
    assert hasattr(args, "func")
    assert args.func == batch_backtest_subcommand
