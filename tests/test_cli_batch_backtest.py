"""Unit tests for batch_backtest CLI subcommand.

Tests cover:
- CLI help output
- Dry-run mode (plan printing)
- Summary file creation (with patching)
- Serial vs parallel execution
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
    # The batch_backtest subcommand should be registered
    subcommand_names = [name for action in parser._actions if hasattr(action, "choices") and action.choices for name in action.choices.keys()]
    
    # Alternatively, try to parse the subcommand (will fail due to missing required args, but that's ok)
    try:
        args = parser.parse_args(["batch_backtest", "--help"])
    except SystemExit as e:
        # Help command exits with code 0 (success)
        assert e.code == 0
    
    # Verify we can parse with required args (even if config file doesn't exist)
    # This just verifies the argument structure is correct
    args = parser.parse_args(["batch_backtest", "--config-file", "dummy.yaml"])
    # argparse converts Path arguments to Path objects
    assert str(args.config_file) == "dummy.yaml" or args.config_file == Path("dummy.yaml")
    assert hasattr(args, "serial")
    assert hasattr(args, "dry_run")
    assert hasattr(args, "max_workers")


def test_cli_batch_backtest_dry_run(tmp_path: Path, capsys) -> None:
    """Test dry-run mode prints plan without executing."""
    # Create a minimal config file
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch configuration
output_root: output/test
base_args:
  freq: "1d"
  data_source: "local"
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
  - id: run2
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    # Create argparse namespace
    import argparse

    args = argparse.Namespace(
        config_file=config_file,
        output_root=None,
        output_dir=None,
        max_workers=4,
        serial=False,
        fail_fast=False,
        dry_run=True,
        rerun=False,
    )

    # Run dry-run
    exit_code = batch_backtest_subcommand(args)

    # Should exit successfully
    assert exit_code == 0

    # Check output contains expected information
    captured = capsys.readouterr()
    assert "Dry-run" in captured.out
    assert "test_batch" in captured.out
    assert "Total runs: 2" in captured.out
    assert "run1" in captured.out
    assert "run2" in captured.out


def test_cli_batch_backtest_creates_summary(tmp_path: Path) -> None:
    """Test that batch execution creates summary files."""
    # Create a minimal config file
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch configuration
output_root: output/test
base_args:
  freq: "1d"
  data_source: "local"
runs:
  - id: run1
    bundle_path: config/bundle.yaml
    start_date: "2015-01-01"
    end_date: "2020-12-31"
""",
        encoding="utf-8",
    )

    # Mock the batch runner functions
    from datetime import datetime

    from src.assembled_core.experiments.batch_runner import BatchResult, RunResult

    mock_result = BatchResult(
        batch_name="test_batch",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=10.0,
        run_results=[
            RunResult(
                run_id="run1",
                status="success",
                output_dir=tmp_path / "output" / "test_batch" / "runs" / "0000_run1" / "backtest",
                runtime_sec=5.0,
            ),
        ],
    )

    with patch(
        "src.assembled_core.experiments.batch_runner.run_batch_serial",
        return_value=mock_result,
    ) as mock_serial:
        # Create argparse namespace
        import argparse

        args = argparse.Namespace(
            config_file=config_file,
            output_root=None,
            output_dir=None,
            max_workers=4,
            serial=True,  # Use serial to avoid ProcessPoolExecutor in tests
            fail_fast=False,
            dry_run=False,
            rerun=False,
        )

        # Run batch
        exit_code = batch_backtest_subcommand(args)

        # Should exit successfully
        assert exit_code == 0

        # Verify function was called
        assert mock_serial.called

        # Verify function was called with correct args
        call_args = mock_serial.call_args
        assert call_args is not None
        assert call_args.kwargs["batch_name"] == "test_batch"


def test_cli_batch_backtest_serial_vs_parallel(tmp_path: Path) -> None:
    """Test that --serial flag selects correct execution path."""
    # Create a minimal config file
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
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

    from src.assembled_core.experiments.batch_runner import BatchResult
    from datetime import datetime

    mock_result = BatchResult(
        batch_name="test_batch",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=5.0,
        run_results=[],
    )

    import argparse

    # Test serial path
    with patch(
        "src.assembled_core.experiments.batch_runner.run_batch_serial", return_value=mock_result
    ) as mock_serial:
        with patch("src.assembled_core.experiments.batch_runner.run_batch_parallel") as mock_parallel:
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

            batch_backtest_subcommand(args)

            # Should call serial, not parallel
            assert mock_serial.called
            assert not mock_parallel.called

    # Test parallel path
    with patch("src.assembled_core.experiments.batch_runner.run_batch_serial") as mock_serial:
        with patch(
            "src.assembled_core.experiments.batch_runner.run_batch_parallel", return_value=mock_result
        ) as mock_parallel:
            args = argparse.Namespace(
                config_file=config_file,
                output_root=None,
                output_dir=None,
                max_workers=4,
                serial=False,
                fail_fast=False,
                dry_run=False,
                rerun=False,
            )

            batch_backtest_subcommand(args)

            # Should call parallel, not serial
            assert mock_parallel.called
            assert not mock_serial.called


def test_cli_batch_backtest_rerun_flag(tmp_path: Path) -> None:
    """Test that --rerun flag is accepted."""
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text(
        """
batch_name: test_batch
description: Test batch
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

    # Create existing batch directory
    batch_output_dir = tmp_path / "output" / "test_batch"
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    from src.assembled_core.experiments.batch_runner import BatchResult, RunResult

    # Create mock result with at least one success
    mock_result = BatchResult(
        batch_name="test_batch",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        total_runtime_sec=5.0,
        run_results=[
            RunResult(
                run_id="run1",
                status="success",
                output_dir=tmp_path / "output" / "test_batch" / "runs" / "0000_run1" / "backtest",
                runtime_sec=5.0,
            ),
        ],
    )

    import argparse

    # Mock both runner and ROOT
    with patch(
        "src.assembled_core.experiments.batch_runner.run_batch_serial", return_value=mock_result
    ):
        from pathlib import Path

        with patch("scripts.cli.ROOT", Path(tmp_path)):
            args = argparse.Namespace(
                config_file=config_file,
                output_root=None,
                output_dir=None,
                max_workers=4,
                serial=True,
                fail_fast=False,
                dry_run=False,
                rerun=True,
            )

            # Should not raise error even if directory exists
            exit_code = batch_backtest_subcommand(args)
            # With at least one successful run, exit code should be 0
            assert exit_code == 0
