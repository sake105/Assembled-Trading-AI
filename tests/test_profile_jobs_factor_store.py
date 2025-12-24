"""Tests for Factor Store integration in profile_jobs.py.

This test module verifies that:
- EOD_SMALL job passes --use-factor-store when flag is set
- BACKTEST_MEDIUM job passes use_factor_store=True when flag is set
- warm_cache flag triggers two runs for EOD_SMALL
- Profile summary log includes cache_warm flag
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_eod_small_with_factor_store_flag(tmp_path: Path) -> None:
    """Test that EOD_SMALL job passes --use-factor-store to run_daily.py when flag is set."""
    from scripts.profile_jobs import run_eod_small_job
    
    # Mock subprocess.run to capture the command
    with patch("scripts.profile_jobs.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock pathlib.Path(__file__) to return a path that resolves correctly
        mock_path_file = MagicMock()
        mock_path_file.resolve.return_value.parent.parent = tmp_path
        
        with patch("scripts.profile_jobs.pathlib.Path", return_value=mock_path_file):
            # Call with use_factor_store=True
            run_eod_small_job(warm_cache=False, use_factor_store=True)
            
            # Verify subprocess.run was called
            assert mock_run.called
            call_args = mock_run.call_args[0][0]  # First positional argument (cmd list)
            
            # Check that --use-factor-store is in the command
            assert "--use-factor-store" in call_args


def test_eod_small_warm_cache_runs_twice(tmp_path: Path) -> None:
    """Test that EOD_SMALL job runs twice when warm_cache=True."""
    from scripts.profile_jobs import run_eod_small_job
    
    # Mock subprocess.run to capture the command
    with patch("scripts.profile_jobs.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock pathlib.Path(__file__) to return a path that resolves correctly
        mock_path_file = MagicMock()
        mock_path_file.resolve.return_value.parent.parent = tmp_path
        
        with patch("scripts.profile_jobs.pathlib.Path", return_value=mock_path_file):
            # Call with warm_cache=True and use_factor_store=True
            run_eod_small_job(warm_cache=True, use_factor_store=True)
            
            # Verify subprocess.run was called twice (cold build, then warm load)
            assert mock_run.call_count == 2


def test_backtest_medium_with_factor_store_flag() -> None:
    """Test that BACKTEST_MEDIUM job passes use_factor_store=True to BacktestArgs when flag is set."""
    from scripts.profile_jobs import run_backtest_medium_job
    
    # Mock run_backtest_from_args to capture the args (it's imported from scripts.run_backtest_strategy)
    with patch("scripts.run_backtest_strategy.run_backtest_from_args") as mock_run_backtest:
        mock_run_backtest.return_value = 0
        
        # Mock numpy.random.seed (numpy is imported inside the function)
        with patch("numpy.random.seed"):
            # Call with use_factor_store=True
            run_backtest_medium_job(use_factor_store=True)
            
            # Verify run_backtest_from_args was called
            assert mock_run_backtest.called
            
            # Get the args object that was passed
            args = mock_run_backtest.call_args[0][0]
            
            # Check that use_factor_store is True
            assert args.use_factor_store is True
            assert args.factor_group == "core_ta"


def test_backtest_medium_without_factor_store_flag() -> None:
    """Test that BACKTEST_MEDIUM job has use_factor_store=False by default."""
    from scripts.profile_jobs import run_backtest_medium_job
    
    # Mock run_backtest_from_args to capture the args (it's imported from scripts.run_backtest_strategy)
    with patch("scripts.run_backtest_strategy.run_backtest_from_args") as mock_run_backtest:
        mock_run_backtest.return_value = 0
        
        # Mock numpy.random.seed (numpy is imported inside the function)
        with patch("numpy.random.seed"):
            # Call without use_factor_store (default False)
            run_backtest_medium_job(use_factor_store=False)
            
            # Verify run_backtest_from_args was called
            assert mock_run_backtest.called
            
            # Get the args object that was passed
            args = mock_run_backtest.call_args[0][0]
            
            # Check that use_factor_store is False
            assert args.use_factor_store is False


def test_profile_job_with_warm_cache_flag(tmp_path: Path) -> None:
    """Test that profile_job function passes warm_cache and use_factor_store to job functions."""
    from scripts.profile_jobs import profile_job, JOB_MAP
    
    # Mock the job function to capture kwargs
    captured_kwargs = {}
    
    def mock_eod_job(**kwargs):
        captured_kwargs.update(kwargs)
    
    # Temporarily replace EOD_SMALL job function
    original_job = JOB_MAP["EOD_SMALL"]
    JOB_MAP["EOD_SMALL"] = mock_eod_job
    
    try:
        profile_job(
            job_name="EOD_SMALL",
            profiler="none",
            profile_out=tmp_path,
            warm_cache=True,
            use_factor_store=True,
        )
        
        # Verify kwargs were passed
        assert captured_kwargs.get("warm_cache") is True
        assert captured_kwargs.get("use_factor_store") is True
    finally:
        # Restore original job function
        JOB_MAP["EOD_SMALL"] = original_job


def test_profile_summary_log_includes_cache_warm(tmp_path: Path) -> None:
    """Test that profile_summary.log includes cache_warm flag when warm_cache and use_factor_store are True."""
    from scripts.profile_jobs import profile_job, JOB_MAP
    
    # Mock the job function to do nothing
    def mock_eod_job(**kwargs):
        pass
    
    # Temporarily replace EOD_SMALL job function
    original_job = JOB_MAP["EOD_SMALL"]
    JOB_MAP["EOD_SMALL"] = mock_eod_job
    
    try:
        # profile_job writes summary log directly to profile_out directory
        profile_out = tmp_path / "profiles"
        profile_job(
            job_name="EOD_SMALL",
            profiler="none",
            profile_out=profile_out,
            warm_cache=True,
            use_factor_store=True,
        )
        
        # Check profile_summary.log exists in profile_out
        summary_log = profile_out / "profile_summary.log"
        if not summary_log.exists():
            # Try searching recursively as fallback
            summary_logs = list(tmp_path.rglob("profile_summary.log"))
            assert len(summary_logs) > 0, f"No profile_summary.log found in {tmp_path}"
            summary_log = summary_logs[0]
        
        assert summary_log.exists(), f"profile_summary.log not found in {profile_out}"
        
        # Read and verify cache_warm is set
        content = summary_log.read_text()
        assert "cache_warm=True" in content
        assert "warm_cache=True" in content
        assert "use_factor_store=True" in content
    finally:
        # Restore original job function
        JOB_MAP["EOD_SMALL"] = original_job

