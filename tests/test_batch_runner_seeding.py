"""Tests for batch runner seeding functionality."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.batch_runner import (
    BatchConfig,
    RunConfig,
    TradingFreq,
    set_random_seeds,
    write_run_manifest,
)


def test_set_random_seeds_python_random():
    """Test that set_random_seeds sets Python random seed."""
    # Generate two random numbers without seed
    random.seed(None)
    val1 = random.random()
    val2 = random.random()
    
    # Set seed and generate again
    set_random_seeds(42)
    val3 = random.random()
    
    # Reset seed and generate again (should be same as val3)
    set_random_seeds(42)
    val4 = random.random()
    
    # val3 and val4 should be identical (same seed)
    assert val3 == val4


def test_set_random_seeds_numpy():
    """Test that set_random_seeds sets NumPy random seed if available."""
    try:
        import numpy as np
        
        # Set seed
        set_random_seeds(42)
        val1 = np.random.random()
        
        # Reset seed and generate again (should be same)
        set_random_seeds(42)
        val2 = np.random.random()
        
        assert val1 == val2
    except ImportError:
        pytest.skip("NumPy not available")


def test_set_random_seeds_torch():
    """Test that set_random_seeds sets PyTorch seed if available."""
    try:
        import torch
        
        # Set seed
        set_random_seeds(42)
        val1 = torch.rand(1).item()
        
        # Reset seed and generate again (should be same)
        set_random_seeds(42)
        val2 = torch.rand(1).item()
        
        assert val1 == val2
    except ImportError:
        pytest.skip("PyTorch not available")


def test_set_random_seeds_graceful_without_numpy():
    """Test that set_random_seeds works gracefully without NumPy."""
    # Mock numpy as None
    with patch("scripts.batch_runner.np", None):
        # Should not raise an error
        set_random_seeds(42)
        # Python random should still be set
        val1 = random.random()
        set_random_seeds(42)
        val2 = random.random()
        assert val1 == val2


def test_set_random_seeds_graceful_without_torch():
    """Test that set_random_seeds works gracefully without PyTorch."""
    # Mock torch as None
    with patch("scripts.batch_runner.torch", None):
        # Should not raise an error
        set_random_seeds(42)
        # Python random should still be set
        val1 = random.random()
        set_random_seeds(42)
        val2 = random.random()
        assert val1 == val2


def test_write_run_manifest_includes_seed():
    """Test that write_run_manifest includes seed in manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_output_dir = Path(tmpdir)
        
        run_cfg = RunConfig(
            id="test_run",
            strategy="trend_baseline",
            freq=TradingFreq.DAILY,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        
        from datetime import datetime
        write_run_manifest(
            run_id="test_run",
            run_cfg=run_cfg,
            run_output_dir=run_output_dir,
            status="success",
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            runtime_sec=10.5,
            exit_code=0,
            seed=42,
        )
        
        # Read manifest
        manifest_path = run_output_dir / "run_manifest.json"
        import json
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # Check that seed is present
        assert "seed" in manifest
        assert manifest["seed"] == 42


def test_run_single_backtest_sets_seed():
    """Test that run_single_backtest sets seed before execution."""
    from scripts.batch_runner import run_single_backtest
    
    run_cfg = RunConfig(
        id="test_run",
        strategy="trend_baseline",
        freq=TradingFreq.DAILY,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_output_root = Path(tmpdir)
        
        # Mock run_backtest_from_args to avoid actual execution
        with patch("scripts.batch_runner.run_backtest_from_args", return_value=0):
            # Track if set_random_seeds was called
            with patch("scripts.batch_runner.set_random_seeds") as mock_set_seed:
                # Run backtest (dry-run to avoid actual execution)
                run_single_backtest(
                    run_cfg=run_cfg,
                    batch_output_root=batch_output_root,
                    seed=42,
                    dry_run=True,  # Skip execution
                )
                
                # set_random_seeds should be called (even in dry_run, but actually no - let's check actual implementation)
                # Actually, in dry_run we skip execution, so seed might not be set. Let's check non-dry-run path.
        
        # Test with actual call (without dry_run, but mock the backtest function)
        with patch("scripts.batch_runner.run_backtest_from_args", return_value=0):
            with patch("scripts.batch_runner.set_random_seeds") as mock_set_seed:
                run_single_backtest(
                    run_cfg=run_cfg,
                    batch_output_root=batch_output_root,
                    seed=42,
                    dry_run=False,
                )
                
                # set_random_seeds should be called with seed=42
                mock_set_seed.assert_called_once_with(42)


def test_run_single_backtest_seed_in_manifest():
    """Test that seed is written to manifest in run_single_backtest."""
    from scripts.batch_runner import run_single_backtest
    
    run_cfg = RunConfig(
        id="test_run",
        strategy="trend_baseline",
        freq=TradingFreq.DAILY,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_output_root = Path(tmpdir)
        
        # Mock run_backtest_from_args to avoid actual execution
        with patch("scripts.batch_runner.run_backtest_from_args", return_value=0):
            run_single_backtest(
                run_cfg=run_cfg,
                batch_output_root=batch_output_root,
                seed=123,
                dry_run=False,
            )
        
        # Check manifest
        run_output_dir = batch_output_root / run_cfg.id
        manifest_path = run_output_dir / "run_manifest.json"
        
        assert manifest_path.exists()
        
        import json
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # Check that seed is present and correct
        assert manifest["seed"] == 123


def test_batch_config_seed_passed_to_runs():
    """Test that batch seed is passed to each run."""
    from scripts.batch_runner import run_batch_serial
    
    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=Path("/tmp"),
        seed=999,
        runs=[
            RunConfig(
                id="run1",
                strategy="trend_baseline",
                freq=TradingFreq.DAILY,
                start_date="2020-01-01",
                end_date="2020-12-31",
            ),
        ],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_output_root = Path(tmpdir)
        
        with patch("scripts.batch_runner.run_backtest_from_args", return_value=0):
            with patch("scripts.batch_runner.set_random_seeds") as mock_set_seed:
                run_batch_serial(
                    batch_cfg=batch_cfg,
                    batch_output_root=batch_output_root,
                    dry_run=False,
                )
                
                # set_random_seeds should be called with batch seed (999)
                mock_set_seed.assert_called_with(999)


def test_deterministic_run_id_with_seed():
    """Test that run ID generation is deterministic with same seed."""
    from scripts.batch_runner import _compute_run_id_hash
    
    run_cfg = RunConfig(
        id="",  # Will be auto-generated
        strategy="trend_baseline",
        freq=TradingFreq.DAILY,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    
    # Generate run ID twice with same seed
    hash1 = _compute_run_id_hash(run_cfg, seed=42)
    hash2 = _compute_run_id_hash(run_cfg, seed=42)
    
    # Should be identical
    assert hash1 == hash2
    
    # Different seed should give different hash
    hash3 = _compute_run_id_hash(run_cfg, seed=43)
    assert hash1 != hash3

