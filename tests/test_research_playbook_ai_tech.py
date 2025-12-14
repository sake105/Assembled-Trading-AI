"""Tests for AI/Tech Multi-Factor ML Alpha Research Playbook.

This module tests the end-to-end research playbook workflow, including
configuration, model selection, and summary generation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook import (
    PlaybookConfig,
    create_default_config,
    select_best_model,
    write_research_summary,
)

pytestmark = pytest.mark.advanced


@pytest.mark.advanced
def test_playbook_config_basic(tmp_path: Path):
    """Test that PlaybookConfig can be created with default values."""
    config = create_default_config()
    
    # Check that all required attributes are present
    assert hasattr(config, "data_root")
    assert hasattr(config, "factor_panel_dir")
    assert hasattr(config, "ml_output_dir")
    assert hasattr(config, "backtest_root")
    assert hasattr(config, "risk_output_dir")
    
    assert hasattr(config, "universe_file")
    assert hasattr(config, "core_bundle_path")
    assert hasattr(config, "core_ml_bundle_path")
    assert hasattr(config, "ml_alpha_bundle_path")
    
    assert hasattr(config, "horizon_days")
    assert hasattr(config, "freq")
    assert hasattr(config, "start_date")
    assert hasattr(config, "end_date")
    assert hasattr(config, "factor_set")
    assert hasattr(config, "n_splits")
    
    # Check default values
    assert config.horizon_days == 20
    assert config.freq == "1d"
    assert config.n_splits == 5
    assert config.rebalance_freq == "M"
    assert config.max_gross_exposure == 1.0
    assert config.start_capital == 100000.0
    
    # Check that paths are Path objects
    assert isinstance(config.universe_file, Path)
    assert isinstance(config.factor_panel_dir, Path)
    assert isinstance(config.ml_output_dir, Path)


@pytest.mark.advanced
def test_select_best_model_prefers_high_ic_ir():
    """Test that select_best_model selects model with highest IC-IR."""
    # Create synthetic model zoo DataFrame
    model_zoo_df = pd.DataFrame({
        "model_name": ["linear", "ridge_0_1", "ridge_1_0"],
        "model_type": ["linear", "ridge", "ridge"],
        "ic_ir": [0.5, 1.2, 0.8],  # ridge_0_1 has highest IC-IR
        "test_r2_mean": [0.3, 0.4, 0.35],
        "test_mse_mean": [0.1, 0.08, 0.09],
        "ls_sharpe": [0.6, 1.1, 0.9],
    })
    
    best_model = select_best_model(model_zoo_df)
    
    # Should select ridge_0_1 (highest IC-IR)
    assert best_model["model_name"] == "ridge_0_1"
    assert best_model["model_type"] == "ridge"
    assert best_model["ic_ir"] == 1.2
    assert "model_params" in best_model
    assert "alpha" in best_model["model_params"]
    assert best_model["model_params"]["alpha"] == 0.1


@pytest.mark.advanced
def test_select_best_model_fallback_to_test_r2():
    """Test that select_best_model falls back to test_r2_mean if IC-IR is missing."""
    # Create synthetic model zoo DataFrame without IC-IR
    model_zoo_df = pd.DataFrame({
        "model_name": ["linear", "ridge_0_1", "rf_depth_5"],
        "model_type": ["linear", "ridge", "random_forest"],
        "test_r2_mean": [0.3, 0.5, 0.4],  # ridge_0_1 has highest R²
        "test_mse_mean": [0.1, 0.08, 0.09],
    })
    
    best_model = select_best_model(model_zoo_df)
    
    # Should select ridge_0_1 (highest test_r2_mean)
    assert best_model["model_name"] == "ridge_0_1"
    assert best_model["test_r2_mean"] == 0.5


@pytest.mark.advanced
def test_select_best_model_handles_nan_values():
    """Test that select_best_model handles NaN values correctly."""
    # Create synthetic model zoo DataFrame with some NaN values
    model_zoo_df = pd.DataFrame({
        "model_name": ["linear", "ridge_0_1", "lasso_0_01"],
        "model_type": ["linear", "ridge", "lasso"],
        "ic_ir": [0.5, np.nan, 0.8],  # ridge_0_1 has NaN
        "test_r2_mean": [0.3, 0.4, 0.35],
    })
    
    best_model = select_best_model(model_zoo_df)
    
    # Should select lasso_0_01 (highest valid IC-IR, ignoring NaN)
    assert best_model["model_name"] == "lasso_0_01"
    assert best_model["ic_ir"] == 0.8


@pytest.mark.advanced
def test_write_research_summary_creates_file(tmp_path: Path):
    """Test that write_research_summary creates a Markdown file with expected content."""
    # Create dummy config
    config = PlaybookConfig(
        data_root=tmp_path / "data",
        factor_panel_dir=tmp_path / "factor_panels",
        ml_output_dir=tmp_path / "ml_output",
        backtest_root=tmp_path / "backtests",
        risk_output_dir=tmp_path / "risk_reports",
        universe_file=tmp_path / "universe.txt",
        core_bundle_path=tmp_path / "core_bundle.yaml",
        core_ml_bundle_path=tmp_path / "core_ml_bundle.yaml",
        ml_alpha_bundle_path=tmp_path / "ml_alpha_bundle.yaml",
        horizon_days=20,
        freq="1d",
        start_date="2015-01-01",
        end_date="2025-12-03",
        factor_set="core+alt_full",
    )
    
    # Create dummy artifacts
    model_zoo_df = pd.DataFrame({
        "model_name": ["linear", "ridge_0_1"],
        "model_type": ["linear", "ridge"],
        "ic_ir": [0.5, 1.2],
        "test_r2_mean": [0.3, 0.4],
        "ls_sharpe": [0.6, 1.1],
    })
    
    best_model = {
        "model_name": "ridge_0_1",
        "model_type": "ridge",
        "model_params": {"alpha": 0.1},
        "label_col": "fwd_return_20d",
        "ic_ir": 1.2,
        "test_r2_mean": 0.4,
        "ls_sharpe": 1.1,
    }
    
    backtest_dir1 = tmp_path / "backtests" / "core_only_20250101_120000"
    backtest_dir1.mkdir(parents=True)
    
    # Create dummy risk_summary.csv
    risk_summary_df = pd.DataFrame({
        "sharpe": [0.8],
        "sortino": [1.0],
        "max_drawdown": [-0.15],
        "mean_return_annualized": [0.12],
    })
    risk_summary_df.to_csv(backtest_dir1 / "risk_summary.csv", index=False)
    
    artifacts = {
        "factor_panel_path": tmp_path / "factor_panel.parquet",
        "model_zoo_df": model_zoo_df,
        "best_model": best_model,
        "ml_alpha_panel_path": tmp_path / "ml_alpha_panel.parquet",
        "backtest_dirs": [backtest_dir1],
        "risk_report_paths": {
            "core_only_20250101_120000": backtest_dir1 / "risk_report.md",
        },
    }
    
    # Call write_research_summary
    summary_path = write_research_summary(config, artifacts)
    
    # Check that file was created
    assert summary_path.exists()
    assert summary_path.suffix == ".md"
    
    # Read and check content
    content = summary_path.read_text(encoding="utf-8")
    
    # Check for key sections
    assert "Research Summary" in content
    assert "Universe" in content or "universe" in content.lower()
    assert "Horizon" in content or "horizon" in content.lower()
    assert "ML Model Comparison" in content
    assert "Best Model Selection" in content
    assert "Backtest Performance Comparison" in content
    assert "ridge_0_1" in content  # Best model name should appear
    assert "1.2" in content  # IC-IR value should appear
    
    # Check that it's valid Markdown (has headers)
    assert "#" in content
    assert "##" in content


@pytest.mark.advanced
def test_write_research_summary_handles_missing_data(tmp_path: Path):
    """Test that write_research_summary handles missing artifacts gracefully."""
    config = PlaybookConfig(
        data_root=tmp_path / "data",
        factor_panel_dir=tmp_path / "factor_panels",
        ml_output_dir=tmp_path / "ml_output",
        backtest_root=tmp_path / "backtests",
        risk_output_dir=tmp_path / "risk_reports",
        universe_file=tmp_path / "universe.txt",
        core_bundle_path=tmp_path / "core_bundle.yaml",
        core_ml_bundle_path=tmp_path / "core_ml_bundle.yaml",
        ml_alpha_bundle_path=tmp_path / "ml_alpha_bundle.yaml",
        horizon_days=20,
        freq="1d",
        start_date="2015-01-01",
        end_date="2025-12-03",
        factor_set="core",
    )
    
    # Minimal artifacts (missing some data)
    artifacts = {
        "factor_panel_path": tmp_path / "factor_panel.parquet",
        # Missing model_zoo_df, best_model, etc.
        "backtest_dirs": [],
        "risk_report_paths": {},
    }
    
    # Should not raise exception
    summary_path = write_research_summary(config, artifacts)
    
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "Research Summary" in content


@pytest.mark.advanced
def test_playbook_main_dry_run(monkeypatch, tmp_path: Path):
    """Test that main() can run in dry-run mode with mocked functions."""
    from research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook import main
    
    # Create dummy config that uses tmp_path
    def mock_create_default_config():
        return PlaybookConfig(
            data_root=tmp_path / "data",
            factor_panel_dir=tmp_path / "factor_panels",
            ml_output_dir=tmp_path / "ml_output",
            backtest_root=tmp_path / "backtests",
            risk_output_dir=tmp_path / "risk_reports",
            universe_file=tmp_path / "universe.txt",
            core_bundle_path=tmp_path / "core_bundle.yaml",
            core_ml_bundle_path=tmp_path / "core_ml_bundle.yaml",
            ml_alpha_bundle_path=tmp_path / "ml_alpha_bundle.yaml",
            horizon_days=20,
            freq="1d",
            start_date="2015-01-01",
            end_date="2025-12-03",
            factor_set="core",
        )
    
    # Mock all heavy functions to return dummy paths
    def mock_run_factor_panel_export(config):
        output_path = config.factor_panel_dir / "factor_panel_test.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty parquet file
        pd.DataFrame({"timestamp": [], "symbol": []}).to_parquet(output_path, index=False)
        return output_path
    
    def mock_run_ml_model_zoo(config, factor_panel_path):
        return pd.DataFrame({
            "model_name": ["ridge_0_1"],
            "model_type": ["ridge"],
            "ic_ir": [1.2],
            "test_r2_mean": [0.4],
        })
    
    def mock_run_ml_alpha_export(config, factor_panel_path, best_model):
        output_path = config.ml_output_dir / "ml_alpha_factors" / "ml_alpha_panel_test.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [], "symbol": []}).to_parquet(output_path, index=False)
        return output_path
    
    def mock_run_backtests_with_bundles(config, ml_alpha_panel_path):
        backtest_dir = config.backtest_root / "core_only_test"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy equity curve
        pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
            "equity": [100000.0] * 10,
        }).to_csv(backtest_dir / "equity_curve.csv", index=False)
        return [backtest_dir]
    
    def mock_run_risk_reports(config, backtest_dirs, ml_alpha_panel_path):
        risk_report_paths = {}
        for backtest_dir in backtest_dirs:
            risk_report_md = backtest_dir / "risk_report.md"
            risk_report_md.write_text("# Risk Report\n\nTest content", encoding="utf-8")
            risk_report_paths[backtest_dir.name] = risk_report_md
        return risk_report_paths
    
    def mock_write_research_summary(config, artifacts):
        output_dir = config.risk_output_dir / "research_summaries"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "research_summary_test.md"
        summary_path.write_text("# Research Summary\n\nTest content", encoding="utf-8")
        return summary_path
    
    # Apply monkeypatches
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.create_default_config",
        mock_create_default_config,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.run_factor_panel_export",
        mock_run_factor_panel_export,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.run_ml_model_zoo",
        mock_run_ml_model_zoo,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.select_best_model",
        lambda df: {
            "model_name": "ridge_0_1",
            "model_type": "ridge",
            "model_params": {"alpha": 0.1},
            "label_col": "fwd_return_20d",
            "ic_ir": 1.2,
            "test_r2_mean": 0.4,
        },
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.run_ml_alpha_export",
        mock_run_ml_alpha_export,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.run_backtests_with_bundles",
        mock_run_backtests_with_bundles,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.run_risk_reports",
        mock_run_risk_reports,
    )
    monkeypatch.setattr(
        "research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook.write_research_summary",
        mock_write_research_summary,
    )
    
    # Create dummy universe file (needed for some checks)
    (tmp_path / "universe.txt").write_text("AAPL\nMSFT\n", encoding="utf-8")
    
    # Run main() - should not raise exceptions
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised {type(e).__name__}: {e}")
    
    # Verify that output directories were created
    assert (tmp_path / "factor_panels").exists()
    assert (tmp_path / "ml_output").exists()
    assert (tmp_path / "backtests").exists()
    assert (tmp_path / "risk_reports").exists()
    
    # Verify that summary was created
    summary_files = list((tmp_path / "risk_reports" / "research_summaries").glob("*.md"))
    assert len(summary_files) > 0

