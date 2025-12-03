"""Tests for experiment tracking module."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.qa.experiment_tracking import ExperimentRun, ExperimentTracker


@pytest.mark.phase12
class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    def test_start_run_creates_directory_and_run_json(self, tmp_path: Path):
        """Test that start_run creates a run directory and run.json."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(
            name="test_experiment",
            config={"param1": 10, "param2": "value"},
            tags=["test", "unit"]
        )
        
        # Check run directory exists
        run_dir = tmp_path / run.run_id
        assert run_dir.exists()
        assert run_dir.is_dir()
        
        # Check artifacts directory exists
        artifacts_dir = run_dir / "artifacts"
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()
        
        # Check run.json exists
        run_json_path = run_dir / "run.json"
        assert run_json_path.exists()
        
        # Check run.json content
        with open(run_json_path, "r", encoding="utf-8") as f:
            run_data = json.load(f)
        
        assert run_data["run_id"] == run.run_id
        assert run_data["name"] == "test_experiment"
        assert run_data["config"] == {"param1": 10, "param2": "value"}
        assert run_data["tags"] == ["test", "unit"]
        assert run_data["status"] == "running"
        assert "created_at" in run_data
    
    def test_log_metrics_creates_and_appends_csv(self, tmp_path: Path):
        """Test that log_metrics creates metrics.csv and appends rows."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        # Log first set of metrics
        tracker.log_metrics(run, {"sharpe": 1.23, "max_drawdown": -0.15}, step=1)
        
        # Check metrics.csv exists
        metrics_csv_path = tmp_path / run.run_id / "metrics.csv"
        assert metrics_csv_path.exists()
        
        # Read and check content
        df = pd.read_csv(metrics_csv_path)
        assert len(df) == 2  # Two metrics
        assert set(df["metric_name"].values) == {"sharpe", "max_drawdown"}
        assert df[df["metric_name"] == "sharpe"]["metric_value"].iloc[0] == pytest.approx(1.23)
        assert df[df["metric_name"] == "max_drawdown"]["metric_value"].iloc[0] == pytest.approx(-0.15)
        # Step is read as string from CSV (empty string for None)
        assert str(df["step"].iloc[0]) == "1"
        
        # Log second set of metrics
        tracker.log_metrics(run, {"sharpe": 1.45, "max_drawdown": -0.12}, step=2)
        
        # Check that rows were appended
        df = pd.read_csv(metrics_csv_path)
        assert len(df) == 4  # Two more metrics
        
        # Check that step=None works (empty string)
        tracker.log_metrics(run, {"final_pf": 1.5})
        df = pd.read_csv(metrics_csv_path)
        final_pf_rows = df[df["metric_name"] == "final_pf"]
        assert len(final_pf_rows) == 1
        # Step is read as string from CSV (empty string for None)
        assert str(final_pf_rows["step"].iloc[0]) == "" or pd.isna(final_pf_rows["step"].iloc[0])
    
    def test_log_artifact_copies_file(self, tmp_path: Path):
        """Test that log_artifact copies a file to artifacts/."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        # Create a test artifact file
        test_file = tmp_path / "test_report.md"
        test_file.write_text("# Test Report\n\nContent here.")
        
        # Log artifact
        tracker.log_artifact(run, test_file, "qa_report.md")
        
        # Check artifact was copied
        artifact_path = tmp_path / run.run_id / "artifacts" / "qa_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == "# Test Report\n\nContent here."
        
        # Test with default target_name (uses original filename)
        test_file2 = tmp_path / "test_data.csv"
        test_file2.write_text("col1,col2\n1,2")
        
        tracker.log_artifact(run, test_file2)
        
        artifact_path2 = tmp_path / run.run_id / "artifacts" / "test_data.csv"
        assert artifact_path2.exists()
    
    def test_log_artifact_raises_if_file_not_found(self, tmp_path: Path):
        """Test that log_artifact raises FileNotFoundError if file doesn't exist."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(run, tmp_path / "nonexistent.txt")
    
    def test_finish_run_updates_status(self, tmp_path: Path):
        """Test that finish_run updates status in run.json."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        # Check initial status
        run_json_path = tmp_path / run.run_id / "run.json"
        with open(run_json_path, "r", encoding="utf-8") as f:
            run_data = json.load(f)
        assert run_data["status"] == "running"
        assert "finished_at" not in run_data
        
        # Finish run
        tracker.finish_run(run, status="finished")
        
        # Check updated status
        with open(run_json_path, "r", encoding="utf-8") as f:
            run_data = json.load(f)
        assert run_data["status"] == "finished"
        assert "finished_at" in run_data
        
        # Test failed status
        run2 = tracker.start_run(name="test_experiment2")
        tracker.finish_run(run2, status="failed")
        
        run_json_path2 = tmp_path / run2.run_id / "run.json"
        with open(run_json_path2, "r", encoding="utf-8") as f:
            run_data2 = json.load(f)
        assert run_data2["status"] == "failed"
    
    def test_finish_run_raises_on_invalid_status(self, tmp_path: Path):
        """Test that finish_run raises ValueError on invalid status."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        with pytest.raises(ValueError, match="Invalid status"):
            tracker.finish_run(run, status="invalid")
    
    def test_list_runs_returns_all_runs(self, tmp_path: Path):
        """Test that list_runs returns all experiment runs."""
        tracker = ExperimentTracker(tmp_path)
        
        # Create multiple runs
        run1 = tracker.start_run(name="experiment1", tags=["tag1"])
        run2 = tracker.start_run(name="experiment2", tags=["tag2"])
        run3 = tracker.start_run(name="experiment3", tags=["tag1", "tag2"])
        
        # List all runs
        runs = tracker.list_runs()
        assert len(runs) == 3
        
        # Check that runs are sorted by created_at (newest first)
        assert runs[0].created_at >= runs[1].created_at
        assert runs[1].created_at >= runs[2].created_at
        
        # Test filtering by tags
        runs_tag1 = tracker.list_runs(tags=["tag1"])
        assert len(runs_tag1) == 2  # run1 and run3
        
        runs_tag2 = tracker.list_runs(tags=["tag2"])
        assert len(runs_tag2) == 2  # run2 and run3
        
        runs_both = tracker.list_runs(tags=["tag1", "tag2"])
        assert len(runs_both) == 1  # Only run3 has both tags
    
    def test_get_run_metrics_returns_dataframe(self, tmp_path: Path):
        """Test that get_run_metrics returns a DataFrame."""
        tracker = ExperimentTracker(tmp_path)
        
        run = tracker.start_run(name="test_experiment")
        
        # Log some metrics
        tracker.log_metrics(run, {"sharpe": 1.23}, step=1)
        tracker.log_metrics(run, {"sharpe": 1.45}, step=2)
        
        # Get metrics as DataFrame
        df = tracker.get_run_metrics(run)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "metric_name" in df.columns
        assert "metric_value" in df.columns
        assert "step" in df.columns
        assert "timestamp" in df.columns
        
        # Test with no metrics (empty DataFrame)
        run2 = tracker.start_run(name="test_experiment2")
        df2 = tracker.get_run_metrics(run2)
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 0
        assert list(df2.columns) == ["step", "timestamp", "metric_name", "metric_value"]


@pytest.mark.phase12
class TestExperimentRun:
    """Tests for ExperimentRun dataclass."""
    
    def test_experiment_run_creation(self):
        """Test that ExperimentRun can be created with required fields."""
        run = ExperimentRun(
            run_id="20250115_143022_abc12345",
            name="test_experiment",
            created_at="2025-01-15T14:30:22",
            config={"param": 10},
            tags=["test"],
            status="running"
        )
        
        assert run.run_id == "20250115_143022_abc12345"
        assert run.name == "test_experiment"
        assert run.config == {"param": 10}
        assert run.tags == ["test"]
        assert run.status == "running"
    
    def test_experiment_run_default_status(self):
        """Test that ExperimentRun defaults to 'running' status."""
        run = ExperimentRun(
            run_id="test",
            name="test",
            created_at="2025-01-15T14:30:22",
            config={},
            tags=[]
        )
        
        assert run.status == "running"

