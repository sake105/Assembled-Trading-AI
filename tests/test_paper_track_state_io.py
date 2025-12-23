"""Unit Tests for Paper Track State IO.

Tests for load_paper_state, save_paper_state, and state persistence functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.config.constants import PAPER_TRACK_STATE_VERSION
from src.assembled_core.paper.paper_track import (
    PaperTrackState,
    load_paper_state,
    save_paper_state,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_state() -> PaperTrackState:
    """Create a sample PaperTrackState for testing."""
    return PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=pd.Timestamp("2025-01-15", tz="UTC"),
        version=PAPER_TRACK_STATE_VERSION,
        positions=pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "qty": [10.0, 20.0],
            }
        ),
        cash=50000.0,
        equity=150000.0,
        seed_capital=100000.0,
        created_at=pd.Timestamp("2025-01-01", tz="UTC"),
        updated_at=pd.Timestamp("2025-01-15", tz="UTC"),
        total_trades=5,
        total_pnl=50000.0,
        last_equity=140000.0,
        last_positions_value=100000.0,
    )


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Create a temporary state file path."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "state.json"


@pytest.mark.advanced
def test_save_paper_state_creates_file(
    sample_state: PaperTrackState, temp_state_file: Path
):
    """Test that save_paper_state creates a valid JSON file."""
    save_paper_state(sample_state, temp_state_file)

    assert temp_state_file.exists(), "State file should be created"

    # Read and validate JSON
    with open(temp_state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["strategy_name"] == "test_strategy"
    assert data["cash"] == 50000.0
    assert data["equity"] == 150000.0
    assert data["seed_capital"] == 100000.0
    assert data["version"] == PAPER_TRACK_STATE_VERSION
    assert data["total_trades"] == 5
    assert data["total_pnl"] == 50000.0
    assert data["last_run_date"] == "2025-01-15T00:00:00+00:00"
    assert len(data["positions"]) == 2
    # v2.0+ fields
    assert "last_equity" in data
    assert "last_positions_value" in data
    assert data["last_equity"] == 140000.0
    assert data["last_positions_value"] == 100000.0


@pytest.mark.advanced
def test_load_paper_state_loads_correctly(
    sample_state: PaperTrackState, temp_state_file: Path
):
    """Test that load_paper_state loads state correctly."""
    # Save state first
    save_paper_state(sample_state, temp_state_file)

    # Load state
    loaded_state = load_paper_state(temp_state_file, "test_strategy")

    assert loaded_state is not None, "State should be loaded"
    assert loaded_state.strategy_name == sample_state.strategy_name
    assert loaded_state.cash == sample_state.cash
    assert loaded_state.equity == sample_state.equity
    assert loaded_state.seed_capital == sample_state.seed_capital
    assert loaded_state.version == sample_state.version
    assert loaded_state.total_trades == sample_state.total_trades
    assert loaded_state.total_pnl == sample_state.total_pnl
    assert loaded_state.last_run_date == sample_state.last_run_date

    # Check positions DataFrame
    pd.testing.assert_frame_equal(
        loaded_state.positions.reset_index(drop=True),
        sample_state.positions.reset_index(drop=True),
    )


@pytest.mark.advanced
def test_load_paper_state_returns_none_if_not_exists(tmp_path: Path):
    """Test that load_paper_state returns None if file doesn't exist."""
    state_file = tmp_path / "nonexistent" / "state.json"

    loaded_state = load_paper_state(state_file, "test_strategy")

    assert loaded_state is None, "Should return None for non-existent file"


@pytest.mark.advanced
def test_load_paper_state_validates_strategy_name(
    sample_state: PaperTrackState, temp_state_file: Path
):
    """Test that load_paper_state validates strategy name."""
    # Save state with one strategy name
    save_paper_state(sample_state, temp_state_file)

    # Try to load with different strategy name
    with pytest.raises(ValueError, match="strategy_name mismatch"):
        load_paper_state(temp_state_file, "different_strategy")


@pytest.mark.advanced
def test_save_paper_state_creates_backup(
    sample_state: PaperTrackState, temp_state_file: Path
):
    """Test that save_paper_state creates a backup when file exists."""
    # Save state first time
    save_paper_state(sample_state, temp_state_file)

    # Modify state
    sample_state.cash = 60000.0
    sample_state.updated_at = pd.Timestamp("2025-01-16", tz="UTC")

    # Save again (should create backup)
    save_paper_state(sample_state, temp_state_file)

    backup_path = temp_state_file.with_suffix(temp_state_file.suffix + ".backup")
    assert backup_path.exists(), "Backup file should be created"

    # Check backup has old data
    with open(backup_path, "r", encoding="utf-8") as f:
        backup_data = json.load(f)

    assert backup_data["cash"] == 50000.0, "Backup should have old cash value"


@pytest.mark.advanced
def test_save_paper_state_with_empty_positions(tmp_path: Path):
    """Test that save_paper_state handles empty positions DataFrame."""
    state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=None,
        positions=pd.DataFrame(columns=["symbol", "qty"]),
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )

    state_file = tmp_path / "state" / "state.json"
    save_paper_state(state, state_file)

    # Load and verify
    loaded_state = load_paper_state(state_file, "test_strategy")
    assert loaded_state is not None
    assert loaded_state.positions.empty
    assert len(loaded_state.positions.columns) == 2  # symbol, qty columns


@pytest.mark.advanced
def test_save_paper_state_atomic_write(
    sample_state: PaperTrackState, temp_state_file: Path
):
    """Test that save_paper_state uses atomic write (temp file then rename)."""
    # Save state
    save_paper_state(sample_state, temp_state_file)

    # Temp file should not exist after successful write
    temp_path = temp_state_file.with_suffix(temp_state_file.suffix + ".tmp")
    assert not temp_path.exists(), "Temp file should be removed after successful write"

    # Final file should exist
    assert temp_state_file.exists(), "State file should exist after save"


@pytest.mark.advanced
def test_load_paper_state_handles_missing_last_run_date(tmp_path: Path):
    """Test that load_paper_state handles missing last_run_date (None)."""
    state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=None,
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )

    state_file = tmp_path / "state" / "state.json"
    save_paper_state(state, state_file)

    # Load and verify
    loaded_state = load_paper_state(state_file, "test_strategy")
    assert loaded_state is not None
    assert loaded_state.last_run_date is None
