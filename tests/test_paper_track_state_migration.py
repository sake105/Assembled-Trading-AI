"""Tests for Paper-Track state versioning and migration."""

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
def v1_state_dict() -> dict:
    """Create a v1.0 state dictionary (without new fields)."""
    return {
        "strategy_name": "test_strategy",
        "last_run_date": "2025-01-15T00:00:00+00:00",
        "version": "1.0",
        "positions": [{"symbol": "AAPL", "qty": 10.0}],
        "cash": 50000.0,
        "equity": 100000.0,
        "seed_capital": 100000.0,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-15T00:00:00+00:00",
        "total_trades": 5,
        "total_pnl": 0.0,
        # Note: v1.0 does NOT have last_equity or last_positions_value
    }


@pytest.fixture
def v2_state_dict() -> dict:
    """Create a v2.0 state dictionary (with new fields)."""
    return {
        "strategy_name": "test_strategy",
        "last_run_date": "2025-01-15T00:00:00+00:00",
        "version": "2.0",
        "positions": [{"symbol": "AAPL", "qty": 10.0}],
        "cash": 50000.0,
        "equity": 100000.0,
        "seed_capital": 100000.0,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-15T00:00:00+00:00",
        "total_trades": 5,
        "total_pnl": 0.0,
        "last_equity": 99000.0,
        "last_positions_value": 50000.0,
    }


def test_load_v1_state_migrates_to_v2(
    tmp_path: Path, v1_state_dict: dict
) -> None:
    """Test that loading a v1.0 state file migrates to v2.0 with default values."""
    state_path = tmp_path / "state_v1.json"

    # Write v1.0 state file
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(v1_state_dict, f)

    # Load state (should migrate from v1.0 to v2.0)
    state = load_paper_state(state_path, strategy_name="test_strategy")

    assert state is not None
    assert state.version == PAPER_TRACK_STATE_VERSION  # Should be migrated to current version
    assert state.strategy_name == "test_strategy"
    assert state.equity == 100000.0
    assert state.cash == 50000.0
    # New v2.0 fields should be None (defaults from migration)
    assert state.last_equity is None
    assert state.last_positions_value is None


def test_load_v2_state_preserves_fields(
    tmp_path: Path, v2_state_dict: dict
) -> None:
    """Test that loading a v2.0 state file preserves all fields."""
    state_path = tmp_path / "state_v2.json"

    # Write v2.0 state file
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(v2_state_dict, f)

    # Load state
    state = load_paper_state(state_path, strategy_name="test_strategy")

    assert state is not None
    assert state.version == "2.0"
    assert state.strategy_name == "test_strategy"
    assert state.equity == 100000.0
    assert state.cash == 50000.0
    # v2.0 fields should be preserved
    assert state.last_equity == 99000.0
    assert state.last_positions_value == 50000.0


def test_save_load_roundtrip_preserves_v2_fields(tmp_path: Path) -> None:
    """Test that save -> load roundtrip preserves all v2.0 fields."""
    state_path = tmp_path / "state.json"

    # Create a v2.0 state
    original_state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=pd.Timestamp("2025-01-15", tz="UTC"),
        version=PAPER_TRACK_STATE_VERSION,
        positions=pd.DataFrame([{"symbol": "AAPL", "qty": 10.0}]),
        cash=50000.0,
        equity=100000.0,
        seed_capital=100000.0,
        created_at=pd.Timestamp("2025-01-01", tz="UTC"),
        updated_at=pd.Timestamp("2025-01-15", tz="UTC"),
        total_trades=5,
        total_pnl=0.0,
        last_equity=99000.0,
        last_positions_value=50000.0,
    )

    # Save state
    save_paper_state(original_state, state_path)

    # Load state
    loaded_state = load_paper_state(state_path, strategy_name="test_strategy")

    assert loaded_state is not None
    assert loaded_state.version == PAPER_TRACK_STATE_VERSION
    assert loaded_state.strategy_name == original_state.strategy_name
    assert loaded_state.equity == original_state.equity
    assert loaded_state.cash == original_state.cash
    assert loaded_state.last_equity == original_state.last_equity
    assert loaded_state.last_positions_value == original_state.last_positions_value
    assert len(loaded_state.positions) == len(original_state.positions)
    assert loaded_state.positions.iloc[0]["symbol"] == "AAPL"
    assert loaded_state.positions.iloc[0]["qty"] == 10.0


def test_load_state_without_version_defaults_to_v1(tmp_path: Path) -> None:
    """Test that state file without version field defaults to v1.0 and migrates."""
    state_path = tmp_path / "state_no_version.json"

    # Write state file without version field (old format)
    state_dict = {
        "strategy_name": "test_strategy",
        "last_run_date": "2025-01-15T00:00:00+00:00",
        "positions": [{"symbol": "AAPL", "qty": 10.0}],
        "cash": 50000.0,
        "equity": 100000.0,
        "seed_capital": 100000.0,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-15T00:00:00+00:00",
        "total_trades": 5,
        "total_pnl": 0.0,
        # No version field
    }

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f)

    # Load state (should default to v1.0 and migrate)
    state = load_paper_state(state_path, strategy_name="test_strategy")

    assert state is not None
    assert state.version == PAPER_TRACK_STATE_VERSION  # Should be migrated
    assert state.last_equity is None  # Default from migration
    assert state.last_positions_value is None  # Default from migration


def test_save_state_updates_version_to_current(tmp_path: Path) -> None:
    """Test that save_paper_state updates version to current if state has old version."""
    state_path = tmp_path / "state.json"

    # Create state with old version
    old_state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=None,
        version="1.0",  # Old version
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )

    # Save state (should update version to current)
    save_paper_state(old_state, state_path)

    # Load and verify version was updated
    loaded_state = load_paper_state(state_path, strategy_name="test_strategy")
    assert loaded_state is not None
    assert loaded_state.version == PAPER_TRACK_STATE_VERSION

    # Verify file contains current version
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == PAPER_TRACK_STATE_VERSION

