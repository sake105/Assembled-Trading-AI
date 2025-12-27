"""Tests for Paper Track CLI catch-up mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_paper_track import compute_date_list


@pytest.fixture
def fake_state_with_last_date(tmp_path: Path) -> Path:
    """Create a fake state file with last_run_date."""
    import json

    state_dir = tmp_path / "output" / "paper_track" / "test_strategy" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"

    state_data = {
        "strategy_name": "test_strategy",
        "last_run_date": "2025-01-10T00:00:00+00:00",
        "version": "1.0",
        "positions": [],
        "cash": 100000.0,
        "equity": 100000.0,
        "seed_capital": 100000.0,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-10T00:00:00+00:00",
        "total_trades": 0,
        "total_pnl": 0.0,
    }

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f)

    return state_path


def test_compute_date_list_catchup_with_state(fake_state_with_last_date: Path) -> None:
    """Test catch-up mode when state exists."""
    dates = compute_date_list(
        as_of=None,
        start_date=None,
        end_date=None,
        catch_up=True,
        state_path=fake_state_with_last_date,
        strategy_name="test_strategy",
    )

    # Should start from last_run_date + 1 (2025-01-11) and end at today
    # Since today is in the future relative to 2025-01-10, we can check the start date
    assert len(dates) > 0
    assert dates[0] == pd.Timestamp("2025-01-11", tz="UTC").normalize()


def test_compute_date_list_catchup_with_state_and_as_of(
    fake_state_with_last_date: Path,
) -> None:
    """Test catch-up mode when state exists and --as-of is provided."""
    dates = compute_date_list(
        as_of="2025-01-15",
        start_date=None,
        end_date=None,
        catch_up=True,
        state_path=fake_state_with_last_date,
        strategy_name="test_strategy",
    )

    # Should start from last_run_date + 1 (2025-01-11) and end at as_of (2025-01-15)
    assert len(dates) == 5  # 11, 12, 13, 14, 15
    assert dates[0] == pd.Timestamp("2025-01-11", tz="UTC").normalize()
    assert dates[-1] == pd.Timestamp("2025-01-15", tz="UTC").normalize()


def test_compute_date_list_catchup_no_state(tmp_path: Path) -> None:
    """Test catch-up mode when state does not exist."""
    state_path = tmp_path / "nonexistent_state.json"

    # Should fall back to as_of (single day)
    dates = compute_date_list(
        as_of="2025-01-15",
        start_date=None,
        end_date=None,
        catch_up=True,
        state_path=state_path,
        strategy_name="test_strategy",
    )

    assert len(dates) == 1
    assert dates[0] == pd.Timestamp("2025-01-15", tz="UTC").normalize()


def test_compute_date_list_catchup_no_state_no_as_of(tmp_path: Path) -> None:
    """Test catch-up mode when state does not exist and no --as-of."""
    state_path = tmp_path / "nonexistent_state.json"

    # Should raise ValueError
    with pytest.raises(ValueError, match="catch_up=True but no state found"):
        compute_date_list(
            as_of=None,
            start_date=None,
            end_date=None,
            catch_up=True,
            state_path=state_path,
            strategy_name="test_strategy",
        )


def test_compute_date_list_catchup_requires_state_path() -> None:
    """Test that catch-up mode requires state_path."""
    with pytest.raises(ValueError, match="catch_up=True requires state_path"):
        compute_date_list(
            as_of=None,
            start_date=None,
            end_date=None,
            catch_up=True,
            state_path=None,
            strategy_name="test_strategy",
        )


def test_compute_date_list_catchup_requires_strategy_name(
    fake_state_with_last_date: Path,
) -> None:
    """Test that catch-up mode requires strategy_name."""
    with pytest.raises(ValueError, match="catch_up=True requires strategy_name"):
        compute_date_list(
            as_of=None,
            start_date=None,
            end_date=None,
            catch_up=True,
            state_path=fake_state_with_last_date,
            strategy_name=None,
        )


def test_compute_date_list_catchup_ignored_with_explicit_range(
    fake_state_with_last_date: Path,
) -> None:
    """Test that catch-up is ignored when explicit start/end are provided."""
    dates = compute_date_list(
        as_of=None,
        start_date="2025-01-20",
        end_date="2025-01-25",
        catch_up=True,  # Should be ignored
        state_path=fake_state_with_last_date,
        strategy_name="test_strategy",
    )

    # Should use explicit range, not catch-up logic
    assert len(dates) == 6  # 20, 21, 22, 23, 24, 25
    assert dates[0] == pd.Timestamp("2025-01-20", tz="UTC").normalize()
    assert dates[-1] == pd.Timestamp("2025-01-25", tz="UTC").normalize()

