"""Tests for get_universe_members_pit (Sprint 1 / C1 — strict PIT lookup).

Verifies that the strict PIT wrapper:
  1. raises UniverseLookupError when as_of is None (no silent fallback)
  2. raises when the universe history file does not exist
  3. raises when no symbols are active at the given as_of
  4. returns the same list as get_universe_members on the happy path
  5. honors require_active_status to block delisted symbols without end_date
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.universe import (
    get_universe_members_pit,
    store_universe_history,
)
from src.assembled_core.errors import UniverseLookupError


@pytest.fixture
def universe_root(tmp_path: Path) -> Path:
    """Build a small universe with known listing/delisting dates."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "LEH", "GHOST"],
            "start_date": ["2000-01-01", "2000-01-01", "2010-01-01"],
            "end_date": [None, "2008-09-15", None],
            "status": ["active", "delisted", "delisted"],
        }
    )
    store_universe_history(df, universe_name="test_sp", root=tmp_path, format="csv")
    return tmp_path


def test_raises_when_as_of_is_none(universe_root: Path) -> None:
    with pytest.raises(UniverseLookupError) as excinfo:
        get_universe_members_pit(
            as_of=None,  # type: ignore[arg-type]
            universe_name="test_sp",
            root=universe_root,
        )
    assert "as_of is required" in str(excinfo.value)


def test_raises_when_universe_file_missing(tmp_path: Path) -> None:
    with pytest.raises(UniverseLookupError):
        get_universe_members_pit(
            as_of="2020-01-01",
            universe_name="nonexistent",
            root=tmp_path,
        )


def test_raises_when_no_active_members_at_as_of(universe_root: Path) -> None:
    # Before any symbol listed
    with pytest.raises(UniverseLookupError) as excinfo:
        get_universe_members_pit(
            as_of="1990-01-01",
            universe_name="test_sp",
            root=universe_root,
        )
    assert "zero members" in str(excinfo.value)


def test_happy_path_returns_active_members(universe_root: Path) -> None:
    # 2005: AAPL + LEH active, GHOST not yet listed
    members = get_universe_members_pit(
        as_of="2005-06-01",
        universe_name="test_sp",
        root=universe_root,
    )
    assert "AAPL" in members
    assert "LEH" in members
    assert "GHOST" not in members


def test_delisted_excluded_after_end_date(universe_root: Path) -> None:
    # 2009: LEH is delisted (end_date=2008-09-15), AAPL still active
    members = get_universe_members_pit(
        as_of="2009-01-01",
        universe_name="test_sp",
        root=universe_root,
    )
    assert "AAPL" in members
    assert "LEH" not in members


def test_require_active_status_blocks_ghost_with_no_end_date(
    universe_root: Path,
) -> None:
    # GHOST has status=delisted, end_date=None. With require_active_status=True
    # (the default), it must be excluded to avoid survivorship bias.
    members = get_universe_members_pit(
        as_of="2020-01-01",
        universe_name="test_sp",
        root=universe_root,
        require_active_status=True,
    )
    assert "GHOST" not in members
    assert "AAPL" in members
