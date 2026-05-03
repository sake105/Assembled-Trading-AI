"""Tests for PITGuard.validate_universe (Sprint 1 / C4b — universe-mode).

Verifies that the universe-mode PIT check correctly flags symbols that
were not part of the stored index at the given as_of timestamp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("src.assembled_core.data.pit_guard")
from src.assembled_core.data.pit_guard import PITGuard, PITViolationError
from src.assembled_core.data.universe import store_universe_history
from src.assembled_core.errors import UniverseLookupError


@pytest.fixture
def universe_root(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "LEH"],
            "start_date": ["2000-01-01", "2000-01-01"],
            "end_date": [None, "2008-09-15"],
            "status": ["active", "delisted"],
        }
    )
    store_universe_history(df, universe_name="uni", root=tmp_path, format="csv")
    return tmp_path


def test_universe_validate_happy_path(universe_root: Path) -> None:
    guard = PITGuard(as_of=pd.Timestamp("2005-06-01", tz="UTC"), mode="assert")
    assert guard.validate_universe(["AAPL", "LEH"], universe_name="uni", root=universe_root)


def test_universe_validate_assert_raises_on_delisted(universe_root: Path) -> None:
    # LEH was delisted 2008-09-15 → cannot be in 2020 universe
    guard = PITGuard(as_of=pd.Timestamp("2020-01-01", tz="UTC"), mode="assert")
    with pytest.raises(PITViolationError) as excinfo:
        guard.validate_universe(["AAPL", "LEH"], universe_name="uni", root=universe_root)
    assert "LEH" in str(excinfo.value)


def test_universe_validate_warn_returns_false(universe_root: Path) -> None:
    guard = PITGuard(as_of=pd.Timestamp("2020-01-01", tz="UTC"), mode="warn")
    ok = guard.validate_universe(["AAPL", "LEH"], universe_name="uni", root=universe_root)
    assert ok is False


def test_universe_validate_rejects_unknown_symbol(universe_root: Path) -> None:
    guard = PITGuard(as_of=pd.Timestamp("2005-06-01", tz="UTC"), mode="assert")
    with pytest.raises(PITViolationError):
        guard.validate_universe(["AAPL", "UNKNOWN_XYZ"], universe_name="uni", root=universe_root)


def test_universe_validate_empty_input_returns_true(universe_root: Path) -> None:
    guard = PITGuard(as_of=pd.Timestamp("2005-06-01", tz="UTC"), mode="assert")
    assert guard.validate_universe([], universe_name="uni", root=universe_root)


def test_universe_validate_missing_universe_raises_lookup_error(tmp_path: Path) -> None:
    guard = PITGuard(as_of=pd.Timestamp("2005-06-01", tz="UTC"), mode="assert")
    with pytest.raises(UniverseLookupError):
        guard.validate_universe(["AAPL"], universe_name="nope", root=tmp_path)
