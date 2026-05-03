"""Tests for PIT universe lookup (A10 — survivorship bias guard).

These tests use synthetic historical membership data to verify that
get_universe_members_pit correctly enforces point-in-time membership:
symbols not yet active at as_of are excluded, delisted symbols are excluded.

Historical membership data (configs/universes/historical_membership.parquet)
is NOT yet available in production. These tests use tmp_path fixtures.

See KNOWN_ISSUES.md §0.1 for the open A10 work item.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.universe import get_universe_members, get_universe_members_pit
from src.assembled_core.errors import UniverseLookupError


def _write_history(tmp_path, rows: list[dict]):
    """Write a synthetic universe history CSV to tmp_path/default.csv.

    Returns the root directory to pass as root= to get_universe_members.
    Layout: <root>/<universe_name>.csv  (matches universe.py _universe_path).
    """
    root_dir = tmp_path / "universe_root"
    root_dir.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "default.csv", index=False)
    return root_dir


class TestGetUniverseMembersPIT:
    def test_symbol_not_yet_listed_is_excluded(self, tmp_path):
        """A symbol with start_date after as_of must not appear."""
        root = _write_history(tmp_path, [
            {"symbol": "AAPL", "start_date": "2000-01-01", "end_date": pd.NaT, "status": "active"},
            {"symbol": "TSLA", "start_date": "2020-12-21", "end_date": pd.NaT, "status": "active"},
        ])
        members = get_universe_members(
            as_of="2020-12-20",
            universe_name="default",
            root=root,
        )
        assert "TSLA" not in members
        assert "AAPL" in members

    def test_symbol_active_on_exact_start_date_is_included(self, tmp_path):
        """A symbol becomes active on its start_date (inclusive)."""
        root = _write_history(tmp_path, [
            {"symbol": "TSLA", "start_date": "2020-12-21", "end_date": pd.NaT, "status": "active"},
        ])
        members = get_universe_members(
            as_of="2020-12-21",
            universe_name="default",
            root=root,
        )
        assert "TSLA" in members

    def test_delisted_symbol_is_excluded(self, tmp_path):
        """A symbol with end_date <= as_of must not appear."""
        root = _write_history(tmp_path, [
            {"symbol": "ENRN", "start_date": "1990-01-01", "end_date": "2001-12-01", "status": "delisted"},
            {"symbol": "AAPL", "start_date": "1980-01-01", "end_date": pd.NaT, "status": "active"},
        ])
        members = get_universe_members(
            as_of="2002-01-01",
            universe_name="default",
            root=root,
        )
        assert "ENRN" not in members
        assert "AAPL" in members

    def test_pit_raises_on_no_as_of(self, tmp_path):
        """get_universe_members_pit without as_of raises UniverseLookupError."""
        with pytest.raises((UniverseLookupError, TypeError)):
            get_universe_members_pit(None, root=tmp_path / "configs")  # type: ignore[arg-type]

    def test_pit_raises_on_empty_universe(self, tmp_path):
        """get_universe_members_pit raises UniverseLookupError when no members match."""
        root = _write_history(tmp_path, [
            {"symbol": "TSLA", "start_date": "2025-01-01", "end_date": pd.NaT, "status": "active"},
        ])
        with pytest.raises(UniverseLookupError):
            get_universe_members_pit(
                as_of="2000-01-01",
                universe_name="default",
                root=root,
            )

    def test_pit_returns_members_when_available(self, tmp_path):
        """get_universe_members_pit returns correct members for a valid as_of."""
        root = _write_history(tmp_path, [
            {"symbol": "AAPL", "start_date": "2000-01-01", "end_date": pd.NaT, "status": "active"},
            {"symbol": "MSFT", "start_date": "1990-01-01", "end_date": pd.NaT, "status": "active"},
            {"symbol": "NEW_CO", "start_date": "2030-01-01", "end_date": pd.NaT, "status": "active"},
        ])
        members = get_universe_members_pit(
            as_of="2024-01-01",
            universe_name="default",
            root=root,
        )
        assert "AAPL" in members
        assert "MSFT" in members
        assert "NEW_CO" not in members

    def test_require_active_status_excludes_implicit_delistings(self, tmp_path):
        """Symbols with end_date=NaT but status != active are excluded when require_active_status=True."""
        root = _write_history(tmp_path, [
            {"symbol": "HALTED", "start_date": "2000-01-01", "end_date": pd.NaT, "status": "suspended"},
            {"symbol": "AAPL", "start_date": "2000-01-01", "end_date": pd.NaT, "status": "active"},
        ])
        members = get_universe_members(
            as_of="2024-01-01",
            universe_name="default",
            root=root,
            require_active_status=True,
        )
        assert "HALTED" not in members
        assert "AAPL" in members
