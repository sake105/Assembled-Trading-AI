"""Tests for survivorship-bias-free universe enforcement.

Covers:
- require_active_status=True excludes delisted symbols
- WARNING emitted when require_active_status=False
- get_universe_members_pit always uses require_active_status=True by default
- EOD-style universe call smoke: require_active_status=True works end-to-end
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.assembled_core.data.universe import (
    get_universe_members,
    get_universe_members_pit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_universe_csv(tmp_path, rows: list[dict]) -> Path:
    """Write synthetic universe history CSV; return the root dir."""
    root = tmp_path / "universe_root"
    root.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "default.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRequireActiveStatusExcludesDelisted:
    def test_true_excludes_symbol_without_end_date_and_not_active_status(
        self, tmp_path
    ):
        """require_active_status=True must exclude symbol whose end_date=NaT but
        status is not 'active' (e.g., delisted with missing end_date)."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "GOOD",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
                {
                    "symbol": "DELISTED",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "delisted",  # no end_date but not active
                },
            ],
        )
        members = get_universe_members(
            as_of="2023-01-01",
            universe_name="default",
            root=root,
            require_active_status=True,
        )
        assert "GOOD" in members
        assert "DELISTED" not in members, (
            "Delisted symbol without end_date must be excluded when require_active_status=True"
        )

    def test_true_excludes_symbol_with_explicit_end_date_before_as_of(self, tmp_path):
        """Symbol with end_date before as_of must always be excluded."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "ACTIVE",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
                {
                    "symbol": "ENDED",
                    "start_date": "2000-01-01",
                    "end_date": "2020-01-01",
                    "status": "active",
                },
            ],
        )
        members = get_universe_members(
            as_of="2023-01-01",
            universe_name="default",
            root=root,
            require_active_status=True,
        )
        assert "ACTIVE" in members
        assert "ENDED" not in members

    def test_false_includes_delisted_without_end_date(self, tmp_path, caplog):
        """require_active_status=False includes all NaT-end symbols regardless of status."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "DELISTED_NO_END",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "delisted",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.data.universe"
        ):
            members = get_universe_members(
                as_of="2023-01-01",
                universe_name="default",
                root=root,
                require_active_status=False,
            )
        assert "DELISTED_NO_END" in members


class TestSurvivorshipBiasWarning:
    def test_warning_emitted_when_require_active_false(self, tmp_path, caplog):
        """[SURVIVORSHIP-BIAS-RISK] must appear in logs when require_active_status=False."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "AAPL",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.data.universe"
        ):
            get_universe_members(
                as_of="2023-01-01",
                universe_name="default",
                root=root,
                require_active_status=False,
            )

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("[SURVIVORSHIP-BIAS-RISK]" in msg for msg in warning_messages), (
            f"Expected [SURVIVORSHIP-BIAS-RISK] warning, got: {warning_messages}"
        )

    def test_no_warning_when_require_active_true(self, tmp_path, caplog):
        """[SURVIVORSHIP-BIAS-RISK] must NOT appear when require_active_status=True."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "AAPL",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.data.universe"
        ):
            get_universe_members(
                as_of="2023-01-01",
                universe_name="default",
                root=root,
                require_active_status=True,
            )

        survivorship_warnings = [
            r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING and "[SURVIVORSHIP-BIAS-RISK]" in r.message
        ]
        assert len(survivorship_warnings) == 0, (
            f"Unexpected [SURVIVORSHIP-BIAS-RISK] warning when require_active_status=True: "
            f"{survivorship_warnings}"
        )


class TestGetUniverseMembersPitDefaultsToRequireActive:
    def test_pit_defaults_require_active_status_true(self, tmp_path):
        """get_universe_members_pit default excludes delisted-with-no-end-date."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "LIVE",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
                {
                    "symbol": "GHOST",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "delisted",
                },
            ],
        )
        members = get_universe_members_pit(
            as_of="2023-01-01",
            universe_name="default",
            root=root,
            # require_active_status intentionally NOT passed — defaults to True
        )
        assert "LIVE" in members
        assert "GHOST" not in members

    def test_pit_explicit_require_active_false_still_warns(self, tmp_path, caplog):
        """Even via get_universe_members_pit, passing require_active_status=False
        triggers the [SURVIVORSHIP-BIAS-RISK] warning."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "LIVE",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
            ],
        )
        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.data.universe"
        ):
            get_universe_members_pit(
                as_of="2023-01-01",
                universe_name="default",
                root=root,
                require_active_status=False,
            )

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("[SURVIVORSHIP-BIAS-RISK]" in msg for msg in warning_messages)

    def test_eod_style_universe_call_integration_smoke(self, tmp_path):
        """Integration smoke: a typical EOD-style call with require_active_status=True
        returns only active symbols without errors."""
        root = _write_universe_csv(
            tmp_path,
            [
                {
                    "symbol": "AAPL",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "active",
                },
                {
                    "symbol": "DELISTED_CO",
                    "start_date": "2000-01-01",
                    "end_date": "2022-06-01",
                    "status": "delisted",
                },
                {
                    "symbol": "GHOST_NO_END",
                    "start_date": "2000-01-01",
                    "end_date": pd.NaT,
                    "status": "inactive",
                },
            ],
        )
        # Simulate EOD-style call: PIT lookup with require_active_status=True
        members = get_universe_members(
            as_of="2023-01-15",
            universe_name="default",
            root=root,
            require_active_status=True,
        )
        assert "AAPL" in members
        assert "DELISTED_CO" not in members
        assert "GHOST_NO_END" not in members
        assert len(members) == 1
