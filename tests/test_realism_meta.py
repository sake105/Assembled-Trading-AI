"""Tests for M7-T04: Realism metadata labeling.

Covers:
- build_realism_label: structure, score computation, level classification
- build_realism_label_from_policy: enabled/disabled sections, mode reading
- Edge cases: unknown modes, empty policy, all-none = "none" level
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase12

import pytest; pytest.importorskip("src.assembled_core.data.realism_meta")
from src.assembled_core.data.realism_meta import (
    REALISM_LEVELS,
    build_realism_label,
    build_realism_label_from_policy,
)


# ---------------------------------------------------------------------------
# build_realism_label
# ---------------------------------------------------------------------------


class TestBuildRealismLabel:
    def test_all_none_gives_none_level(self):
        label = build_realism_label()
        assert label["realism_level"] == "none"
        assert label["realism_score"] == 0

    def test_all_highest_gives_high_level(self):
        label = build_realism_label(
            calendar_mode="nyse",
            corporate_actions_mode="splits+dividends",
            cost_model_mode="tca",
            universe_mode="snapshot",
            data_source="real",
        )
        assert label["realism_level"] == "high"
        assert label["realism_score"] == 10

    def test_label_contains_all_required_keys(self):
        label = build_realism_label()
        required = {
            "calendar_mode",
            "corporate_actions_mode",
            "cost_model_mode",
            "universe_mode",
            "data_source",
            "notes",
            "realism_score",
            "realism_level",
        }
        assert required.issubset(label.keys())

    def test_calendar_nyse_adds_score_2(self):
        label_none = build_realism_label(calendar_mode="none")
        label_nyse = build_realism_label(calendar_mode="nyse")
        assert label_nyse["realism_score"] - label_none["realism_score"] == 2

    def test_calendar_fallback_adds_score_1(self):
        label_none = build_realism_label(calendar_mode="none")
        label_fb = build_realism_label(calendar_mode="fallback")
        assert label_fb["realism_score"] - label_none["realism_score"] == 1

    def test_splits_only_adds_score_1(self):
        base = build_realism_label()["realism_score"]
        with_splits = build_realism_label(corporate_actions_mode="splits_only")
        assert with_splits["realism_score"] - base == 1

    def test_splits_and_dividends_adds_score_2(self):
        base = build_realism_label()["realism_score"]
        with_ca = build_realism_label(corporate_actions_mode="splits+dividends")
        assert with_ca["realism_score"] - base == 2

    def test_real_data_adds_score_2(self):
        base = build_realism_label(data_source="synthetic")["realism_score"]
        real = build_realism_label(data_source="real")["realism_score"]
        assert real - base == 2

    def test_notes_preserved(self):
        label = build_realism_label(notes="custom note here")
        assert label["notes"] == "custom note here"

    def test_realism_level_is_valid(self):
        for cal in ("none", "fallback", "nyse"):
            label = build_realism_label(calendar_mode=cal)
            assert label["realism_level"] in REALISM_LEVELS

    def test_unknown_mode_treated_as_zero_score(self):
        # Unknown modes fall back to 0 in scoring
        label = build_realism_label(calendar_mode="unknown_mode")
        assert label["realism_score"] == 0

    def test_minimal_level_range(self):
        # Score 1–3 → minimal
        label = build_realism_label(calendar_mode="fallback")  # score=1
        assert label["realism_level"] == "minimal"

    def test_standard_level_range(self):
        # score 4–6 → standard
        label = build_realism_label(
            calendar_mode="nyse",  # +2
            cost_model_mode="tca",  # +2
        )
        assert label["realism_score"] == 4
        assert label["realism_level"] == "standard"

    def test_policy_mode_and_tca_both_score_1_or_2(self):
        policy = build_realism_label(cost_model_mode="policy")["realism_score"]
        tca = build_realism_label(cost_model_mode="tca")["realism_score"]
        assert tca >= policy


# ---------------------------------------------------------------------------
# build_realism_label_from_policy
# ---------------------------------------------------------------------------


class TestBuildRealismLabelFromPolicy:
    def test_empty_policy_all_disabled(self):
        label = build_realism_label_from_policy({})
        assert label["calendar_mode"] == "none"
        assert label["cost_model_mode"] == "none"
        assert label["universe_mode"] == "none"

    def test_none_policy_all_disabled(self):
        label = build_realism_label_from_policy(None)
        assert label["realism_score"] == 0

    def test_calendar_enabled_uses_nyse_default(self):
        policy = {"calendar": {"enabled": True}}
        label = build_realism_label_from_policy(policy)
        assert label["calendar_mode"] == "nyse"

    def test_calendar_enabled_custom_mode(self):
        policy = {"calendar": {"enabled": True, "mode": "fallback"}}
        label = build_realism_label_from_policy(policy)
        assert label["calendar_mode"] == "fallback"

    def test_calendar_disabled(self):
        policy = {"calendar": {"enabled": False, "mode": "nyse"}}
        label = build_realism_label_from_policy(policy)
        assert label["calendar_mode"] == "none"

    def test_cost_model_enabled(self):
        policy = {"cost_model": {"enabled": True}}
        label = build_realism_label_from_policy(policy)
        assert label["cost_model_mode"] == "policy"

    def test_universe_snapshot(self):
        policy = {"universe": {"enabled": True, "mode": "snapshot"}}
        label = build_realism_label_from_policy(policy)
        assert label["universe_mode"] == "snapshot"

    def test_all_enabled_raises_score(self):
        policy = {
            "calendar": {"enabled": True, "mode": "nyse"},
            "corporate_actions": {"enabled": True, "mode": "splits+dividends"},
            "cost_model": {"enabled": True, "mode": "tca"},
            "universe": {"enabled": True, "mode": "snapshot"},
        }
        label = build_realism_label_from_policy(policy, data_source="real")
        assert label["realism_score"] == 10
        assert label["realism_level"] == "high"

    def test_data_source_passed_through(self):
        label = build_realism_label_from_policy({}, data_source="real_partial")
        assert label["data_source"] == "real_partial"
