"""Tests for news_taxonomy: categorize_event + aggregate_categories_by_window."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.intel.news_taxonomy import (
    aggregate_categories_by_window,
    categorize_event,
)


class TestCategorizeEvent:
    def test_direct_trigger_banking_crisis(self):
        assert categorize_event(trigger_type="BANKING_CRISIS") == "FINANZEN"

    def test_direct_trigger_war_escalation(self):
        assert categorize_event(trigger_type="WAR_ESCALATION") == "KONFLIKTE"

    def test_direct_trigger_energy_supply_risk(self):
        assert categorize_event(trigger_type="ENERGY_SUPPLY_RISK") == "ROHSTOFFE"

    def test_direct_trigger_new_export_control(self):
        assert categorize_event(trigger_type="NEW_EXPORT_CONTROL") == "TECHNOLOGIE"

    def test_direct_trigger_policy_shift(self):
        assert categorize_event(trigger_type="POLICY_SHIFT") == "POLITIK"

    def test_direct_trigger_sanctions(self):
        assert categorize_event(trigger_type="SANCTIONS_ESCALATION") == "GEOPOLITIK"

    def test_trigger_case_insensitive(self):
        assert categorize_event(trigger_type="banking_crisis") == "FINANZEN"

    def test_voting_majority_financial(self):
        # 2 financial + 1 military → FINANZEN
        result = categorize_event(
            event_types=["banking_crisis", "rate_surprise", "war_escalation"]
        )
        assert result == "FINANZEN"

    def test_voting_single_label(self):
        assert categorize_event(event_types=["oil"]) == "ROHSTOFFE"

    def test_priority_tiebreak_finanzen_over_konflikte(self):
        # Tie: 1 finanzen + 1 konflikte → FINANZEN wins (higher priority)
        result = categorize_event(event_types=["banking_crisis", "war_escalation"])
        assert result == "FINANZEN"

    def test_priority_tiebreak_konflikte_over_geopolitik(self):
        result = categorize_event(event_types=["military_buildup", "sanctions"])
        assert result == "KONFLIKTE"

    def test_fallback_unknown_trigger(self):
        assert categorize_event(trigger_type="COMPLETELY_UNKNOWN") == "SONSTIGE"

    def test_fallback_no_args(self):
        assert categorize_event() == "SONSTIGE"

    def test_fallback_empty_event_types(self):
        assert categorize_event(event_types=[]) == "SONSTIGE"

    def test_trigger_overrides_event_types(self):
        # trigger_type takes priority over event_types vote
        result = categorize_event(
            event_types=["war_escalation", "military_buildup"],
            trigger_type="BANKING_CRISIS",
        )
        assert result == "FINANZEN"


class TestAggregateCategories:
    def _make_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "published_at": pd.to_datetime(
                    ["2026-05-01 10:00", "2026-05-01 14:00", "2026-05-02 09:00"],
                    utc=True,
                ),
                "symbol": ["AAPL", "NVDA", "AAPL"],
                "category": ["FINANZEN", "KONFLIKTE", "ROHSTOFFE"],
            }
        )

    def test_returns_dataframe(self):
        result = aggregate_categories_by_window(self._make_events())
        assert isinstance(result, pd.DataFrame)

    def test_unique_date_symbol_pairs(self):
        result = aggregate_categories_by_window(self._make_events())
        assert (
            len(result) == 3
        )  # (2026-05-01, AAPL), (2026-05-01, NVDA), (2026-05-02, AAPL)

    def test_category_columns_present(self):
        result = aggregate_categories_by_window(self._make_events())
        for cat in [
            "finanzen",
            "konflikte",
            "geopolitik",
            "rohstoffe",
            "technologie",
            "politik",
        ]:
            col = f"news_count_{cat}_24h"
            assert col in result.columns, f"Missing column: {col}"

    def test_counts_correct(self):
        result = aggregate_categories_by_window(self._make_events())
        result = result.set_index(["date", "symbol"])
        aapl_may1 = result.loc[
            (pd.Timestamp("2026-05-01", tz="UTC"), "AAPL"), "news_count_finanzen_24h"
        ]
        assert aapl_may1 == 1
        nvda_may1 = result.loc[
            (pd.Timestamp("2026-05-01", tz="UTC"), "NVDA"), "news_count_konflikte_24h"
        ]
        assert nvda_may1 == 1

    def test_empty_input_returns_empty(self):
        result = aggregate_categories_by_window(pd.DataFrame())
        assert result.empty

    def test_list_tickers_exploded(self):
        df = pd.DataFrame(
            {
                "published_at": pd.to_datetime(["2026-05-01 10:00"], utc=True),
                "tickers": [["AAPL", "MSFT"]],
                "category": ["FINANZEN"],
            }
        )
        result = aggregate_categories_by_window(df)
        assert len(result) == 2  # one row per symbol
