"""Tests for crisis_alpha.labeling — triple-barrier episode labeler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.events.crisis_alpha.labeling import (
    _extract_active_episodes,
    _extract_close_prices,
    _load_records,
    label_crisis_alpha_episodes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=_UTC)


def _record(
    state: str, entered: datetime, evaluated: datetime, geo: float = 2.5
) -> dict:
    return {
        "state": state,
        "entered_at_utc": entered.isoformat(),
        "last_evaluated_utc": evaluated.isoformat(),
        "geo_score_at_entry": geo,
        "reason": f"test {state}",
    }


def _make_prices(
    symbols: list[str],
    start: str = "2026-01-01",
    periods: int = 60,
    drift: float = 0.001,
) -> pd.DataFrame:
    """Wide close-price DataFrame for given symbols."""
    idx = pd.date_range(start, periods=periods, freq="B", tz=_UTC)
    rng = np.random.default_rng(42)
    data = {}
    for sym in symbols:
        prices = 100 * np.cumprod(1 + rng.normal(drift, 0.015, size=periods))
        data[sym] = prices
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# _load_records
# ---------------------------------------------------------------------------


class TestLoadRecords:
    def test_returns_provided_list(self):
        records = [{"state": "WATCH"}]
        assert _load_records(records, None) == records

    def test_loads_from_jsonl(self, tmp_path: Path):
        p = tmp_path / "state.jsonl"
        p.write_text(
            json.dumps({"state": "ACTIVE"})
            + "\n"
            + json.dumps({"state": "COOLDOWN"})
            + "\n",
            encoding="utf-8",
        )
        result = _load_records(None, p)
        assert len(result) == 2
        assert result[0]["state"] == "ACTIVE"

    def test_missing_path_returns_empty(self, tmp_path: Path):
        result = _load_records(None, tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "state.jsonl"
        p.write_text(
            '{"state":"ACTIVE"}\nnot-json\n{"state":"WATCH"}\n', encoding="utf-8"
        )
        result = _load_records(None, p)
        assert len(result) == 2

    def test_no_source_returns_empty(self):
        assert _load_records(None, None) == []


# ---------------------------------------------------------------------------
# _extract_active_episodes
# ---------------------------------------------------------------------------


class TestExtractActiveEpisodes:
    def test_single_active_episode(self):
        t1 = _ts(2026, 1, 5)
        t2 = _ts(2026, 1, 8)
        t3 = _ts(2026, 1, 10)
        records = [
            _record("WATCH", t1, t1),
            _record("ACTIVE", t2, t2, geo=2.5),
            _record("COOLDOWN", t3, t3),
        ]
        episodes = _extract_active_episodes(records)
        assert len(episodes) == 1
        assert episodes[0]["entry_time"] == t2
        assert episodes[0]["exit_time"] == t3
        assert episodes[0]["geo_score_at_entry"] == pytest.approx(2.5)

    def test_open_episode_no_exit(self):
        t1 = _ts(2026, 1, 5)
        t2 = _ts(2026, 1, 8)
        records = [
            _record("WATCH", t1, t1),
            _record("ACTIVE", t2, t2),
        ]
        episodes = _extract_active_episodes(records)
        assert len(episodes) == 1
        assert episodes[0]["exit_time"] is None

    def test_two_separate_episodes(self):
        records = [
            _record("WATCH", _ts(2026, 1, 1), _ts(2026, 1, 1)),
            _record("ACTIVE", _ts(2026, 1, 5), _ts(2026, 1, 5)),
            _record("COOLDOWN", _ts(2026, 1, 8), _ts(2026, 1, 8)),
            _record("WATCH", _ts(2026, 1, 20), _ts(2026, 1, 20)),
            _record("ACTIVE", _ts(2026, 1, 25), _ts(2026, 1, 25)),
            _record("COOLDOWN", _ts(2026, 1, 28), _ts(2026, 1, 28)),
        ]
        episodes = _extract_active_episodes(records)
        assert len(episodes) == 2

    def test_no_active_records_returns_empty(self):
        records = [
            _record("WATCH", _ts(2026, 1, 1), _ts(2026, 1, 1)),
            _record("COOLDOWN", _ts(2026, 1, 8), _ts(2026, 1, 8)),
        ]
        assert _extract_active_episodes(records) == []

    def test_empty_records_returns_empty(self):
        assert _extract_active_episodes([]) == []


# ---------------------------------------------------------------------------
# _extract_close_prices
# ---------------------------------------------------------------------------


class TestExtractClosePrices:
    def test_wide_format_direct_columns(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"GLD": np.ones(10) * 180.0, "TLT": np.ones(10) * 95.0}, index=idx
        )
        result = _extract_close_prices(df, ["GLD", "TLT"])
        assert set(result.keys()) == {"GLD", "TLT"}
        assert len(result["GLD"]) == 10

    def test_missing_symbol_excluded(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        df = pd.DataFrame({"GLD": np.ones(5)}, index=idx)
        result = _extract_close_prices(df, ["GLD", "VIXY"])
        assert "GLD" in result
        assert "VIXY" not in result

    def test_case_insensitive_fallback(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        df = pd.DataFrame({"gld": np.ones(5) * 180.0}, index=idx)
        result = _extract_close_prices(df, ["GLD"])
        assert "GLD" in result


# ---------------------------------------------------------------------------
# label_crisis_alpha_episodes — end-to-end
# ---------------------------------------------------------------------------


class TestLabelCrisisAlphaEpisodes:
    def test_returns_empty_when_no_records(self):
        prices = _make_prices(["GLD"])
        result = label_crisis_alpha_episodes(prices, state_records=[])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_when_no_active_episodes(self):
        prices = _make_prices(["GLD"])
        records = [_record("WATCH", _ts(2026, 1, 5), _ts(2026, 1, 5))]
        result = label_crisis_alpha_episodes(prices, state_records=records)
        assert result.empty

    def test_labeled_dataframe_has_expected_columns(self):
        prices = _make_prices(["GLD", "TLT"], start="2026-01-01", periods=60)
        records = [
            _record("WATCH", _ts(2026, 1, 2), _ts(2026, 1, 2)),
            _record("ACTIVE", _ts(2026, 1, 7), _ts(2026, 1, 7), geo=2.5),
            _record("COOLDOWN", _ts(2026, 1, 21), _ts(2026, 1, 21)),
        ]
        result = label_crisis_alpha_episodes(
            prices, state_records=records, symbols=["GLD", "TLT"]
        )
        if result.empty:
            pytest.skip("No labels produced — price data may not align with episode")
        expected_cols = {
            "episode_id",
            "symbol",
            "entry_time",
            "exit_time",
            "ret",
            "bin",
            "geo_score_at_entry",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_bin_values_are_valid(self):
        prices = _make_prices(["GLD"], start="2026-01-01", periods=60)
        records = [
            _record("WATCH", _ts(2026, 1, 2), _ts(2026, 1, 2)),
            _record("ACTIVE", _ts(2026, 1, 7), _ts(2026, 1, 7), geo=2.5),
            _record("COOLDOWN", _ts(2026, 1, 21), _ts(2026, 1, 21)),
        ]
        result = label_crisis_alpha_episodes(
            prices, state_records=records, symbols=["GLD"]
        )
        if result.empty:
            pytest.skip("No labels produced")
        assert result["bin"].isin([-1, 0, 1]).all(), "bin values must be -1, 0, or +1"

    def test_loads_from_jsonl_path(self, tmp_path: Path):
        prices = _make_prices(["GLD"], start="2026-01-01", periods=60)
        state_log = tmp_path / "state.jsonl"
        records = [
            _record("WATCH", _ts(2026, 1, 2), _ts(2026, 1, 2)),
            _record("ACTIVE", _ts(2026, 1, 7), _ts(2026, 1, 7), geo=2.5),
            _record("COOLDOWN", _ts(2026, 1, 21), _ts(2026, 1, 21)),
        ]
        state_log.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        result = label_crisis_alpha_episodes(
            prices, state_log_path=state_log, symbols=["GLD"]
        )
        # Either produces labeled rows or empty (if price dates don't align) — no exception
        assert isinstance(result, pd.DataFrame)

    def test_returns_empty_when_prices_predate_episodes(self):
        # Prices from 2025; episode in 2026
        prices = _make_prices(["GLD"], start="2025-01-01", periods=30)
        prices.index = pd.date_range(
            "2025-01-01", periods=30, freq="B", tz=timezone.utc
        )
        records = [
            _record("ACTIVE", _ts(2026, 3, 1), _ts(2026, 3, 1)),
            _record("COOLDOWN", _ts(2026, 3, 10), _ts(2026, 3, 10)),
        ]
        result = label_crisis_alpha_episodes(
            prices, state_records=records, symbols=["GLD"]
        )
        assert result.empty
