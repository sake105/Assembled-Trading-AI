"""Tests for ranking hysteresis (symbol rotation churn reduction)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.paper.ranking_hysteresis import apply_ranking_hysteresis

pytestmark = [pytest.mark.unit]


def _make_signals(entries: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": "2025-10-15", "symbol": s, "direction": d, "score": sc}
        for s, d, sc in entries
    ])


class TestApplyRankingHysteresis:
    def test_new_symbol_rank6_blocked_at_entry5(self):
        """Symbol not held, rank 6 > entry_n=5 → set FLAT."""
        signals = _make_signals([
            ("A", "LONG", 0.9), ("B", "LONG", 0.8), ("C", "LONG", 0.7),
            ("D", "LONG", 0.6), ("E", "LONG", 0.5), ("F", "LONG", 0.4),
        ])
        held = set()
        filtered, meta = apply_ranking_hysteresis(signals, held, entry_n=5, hold_n=7)

        long_syms = set(filtered[filtered["direction"] == "LONG"]["symbol"])
        assert "F" not in long_syms
        assert len(long_syms) == 5
        assert meta["blocked_entry"] == 1

    def test_held_symbol_rank6_kept_at_hold7(self):
        """Symbol currently held, rank 6 <= hold_n=7 → stays LONG."""
        signals = _make_signals([
            ("A", "LONG", 0.9), ("B", "LONG", 0.8), ("C", "LONG", 0.7),
            ("D", "LONG", 0.6), ("E", "LONG", 0.5), ("F", "LONG", 0.4),
        ])
        held = {"F"}
        filtered, meta = apply_ranking_hysteresis(signals, held, entry_n=5, hold_n=7)

        long_syms = set(filtered[filtered["direction"] == "LONG"]["symbol"])
        assert "F" in long_syms
        assert meta["kept_by_hysteresis"] == 1

    def test_held_symbol_rank8_dropped_at_hold7(self):
        """Symbol currently held, rank 8 > hold_n=7 → set FLAT."""
        signals = _make_signals([
            ("A", "LONG", 0.9), ("B", "LONG", 0.8), ("C", "LONG", 0.7),
            ("D", "LONG", 0.6), ("E", "LONG", 0.5), ("F", "LONG", 0.4),
            ("G", "LONG", 0.3), ("H", "LONG", 0.2),
        ])
        held = {"H"}
        filtered, meta = apply_ranking_hysteresis(signals, held, entry_n=5, hold_n=7)

        long_syms = set(filtered[filtered["direction"] == "LONG"]["symbol"])
        assert "H" not in long_syms

    def test_disabled_passes_all(self):
        """Without hysteresis, all LONG signals pass unchanged."""
        signals = _make_signals([
            ("A", "LONG", 0.9), ("B", "LONG", 0.8), ("C", "LONG", 0.7),
            ("D", "LONG", 0.6), ("E", "LONG", 0.5), ("F", "LONG", 0.4),
        ])
        long_count = len(signals[signals["direction"] == "LONG"])
        assert long_count == 6

    def test_held_within_entry_not_counted_as_hysteresis(self):
        """Held symbol within entry_n is not counted as kept_by_hysteresis."""
        signals = _make_signals([
            ("A", "LONG", 0.9), ("B", "LONG", 0.8), ("C", "LONG", 0.7),
        ])
        held = {"A"}
        _, meta = apply_ranking_hysteresis(signals, held, entry_n=5, hold_n=7)
        assert meta["kept_by_hysteresis"] == 0

    def test_empty_signals(self):
        signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        filtered, meta = apply_ranking_hysteresis(signals, set(), entry_n=5, hold_n=7)
        assert len(filtered) == 0

    def test_no_long_signals(self):
        signals = _make_signals([("A", "FLAT", 0.0)])
        filtered, meta = apply_ranking_hysteresis(signals, {"A"}, entry_n=5, hold_n=7)
        assert meta["kept_by_hysteresis"] == 0

    def test_meta_fields_complete(self):
        signals = _make_signals([("A", "LONG", 0.9)])
        _, meta = apply_ranking_hysteresis(signals, set(), entry_n=5, hold_n=7)
        assert "entry_n" in meta
        assert "hold_n" in meta
        assert "kept_by_hysteresis" in meta
        assert "blocked_entry" in meta


class TestConfigDefault:
    def test_default_hysteresis_disabled(self):
        from src.assembled_core.paper.paper_track import PaperTrackConfig
        from pathlib import Path

        cfg = PaperTrackConfig(
            strategy_name="test",
            strategy_type="trend_baseline",
            universe_file=Path("watchlist.txt"),
            freq="1d",
        )
        assert cfg.ranking_hysteresis_enabled is False
        assert cfg.ranking_entry_n == 5
        assert cfg.ranking_hold_n == 7
