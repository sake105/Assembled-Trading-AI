"""Tests for _add_pairs_signals_if_enabled — pairs trading wiring in _tc_signals.

Verifies the pairs signal layer:
- Is a passthrough when policy.pairs_trading.enabled=False (default)
- Appends LONG/SHORT/FLAT rows when enabled and mock pairs return active signals
- Correctly converts LONG_A / SHORT_A / EXIT / HOLD directions
- Includes hedge leg when include_hedge_leg=True
- Skips gracefully when ctx.prices is None or missing columns
- Handles exceptions without crashing (fail-safe)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.pipeline._tc_signals import _add_pairs_signals_if_enabled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(prices_df: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None):
    """Minimal TradingContext-like object."""

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.prices = prices_df
    ctx.as_of = as_of or pd.Timestamp("2026-01-15", tz="UTC")
    ctx._policy_cache = None
    ctx.mode = "backtest"
    return ctx


def _make_prices(symbols: list[str], periods: int = 150) -> pd.DataFrame:
    """Long-format OHLCV DataFrame."""
    idx = pd.date_range("2025-07-01", periods=periods, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    rows = []
    for sym in symbols:
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=periods))
        for i, ts in enumerate(idx):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "close": prices[i],
                    "open": prices[i],
                    "high": prices[i],
                    "low": prices[i],
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _make_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-15", tz="UTC"),
                "symbol": "SPY",
                "direction": "LONG",
                "score": 0.8,
            }
        ]
    )


def _pairs_df_factory(direction: str, z: float = 2.5) -> pd.DataFrame:
    """Return a fake pairs DataFrame for (AAPL, MSFT)."""
    return pd.DataFrame(
        [
            {
                "symbol_a": "AAPL",
                "symbol_b": "MSFT",
                "direction": direction,
                "z_score": z,
                "spread": 0.5,
                "beta": 1.1,
            }
        ]
    )


_DISABLED_POLICY = {"pairs_trading": {"enabled": False}}
_ENABLED_POLICY = {
    "pairs_trading": {
        "enabled": True,
        "include_hedge_leg": True,
        "min_history": 10,
        "coint_pval_threshold": 0.05,
        "max_pairs": 5,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 4.0,
    },
    "scope": {"shorts_allowed": True, "min_short_signal_confidence": 0.0},
}


# ---------------------------------------------------------------------------
# Passthrough when disabled
# ---------------------------------------------------------------------------


class TestPairsDisabled:
    def test_passthrough_when_disabled(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value=_DISABLED_POLICY,
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        assert len(result) == 1
        assert list(result["symbol"]) == ["SPY"]

    def test_passthrough_when_pairs_key_missing(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value={},
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Graceful skip when data unavailable
# ---------------------------------------------------------------------------


class TestPairsDataGuards:
    def test_skips_when_prices_none(self):
        signals = _make_signals()
        ctx = _make_ctx(prices_df=None)
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value=_ENABLED_POLICY,
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        assert len(result) == 1

    def test_skips_when_prices_missing_close_column(self):
        prices = _make_prices(["AAPL", "MSFT"]).drop(columns=["close"])
        ctx = _make_ctx(prices)
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value=_ENABLED_POLICY,
        ):
            result = _add_pairs_signals_if_enabled(
                signals=_make_signals(), ctx=ctx, log=_logger()
            )
        assert len(result) == 1

    def test_skips_when_as_of_is_none(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]), as_of=None)
        ctx.as_of = None  # override the default set by _make_ctx
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value=_ENABLED_POLICY,
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        assert len(result) == 1  # passthrough — PIT guard fired

    def test_skips_when_insufficient_history(self):
        prices = _make_prices(["AAPL", "MSFT"], periods=5)
        ctx = _make_ctx(prices)
        policy = {
            **_ENABLED_POLICY,
            "pairs_trading": {**_ENABLED_POLICY["pairs_trading"], "min_history": 50},
        }
        with patch(
            "src.assembled_core.pipeline._tc_signals.load_policy",
            return_value=policy,
        ):
            result = _add_pairs_signals_if_enabled(_make_signals(), ctx, _logger())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Direction conversion
# ---------------------------------------------------------------------------


class TestPairsDirectionConversion:
    def _run(
        self, direction: str, include_hedge: bool = True, z: float = 2.5
    ) -> pd.DataFrame:
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        policy = {
            "pairs_trading": {
                **_ENABLED_POLICY["pairs_trading"],
                "include_hedge_leg": include_hedge,
            },
            "scope": {"shorts_allowed": True, "min_short_signal_confidence": 0.0},
        }
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=policy,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=_pairs_df_factory(direction, z),
            ),
        ):
            return _add_pairs_signals_if_enabled(signals, ctx, _logger())

    def test_long_a_adds_long_for_symbol_a(self):
        result = self._run("LONG_A")
        new_rows = result[result["symbol"].isin(["AAPL", "MSFT"])]
        aapl = new_rows[new_rows["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["direction"] == "LONG"
        assert aapl.iloc[0]["score"] > 0

    def test_long_a_with_hedge_adds_short_for_symbol_b(self):
        result = self._run("LONG_A", include_hedge=True)
        msft = result[result["symbol"] == "MSFT"]
        assert len(msft) == 1
        assert msft.iloc[0]["direction"] == "SHORT"
        assert msft.iloc[0]["score"] > 0

    def test_long_a_without_hedge_skips_symbol_b(self):
        result = self._run("LONG_A", include_hedge=False)
        assert "MSFT" not in result["symbol"].values

    def test_short_a_adds_short_for_symbol_a(self):
        result = self._run("SHORT_A")
        aapl = result[result["symbol"] == "AAPL"]
        assert aapl.iloc[0]["direction"] == "SHORT"
        assert aapl.iloc[0]["score"] > 0

    def test_short_a_with_hedge_adds_long_for_symbol_b(self):
        result = self._run("SHORT_A", include_hedge=True)
        msft = result[result["symbol"] == "MSFT"]
        assert msft.iloc[0]["direction"] == "LONG"
        assert msft.iloc[0]["score"] > 0

    def test_exit_adds_flat_for_both_legs(self):
        result = self._run("EXIT")
        new_rows = result[result["symbol"].isin(["AAPL", "MSFT"])]
        assert len(new_rows) == 2
        assert set(new_rows["direction"]) == {"FLAT"}
        assert (new_rows["score"] == 0.0).all()

    def test_hold_adds_no_new_rows(self):
        result = self._run("HOLD")
        assert "AAPL" not in result["symbol"].values
        assert "MSFT" not in result["symbol"].values

    def test_score_normalised_from_z(self):
        result = self._run("LONG_A", z=4.0)
        aapl = result[result["symbol"] == "AAPL"]
        assert aapl.iloc[0]["score"] == pytest.approx(1.0)

    def test_score_capped_at_one_for_large_z(self):
        result = self._run("LONG_A", z=8.0)
        aapl = result[result["symbol"] == "AAPL"]
        assert aapl.iloc[0]["score"] <= 1.0

    def test_existing_signals_preserved(self):
        result = self._run("LONG_A")
        assert "SPY" in result["symbol"].values

    def test_min_short_signal_confidence_suppresses_low_score_short(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        policy = {
            "pairs_trading": {
                **_ENABLED_POLICY["pairs_trading"],
                "include_hedge_leg": True,
            },
            # shorts allowed but confidence threshold is high; z=2.5 → score=0.625 < 0.9
            "scope": {"shorts_allowed": True, "min_short_signal_confidence": 0.9},
        }
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=policy,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=_pairs_df_factory("LONG_A", 2.5),
            ),
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        # AAPL LONG still emitted; MSFT SHORT hedge suppressed (score 0.625 < 0.9)
        aapl = result[result["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["direction"] == "LONG"
        assert "MSFT" not in result["symbol"].values

    def test_short_gate_suppresses_short_when_not_allowed(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        policy = {
            "pairs_trading": {
                **_ENABLED_POLICY["pairs_trading"],
                "include_hedge_leg": True,
            },
            "scope": {"shorts_allowed": False},
        }
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=policy,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=_pairs_df_factory("LONG_A", 2.5),
            ),
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        # LONG_A: AAPL LONG still emitted; MSFT SHORT hedge suppressed by gate
        aapl = result[result["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["direction"] == "LONG"
        assert "MSFT" not in result["symbol"].values

    def test_short_a_fully_suppressed_when_shorts_not_allowed(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        policy = {
            "pairs_trading": {
                **_ENABLED_POLICY["pairs_trading"],
                "include_hedge_leg": True,
            },
            "scope": {"shorts_allowed": False},
        }
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=policy,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=_pairs_df_factory("SHORT_A", 2.5),
            ),
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        # Both legs suppressed: a one-sided LONG without the paired SHORT is not a pairs position.
        assert "AAPL" not in result["symbol"].values
        assert "MSFT" not in result["symbol"].values


# ---------------------------------------------------------------------------
# Duplicate-symbol collision guards
# ---------------------------------------------------------------------------


class TestPairsSymbolCollision:
    def test_pairs_symbol_skipped_when_already_in_main_signals(self):
        """Pairs do not override existing main-signal rows for the same symbol."""
        # Base signals already contain AAPL — pairs should not append a second AAPL row.
        base = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-01-15", tz="UTC"),
                    "symbol": "AAPL",
                    "direction": "LONG",
                    "score": 0.7,
                },
                {
                    "timestamp": pd.Timestamp("2026-01-15", tz="UTC"),
                    "symbol": "SPY",
                    "direction": "LONG",
                    "score": 0.8,
                },
            ]
        )
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=_ENABLED_POLICY,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=_pairs_df_factory("LONG_A", 2.5),  # AAPL=symbol_a
            ),
        ):
            result = _add_pairs_signals_if_enabled(base, ctx, _logger())
        aapl_rows = result[result["symbol"] == "AAPL"]
        assert len(aapl_rows) == 1, "existing AAPL row must not be duplicated"
        assert aapl_rows.iloc[0]["direction"] == "LONG"  # original direction preserved

    def test_two_pairs_sharing_one_leg_emits_symbol_once(self):
        """When the same symbol appears in multiple pairs, it is only emitted once."""
        two_pairs = pd.DataFrame(
            [
                {
                    "symbol_a": "AAPL",
                    "symbol_b": "MSFT",
                    "direction": "LONG_A",
                    "z_score": 2.5,
                    "spread": 0.5,
                    "beta": 1.0,
                },
                {
                    "symbol_a": "AAPL",
                    "symbol_b": "GOOG",
                    "direction": "LONG_A",
                    "z_score": 2.0,
                    "spread": 0.3,
                    "beta": 0.9,
                },
            ]
        )
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT", "GOOG"]))
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=_ENABLED_POLICY,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                return_value=two_pairs,
            ),
        ):
            result = _add_pairs_signals_if_enabled(_make_signals(), ctx, _logger())
        aapl_rows = result[result["symbol"] == "AAPL"]
        assert len(aapl_rows) == 1, (
            "AAPL must appear exactly once even across multiple pairs"
        )


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


class TestPairsExceptionSafety:
    def test_exception_in_pairs_fn_is_swallowed(self):
        signals = _make_signals()
        ctx = _make_ctx(_make_prices(["AAPL", "MSFT"]))
        with (
            patch(
                "src.assembled_core.pipeline._tc_signals.load_policy",
                return_value=_ENABLED_POLICY,
            ),
            patch(
                "src.assembled_core.signals.pairs_trading.generate_pairs_signals_from_panel",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = _add_pairs_signals_if_enabled(signals, ctx, _logger())
        assert len(result) == 1  # original signals preserved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logger():
    import logging

    return logging.getLogger("test_pairs_wiring")
