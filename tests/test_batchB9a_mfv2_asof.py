"""Batch B9a — multifactor_v2.compute_signals as_of PIT anchor (Diagnostik §strategies MAJOR-b).

E-002 class leak: compute_signals previously had NO as_of parameter and anchored
ALL altdata factors at the panel tail (latest["timestamp"].max()). On a full
price panel (backtest/replay) that anchors altdata at the LAST bar, leaking
future altdata relative to the cycle date.

This suite verifies:
  1. An explicit as_of in the MIDDLE of a panel anchors the altdata factors at
     that as_of (NOT the panel tail) and slices the panel to <= as_of, so a
     future altdata-bearing row after as_of cannot contribute.
  2. WITHOUT as_of, the output is byte-identical to the current panel-max-anchor
     behaviour (backward-compat guard).
  3. The caller (_shared_eod.compute_signals_by_mode) forwards as_of; a backtest
     as_of mid-panel yields as_of-correct signals while a live as_of == panel-max
     reproduces the no-as_of output.

Mirrors the as_of-param pattern already shipped on the per-factor helpers and the
existing tests in tests/test_multifactor_v2_pit_as_of.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.assembled_core.strategies.multifactor_v2 import compute_signals

pytestmark = pytest.mark.fast


_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


def _synth_panel(
    symbols: list[str] | None = None,
    n_days: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic price panel with TA-like feature columns.

    Mirrors tests/strategies/test_multifactor_v2.py::_synth_panel so the
    backward-compat assertions exercise the same factor paths as the main suite.
    """
    rng = np.random.default_rng(seed)
    symbols = symbols or _SYMBOLS
    rows = []
    for sym in symbols:
        base = 100.0 + rng.normal(0, 10)
        prices = [base]
        for _ in range(n_days - 1):
            prices.append(prices[-1] * (1 + rng.normal(0.0005, 0.015)))
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        for i, d in enumerate(dates):
            p = prices[i]
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "close": p,
                    "high": p * (1 + abs(rng.normal(0, 0.01))),
                    "low": p * (1 - abs(rng.normal(0, 0.01))),
                    "volume": int(rng.uniform(1e6, 5e6)),
                    "ta_rsi_14_v1": rng.uniform(30, 70),
                    "ta_adx_v1": rng.uniform(15, 40),
                    "ta_macd_hist_v1": rng.normal(0, 0.5),
                    "ta_ma_200_v1": p * (1 + rng.normal(0, 0.05)),
                    "ta_ma_50_v1": p * (1 + rng.normal(0, 0.02)),
                    "ta_bb_pctb_v1": rng.uniform(0.1, 0.9),
                    "ta_stoch_k_v1": rng.uniform(20, 80),
                    "ta_obv_v1": rng.uniform(1e7, 5e7),
                    "ta_vol_weighted_mom_20d_v1": rng.normal(0, 0.02),
                    "tick_imbalance_20d": rng.uniform(0.3, 0.7),
                    "abnormal_vol_20d": rng.uniform(0.5, 2.0),
                    "rv_20": rng.uniform(0.10, 0.30),
                    "vov_20_60": rng.uniform(0.0, 0.05),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. as_of anchors altdata at the cycle date (NOT the panel tail)
# ---------------------------------------------------------------------------


class TestAsOfAnchorsAltdata:
    def test_midpanel_as_of_anchors_altdata_not_panel_tail(self) -> None:
        """An as_of in the middle of the panel must anchor altdata at as_of.

        Pre-fix, _bar_as_of = panel max, so the captured altdata as_of would be
        the panel tail (a future date). Post-fix it must equal the passed as_of.
        """
        panel = _synth_panel(n_days=120)
        all_dates = sorted(pd.to_datetime(panel["timestamp"]).unique())
        mid_as_of = pd.Timestamp(all_dates[60])  # strictly inside the panel
        panel_tail = pd.Timestamp(all_dates[-1])
        assert mid_as_of < panel_tail  # sanity: as_of is genuinely mid-panel

        captured_earn: list[pd.Timestamp] = []
        captured_news: list[pd.Timestamp] = []

        def fake_earn(symbols, as_of, lookback_days=90):
            captured_earn.append(pd.Timestamp(as_of))
            return None

        def fake_news(symbols, as_of, lookback_days=30):
            captured_news.append(pd.Timestamp(as_of))
            return None

        with (
            patch(
                "src.assembled_core.data.altdata_loader.load_earnings_history",
                side_effect=fake_earn,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_insider_filings",
                return_value=None,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_news_sentiment",
                side_effect=fake_news,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_macro_indicators",
                return_value=None,
            ),
        ):
            compute_signals(panel, strategy_cfg={}, as_of=mid_as_of)

        captured = captured_earn + captured_news
        assert captured, "expected at least one altdata load with an as_of"
        for ts in captured:
            assert ts.normalize() == mid_as_of.normalize(), (
                f"altdata anchored at {ts} != as_of {mid_as_of} (panel tail "
                f"was {panel_tail}) — look-ahead leak not closed"
            )
            assert ts.normalize() != panel_tail.normalize(), (
                "altdata still anchored at the panel tail — leak active"
            )

    def test_future_altdata_row_after_as_of_does_not_contribute(self) -> None:
        """A pre-computed altdata column present only on bars AFTER as_of must
        not feed the factor — the panel is sliced to <= as_of first.

        sue_score (PEAD factor 33) reads a pre-merged panel column when present;
        we plant it only on the post-as_of tail. After the as_of slice the
        latest bar (<= as_of) carries NO sue_score, so the factor must fall back
        to neutral (no contribution from the future value).
        """
        panel = _synth_panel(n_days=120).sort_values("timestamp").reset_index(drop=True)
        all_dates = sorted(pd.to_datetime(panel["timestamp"]).unique())
        mid_as_of = pd.Timestamp(all_dates[60])

        # Plant a strong future SUE signal ONLY on bars strictly after as_of.
        panel["sue_score"] = np.nan
        post_mask = pd.to_datetime(panel["timestamp"]) > mid_as_of
        panel.loc[post_mask, "sue_score"] = 9.99  # extreme future value

        # If the future row leaked, load_earnings_history would be skipped
        # because the column is present; instead, after the as_of slice the
        # latest bar has no sue_score, so the altdata fallback path runs.
        load_called: list = []

        def spy_load(symbols, as_of, lookback_days=90):
            load_called.append(pd.Timestamp(as_of))
            return None

        with patch(
            "src.assembled_core.data.altdata_loader.load_earnings_history",
            side_effect=spy_load,
        ):
            out = compute_signals(panel, strategy_cfg={}, as_of=mid_as_of)

        # Fallback load must have run with the as_of anchor (column not seen on
        # the <= as_of latest bar) — proves the future sue_score was sliced out.
        assert load_called, (
            "PEAD altdata fallback did not run — future sue_score row leaked past as_of"
        )
        for ts in load_called:
            assert ts.normalize() == mid_as_of.normalize()
        # And the output is a well-formed frame (no crash from the slice path).
        assert isinstance(out, pd.DataFrame)

    def test_as_of_slices_panel_latest_bar_to_cutoff(self) -> None:
        """With as_of mid-panel, emitted signals must carry the <= as_of latest
        bar timestamp, never the panel tail."""
        panel = _synth_panel(n_days=120)
        all_dates = sorted(pd.to_datetime(panel["timestamp"]).unique())
        mid_as_of = pd.Timestamp(all_dates[60])

        out = compute_signals(
            panel, strategy_cfg={"min_signal_score": -1e9}, as_of=mid_as_of
        )
        if not out.empty and "timestamp" in out.columns:
            out_ts = pd.to_datetime(out["timestamp"])
            assert (out_ts <= mid_as_of).all(), (
                "emitted signal timestamps exceed as_of — panel not sliced"
            )


# ---------------------------------------------------------------------------
# 2. Backward-compat: no as_of reproduces panel-max behaviour exactly
# ---------------------------------------------------------------------------


class TestBackwardCompatNoAsOf:
    def test_no_as_of_output_is_byte_identical_to_panel_max_as_of(self) -> None:
        """compute_signals(panel) (no as_of) must equal compute_signals(panel,
        as_of=panel_max): the explicit panel-max anchor + full-panel slice is a
        no-op, proving live (as_of == panel tail) is unchanged."""
        panel = _synth_panel(n_days=200)
        panel_max = pd.Timestamp(pd.to_datetime(panel["timestamp"]).max())

        out_default = compute_signals(panel.copy(), strategy_cfg={})
        out_explicit = compute_signals(panel.copy(), strategy_cfg={}, as_of=panel_max)

        pd.testing.assert_frame_equal(
            out_default.reset_index(drop=True),
            out_explicit.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )

    def test_no_as_of_anchor_is_panel_max(self) -> None:
        """Without as_of, the altdata anchor remains the panel max timestamp."""
        panel = _synth_panel(n_days=120)
        panel_max = pd.Timestamp(pd.to_datetime(panel["timestamp"]).max())

        captured: list[pd.Timestamp] = []

        def fake_earn(symbols, as_of, lookback_days=90):
            captured.append(pd.Timestamp(as_of))
            return None

        with (
            patch(
                "src.assembled_core.data.altdata_loader.load_earnings_history",
                side_effect=fake_earn,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_insider_filings",
                return_value=None,
            ),
        ):
            compute_signals(panel, strategy_cfg={})  # no as_of

        for ts in captured:
            assert ts.normalize() == panel_max.normalize(), (
                "no-as_of anchor drifted from panel max — backward-compat broken"
            )


# ---------------------------------------------------------------------------
# 3. Caller wiring: compute_signals_by_mode forwards as_of
# ---------------------------------------------------------------------------


class TestCallerWiring:
    _POLICY = {"signal_generation": {"mode": "multifactor"}}

    def test_compute_signals_by_mode_forwards_as_of(self) -> None:
        """The shared EOD dispatch must thread as_of into compute_signals."""
        from src.assembled_core.pipeline import _shared_eod

        panel = _synth_panel(n_days=120)
        all_dates = sorted(pd.to_datetime(panel["timestamp"]).unique())
        mid_as_of = pd.Timestamp(all_dates[60])

        seen: dict = {}

        def spy_compute(prices, strategy_cfg=None, *, as_of=None):
            seen["as_of"] = as_of
            return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

        with patch(
            "src.assembled_core.strategies.multifactor_v2.compute_signals",
            side_effect=spy_compute,
        ):
            _shared_eod.compute_signals_by_mode(
                panel, self._POLICY, freq="1d", as_of=mid_as_of
            )

        assert seen.get("as_of") is not None
        assert pd.Timestamp(seen["as_of"]).normalize() == mid_as_of.normalize(), (
            "compute_signals_by_mode did not forward as_of to compute_signals"
        )

    def test_compute_signals_by_mode_default_as_of_is_none(self) -> None:
        """Default (no as_of kwarg) forwards None → byte-identical legacy path."""
        from src.assembled_core.pipeline import _shared_eod

        panel = _synth_panel(n_days=80)
        seen: dict = {}

        def spy_compute(prices, strategy_cfg=None, *, as_of=None):
            seen["as_of"] = as_of
            return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

        with patch(
            "src.assembled_core.strategies.multifactor_v2.compute_signals",
            side_effect=spy_compute,
        ):
            _shared_eod.compute_signals_by_mode(panel, self._POLICY, freq="1d")

        assert seen.get("as_of") is None, (
            "default compute_signals_by_mode must forward as_of=None"
        )

    def test_live_panel_max_as_of_matches_no_as_of_via_dispatch(self) -> None:
        """Through the real dispatch: passing panel-max as_of (live equivalent)
        yields the same signals as the legacy no-as_of call."""
        from src.assembled_core.pipeline import _shared_eod

        panel = _synth_panel(n_days=180)
        panel_max = pd.Timestamp(pd.to_datetime(panel["timestamp"]).max())

        out_live = _shared_eod.compute_signals_by_mode(
            panel.copy(), self._POLICY, freq="1d", as_of=panel_max
        )
        out_legacy = _shared_eod.compute_signals_by_mode(
            panel.copy(), self._POLICY, freq="1d"
        )

        pd.testing.assert_frame_equal(
            out_live.reset_index(drop=True),
            out_legacy.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
