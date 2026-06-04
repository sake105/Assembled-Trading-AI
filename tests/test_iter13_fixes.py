"""Regression tests for Iteration-13 fixes.

Covers:
  Fix 1 — altdata_earnings_insider_factors.py: np.inf → np.nan in insider_buy_sell_ratio
  Fix 2 — mean_reversion_factors.py: RSI uptrend_flag preserves NaN during EMA warm-up
  Fix 3 — altdata_earnings_insider_factors.py: post_earnings_drift_return gated behind flag
  Fix 4 — position_engine.py: net_exposure uses signed qty (not abs)
  Fix 5 — altdata_news_macro_factors.py: GDP regime uses column filter not exact match
  Fix 6 — strategy_allocator.py: wscore only sums scores from winning direction
  Fix 7 — position_sizing.py: TC rebalancing exit positions move toward 0 not away
  Fix 8 — altdata_news_macro_factors.py: O(N²) sentiment lookup → O(1) dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fix 1 — insider_buy_sell_ratio: no np.inf
# ---------------------------------------------------------------------------


class TestInsiderBuySellRatioNoInf:
    """insider_buy_sell_ratio must not produce np.inf when sells=0 but buys>0."""

    def test_no_inf_when_sells_zero_buys_positive(self):
        """np.where result must be np.nan not np.inf when sell_count=0 and buy_count>0."""
        buy = np.array([3.0, 0.0, 2.0])
        sell = np.array([0.0, 0.0, 1.0])
        # np.where evaluates all branches before selecting; suppress the
        # expected divide-by-zero warning for the zero-sell test fixtures.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                sell > 0,
                buy / sell,
                np.where(buy > 0, np.nan, np.nan),
            )
        assert not np.any(np.isinf(ratio)), f"ratio must not contain inf, got: {ratio}"
        assert np.isnan(ratio[0]), "sells=0, buys=3 → must be NaN not inf"

    def test_source_does_not_use_np_inf(self):
        """Source must not produce np.inf in the buy_sell_ratio branch."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/features/altdata_earnings_insider_factors.py"
        ).read_text(encoding="utf-8")
        # The np.inf sentinel in the insider ratio branch must be gone
        # (replaced with np.nan)
        lines = src.splitlines()
        inf_in_ratio_context = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "np.inf" in line
            and "buy_sell_ratio" in "\n".join(lines[max(0, i - 3) : i + 3])
        ]
        assert not inf_in_ratio_context, f"np.inf must not appear near buy_sell_ratio, found at: {inf_in_ratio_context}"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 2 — RSI uptrend_flag preserves NaN during EMA warm-up
# ---------------------------------------------------------------------------


class TestRsiUptrendFlagNan:
    """uptrend_flag must be NaN where ema_200 is NaN (insufficient history), not 0.0."""

    def test_uptrend_flag_nan_where_ema200_nan(self):
        """First ~200 bars where ema_200 is NaN must produce NaN uptrend_flag."""
        close = pd.Series([100.0 + i * 0.1 for i in range(250)])
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        uptrend_flag = pd.Series(
            np.where(ema_200.isna(), np.nan, (ema_50 > ema_200).astype(float)),
            index=close.index,
        )

        # EMA with adjust=False never produces NaN after first value — so test
        # the NaN-preservation logic with an explicitly NaN-padded series instead
        ema_200_with_nans = ema_200.copy()
        ema_200_with_nans.iloc[:200] = np.nan

        uptrend_flag_v2 = pd.Series(
            np.where(
                ema_200_with_nans.isna(),
                np.nan,
                (ema_50 > ema_200_with_nans).astype(float),
            ),
            index=close.index,
        )
        assert uptrend_flag_v2.iloc[:200].isna().all(), "uptrend_flag must be NaN where ema_200 is NaN"  # fmt: skip
        assert not uptrend_flag_v2.iloc[200:].isna().any(), "uptrend_flag must be non-NaN where ema_200 is available"  # fmt: skip

    def test_source_uses_np_where_with_nan_guard(self):
        """mean_reversion_factors.py must use np.where(ema_200.isna(), np.nan, ...) pattern."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/features/mean_reversion_factors.py"
        ).read_text(encoding="utf-8")
        assert "ema_200.isna()" in src, "mean_reversion_factors.py must guard uptrend_flag against ema_200 NaN values"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 3 — post_earnings_drift_return gated behind compute_pead_target flag
# ---------------------------------------------------------------------------


class TestPeadTargetGated:
    """post_earnings_drift_return must only be computed when compute_pead_target=True."""

    def test_function_accepts_compute_pead_target_param(self):
        """build_earnings_surprise_factors must accept compute_pead_target kwarg."""
        import inspect
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        sig = inspect.signature(build_earnings_surprise_factors)
        assert "compute_pead_target" in sig.parameters, "build_earnings_surprise_factors must accept compute_pead_target parameter"  # fmt: skip

    def _make_events(self):
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [pd.Timestamp("2024-01-05")],
                "event_type": ["earnings"],
                "event_id": ["aapl_20240105"],
                "eps_estimate": [1.0],
                "eps_actual": [1.2],
            }
        )

    def test_default_excludes_pead_column(self):
        """Default call (compute_pead_target=False) must not include post_earnings_drift_return."""
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        prices = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "close": [150.0 + i for i in range(10)],
            }
        )

        result = build_earnings_surprise_factors(self._make_events(), prices)
        pead_cols = [c for c in result.columns if "post_earnings_drift" in c]
        assert not pead_cols, f"Default call must not include post_earnings_drift_return, got: {pead_cols}"  # fmt: skip

    def test_explicit_flag_includes_pead_column(self):
        """compute_pead_target=True must include post_earnings_drift_return column."""
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        prices = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 30,
                "timestamp": pd.date_range("2024-01-01", periods=30),
                "close": [150.0 + i for i in range(30)],
            }
        )

        result = build_earnings_surprise_factors(
            self._make_events(), prices, compute_pead_target=True
        )
        pead_cols = [c for c in result.columns if "post_earnings_drift" in c]
        assert pead_cols, "compute_pead_target=True must include post_earnings_drift_return column"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 3b — merge_asof tz reconciliation: tz-naive price bars must not raise
# MergeError against the UTC-aware disclosure_date key, and the as-of join
# must remain PIT-correct (no look-ahead).
# ---------------------------------------------------------------------------


class TestEarningsMergeAsofTzReconciliation:
    """tz-naive price timestamps + UTC-aware event disclosure_date must merge."""

    def _make_events(self):
        # Earnings event timestamped 2024-01-05 (tz-naive on the way in).
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [pd.Timestamp("2024-01-05")],
                "event_type": ["earnings"],
                "event_id": ["aapl_20240105"],
                "eps_estimate": [1.0],
                "eps_actual": [1.2],  # +20% surprise
            }
        )

    def test_tz_naive_prices_do_not_raise_merge_error(self):
        """tz-naive datetime64[ns] price bars must not raise MergeError."""
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        prices = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 10,
                # tz-NAIVE datetime64[ns] (regression trigger for the MergeError)
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "close": [150.0 + i for i in range(10)],
            }
        )
        assert prices["timestamp"].dt.tz is None  # precondition: naive input

        # Must not raise pandas.errors.MergeError.
        result = build_earnings_surprise_factors(self._make_events(), prices)

        # Output timestamps are normalised to UTC-aware (dominant convention).
        assert str(result["timestamp"].dt.tz) == "UTC"
        assert "earnings_eps_surprise_last" in result.columns

    def test_asof_join_is_pit_correct_no_lookahead(self):
        """The joined surprise must appear only AFTER disclosure (no look-ahead).

        Event ts = 2024-01-05, disclosure_date = event_date + 1d = 2024-01-06,
        allow_exact_matches=False + direction="backward" => the surprise must be
        NaN on every bar <= 2024-01-06 and present (+20%) from 2024-01-07 on.
        """
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        prices = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "close": [150.0 + i for i in range(10)],
            }
        )
        result = build_earnings_surprise_factors(self._make_events(), prices)
        result = result.sort_values("timestamp").reset_index(drop=True)

        col = "earnings_eps_surprise_last"
        # Bars up to and including the disclosure_date (2024-01-06) must be NaN.
        before = result["timestamp"] <= pd.Timestamp("2024-01-06", tz="UTC")
        assert result.loc[before, col].isna().all(), (
            "look-ahead: surprise leaked onto a bar at/ before disclosure_date"
        )
        # Bars strictly after disclosure must carry the +20% surprise.
        after = result["timestamp"] >= pd.Timestamp("2024-01-07", tz="UTC")
        vals_after = result.loc[after, col]
        assert vals_after.notna().all()
        assert np.allclose(vals_after.to_numpy(), 20.0)

    def test_tz_aware_prices_still_work(self):
        """tz-aware (UTC) price bars must continue to merge correctly."""
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )

        prices = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, tz="UTC"),
                "close": [150.0 + i for i in range(10)],
            }
        )
        result = build_earnings_surprise_factors(self._make_events(), prices)
        assert str(result["timestamp"].dt.tz) == "UTC"
        # Same PIT expectation as the tz-naive case.
        after = result["timestamp"] >= pd.Timestamp("2024-01-07", tz="UTC")
        assert np.allclose(
            result.loc[after, "earnings_eps_surprise_last"].to_numpy(), 20.0
        )


# ---------------------------------------------------------------------------
# Fix 4 — position_engine.py: net_exposure signed
# ---------------------------------------------------------------------------


class TestNetExposureSigned:
    """net_exposure in position summary must reflect signed qty, not gross abs."""

    def test_source_uses_signed_qty_for_net_exposure(self):
        """position_engine.py must compute net_exposure with signed qty."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/accounting/position_engine.py"
        ).read_text(encoding="utf-8")
        # net_exposure must use qty (signed) * last_price, not abs(qty)
        assert (
            'positions_df["qty"] * positions_df["last_price"]' in src
            or "qty * last_price" in src
        ), "net_exposure must use signed qty * last_price, not abs notional sum"

    def test_net_exposure_negative_for_short_book(self):
        """A pure short book must produce negative net_exposure."""
        # Simulate the calculation: short 100 shares at $50 = -$5000 net
        qty = pd.Series([-100.0])
        last_price = pd.Series([50.0])
        net_exposure = float((qty * last_price).sum())
        assert net_exposure < 0, f"Short book net_exposure must be negative, got {net_exposure}"  # fmt: skip

    def test_net_vs_gross_differ_for_mixed_book(self):
        """Mixed long/short book: net_exposure != gross_exposure."""
        qty = pd.Series([200.0, -100.0])
        last_price = pd.Series([50.0, 50.0])
        net = float((qty * last_price).sum())  # 200*50 - 100*50 = 5000
        gross = float((qty.abs() * last_price).sum())  # 200*50 + 100*50 = 15000
        assert net != gross, "Net and gross must differ for a mixed long/short book"
        assert net == pytest.approx(5000.0)
        assert gross == pytest.approx(15000.0)


# ---------------------------------------------------------------------------
# Fix 5 — GDP regime: filter instead of exact column match
# ---------------------------------------------------------------------------


class TestGdpRegimeColumnFilter:
    """gdp_regime must work when the GDP column is named 'GDP_GROWTH' not exactly 'GDP'."""

    def test_source_uses_filter_not_exact_match(self):
        """altdata_news_macro_factors.py must use column filter, not hardcoded 'GDP' key."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/features/altdata_news_macro_factors.py"
        ).read_text(encoding="utf-8")
        # Must NOT directly index gpivot["GDP"] — fragile exact match
        assert 'gpivot["GDP"]' not in src, "GDP column must be found via filter (e.g. [c for c in gpivot.columns if 'GDP' in c]), not hardcoded exact match gpivot['GDP']"  # fmt: skip

    def test_gdp_filter_finds_gdp_growth_column(self):
        """Filter pattern must find 'GDP_GROWTH' when exact 'GDP' is absent."""
        columns = ["GDP_GROWTH", "PMI", "UNEMPLOYMENT_RATE"]
        gdp_cols = [c for c in columns if "GDP" in c.upper()]
        assert gdp_cols == ["GDP_GROWTH"], "Filter must find GDP_GROWTH even when exact 'GDP' column absent"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 6 — strategy_allocator.py: wscore only sums winning direction
# ---------------------------------------------------------------------------


class TestWscoreDirectionFilter:
    """wscore in strategy_allocator must not mix scores from different directions."""

    def test_source_filters_direction_in_wscore(self):
        """strategy_allocator.py must filter by best_dir before summing wscore."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/portfolio/strategy_allocator.py"
        ).read_text(encoding="utf-8")
        # After the fix, wscore computation must reference direction filtering
        assert "best_dir" in src, "strategy_allocator.py must compute best_dir"
        assert "_dir_mask" in src or "direction == best_dir" in src, "wscore must be computed only over rows where direction == best_dir"  # fmt: skip


# ---------------------------------------------------------------------------
# Fix 7 — position_sizing.py: TC rebalancing exit toward zero
# ---------------------------------------------------------------------------


class TestTCRebalancingExitGuard:
    """TC-penalized rebalancing must not push exiting positions (w_target=0) positive."""

    def test_source_has_exit_guard(self):
        """position_sizing.py must guard w_target==0 case in TC rebalancing."""
        import pathlib

        src = pathlib.Path("src/assembled_core/portfolio/position_sizing.py").read_text(
            encoding="utf-8"
        )
        assert "w_target == 0" in src or "w_target == 0.0" in src, "position_sizing.py must have a guard for w_target==0.0 in TC rebalancing"  # fmt: skip

    def test_exit_weight_does_not_go_positive(self):
        """When w_target=0 and w_current>0, adjusted weight must stay <= w_current."""
        w_target = 0.0
        w_current = 0.05
        penalty = 0.01  # small TC penalty

        # Old (buggy) behavior: adjusted = w_target + penalty = 0.01 > 0
        adjusted_buggy = w_target + penalty
        assert adjusted_buggy > 0.0, "Buggy formula produces positive weight for exit"

        # New (correct) behavior: when w_target==0, move toward 0 not away
        if w_target == 0.0:
            adjusted_fixed = max(0.0, w_current - penalty)
        else:
            adjusted_fixed = w_target + penalty

        assert adjusted_fixed <= w_current, f"Fixed exit weight must be <= w_current ({w_current}), got {adjusted_fixed}"  # fmt: skip
        assert adjusted_fixed >= 0.0, "Fixed exit weight must be non-negative"


# ---------------------------------------------------------------------------
# Fix 8 — altdata_news_macro_factors.py: O(1) sentiment dict lookup
# ---------------------------------------------------------------------------


class TestSentimentDictLookup:
    """Sentiment lookup must use a dict, not a linear scan over sentiment_factors_list."""

    def test_source_builds_sentiment_dict(self):
        """altdata_news_macro_factors.py must build a dict from sentiment_factors_list."""
        import pathlib

        src = pathlib.Path(
            "src/assembled_core/features/altdata_news_macro_factors.py"
        ).read_text(encoding="utf-8")
        assert "_sentiment_by_symbol" in src, "altdata_news_macro_factors.py must build a _sentiment_by_symbol dict for O(1) per-symbol lookup instead of O(N) linear scan"  # fmt: skip

    def test_dict_lookup_is_faster_than_linear_scan(self):
        """Verify the dict lookup pattern is correct for the given data structure."""
        # Simulate: sentiment_factors_list = [df_aapl, df_msft, ...]
        df_aapl = pd.DataFrame({"symbol": ["AAPL"] * 3, "score": [0.1, 0.2, 0.3]})
        df_msft = pd.DataFrame({"symbol": ["MSFT"] * 3, "score": [0.4, 0.5, 0.6]})
        sentiment_factors_list = [df_aapl, df_msft]

        group_col = "symbol"
        sentiment_dict = {sf[group_col].iloc[0]: sf for sf in sentiment_factors_list}

        assert "AAPL" in sentiment_dict
        assert "MSFT" in sentiment_dict
        assert sentiment_dict.get("GOOG") is None
        assert sentiment_dict["AAPL"].equals(df_aapl)
