"""Batch-12 PIT regression tests: alt-data disclosure lag + exact-match guard.

These tests are DISCRIMINATING — they fail against the pre-Batch-12 T+0 merge
behaviour and pass only with the PIT-correct fixes:

F1  features/altdata_earnings_insider_factors.build_earnings_surprise_factors
    - disclosure_date derived as event_date + conservative lag (was T+0)
    - merge_asof on disclosure_date with allow_exact_matches=False (was raw
      event timestamp + allow_exact_matches=True)

F2  features/altdata_news_macro_factors.{build_news_sentiment_factors,
    build_macro_regime_factors}
    - news disclosure_date derived as event_date + conservative lag (was T+0)
    - news + macro merges on disclosure/availability key, exact-match disabled
    - macro regime aligned by a release-lagged availability date

F3  data/altdata_loader.load_earnings_history
    - event_date-only feed gets a conservative disclosure lag before the as_of
      cutoff (no shift when a real disclosure_date column is present)

F4  data/altdata_loader.load_macro_indicators
    - SPLIT-BOUND release lag: release_lag_days delays only the upper (as_of)
      bound; the lower (lookback) bound keeps comparing the RAW observation date,
      so the lag can only hide recent unreleased obs, never pull older history
      into the window. release_lag_days=0 reproduces the raw-observation filter
      byte-for-byte. The returned timestamp stays the raw observation date.

PRODUCTION-INVARIANCE: each builder's empty/dead-feed path must still return the
unchanged zero/empty factor output — the fixes must not activate a dead factor.

Marker: fast (always collected; no optional deps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.data import altdata_loader
from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_macro_regime_factors,
    build_news_sentiment_factors,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _daily_prices(symbol: str = "AAPL", n: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"timestamp": dates, "symbol": [symbol] * n, "close": [100.0] * n}
    )


# --------------------------------------------------------------------------- #
# F1 — earnings disclosure lag + exact-match guard
# --------------------------------------------------------------------------- #
class TestF1EarningsDisclosureLag:
    def test_event_bar_excluded_and_disclosure_lagged(self):
        """Event on day T must NOT feed bar T; first visible bar is post-disclosure.

        Pre-fix (T+0 + allow_exact_matches=True) the surprise leaked onto the
        event bar 2024-01-03. Post-fix disclosure_date = T+1 and the bar must be
        strictly after disclosure (exact-match disabled), so 2024-01-03 and
        2024-01-04 are NaN and the value first appears 2024-01-05.
        """
        prices = _daily_prices(n=10)
        events = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-03", tz="UTC"),
                    "symbol": "AAPL",
                    "event_type": "earnings",
                    "eps_actual": 2.1,
                    "eps_estimate": 2.0,
                }
            ]
        )
        r = build_earnings_surprise_factors(events, prices, window_days=5).set_index(
            "timestamp"
        )["earnings_eps_surprise_last"]

        # Event bar and disclosure bar are NaN (no same-bar / disclosure-bar leak).
        assert pd.isna(r.loc[pd.Timestamp("2024-01-03", tz="UTC")])
        assert pd.isna(r.loc[pd.Timestamp("2024-01-04", tz="UTC")])
        # First post-disclosure bar carries the ~5% surprise.
        assert abs(r.loc[pd.Timestamp("2024-01-05", tz="UTC")] - 5.0) < 0.1

    def test_caller_supplied_disclosure_not_double_shifted(self):
        """A caller-provided disclosure_date is used as-is (no extra latency)."""
        prices = _daily_prices(n=12)
        events = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-02", tz="UTC"),
                    "symbol": "AAPL",
                    "event_type": "earnings",
                    "eps_actual": 2.1,
                    "eps_estimate": 2.0,
                    # Vendor-supplied real disclosure date.
                    "disclosure_date": pd.Timestamp("2024-01-06", tz="UTC"),
                }
            ]
        )
        r = build_earnings_surprise_factors(events, prices, window_days=5).set_index(
            "timestamp"
        )["earnings_eps_surprise_last"]

        # Disclosure is 01-06; exact-match disabled => first visible bar 01-07,
        # NOT shifted further by the derived +1 lag (override preserved).
        assert pd.isna(r.loc[pd.Timestamp("2024-01-06", tz="UTC")])
        assert abs(r.loc[pd.Timestamp("2024-01-07", tz="UTC")] - 5.0) < 0.1
        # And pre-disclosure bars stay NaN.
        assert pd.isna(r.loc[pd.Timestamp("2024-01-05", tz="UTC")])

    def test_dead_feed_stays_dead(self):
        """PRODUCTION-INVARIANCE: empty events -> unchanged NaN/zero factors."""
        prices = _daily_prices(n=8)
        empty = pd.DataFrame(columns=["timestamp", "symbol", "event_type"])
        r = build_earnings_surprise_factors(empty, prices, window_days=5)
        assert len(r) == len(prices)
        assert r["earnings_eps_surprise_last"].isna().all()
        assert (r["earnings_positive_surprise_flag"] == 0.0).all()
        assert (r["earnings_negative_surprise_flag"] == 0.0).all()


# --------------------------------------------------------------------------- #
# F2 — news + macro disclosure / release lag + exact-match guard
# --------------------------------------------------------------------------- #
class TestF2NewsMacroDisclosureLag:
    def test_news_event_bar_excluded_and_lagged(self):
        """News on day T must not feed bar T; visible only post-disclosure."""
        prices = _daily_prices(n=10)
        news = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-03", tz="UTC"),
                    "symbol": "AAPL",
                    "sentiment_score": 0.9,
                    "sentiment_volume": 10,
                }
            ]
        )
        r = build_news_sentiment_factors(news, prices, lookback_days=3).set_index(
            "timestamp"
        )["news_sentiment_mean_3d"]
        # disclosure = 01-04, exact-match disabled => first visible bar 01-05.
        assert pd.isna(r.loc[pd.Timestamp("2024-01-03", tz="UTC")])
        assert pd.isna(r.loc[pd.Timestamp("2024-01-04", tz="UTC")])
        assert abs(r.loc[pd.Timestamp("2024-01-05", tz="UTC")] - 0.9) < 1e-9

    def test_news_caller_disclosure_preserved(self):
        """Caller-supplied news disclosure_date is not double-shifted."""
        prices = _daily_prices(n=10)
        news = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-02", tz="UTC"),
                    "symbol": "AAPL",
                    "sentiment_score": 0.7,
                    "sentiment_volume": 5,
                    "disclosure_date": pd.Timestamp("2024-01-05", tz="UTC"),
                }
            ]
        )
        r = build_news_sentiment_factors(news, prices, lookback_days=3).set_index(
            "timestamp"
        )["news_sentiment_mean_3d"]
        # Disclosure 01-05, exact-match disabled => first visible 01-06.
        assert pd.isna(r.loc[pd.Timestamp("2024-01-05", tz="UTC")])
        assert abs(r.loc[pd.Timestamp("2024-01-06", tz="UTC")] - 0.7) < 1e-9

    def test_macro_release_lag_hides_recent_observation(self):
        """Macro obs within release_lag_days of the panel must be invisible."""
        prices = _daily_prices(n=12)
        macro = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "macro_code": "GDP",
                    "value": 3.0,
                    "country": "US",
                }
            ]
        )
        # Default release_lag_days=32 => obs 01-01 becomes available ~02-02,
        # outside the 12-day panel => entirely NaN.
        r = build_macro_regime_factors(macro, prices)
        assert r["macro_growth_regime"].notna().sum() == 0

    def test_macro_exact_match_excluded_without_lag(self):
        """Even with release_lag_days=0, obs must not feed its own bar."""
        prices = _daily_prices(n=6)
        macro = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "macro_code": "GDP",
                    "value": 3.0,
                    "country": "US",
                }
            ]
        )
        r = build_macro_regime_factors(macro, prices, release_lag_days=0).set_index(
            "timestamp"
        )["macro_growth_regime"]
        # Observation bar 01-01 excluded; first visible 01-02.
        assert pd.isna(r.loc[pd.Timestamp("2024-01-01", tz="UTC")])
        assert r.loc[pd.Timestamp("2024-01-02", tz="UTC")] == 1.0

    def test_news_dead_feed_stays_dead(self):
        """PRODUCTION-INVARIANCE: empty news -> unchanged NaN factors."""
        prices = _daily_prices(n=8)
        empty = pd.DataFrame(
            columns=["timestamp", "symbol", "sentiment_score", "sentiment_volume"]
        )
        r = build_news_sentiment_factors(empty, prices, lookback_days=3)
        assert len(r) == len(prices)
        assert r["news_sentiment_mean_3d"].isna().all()

    def test_macro_dead_feed_stays_dead(self):
        """PRODUCTION-INVARIANCE: empty macro -> unchanged NaN factors."""
        prices = _daily_prices(n=8)
        empty = pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])
        r = build_macro_regime_factors(empty, prices)
        assert len(r) == len(prices)
        assert r["macro_growth_regime"].isna().all()
        assert r["macro_inflation_regime"].isna().all()
        assert r["macro_risk_aversion_proxy"].isna().all()


# --------------------------------------------------------------------------- #
# F3 — loader earnings event_date fallback lag
# --------------------------------------------------------------------------- #
class TestF3LoaderEarningsEventDateFallback:
    def test_event_date_only_feed_gets_disclosure_lag(self, tmp_path):
        """event_date-only feed: event excluded at as_of=event_date, visible at +1."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "event_date": pd.to_datetime(["2024-03-10"], utc=True),
                "eps_surprise_pct": [4.0],
            }
        )
        df.to_parquet(tmp_path / "events_earnings.parquet")

        # as_of == event_date: pre-fix the event (event_date used raw) would be
        # included; post-fix the +1 disclosure lag excludes it.
        at_event = altdata_loader.load_earnings_history(
            ["AAA"], pd.Timestamp("2024-03-10"), lookback_days=30, root=tmp_path
        )
        assert at_event.empty

        # as_of == event_date + 1: now visible.
        after = altdata_loader.load_earnings_history(
            ["AAA"], pd.Timestamp("2024-03-11"), lookback_days=30, root=tmp_path
        )
        assert len(after) == 1
        assert after["symbol"].iloc[0] == "AAA"

    def test_real_disclosure_date_not_shifted(self, tmp_path):
        """A real disclosure_date column is used as-is (no extra lag)."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "event_date": pd.to_datetime(["2024-03-01"], utc=True),
                "disclosure_date": pd.to_datetime(["2024-03-10"], utc=True),
                "eps_surprise_pct": [4.0],
            }
        )
        df.to_parquet(tmp_path / "events_earnings.parquet")

        # Visible exactly at disclosure_date (no +1 shift applied).
        at_disc = altdata_loader.load_earnings_history(
            ["AAA"], pd.Timestamp("2024-03-10"), lookback_days=60, root=tmp_path
        )
        assert len(at_disc) == 1
        # Not visible the day before disclosure.
        before = altdata_loader.load_earnings_history(
            ["AAA"], pd.Timestamp("2024-03-09"), lookback_days=60, root=tmp_path
        )
        assert before.empty


# --------------------------------------------------------------------------- #
# F4 — loader macro release lag
# --------------------------------------------------------------------------- #
class TestF4LoaderMacroReleaseLag:
    def _write_macro(self, tmp_path):
        # Wide-format macro parquet: timestamp + indicator columns.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
                "GDP": [3.0],
            }
        )
        df.to_parquet(tmp_path / "macro.parquet")

    def _write_macro_two_obs(self, tmp_path):
        # Two obs designed to discriminate the SPLIT-BOUND fix from the buggy
        # two-sided forward-shift. With as_of=2024-06-15, lookback_days=90 the
        # raw cutoff is 2024-03-17:
        #   - 2024-03-01: RAW date is OLDER than cutoff => must stay EXCLUDED.
        #       But available = 03-01 + 32 = 04-02 lands INSIDE [cutoff, as_of],
        #       so the buggy shift-then-two-sided-mask wrongly ADMITS it.
        #   - 2024-04-01: RAW date is inside the window and available 05-03 <=
        #       as_of => legitimately INCLUDED (positive control).
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-03-01", "2024-04-01"], utc=True),
                "GDP": [3.0, 4.0],
            }
        )
        df.to_parquet(tmp_path / "macro.parquet")

    def test_macro_observation_hidden_until_release(self, tmp_path):
        """Obs 01-01 with default lag 32 is invisible at as_of=01-15."""
        self._write_macro(tmp_path)
        at_obs_window = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-01-15"), lookback_days=365, root=tmp_path
        )
        # Pre-fix: obs 01-01 <= 01-15 => included. Post-fix: available ~02-02 => excluded.
        assert at_obs_window.empty

    def test_macro_visible_after_release_lag(self, tmp_path):
        """Obs 01-01 becomes visible once as_of passes obs + release_lag."""
        self._write_macro(tmp_path)
        after_release = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-02-15"), lookback_days=365, root=tmp_path
        )
        assert not after_release.empty
        assert (after_release["macro_code"] == "GDP").all()

    def test_macro_release_lag_zero_is_legacy_behaviour(self, tmp_path):
        """release_lag_days=0 reproduces the raw observation-date cutoff."""
        self._write_macro(tmp_path)
        raw = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-01-15"),
            lookback_days=365,
            root=tmp_path,
            release_lag_days=0,
        )
        assert not raw.empty

    def test_macro_lower_bound_not_pulled_in_by_lag(self, tmp_path):
        """LOWER-BOUND regression: an obs whose RAW date precedes the cutoff must
        stay EXCLUDED at as_of even with release_lag_days=32.

        This FAILS against the buggy shift-then-two-sided-mask (which pulls the
        2024-03-01 obs up to available 2024-04-02 and wrongly admits it) and
        PASSES only with the split-bound fix that keeps the lower bound on the
        raw observation date. The 2024-04-01 obs is the positive control.
        """
        self._write_macro_two_obs(tmp_path)
        out = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-06-15"),
            lookback_days=90,
            root=tmp_path,
            release_lag_days=32,
        )
        # The pre-cutoff obs (raw 2024-03-01) must NOT appear; only 2024-04-01.
        seen = set(pd.to_datetime(out["timestamp"]).dt.normalize())
        assert pd.Timestamp("2024-03-01") not in seen
        assert pd.Timestamp("2024-04-01") in seen
        assert out["value"].tolist() == [4.0]

    def test_macro_lag_zero_equals_raw_window_selection(self, tmp_path):
        """LEGACY-EQUIVALENCE: with release_lag_days=0 the returned obs set equals
        the plain raw-window selection (no shift artifact on either bound).

        At as_of=2024-06-15, lookback_days=90 (cutoff 2024-03-17) the raw window
        admits exactly 2024-04-01 (2024-03-01 is older than the cutoff).
        """
        self._write_macro_two_obs(tmp_path)
        out = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-06-15"),
            lookback_days=90,
            root=tmp_path,
            release_lag_days=0,
        )
        # Expected = raw mask (raw_ts <= as_of) & (raw_ts >= cutoff) on the source.
        raw_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-03-01", "2024-04-01"]),
                "GDP": [3.0, 4.0],
            }
        )
        as_of = pd.Timestamp("2024-06-15")
        cutoff = as_of - pd.Timedelta(days=90)
        expected = raw_df[
            (raw_df["timestamp"] <= as_of) & (raw_df["timestamp"] >= cutoff)
        ]
        assert sorted(
            pd.to_datetime(out["timestamp"]).dt.normalize().tolist()
        ) == sorted(expected["timestamp"].dt.normalize().tolist())
        assert sorted(out["value"].tolist()) == sorted(expected["GDP"].tolist())

    def test_macro_returns_raw_observation_timestamp(self, tmp_path):
        """Returned timestamp is the RAW observation date, NOT the shifted
        availability date (downstream applies its own availability lag)."""
        self._write_macro(tmp_path)
        out = altdata_loader.load_macro_indicators(
            pd.Timestamp("2024-02-15"),
            lookback_days=365,
            root=tmp_path,
            release_lag_days=32,
        )
        assert not out.empty
        ts = pd.to_datetime(out["timestamp"]).dt.normalize().tolist()
        # Raw observation date 2024-01-01, not 2024-02-02 (raw + 32d).
        assert ts == [pd.Timestamp("2024-01-01")]


# --------------------------------------------------------------------------- #
# Cross-fix: disclosure-lag never injects signal where there was none
# --------------------------------------------------------------------------- #
def test_no_factor_activation_on_zero_value_feed():
    """A feed of genuinely zero-surprise events stays zero post-fix."""
    prices = _daily_prices(n=10)
    events = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-03", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "earnings",
                "eps_actual": 2.0,
                "eps_estimate": 2.0,  # zero surprise
            }
        ]
    )
    r = build_earnings_surprise_factors(events, prices, window_days=5)
    vals = r["earnings_eps_surprise_last"].dropna()
    assert (np.abs(vals) < 1e-9).all()
    assert (r["earnings_positive_surprise_flag"] == 0.0).all()
    assert (r["earnings_negative_surprise_flag"] == 0.0).all()
