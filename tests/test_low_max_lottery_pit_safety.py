"""PIT-safety and correctness tests for the Low-MAX / Lottery-Avoidance strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.strategies.low_max_lottery import (
    compute_signals,
    generate_low_max_signals_from_prices,
)

_LOOKBACK = 21
_QUANTILES = 5
_MIN_STOCKS = 3  # 20 stocks / 5 quantiles = 4 per bucket; default=3 allows this
_PAIR = [
    "SPY",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOG",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "BRK-B",
    "JNJ",
    "UNH",
    "PG",
    "HD",
    "MA",
    "V",
    "ABBV",
    "CVX",
    "LLY",
    "MRK",
]  # 20 stocks


def _make_prices(
    n_bars: int = 300,
    rng_seed: int = 42,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic long-format price data via geometric Brownian motion.

    Each symbol is an independent random walk so there is natural cross-sectional
    spread in MAX values.
    """
    if symbols is None:
        symbols = _PAIR

    rng = np.random.default_rng(rng_seed)
    base_ts = pd.Timestamp("2023-01-02", tz="UTC")
    # Business-day timestamps
    dates = pd.bdate_range(start=base_ts, periods=n_bars, tz="UTC")

    rows: list[dict] = []
    for sym in symbols:
        # Independent GBM per symbol — different mu/sigma so MAX spreads cross-sectionally
        mu = rng.uniform(-0.0002, 0.0008)
        sigma = rng.uniform(0.008, 0.025)
        log_returns = rng.normal(mu, sigma, size=n_bars)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(prices[i])})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_schema_valid_columns() -> None:
    prices = _make_prices()
    sigs = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    assert not sigs.empty, "Expected non-empty signals for 300 bars / 20 stocks"
    assert set(sigs.columns) >= {"timestamp", "symbol", "direction", "score"}
    assert set(sigs["direction"].unique()) == {"LONG"}
    assert (sigs["score"] > 0).all()
    assert (sigs["score"] <= 1.0).all()


def test_causal_no_future_leak() -> None:
    """Signals at the last bar of prices[0:200] must match those in prices[0:300]."""
    prices_full = _make_prices(n_bars=300)
    prices_short = prices_full[
        prices_full["timestamp"] <= prices_full["timestamp"].unique()[199]
    ].copy()

    sigs_full = generate_low_max_signals_from_prices(
        prices_full, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    sigs_short = generate_low_max_signals_from_prices(
        prices_short, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )

    assert not sigs_full.empty
    assert not sigs_short.empty

    # The rebalancing that last fired on or before bar 199 should produce identical
    # symbol sets in both runs.
    cutoff_ts = prices_short["timestamp"].max()
    # Find the latest rebalancing date <= cutoff_ts in each signal frame
    full_at_cutoff = sigs_full[sigs_full["timestamp"] <= cutoff_ts]
    short_at_cutoff = sigs_short[sigs_short["timestamp"] <= cutoff_ts]

    if not full_at_cutoff.empty and not short_at_cutoff.empty:
        last_rb_full = full_at_cutoff["timestamp"].max()
        last_rb_short = short_at_cutoff["timestamp"].max()

        syms_full = set(
            full_at_cutoff[full_at_cutoff["timestamp"] == last_rb_full]["symbol"]
        )
        syms_short = set(
            short_at_cutoff[short_at_cutoff["timestamp"] == last_rb_short]["symbol"]
        )
        assert syms_full == syms_short, (
            f"Future-leak detected: portfolios differ at last common rebalancing.\n"
            f"  full={sorted(syms_full)}\n  short={sorted(syms_short)}"
        )


def test_insufficient_bars_returns_empty() -> None:
    prices = _make_prices(n_bars=_LOOKBACK - 1)
    sigs = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    assert sigs.empty
    assert list(sigs.columns) == ["timestamp", "symbol", "direction", "score"]


def test_monthly_rebalance_consistency() -> None:
    """Within a calendar month the portfolio must not change."""
    prices = _make_prices(n_bars=300)
    sigs = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    assert not sigs.empty

    sigs = sigs.copy()
    sigs["ym"] = sigs["timestamp"].dt.tz_convert(None).dt.to_period("M")

    for period, group in sigs.groupby("ym"):
        unique_portfolios = (
            group.groupby("timestamp")["symbol"].apply(frozenset).unique()
        )
        assert len(unique_portfolios) == 1, (
            f"Portfolio changed within month {period}: {unique_portfolios}"
        )


def test_equal_weight_within_portfolio() -> None:
    """At each timestamp all score values must be identical (equal weight = 1/n)."""
    prices = _make_prices(n_bars=300)
    sigs = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    assert not sigs.empty

    for ts, group in sigs.groupby("timestamp"):
        scores = group["score"].values
        assert np.allclose(scores, scores[0], atol=1e-9), (
            f"Unequal weights at {ts}: {scores}"
        )
        expected = 1.0 / len(scores)
        assert abs(scores[0] - expected) < 1e-9, (
            f"Score {scores[0]} != 1/{len(scores)} at {ts}"
        )


def test_long_only_no_short() -> None:
    prices = _make_prices(n_bars=300)
    sigs = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES
    )
    assert not sigs.empty
    assert "SHORT" not in sigs["direction"].values


def test_quantile_select_top_differs_from_bottom() -> None:
    """Bottom quintile (quantile_select=1) and top quintile (=5) must hold different symbols."""
    prices = _make_prices(n_bars=300)

    sigs_bottom = generate_low_max_signals_from_prices(
        prices, max_lookback=_LOOKBACK, n_quantiles=_QUANTILES, quantile_select=1
    )
    sigs_top = generate_low_max_signals_from_prices(
        prices,
        max_lookback=_LOOKBACK,
        n_quantiles=_QUANTILES,
        quantile_select=_QUANTILES,
    )

    assert not sigs_bottom.empty
    assert not sigs_top.empty

    # Compare at a common timestamp
    common_ts = set(sigs_bottom["timestamp"].unique()) & set(
        sigs_top["timestamp"].unique()
    )
    assert common_ts, (
        "No overlapping timestamps between bottom and top quintile signals"
    )

    sample_ts = max(common_ts)
    syms_bottom = set(sigs_bottom[sigs_bottom["timestamp"] == sample_ts]["symbol"])
    syms_top = set(sigs_top[sigs_top["timestamp"] == sample_ts]["symbol"])

    assert syms_bottom != syms_top, (
        "Bottom and top quintile portfolios are identical — quantile selection has no effect"
    )


def test_compute_signals_stale_guard() -> None:
    """compute_signals must return either empty or signals at the latest price timestamp.

    Regime A (stale guard fires): signal series ends before prices_latest_ts → empty.
    Regime B (no stale): carry-forward reaches the latest bar → signals at prices_latest_ts.
    Both are valid outcomes.  The invariant is: if non-empty, timestamps must equal the
    latest price bar (never stale historical rows).
    """
    prices = _make_prices(n_bars=300)

    # Append a future bar with no price change — forces prices_latest_ts > latest signal ts
    last_ts = prices["timestamp"].max()
    future_ts = last_ts + pd.tseries.offsets.BusinessDay(1)
    extra_rows = pd.DataFrame(
        [{"timestamp": future_ts, "symbol": sym, "close": 100.0} for sym in _PAIR]
    )
    prices_extended = pd.concat([prices, extra_rows], ignore_index=True)

    sigs = compute_signals(
        prices_extended,
        max_lookback=_LOOKBACK,
        n_quantiles=_QUANTILES,
    )
    assert list(sigs.columns) == ["timestamp", "symbol", "direction", "score"]
    # KEY invariant: if signals are returned they must be at the latest price bar —
    # never stale historical rows.  An empty result means the stale guard fired correctly.
    if not sigs.empty:
        assert sigs["timestamp"].max() == prices_extended["timestamp"].max(), (
            "compute_signals returned signals older than the latest price bar — stale signals leaked"
        )
