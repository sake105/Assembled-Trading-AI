"""Low-MAX / Lottery-Avoidance strategy.

Exploits the lottery anomaly documented by Bali, Cakici, and Whitelaw (2011):
stocks with the highest recent single-day return (MAX) — the "lottery" stocks —
earn significantly lower future returns than stocks with low MAX values.  The
anomaly is robust across size, value, and momentum factors.

Strategy logic:
  1. For each stock compute MAX_i = max(daily_return) over the last
     ``max_lookback`` trading days (default 21 ≈ one calendar month).
  2. At each monthly rebalancing date (first trading day of each calendar month)
     sort the cross-section by MAX and assign stocks to ``n_quantiles`` buckets
     via pd.qcut.
  3. ``quantile_select=1`` → bottom quintile (lowest MAX, anti-lottery portfolio):
     this is the main production strategy and the source of the anomaly alpha.
     ``quantile_select=n_quantiles`` → top quintile (highest MAX): useful for
     MAX-spread comparison and short-leg backtests.
  4. Equal-weight the selected portfolio.  Carry forward between rebalancings.

Signal contract: DataFrame[timestamp, symbol, direction, score]
  direction is always "LONG" (long-only, gross ≈ 1× capital).
  score = 1/n for each of the n selected stocks (equal weight).

Causality guarantee: at each rebalancing date T, only returns up to and
including T are used (strictly causal, PIT-safe).  No bfill (E-030 anti-pattern).
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LOOKBACK = 21
_DEFAULT_N_QUANTILES = 5
_DEFAULT_REBALANCE = "monthly"
_DEFAULT_QUANTILE_SELECT = 1
_DEFAULT_MIN_STOCKS = 3

_EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
_EMPTY_POS = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _monthly_rebalance_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the first trading date of each calendar month present in *dates*.

    Uses (year, month) tuple keys — NOT bare month numbers — to avoid the
    E-031 year-strip bug where December of year N and December of year N+1
    would collapse to the same rebalancing date.
    """
    seen: dict[tuple[int, int], pd.Timestamp] = {}
    for ts in dates:
        key = (ts.year, ts.month)
        if key not in seen:
            seen[key] = ts
    return sorted(seen.values())


def _iter_portfolio(
    rebalance_dates: list[pd.Timestamp],
    portfolios: dict[pd.Timestamp, list[str]],
    all_dates: pd.DatetimeIndex,
) -> Iterator[tuple[pd.Timestamp, list[str]]]:
    """Yield (date, symbols) for every trading date, carrying forward the
    most-recently-computed portfolio between rebalancing dates.
    """
    rb_sorted = sorted(rebalance_dates)
    current_portfolio: list[str] = []
    rb_idx = 0
    n_rb = len(rb_sorted)

    for ts in all_dates:
        # Advance to the latest rebalancing date that is <= ts.
        # Only update current_portfolio when the rebalancing actually produced one;
        # absent entries (skipped months) keep the prior portfolio (carry-forward).
        while rb_idx < n_rb and rb_sorted[rb_idx] <= ts:
            if rb_sorted[rb_idx] in portfolios:
                current_portfolio = portfolios[rb_sorted[rb_idx]]
            rb_idx += 1
        if current_portfolio:
            yield ts, current_portfolio


def generate_low_max_signals_from_prices(
    prices: pd.DataFrame,
    *,
    max_lookback: int = _DEFAULT_MAX_LOOKBACK,
    n_quantiles: int = _DEFAULT_N_QUANTILES,
    rebalance: str = _DEFAULT_REBALANCE,
    quantile_select: int = _DEFAULT_QUANTILE_SELECT,
    min_stocks: int = _DEFAULT_MIN_STOCKS,
) -> pd.DataFrame:
    """Generate full time-series Low-MAX signals from a long-format price panel.

    Args:
        prices: long-format DataFrame with columns timestamp, symbol, close.
            timestamp must be tz-aware.
        max_lookback: rolling trading-day window for MAX computation.
        n_quantiles: number of cross-sectional quantile buckets (default 5 = quintiles).
        rebalance: rebalancing frequency.  Only ``"monthly"`` is supported.
        quantile_select: which quantile bucket to trade.  1 = lowest MAX (main
            strategy), n_quantiles = highest MAX.
        min_stocks: minimum eligible stocks required to form a portfolio at a
            rebalancing date.  If fewer survive the NaN-drop, the prior portfolio
            is carried forward unchanged.

    Returns:
        DataFrame[timestamp, symbol, direction, score] sorted by timestamp.
        Empty schema DataFrame if data is insufficient.
    """
    if rebalance != "monthly":
        logger.warning(
            "[low_max] unsupported rebalance='%s'; only 'monthly' is implemented",
            rebalance,
        )
        return _EMPTY.copy()

    if not 1 <= quantile_select <= n_quantiles:
        logger.warning(
            "[low_max] quantile_select=%d out of range [1, %d]",
            quantile_select,
            n_quantiles,
        )
        return _EMPTY.copy()

    required_cols = {"timestamp", "symbol", "close"}
    if not required_cols.issubset(prices.columns):
        logger.warning(
            "[low_max] prices missing required columns: %s",
            required_cols - set(prices.columns),
        )
        return _EMPTY.copy()

    subset = prices[["timestamp", "symbol", "close"]].copy()

    n_dupes = subset.duplicated(["timestamp", "symbol"]).sum()
    if n_dupes:
        logger.warning(
            "[low_max] %d duplicate (timestamp, symbol) rows — using 'last'",
            n_dupes,
        )

    # Pivot to wide format; ffill only — no bfill to avoid E-030 look-ahead bias
    pivot = subset.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    pivot = pivot.ffill()

    dates = pivot.index
    n = len(dates)

    if n <= max_lookback:
        logger.debug(
            "[low_max] insufficient bars: %d <= max_lookback=%d",
            n,
            max_lookback,
        )
        return _EMPTY.copy()

    # Warn on non-positive prices (corporate-action artefact / bad feed)
    n_nonpos = int((pivot <= 0).to_numpy().sum())
    if n_nonpos:
        logger.warning(
            "[low_max] %d non-positive prices in panel — these rows will produce NaN returns",
            n_nonpos,
        )

    # Daily returns (pct_change does not require log; stays causal)
    returns = pivot.pct_change()  # shape (n, n_symbols); row 0 is NaN

    rebalance_dates = _monthly_rebalance_dates(dates)
    portfolios: dict[pd.Timestamp, list[str]] = {}

    for rb_ts in rebalance_dates:
        t_pos = dates.get_loc(rb_ts)
        if isinstance(t_pos, slice):
            t_pos = t_pos.start

        # Need at least max_lookback bars ending at t_pos (inclusive)
        if t_pos < max_lookback:
            continue

        # Strictly causal window: [t_pos - max_lookback + 1, t_pos] inclusive
        window = returns.iloc[t_pos - max_lookback + 1 : t_pos + 1]

        # Drop any stock with a NaN in the window (late inception or data gap)
        valid = window.dropna(axis=1, how="any")

        # Need at least n_quantiles stocks to form non-degenerate quantile buckets,
        # and at least min_stocks in the universe so the selected bucket can reach
        # the min_stocks threshold.  The per-bucket check comes after qcut below.
        if valid.shape[1] < n_quantiles:
            logger.debug(
                "[low_max] %s: only %d valid stocks after NaN-drop (< n_quantiles=%d) — carrying prior portfolio",
                rb_ts.date(),
                valid.shape[1],
                n_quantiles,
            )
            # Carry forward: do not insert a new entry; _iter_portfolio uses prior
            continue

        max_vals: pd.Series = valid.max(axis=0)  # MAX_i per stock

        try:
            labels = pd.qcut(max_vals, q=n_quantiles, labels=False, duplicates="drop")
        except ValueError as exc:
            logger.warning(
                "[low_max] %s: qcut failed (%s) — carrying prior portfolio",
                rb_ts.date(),
                exc,
            )
            continue

        # qcut labels are 0-based; convert to 1-based bucket number
        selected_mask = labels == (quantile_select - 1)
        selected_symbols = list(max_vals[selected_mask].index)

        if len(selected_symbols) < min_stocks:
            logger.debug(
                "[low_max] %s: quantile %d has only %d stocks < min_stocks=%d — carrying prior",
                rb_ts.date(),
                quantile_select,
                len(selected_symbols),
                min_stocks,
            )
            continue

        portfolios[rb_ts] = selected_symbols

    if not portfolios:
        logger.debug("[low_max] no valid rebalancing portfolios computed")
        return _EMPTY.copy()

    rows: list[dict] = []
    for ts, syms in _iter_portfolio(rebalance_dates, portfolios, dates):
        weight = 1.0 / len(syms)
        for sym in syms:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "direction": "LONG",
                    "score": weight,
                }
            )

    if not rows:
        logger.debug("[low_max] no signal rows produced")
        return _EMPTY.copy()

    result = pd.DataFrame(rows)
    return result.sort_values("timestamp").reset_index(drop=True)


def compute_signals(
    prices: pd.DataFrame,
    *,
    max_lookback: int = _DEFAULT_MAX_LOOKBACK,
    n_quantiles: int = _DEFAULT_N_QUANTILES,
    rebalance: str = _DEFAULT_REBALANCE,
    quantile_select: int = _DEFAULT_QUANTILE_SELECT,
    min_stocks: int = _DEFAULT_MIN_STOCKS,
) -> pd.DataFrame:
    """Return latest-bar signals for the paper trading cycle.

    Delegates to generate_low_max_signals_from_prices and returns only the
    rows for the most recent timestamp.  If the most recent bar in *prices*
    has no active position (flat day or insufficient data), returns an empty
    DataFrame rather than stale historical signals.
    """
    full = generate_low_max_signals_from_prices(
        prices,
        max_lookback=max_lookback,
        n_quantiles=n_quantiles,
        rebalance=rebalance,
        quantile_select=quantile_select,
        min_stocks=min_stocks,
    )
    if full is None or full.empty:
        return _EMPTY.copy()

    prices_latest_ts = prices["timestamp"].max()
    latest_ts = full["timestamp"].max()

    # Stale-signal guard: if the signal series ends before the price series,
    # the strategy is flat today — do not replay historical positions.
    if latest_ts < prices_latest_ts:
        logger.debug(
            "[low_max] stale signal (signal_ts=%s < prices_ts=%s) — returning flat",
            latest_ts,
            prices_latest_ts,
        )
        return _EMPTY.copy()

    return full[full["timestamp"] == latest_ts].reset_index(drop=True)


def compute_target_positions(
    signals: pd.DataFrame,
    capital: float,
    **_kwargs,
) -> pd.DataFrame:
    """Derive target positions from Low-MAX signals.

    All signals are LONG; target_weight = score.  target_qty = 0.0 — the
    downstream pipeline converts target_weight → share quantities using
    current prices (same contract as trend_baseline and etf_pairs_meanrev).

    Gross exposure: long-only, gross ≈ 1× capital (scores sum to 1.0 within
    a selected quantile when equal-weighted).

    Args:
        signals: DataFrame from compute_signals (columns symbol, direction, score).
        capital: current capital base (USD); unused here, included for interface
            compatibility with the pipeline contract.

    Returns:
        DataFrame[symbol, target_weight, target_qty].
    """
    if signals is None or signals.empty:
        return _EMPTY_POS.copy()

    required = {"symbol", "direction", "score"}
    if not required.issubset(signals.columns):
        return _EMPTY_POS.copy()

    rows = []
    for _, row in signals.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "target_weight": float(row["score"]),
                "target_qty": 0.0,
            }
        )

    if not rows:
        return _EMPTY_POS.copy()
    return pd.DataFrame(rows)
