"""Dual Momentum strategy — Antonacci-variant (Gary Antonacci 2014).

Monthly rebalancing, long-only, two-step selection:
  1. Relative momentum: pick the best performer among risk assets (SPY vs VEU)
     over the past lookback_months calendar months.
  2. Absolute momentum (trend filter): if the relative winner beats the
     cash/T-bill hurdle (BIL) → invest in winner.
     Otherwise → retreat to the safe asset (AGG).

The strategy holds exactly one asset at a time, fully invested (score=1.0).
Rebalancing occurs on the last trading day of each calendar month.

All lookbacks are strictly causal:
  - A bar is identified as the last trading day of its month only when the
    next bar arrives in a different calendar month (no future-price data
    needed — only the calendar structure of the price index).
  - The 12-month base price uses the last bar at or before
    (rebalance_date − lookback_months calendar months).
  - Between rebalance dates the previous holding is forward-filled.

Parameters (all configurable):
  lookback_months  (int):        12       — momentum window
  risk_assets      (list[str]):  ["SPY", "VEU"]
  hurdle_asset     (str):        "BIL"    — 3-month T-bill ETF (cash proxy)
  safe_asset       (str):        "AGG"    — US aggregate bond ETF
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_MONTHS = 12
_DEFAULT_RISK_ASSETS: list[str] = ["SPY", "VEU"]
_DEFAULT_HURDLE_ASSET = "BIL"
_DEFAULT_SAFE_ASSET = "AGG"

_EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def generate_dual_momentum_signals_from_prices(
    prices: pd.DataFrame,
    *,
    lookback_months: int = _DEFAULT_LOOKBACK_MONTHS,
    risk_assets: list[str] | None = None,
    hurdle_asset: str = _DEFAULT_HURDLE_ASSET,
    safe_asset: str = _DEFAULT_SAFE_ASSET,
) -> pd.DataFrame:
    """Generate full time-series dual-momentum signals.

    Returns one row per daily bar from the first valid rebalance date onward.
    Each bar carries direction="LONG", score=1.0 for the held asset.
    Holding is forward-filled between monthly rebalance dates.

    Minimum bars required: lookback_months × 21 + 1 calendar month of bars
    (so the first EOM bar has a complete lookback_months of prior data).

    Args:
        prices: long-format DataFrame with columns timestamp, symbol, close.
                Must contain all of risk_assets + hurdle_asset + safe_asset.
        lookback_months: momentum lookback window in calendar months.
        risk_assets: equity candidates for relative momentum comparison.
        hurdle_asset: cash proxy for absolute momentum filter.
        safe_asset: defensive asset when absolute momentum is negative.

    Returns:
        DataFrame[timestamp, symbol, direction, score] sorted by timestamp.
        Empty schema DataFrame if data is insufficient.
    """
    if risk_assets is None:
        risk_assets = list(_DEFAULT_RISK_ASSETS)

    all_required = set(risk_assets) | {hurdle_asset, safe_asset}
    present = set(prices["symbol"].unique())
    missing = all_required - present
    if missing:
        logger.warning("[dual_momentum] missing symbols in prices panel: %s", missing)
        return _EMPTY

    # Pivot to wide format: index=timestamp, columns=symbol
    pivot = (
        prices[prices["symbol"].isin(all_required)]
        .pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
    )
    # Forward-fill internal gaps only.  bfill is intentionally omitted: back-filling
    # would propagate future prices backward into leading NaN rows for symbols whose
    # inception post-dates the panel start (e.g. BIL started 2007-05-25 while the
    # panel may begin 2007-01-01), introducing look-ahead bias.
    pivot = pivot.ffill()

    dates = pivot.index
    if len(dates) < 2:
        return _EMPTY

    # Identify EOM rebalance bars: bar i is EOM iff bar i+1 is in a new calendar month.
    # The very last bar in the dataset is never tagged as EOM (we cannot confirm it is
    # the true month-end without seeing a bar in the next month).
    # Use year*12+month keys (not month alone) so that the same calendar month in
    # different years is correctly treated as distinct (e.g. Jan-2021 vs Jan-2022).
    # Requires dates to be sorted ascending — guaranteed by pivot.sort_index() above.
    keys = dates.year.values * 12 + dates.month.values  # int array, tz-safe
    eom_flags = np.zeros(len(dates), dtype=bool)
    eom_flags[:-1] = keys[:-1] != keys[1:]
    eom_dates = dates[eom_flags]

    if len(eom_dates) == 0:
        logger.debug("[dual_momentum] no complete calendar months in price panel")
        return _EMPTY

    lookback_offset = pd.DateOffset(months=lookback_months)
    rebalance_signals: list[dict] = []

    for rb_date in eom_dates:
        # Strictly causal: only bars at or before rb_date
        sub = pivot.loc[:rb_date]

        # Base price: last bar at or before (rb_date − lookback_months)
        base_cutoff = rb_date - lookback_offset
        base_sub = sub.loc[:base_cutoff]
        if len(base_sub) == 0:
            continue  # Not enough history for this rebalance bar

        p_base = base_sub.iloc[-1]
        p_now = sub.iloc[-1]  # = pivot.loc[rb_date] since sub ends at rb_date

        # 12M total returns for risk assets and hurdle
        def _ret(sym: str) -> float:
            b = float(p_base[sym])
            n = float(p_now[sym])
            if np.isnan(b) or b < 1e-9:  # NaN: symbol not yet incepted at base date
                return float("nan")
            return n / b - 1.0

        rets = {sym: _ret(sym) for sym in list(risk_assets) + [hurdle_asset]}

        # Step 1: relative momentum — pick the outperformer among risk assets
        valid_risk = {sym: rets[sym] for sym in risk_assets if not np.isnan(rets[sym])}
        if not valid_risk:
            continue
        outperformer = max(valid_risk, key=valid_risk.__getitem__)

        # Step 2: absolute momentum — compare outperformer vs hurdle
        hurdle_ret = rets.get(hurdle_asset, float("nan"))
        if not np.isnan(hurdle_ret) and valid_risk[outperformer] > hurdle_ret:
            selected = outperformer
        else:
            selected = safe_asset

        rebalance_signals.append({"_rb_date": rb_date, "_symbol": selected})

    if not rebalance_signals:
        logger.debug(
            "[dual_momentum] no valid rebalance signals "
            "(insufficient lookback in all EOM bars)"
        )
        return _EMPTY

    # Build sparse signal Series on EOM dates, then forward-fill to every bar
    rb_df = pd.DataFrame(rebalance_signals).set_index("_rb_date")["_symbol"]
    first_signal_date = rb_df.index.min()

    all_bars_after_first = dates[dates >= first_signal_date]
    holding_series = rb_df.reindex(all_bars_after_first).ffill().dropna()

    ts_list = holding_series.index.tolist()
    result = pd.DataFrame(
        {
            "timestamp": ts_list,
            "symbol": holding_series.values.tolist(),
            "direction": "LONG",
            "score": 1.0,
        }
    )
    return result.sort_values("timestamp").reset_index(drop=True)


def compute_signals(
    prices: pd.DataFrame,
    *,
    lookback_months: int = _DEFAULT_LOOKBACK_MONTHS,
    risk_assets: list[str] | None = None,
    hurdle_asset: str = _DEFAULT_HURDLE_ASSET,
    safe_asset: str = _DEFAULT_SAFE_ASSET,
) -> pd.DataFrame:
    """Return the latest-bar signal (the current holding).

    Delegates to generate_dual_momentum_signals_from_prices and returns the
    last row.  Matches the paper trading cycle contract: one row per symbol
    with direction="LONG" for the currently held asset.
    """
    if risk_assets is None:
        risk_assets = list(_DEFAULT_RISK_ASSETS)
    full = generate_dual_momentum_signals_from_prices(
        prices,
        lookback_months=lookback_months,
        risk_assets=risk_assets,
        hurdle_asset=hurdle_asset,
        safe_asset=safe_asset,
    )
    if full is None or full.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    latest = full.sort_values("timestamp").tail(1).reset_index(drop=True)
    return latest[["timestamp", "symbol", "direction", "score"]]


def compute_target_positions(
    signals: pd.DataFrame,
    capital: float,
    **_kwargs,
) -> pd.DataFrame:
    """Derive target positions from dual-momentum signals.

    Dual momentum holds exactly one asset at weight=1.0.  target_qty is 0.0
    — downstream pipeline converts target_weight → shares using current prices
    (same contract as trend_baseline and vol_target_overlay).

    Args:
        signals: DataFrame from compute_signals with columns symbol, direction, score.
        capital: current capital base (USD).

    Returns:
        DataFrame[symbol, target_weight, target_qty].
    """
    _EMPTY_POS = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty or "score" not in signals.columns:
        return _EMPTY_POS
    longs = signals[signals["direction"] == "LONG"].copy()
    if longs.empty:
        return _EMPTY_POS
    return pd.DataFrame(
        {
            "symbol": longs["symbol"].values,
            "target_weight": longs["score"].values.astype(float),
            "target_qty": [0.0] * len(longs),
        }
    )
