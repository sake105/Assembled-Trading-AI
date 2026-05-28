"""Vol-Target Overlay strategy — volatility-targeting on SPY with IEF as defensive asset.

Strategy logic (per trading day, long-only, two-asset):
  - realized_vol = std(daily_ret[-vol_lookback:]) * sqrt(252)  [annualised, causal]
  - raw_weight_spy = min(1.0, target_vol / realized_vol)
  - Trend filter: if SPY close < SPY-sma_window SMA → raw_weight_spy *= 0.5
  - weight_ief = 1 - weight_spy  (always fully invested)

All lookbacks are strictly causal: only bars at or before the current bar are
used.  Rolling windows use min_periods=window so no partial-window values leak.

Parameters (all configurable):
  target_vol     (float): 0.12  — annualised target volatility
  vol_lookback   (int):   20    — realised-vol window in trading days
  sma_window     (int):   200   — SMA trend-filter window in trading days
  defensive_asset (str): "IEF" — Barclays 7–10Y Treasury ETF
  risk_asset      (str): "SPY" — S&P 500 ETF
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_VOL = 0.12
_DEFAULT_VOL_LOOKBACK = 20
_DEFAULT_SMA_WINDOW = 200
_DEFAULT_DEFENSIVE_ASSET = "IEF"
_DEFAULT_RISK_ASSET = "SPY"


def generate_vol_target_signals_from_prices(
    prices: pd.DataFrame,
    *,
    target_vol: float = _DEFAULT_TARGET_VOL,
    vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
    sma_window: int = _DEFAULT_SMA_WINDOW,
    defensive_asset: str = _DEFAULT_DEFENSIVE_ASSET,
    risk_asset: str = _DEFAULT_RISK_ASSET,
) -> pd.DataFrame:
    """Generate full time-series vol-target signals for every bar once warmup is complete.

    Returns one row per (timestamp, symbol) for both risk_asset and defensive_asset.
    direction is always "LONG"; score carries the fractional weight [0, 1].

    Only rows where both realized_vol and SMA warmup are satisfied are returned.
    Minimum bars required: max(vol_lookback, sma_window) + 1
    (the +1 is for pct_change which needs one baseline bar before the rolling window).

    Args:
        prices: long-format DataFrame with columns timestamp, symbol, close.
                Must contain risk_asset rows.  defensive_asset rows are not
                required here — weights are derived purely from risk_asset data.
        target_vol: annualised target volatility.
        vol_lookback: rolling window for realised vol (trading days).
        sma_window: rolling window for trend-filter SMA (trading days).
        defensive_asset: ticker for the defensive leg.
        risk_asset: ticker for the risk leg.

    Returns:
        DataFrame[timestamp, symbol, direction, score] sorted by
        (timestamp, symbol).  Empty schema DataFrame if data is insufficient.
    """
    _EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    warmup_needed = max(vol_lookback, sma_window)

    # F-senior-9: warn if defensive_asset is absent — weights are derived from risk_asset
    # only, but downstream callers typically need both in their price panels.
    if defensive_asset not in prices["symbol"].values:
        logger.warning(
            "[vol_target] defensive_asset '%s' not found in prices panel; "
            "IEF weight rows will be synthesised from 1 - w_spy",
            defensive_asset,
        )

    spy = (
        prices[prices["symbol"] == risk_asset]
        .sort_values("timestamp")
        .copy()
        .reset_index(drop=True)
    )

    if len(spy) < warmup_needed + 1:
        logger.debug(
            "[vol_target] insufficient %s bars (%d < %d warmup+1)",
            risk_asset,
            len(spy),
            warmup_needed + 1,
        )
        return _EMPTY

    # Daily return — causal: ret[i] = close[i]/close[i-1] - 1.  Row 0 is NaN.
    spy["_ret"] = spy["close"].pct_change()

    # Realised vol: rolling std annualised.  min_periods=vol_lookback means we
    # only get a value once we have a full window — no partial-window leakage.
    spy["_rvol"] = spy["_ret"].rolling(
        window=vol_lookback, min_periods=vol_lookback
    ).std() * np.sqrt(252)

    # SMA trend filter — same min_periods discipline.
    spy["_sma"] = spy["close"].rolling(window=sma_window, min_periods=sma_window).mean()

    # Vol-target weight: clamp to [0, 1].  Clip denominator to avoid div/0.
    spy["_w_spy"] = np.minimum(1.0, target_vol / spy["_rvol"].clip(lower=1e-9))

    # Trend filter: halve SPY weight when below its SMA.
    below_sma = spy["close"] < spy["_sma"]
    spy.loc[below_sma, "_w_spy"] = spy.loc[below_sma, "_w_spy"] * 0.5

    # Defensive complement.
    spy["_w_def"] = 1.0 - spy["_w_spy"]

    # Drop warmup rows (NaN in either _rvol or _sma).
    valid = spy.dropna(subset=["_rvol", "_sma"]).copy()
    if valid.empty:
        return _EMPTY

    # Use tolist() to preserve TZ-awareness (DatetimeArray.values may strip tz in some pandas builds).
    ts_list = valid["timestamp"].tolist()
    risk_rows = pd.DataFrame(
        {
            "timestamp": ts_list,
            "symbol": risk_asset,
            "direction": "LONG",
            "score": valid["_w_spy"].values.astype(float),
        }
    )
    def_rows = pd.DataFrame(
        {
            "timestamp": ts_list,
            "symbol": defensive_asset,
            "direction": "LONG",
            "score": valid["_w_def"].values.astype(float),
        }
    )

    result = pd.concat([risk_rows, def_rows], ignore_index=True)
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def compute_signals(
    prices: pd.DataFrame,
    *,
    target_vol: float = _DEFAULT_TARGET_VOL,
    vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
    sma_window: int = _DEFAULT_SMA_WINDOW,
    defensive_asset: str = _DEFAULT_DEFENSIVE_ASSET,
    risk_asset: str = _DEFAULT_RISK_ASSET,
) -> pd.DataFrame:
    """Return the latest-bar signal for {risk_asset, defensive_asset}.

    Delegates to generate_vol_target_signals_from_prices and selects the last
    row per symbol.  Both symbols carry direction="LONG" and score=weight once
    warmup completes.  Matches the interface expected by the paper trading cycle
    (one row per symbol, latest bar only).

    Returns DataFrame[timestamp, symbol, direction, score].
    """
    full = generate_vol_target_signals_from_prices(
        prices,
        target_vol=target_vol,
        vol_lookback=vol_lookback,
        sma_window=sma_window,
        defensive_asset=defensive_asset,
        risk_asset=risk_asset,
    )
    if full is None or full.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    latest = (
        full.sort_values("timestamp")
        .groupby("symbol", group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest[["timestamp", "symbol", "direction", "score"]]


def compute_target_positions(
    signals: pd.DataFrame,
    capital: float,
    **_kwargs,
) -> pd.DataFrame:
    """Derive target positions from vol-target signals.

    The score column carries the fractional target weight.  target_qty is 0.0
    — downstream pipeline converts target_weight → shares using current prices
    (same contract as trend_baseline).  Callers that need notional dollars must
    compute weight × capital themselves.

    Args:
        signals: DataFrame from compute_signals / generate_vol_target_signals
                 with columns symbol, direction, score.
        capital: current capital base (USD).

    Returns:
        DataFrame[symbol, target_weight, target_qty].
    """
    _EMPTY = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty or "score" not in signals.columns:
        return _EMPTY

    longs = signals[signals["direction"] == "LONG"].copy()
    if longs.empty:
        return _EMPTY

    # target_qty is intentionally 0.0 — the downstream pipeline converts
    # target_weight → shares using current prices (same contract as trend_baseline).
    # Callers that need notional dollars must compute weight × capital themselves.
    return pd.DataFrame(
        {
            "symbol": longs["symbol"].values,
            "target_weight": longs["score"].values.astype(float),
            "target_qty": [0.0] * len(longs),
        }
    )
