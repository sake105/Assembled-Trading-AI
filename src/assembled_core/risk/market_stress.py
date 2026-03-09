"""Market stress signal (price-based, no external APIs).

Computes stress_ok and stress_score from volatility z-score and drawdown over lookback.
Used by risk state machine for activation confirmation (INT-5.2).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _select_price_series(prices: pd.DataFrame, benchmark_symbol: str | None) -> pd.Series:
    """Return a single close series sorted by timestamp. Multi-symbol: use benchmark or first."""
    if prices.empty or "close" not in prices.columns:
        return pd.Series(dtype=float)
    if "symbol" in prices.columns and prices["symbol"].nunique() > 1:
        sym = benchmark_symbol
        if sym is None or sym not in prices["symbol"].values:
            sym = sorted(prices["symbol"].unique())[0]
        prices = prices[prices["symbol"] == sym].copy()
    out = prices[["timestamp", "close"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.set_index("timestamp")["close"]


def compute_market_stress(prices: pd.DataFrame, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Compute market stress from price series (deterministic, no external deps).

    Args:
        prices: DataFrame with timestamp, close; optional symbol.
        policy: Config with market_stress.lookback_days, metrics, confirm_rule, qc.

    Returns:
        dict with stress_ok (bool), stress_score (0..2), details (vol_z, stress_vol, min_dd, stress_dd).
    """
    cfg = (policy or {}).get("market_stress") or {}
    if not cfg.get("enabled", True):
        return {"stress_ok": False, "stress_score": 0, "details": {}}

    lookback = int(cfg.get("lookback_days", 20) or 20)
    qc = cfg.get("qc") or {}
    if_data_missing = qc.get("if_data_missing", False)

    close = _select_price_series(prices, cfg.get("benchmark_symbol"))
    if close.empty or len(close) < 2:
        return {
            "stress_ok": if_data_missing,
            "stress_score": 0,
            "details": {"vol_z": None, "stress_vol": False, "min_dd": None, "stress_dd": False},
        }

    returns = close.pct_change().dropna()
    if returns.empty:
        return {
            "stress_ok": if_data_missing,
            "stress_score": 0,
            "details": {"vol_z": None, "stress_vol": False, "min_dd": None, "stress_dd": False},
        }

    metrics = cfg.get("metrics") or {}
    vol_cfg = metrics.get("vol_z") or {}
    dd_cfg = metrics.get("dd_lookback") or {}
    confirm = cfg.get("confirm_rule") or {}
    mode = str(confirm.get("mode", "any"))

    stress_vol = False
    vol_z_val: float | None = None
    if vol_cfg.get("enabled", True):
        z_threshold = float(vol_cfg.get("z_threshold", 1.5) or 1.5)
        # Current vol = std of last lookback returns
        last_returns = returns.iloc[-lookback:] if len(returns) >= lookback else returns
        vol = float(last_returns.std())
        # Historical vols = rolling std over 5*lookback
        roll_win = min(5 * lookback, len(returns) - 1)
        if roll_win >= 1:
            vol_hist = returns.rolling(roll_win, min_periods=1).std()
            vol_hist = vol_hist.dropna()
            if len(vol_hist) >= 1:
                vh_mean = float(vol_hist.mean())
                vh_std = float(vol_hist.std())
                if vh_std > 1e-12:
                    vol_z_val = (vol - vh_mean) / vh_std
                    stress_vol = vol_z_val >= z_threshold
                else:
                    vol_z_val = 0.0

    stress_dd = False
    min_dd_val: float | None = None
    if dd_cfg.get("enabled", True):
        dd_threshold = float(dd_cfg.get("dd_threshold", -0.05) or -0.05)
        # Drawdown from rolling max
        roll_max = close.rolling(lookback, min_periods=1).max()
        dd = (close / roll_max) - 1.0
        dd = dd.dropna()
        if len(dd) >= 1:
            last_dd = dd.iloc[-lookback:] if len(dd) >= lookback else dd
            min_dd_val = float(last_dd.min())
            stress_dd = min_dd_val <= dd_threshold

    if mode == "all":
        stress_ok = stress_vol and stress_dd
    else:
        stress_ok = stress_vol or stress_dd

    stress_score = (1 if stress_vol else 0) + (1 if stress_dd else 0)

    return {
        "stress_ok": bool(stress_ok),
        "stress_score": stress_score,
        "details": {
            "vol_z": vol_z_val,
            "stress_vol": stress_vol,
            "min_dd": min_dd_val,
            "stress_dd": stress_dd,
        },
    }


__all__ = ["compute_market_stress"]
