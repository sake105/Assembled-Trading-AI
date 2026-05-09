"""Cross-Asset Carry Signal via ETF proxies.

From 13_FREE_MODULE.md §13.15.
CTA-style macro overlay — carry across equity, bond, FX, commodity.
All free: ETF prices via yfinance + FRED for risk-free rate.

Carry components:
- Equity carry: SPY dividend yield - T-Bill rate
- Bond carry: TLT yield - SHY yield (term premium)
- FX carry: UUP (USD) vs FXE (EUR) vs FXY (JPY) — 3-month rate differentials
- Commodity carry: Roll-yield proxy via USO/UNG front spread
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ETF tickers used as proxies
CARRY_ETFS = {
    "equity_risky": "SPY",
    "equity_safe": "SHY",
    "bond_long": "TLT",
    "bond_short": "SHY",
    "fx_usd": "UUP",
    "fx_eur": "FXE",
    "fx_jpy": "FXY",
    "commodity_oil": "USO",
    "commodity_gas": "UNG",
}


def _try_yfinance():
    try:
        import yfinance as yf

        return yf
    except ImportError:
        logger.warning("yfinance not installed — pip install yfinance")
        return None


def _get_returns(ticker: str, period: str = "3mo") -> pd.Series:
    """Fetch trailing returns for carry computation."""
    yf = _try_yfinance()
    if yf is None:
        return pd.Series(dtype=float)

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return pd.Series(dtype=float)
        prices = hist["Close"]
        return prices.pct_change().dropna()
    except Exception as exc:
        logger.debug("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.Series(dtype=float)


def equity_carry(lookback_days: int = 63) -> float:
    """Equity carry: SPY trailing dividend yield proxy minus SHY return.

    Positive = equities offer positive carry vs cash.
    """
    spy_ret = _get_returns("SPY", period="3mo")
    shy_ret = _get_returns("SHY", period="3mo")

    if spy_ret.empty or shy_ret.empty:
        return 0.0

    common = spy_ret.index.intersection(shy_ret.index)
    n = min(lookback_days, len(common))
    if n < 5:
        return 0.0

    spy_ann = float(spy_ret.iloc[-n:].mean() * 252)
    shy_ann = float(shy_ret.iloc[-n:].mean() * 252)
    return spy_ann - shy_ann


def bond_carry(lookback_days: int = 63) -> float:
    """Bond carry: TLT return minus SHY return (term premium proxy)."""
    tlt_ret = _get_returns("TLT", period="3mo")
    shy_ret = _get_returns("SHY", period="3mo")

    if tlt_ret.empty or shy_ret.empty:
        return 0.0

    common = tlt_ret.index.intersection(shy_ret.index)
    n = min(lookback_days, len(common))
    if n < 5:
        return 0.0

    tlt_ann = float(tlt_ret.iloc[-n:].mean() * 252)
    shy_ann = float(shy_ret.iloc[-n:].mean() * 252)
    return tlt_ann - shy_ann


def fx_carry_usd_eur(lookback_days: int = 63) -> float:
    """FX carry: USD (UUP) return minus EUR (FXE) return.

    Positive = USD carries better than EUR (USD attractive, EUR weak).
    """
    uup_ret = _get_returns("UUP", period="3mo")
    fxe_ret = _get_returns("FXE", period="3mo")

    if uup_ret.empty or fxe_ret.empty:
        return 0.0

    common = uup_ret.index.intersection(fxe_ret.index)
    n = min(lookback_days, len(common))
    if n < 5:
        return 0.0

    uup_ann = float(uup_ret.iloc[-n:].mean() * 252)
    fxe_ann = float(fxe_ret.iloc[-n:].mean() * 252)
    return uup_ann - fxe_ann


def commodity_roll_proxy(lookback_days: int = 21) -> float:
    """Commodity carry proxy: USO (oil) trailing return as roll-yield signal.

    Positive = contango (futures above spot, negative carry for longs).
    Negative = backwardation (positive carry for longs).
    We proxy with price return — positive price = spot moving up = backwardation.
    """
    uso_ret = _get_returns("USO", period="1mo")
    if uso_ret.empty:
        return 0.0
    n = min(lookback_days, len(uso_ret))
    return float(uso_ret.iloc[-n:].mean() * 252)


def cross_asset_carry_score(lookback_days: int = 63) -> dict[str, float]:
    """Compute carry scores for all asset classes.

    Returns:
        Dict with keys: equity, bond, fx_usd_eur, commodity, composite.
        All values are annualized carry estimates.
        composite = equal-weight average (sign-adjusted).
    """
    eq = equity_carry(lookback_days)
    bd = bond_carry(lookback_days)
    fx = fx_carry_usd_eur(lookback_days)
    cm = commodity_roll_proxy(min(lookback_days, 21))

    # Normalize to [-1, +1] with a 10% annual scale
    def _norm(x: float, scale: float = 0.10) -> float:
        return max(-1.0, min(1.0, x / scale))

    composite = np.mean([_norm(eq), _norm(bd), _norm(fx), _norm(cm)])

    return {
        "equity": eq,
        "bond": bd,
        "fx_usd_eur": fx,
        "commodity": cm,
        "composite": float(composite),
    }


def carry_exposure_multiplier(composite_carry: float, threshold: float = 0.2) -> float:
    """Return long-bias multiplier based on composite carry.

    Args:
        composite_carry: composite score from cross_asset_carry_score() [-1, +1].
        threshold: Below -threshold → reduce; above +threshold → increase.

    Returns:
        Multiplier in [0.7, 1.3].
    """
    if composite_carry > threshold:
        return 1.2
    if composite_carry < -threshold:
        return 0.8
    return 1.0


__all__ = [
    "CARRY_ETFS",
    "equity_carry",
    "bond_carry",
    "fx_carry_usd_eur",
    "commodity_roll_proxy",
    "cross_asset_carry_score",
    "carry_exposure_multiplier",
]
