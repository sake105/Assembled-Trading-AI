"""Sector rotation signals based on ETF momentum and relative strength (M16).

Ranks 8 SPDR sector ETFs by momentum and relative strength vs SPY.
Generates LONG (top sectors), SHORT (bottom sectors), FLAT signals.

Designed for use in grand backtest pipeline — called once pre-loop to build
the full score panel, then queried per date in the trading loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY"]

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
}


@dataclass
class SectorRotationConfig:
    lookback_3m: int = 63         # ~3 months
    lookback_6m: int = 126        # ~6 months
    rs_window: int = 20           # relative strength vs SPY
    top_n_long: int = 3           # top N sectors to long
    bottom_n_short: int = 2       # bottom N sectors to short
    risk_off_threshold: int = 5   # if >= N sectors negative → risk-off
    weight_3m: float = 0.50
    weight_6m: float = 0.30
    weight_rs: float = 0.20


@dataclass
class SectorSignals:
    """Per-date sector rotation signals."""
    date: pd.Timestamp
    scores: dict[str, float]        # ETF → composite score
    longs: list[str]                # ETFs to long
    shorts: list[str]               # ETFs to short
    is_risk_off: bool               # True → rotate to bonds/gold
    negative_count: int             # how many sectors are negative momentum


def compute_sector_scores(
    sector_prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    config: SectorRotationConfig | None = None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute daily composite sector scores for all sector ETFs.

    Args:
        sector_prices: Long-format price DataFrame for sector ETFs.
        spy_prices: Long-format or wide-format SPY price data.
        config: Rotation configuration.

    Returns:
        DataFrame with columns: timestamp, {etf}_score for each sector ETF,
        plus {etf}_3m, {etf}_6m, {etf}_rs columns.
    """
    config = config or SectorRotationConfig()

    # Pivot sector prices to wide format
    sector_wide = sector_prices.pivot(
        index=timestamp_col, columns=symbol_col, values=price_col
    )
    sector_wide = sector_wide.sort_index()

    # Get SPY close series
    if symbol_col in spy_prices.columns:
        spy_series = (
            spy_prices[spy_prices[symbol_col] == "SPY"]
            .set_index(timestamp_col)[price_col]
            .sort_index()
        )
    else:
        spy_series = spy_prices[price_col].sort_index() if price_col in spy_prices.columns else spy_prices.squeeze()

    # Align SPY to sector dates
    spy_series = spy_series.reindex(sector_wide.index).ffill()

    available_etfs = [e for e in SECTOR_ETFS if e in sector_wide.columns]
    if not available_etfs:
        _log.warning("No sector ETFs found in price data. Columns: %s", list(sector_wide.columns))
        return pd.DataFrame()

    records = []
    dates = sector_wide.index

    for date in dates:
        row: dict = {timestamp_col: date}

        for etf in available_etfs:
            etf_series = sector_wide[etf].loc[:date]
            spy_slice = spy_series.loc[:date]

            # 3M momentum
            if len(etf_series) >= config.lookback_3m:
                ret_3m = (etf_series.iloc[-1] / etf_series.iloc[-config.lookback_3m] - 1)
            else:
                ret_3m = np.nan

            # 6M momentum
            if len(etf_series) >= config.lookback_6m:
                ret_6m = (etf_series.iloc[-1] / etf_series.iloc[-config.lookback_6m] - 1)
            else:
                ret_6m = np.nan

            # Relative strength vs SPY (RS window)
            if len(etf_series) >= config.rs_window and len(spy_slice) >= config.rs_window:
                etf_rs = (etf_series.iloc[-1] / etf_series.iloc[-config.rs_window] - 1)
                spy_rs = (spy_slice.iloc[-1] / spy_slice.iloc[-config.rs_window] - 1)
                rs = etf_rs - spy_rs
            else:
                rs = np.nan

            row[f"{etf}_3m"] = ret_3m
            row[f"{etf}_6m"] = ret_6m
            row[f"{etf}_rs"] = rs

        records.append(row)

    scores_df = pd.DataFrame(records).set_index(timestamp_col)

    # Cross-sectional rank → composite score (per date)
    result_rows = []
    for row in scores_df.itertuples(index=True):
        date = row.Index
        score_row: dict = {timestamp_col: date}
        scores_raw = {}
        for etf in available_etfs:
            m3 = getattr(row, f"{etf}_3m", np.nan)
            m6 = getattr(row, f"{etf}_6m", np.nan)
            rs = getattr(row, f"{etf}_rs", np.nan)
            # Simple composite (weights sum to 1.0)
            parts = [
                (m3, config.weight_3m),
                (m6, config.weight_6m),
                (rs, config.weight_rs),
            ]
            valid = [(v, w) for v, w in parts if not np.isnan(v)]
            if valid:
                total_w = sum(w for _, w in valid)
                composite = sum(v * w for v, w in valid) / total_w
            else:
                composite = np.nan
            scores_raw[etf] = composite
            score_row[f"{etf}_score"] = composite
            score_row[f"{etf}_3m"] = m3
            score_row[f"{etf}_6m"] = m6
            score_row[f"{etf}_rs"] = rs

        result_rows.append(score_row)

    result_df = pd.DataFrame(result_rows)
    _log.info("Sector scores computed: %d dates, %d ETFs", len(result_df), len(available_etfs))
    return result_df


def generate_sector_rotation_signals(
    scores_row: pd.Series | dict,
    available_etfs: list[str] | None = None,
    config: SectorRotationConfig | None = None,
) -> SectorSignals:
    """Generate LONG/SHORT/FLAT signals from a single date's score row.

    Args:
        scores_row: Series/dict with {etf}_score columns for one date.
        available_etfs: Which ETFs to consider.
        config: Rotation config.

    Returns:
        SectorSignals with longs, shorts, and risk-off flag.
    """
    config = config or SectorRotationConfig()
    etfs = available_etfs or SECTOR_ETFS

    if isinstance(scores_row, pd.Series):
        date = scores_row.name if hasattr(scores_row, "name") else pd.Timestamp.now()
    else:
        date = pd.Timestamp.now()

    etf_scores: dict[str, float] = {}
    for etf in etfs:
        key = f"{etf}_score"
        val = scores_row.get(key, np.nan) if isinstance(scores_row, dict) else scores_row.get(key, np.nan)
        if not np.isnan(val):
            etf_scores[etf] = val

    if not etf_scores:
        return SectorSignals(date=date, scores={}, longs=[], shorts=[], is_risk_off=False, negative_count=0)

    # Count negative momentum sectors
    negative_count = sum(1 for v in etf_scores.values() if v < 0)
    is_risk_off = negative_count >= config.risk_off_threshold

    if is_risk_off:
        # All sectors weak → rotate to bonds/gold (signals empty, handled upstream)
        return SectorSignals(
            date=date, scores=etf_scores, longs=[], shorts=[],
            is_risk_off=True, negative_count=negative_count,
        )

    # Rank by score
    ranked = sorted(etf_scores.items(), key=lambda x: x[1], reverse=True)

    longs = [etf for etf, _ in ranked[:config.top_n_long]]
    # Only short if score is negative
    shorts = [etf for etf, score in ranked[-config.bottom_n_short:] if score < 0]

    return SectorSignals(
        date=date,
        scores=etf_scores,
        longs=longs,
        shorts=shorts,
        is_risk_off=False,
        negative_count=negative_count,
    )


def get_sector_weights(
    signals: SectorSignals,
    long_weight: float = 0.12,
    short_weight: float = 0.08,
) -> dict[str, float]:
    """Convert SectorSignals into weight dict for portfolio.

    Args:
        signals: Output of generate_sector_rotation_signals().
        long_weight: Weight per long sector ETF.
        short_weight: Absolute weight per short sector ETF (will be negative).

    Returns:
        Dict: {symbol: weight} — positive for longs, negative for shorts.
    """
    if signals.is_risk_off:
        return {}

    weights: dict[str, float] = {}
    for etf in signals.longs:
        weights[etf] = long_weight
    for etf in signals.shorts:
        weights[etf] = -short_weight
    return weights


__all__ = [
    "SECTOR_ETFS",
    "SECTOR_NAMES",
    "SectorRotationConfig",
    "SectorSignals",
    "compute_sector_scores",
    "generate_sector_rotation_signals",
    "get_sector_weights",
]
