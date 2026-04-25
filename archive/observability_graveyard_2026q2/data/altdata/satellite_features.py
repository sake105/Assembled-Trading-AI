"""Satellite / Geospatial Alternative Data Features (M38c).

Processes satellite-derived economic indicators for alpha generation.
Designed to consume pre-processed satellite data (parking lot counts,
shipping traffic, nighttime lights, crop health indices).

Features produced:
    parking_lot_occupancy   — normalized parking lot fill rate [0, 1]
    parking_lot_trend_4w    — 4-week trend in occupancy
    shipping_volume_index   — container/vessel activity index
    shipping_trend_4w       — 4-week trend in shipping
    nightlight_intensity    — economic activity proxy from nighttime lights
    nightlight_yoy_change   — year-over-year change in nightlight intensity
    crop_health_index       — NDVI-based crop health [0, 1]

All features are PIT-safe: only satellite data available by processing
date (typically T+1 to T+3) is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SatelliteConfig:
    """Configuration for satellite feature processing."""

    processing_lag_days: int = 2
    trend_window_weeks: int = 4
    min_observations: int = 3
    yoy_lookback_days: int = 365


def process_parking_lot_data(
    raw_data: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: SatelliteConfig | None = None,
    symbol_col: str = "symbol",
    date_col: str = "observation_date",
    occupancy_col: str = "occupancy_rate",
) -> pd.DataFrame:
    """Process parking lot satellite data into features.

    Args:
        raw_data: DataFrame with parking lot observations per symbol/date.
        as_of: Reference date (PIT cutoff).
        config: SatelliteConfig.
        symbol_col: Symbol column.
        date_col: Observation date column.
        occupancy_col: Occupancy rate column [0, 1].

    Returns:
        DataFrame with parking_lot_occupancy and parking_lot_trend_4w per symbol.
    """
    cfg = config or SatelliteConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if raw_data.empty:
        return pd.DataFrame(columns=[symbol_col, "parking_lot_occupancy", "parking_lot_trend_4w"])

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    if df.empty:
        return pd.DataFrame(columns=[symbol_col, "parking_lot_occupancy", "parking_lot_trend_4w"])

    trend_cutoff = pit_cutoff - pd.Timedelta(weeks=cfg.trend_window_weeks)
    rows = []

    for sym, grp in df.groupby(symbol_col):
        recent = grp.sort_values(date_col)

        if len(recent) < cfg.min_observations:
            continue

        # Latest occupancy
        latest = float(recent[occupancy_col].iloc[-1])

        # 4-week trend
        recent_window = recent[recent[date_col] > trend_cutoff]
        if len(recent_window) >= 2:
            first_half = recent_window[occupancy_col].iloc[:len(recent_window)//2].mean()
            second_half = recent_window[occupancy_col].iloc[len(recent_window)//2:].mean()
            trend = float(second_half - first_half)
        else:
            trend = 0.0

        rows.append({
            symbol_col: sym,
            "parking_lot_occupancy": np.clip(latest, 0.0, 1.0),
            "parking_lot_trend_4w": round(trend, 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[Satellite] Processed parking lot data for %d symbols", len(result))
    return result


def process_shipping_data(
    raw_data: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: SatelliteConfig | None = None,
    region_col: str = "region",
    date_col: str = "observation_date",
    volume_col: str = "vessel_count",
) -> pd.DataFrame:
    """Process shipping/port activity data into features.

    Args:
        raw_data: DataFrame with shipping observations per region/date.
        as_of: Reference date.
        config: SatelliteConfig.
        region_col: Region/port column.
        date_col: Observation date column.
        volume_col: Vessel count or volume column.

    Returns:
        DataFrame with shipping_volume_index and shipping_trend_4w per region.
    """
    cfg = config or SatelliteConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if raw_data.empty:
        return pd.DataFrame(columns=[region_col, "shipping_volume_index", "shipping_trend_4w"])

    df = raw_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    if df.empty:
        return pd.DataFrame(columns=[region_col, "shipping_volume_index", "shipping_trend_4w"])

    trend_cutoff = pit_cutoff - pd.Timedelta(weeks=cfg.trend_window_weeks)
    rows = []

    for region, grp in df.groupby(region_col):
        sorted_grp = grp.sort_values(date_col)

        if len(sorted_grp) < cfg.min_observations:
            continue

        # Normalize to index (latest / rolling mean)
        mean_vol = sorted_grp[volume_col].mean()
        if mean_vol < 1e-10:
            continue
        latest = float(sorted_grp[volume_col].iloc[-1])
        index_val = latest / mean_vol

        # Trend
        window = sorted_grp[sorted_grp[date_col] > trend_cutoff]
        if len(window) >= 2:
            first = window[volume_col].iloc[:len(window)//2].mean()
            second = window[volume_col].iloc[len(window)//2:].mean()
            trend = (second - first) / max(mean_vol, 1e-10)
        else:
            trend = 0.0

        rows.append({
            region_col: region,
            "shipping_volume_index": round(float(index_val), 4),
            "shipping_trend_4w": round(float(trend), 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[Satellite] Processed shipping data for %d regions", len(result))
    return result


def compute_nightlight_features(
    observations: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: SatelliteConfig | None = None,
    region_col: str = "region",
    date_col: str = "observation_date",
    intensity_col: str = "light_intensity",
) -> pd.DataFrame:
    """Compute nighttime light intensity features.

    Args:
        observations: DataFrame with nightlight observations.
        as_of: Reference date.
        config: SatelliteConfig.

    Returns:
        DataFrame with nightlight_intensity and nightlight_yoy_change per region.
    """
    cfg = config or SatelliteConfig()
    as_of_dt = pd.Timestamp(as_of)
    pit_cutoff = as_of_dt - pd.Timedelta(days=cfg.processing_lag_days)

    if observations.empty:
        return pd.DataFrame(columns=[region_col, "nightlight_intensity", "nightlight_yoy_change"])

    df = observations.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] <= pit_cutoff]

    yoy_cutoff = pit_cutoff - pd.Timedelta(days=cfg.yoy_lookback_days)
    rows = []

    for region, grp in df.groupby(region_col):
        sorted_grp = grp.sort_values(date_col)
        if len(sorted_grp) < 1:
            continue

        latest = float(sorted_grp[intensity_col].iloc[-1])

        # YoY change
        prior = sorted_grp[sorted_grp[date_col] <= yoy_cutoff]
        if len(prior) > 0:
            prior_val = float(prior[intensity_col].iloc[-1])
            yoy = (latest - prior_val) / max(abs(prior_val), 1e-10)
        else:
            yoy = 0.0

        rows.append({
            region_col: region,
            "nightlight_intensity": round(latest, 4),
            "nightlight_yoy_change": round(float(yoy), 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[Satellite] Computed nightlight features for %d regions", len(result))
    return result


__all__ = [
    "SatelliteConfig",
    "process_parking_lot_data",
    "process_shipping_data",
    "compute_nightlight_features",
]
