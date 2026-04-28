"""Data-Quality-Gate — validates OHLCV data before the feature pipeline."""
from .gate import DataQualityError, DataQualityGate
from .schemas.ohlcv import OHLCVSchema
from .checks import (
    detect_missing_trading_days,
    detect_price_spikes,
    detect_unadjusted_splits,
    detect_volume_anomalies,
)

__all__ = [
    "DataQualityGate",
    "DataQualityError",
    "OHLCVSchema",
    "detect_price_spikes",
    "detect_missing_trading_days",
    "detect_volume_anomalies",
    "detect_unadjusted_splits",
]
