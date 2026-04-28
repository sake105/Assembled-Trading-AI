"""Pandera schema for OHLCV bars. From 37_DATA_QUALITY_GATE.md §2."""
from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series


class OHLCVSchema(pa.DataFrameModel):
    """One row = one bar for one ticker."""

    ticker: Series[str] = pa.Field(
        str_matches=r"^[A-Z0-9\.\-\^]{1,12}$",
        nullable=False,
    )
    open: Series[float] = pa.Field(gt=0, lt=1_000_000, nullable=False)
    high: Series[float] = pa.Field(gt=0, lt=1_000_000, nullable=False)
    low: Series[float] = pa.Field(gt=0, lt=1_000_000, nullable=False)
    close: Series[float] = pa.Field(gt=0, lt=1_000_000, nullable=False)
    volume: Series[int] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = False   # allow extra columns (vwap/trade_count/timestamp handled per-use)
        coerce = True

    @pa.dataframe_check
    def high_ge_low(cls, df: pd.DataFrame) -> Series[bool]:
        return df["high"] >= df["low"]

    @pa.dataframe_check
    def high_ge_open_close(cls, df: pd.DataFrame) -> Series[bool]:
        return (df["high"] >= df["open"]) & (df["high"] >= df["close"])

    @pa.dataframe_check
    def low_le_open_close(cls, df: pd.DataFrame) -> Series[bool]:
        return (df["low"] <= df["open"]) & (df["low"] <= df["close"])
