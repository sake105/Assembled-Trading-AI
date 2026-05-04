"""Pandera-based OHLCV data quality gate.

From 37_DATA_QUALITY_GATE.md.

Every incoming OHLCV batch is validated before features are computed.
Invalid batches are quarantined; a structured result is returned so the
pipeline can decide to block, warn, or degrade gracefully.

Install: pip install pandera==0.21.0

Falls back to custom checks when pandera is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class QualityStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class QualityResult:
    """Structured output of the quality gate."""

    status: QualityStatus
    ticker: str
    n_rows: int
    checks_failed: list[str] = field(default_factory=list)
    checks_warned: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == QualityStatus.PASS

    @property
    def blocked(self) -> bool:
        return self.status == QualityStatus.FAIL


# ---------------------------------------------------------------------------
# Custom OHLCV checks (no pandera dependency)
# ---------------------------------------------------------------------------


def _check_ohlcv_structure(
    df: pd.DataFrame, ticker: str, result: QualityResult
) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        result.checks_failed.append(f"missing_columns:{missing}")


def _check_no_zero_prices(df: pd.DataFrame, result: QualityResult) -> None:
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns and (df[col] == 0).any():
            result.checks_failed.append(f"zero_price:{col}")


def _check_no_negative_prices(df: pd.DataFrame, result: QualityResult) -> None:
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns and (df[col] < 0).any():
            result.checks_failed.append(f"negative_price:{col}")


def _check_high_ge_low(df: pd.DataFrame, result: QualityResult) -> None:
    if {"High", "Low"} <= set(df.columns):
        violations = (df["High"] < df["Low"]).sum()
        if violations > 0:
            result.checks_failed.append(f"high_lt_low:{violations}_rows")


def _check_close_within_range(df: pd.DataFrame, result: QualityResult) -> None:
    if {"High", "Low", "Close"} <= set(df.columns):
        out_of_range = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
        if out_of_range > 0:
            result.checks_warned.append(f"close_outside_hl:{out_of_range}_rows")


def _check_price_spikes(
    df: pd.DataFrame, result: QualityResult, spike_threshold: float = 3.0
) -> None:
    """Flag rows where intrabar return > spike_threshold × daily vol."""
    if "Close" not in df.columns or len(df) < 5:
        return
    returns = df["Close"].pct_change(fill_method=None).dropna()
    if returns.std() < 1e-9:
        return
    z = returns.abs() / returns.std()
    spikes = int((z > spike_threshold * 10).sum())  # 10× daily vol = spike
    if spikes > 0:
        result.checks_warned.append(f"price_spikes:{spikes}_rows")


def _check_volume(df: pd.DataFrame, result: QualityResult) -> None:
    if "Volume" in df.columns:
        neg = int((df["Volume"] < 0).sum())
        if neg:
            result.checks_failed.append(f"negative_volume:{neg}_rows")


def _check_timestamps_monotonic(df: pd.DataFrame, result: QualityResult) -> None:
    if not df.index.is_monotonic_increasing:
        result.checks_failed.append("timestamps_not_monotonic")


def _check_no_null_prices(df: pd.DataFrame, result: QualityResult) -> None:
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                result.checks_warned.append(f"null_{col.lower()}:{n_null}")


# ---------------------------------------------------------------------------
# Pandera schema (optional)
# ---------------------------------------------------------------------------


def _build_pandera_schema():
    """Build Pandera DataFrameSchema for OHLCV validation."""
    try:
        import pandera as pa

        schema = pa.DataFrameSchema(
            columns={
                "Open": pa.Column(float, pa.Check.gt(0), nullable=False),
                "High": pa.Column(float, pa.Check.gt(0), nullable=False),
                "Low": pa.Column(float, pa.Check.gt(0), nullable=False),
                "Close": pa.Column(float, pa.Check.gt(0), nullable=False),
                "Volume": pa.Column(float, pa.Check.ge(0), nullable=True),
            },
            checks=[
                pa.Check(
                    lambda df: (df["High"] >= df["Low"]).all(), error="high_lt_low"
                ),
                pa.Check(
                    lambda df: (df["Close"] <= df["High"]).all()
                    & (df["Close"] >= df["Low"]).all(),
                    error="close_outside_hl",
                ),
            ],
            index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), coerce=True, name=None),
        )
        return schema
    except ImportError:
        return None


_PANDERA_SCHEMA = None  # lazily built


# ---------------------------------------------------------------------------
# Main gate entry point
# ---------------------------------------------------------------------------


def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    use_pandera: bool = True,
    quarantine_dir: str | None = None,
) -> QualityResult:
    """Run the full OHLCV quality gate.

    Args:
        df: OHLCV DataFrame (DatetimeIndex, columns Open/High/Low/Close/Volume).
        ticker: Ticker symbol for logging.
        use_pandera: Whether to attempt Pandera schema validation.
        quarantine_dir: If set and status is FAIL, save the bad data there.

    Returns:
        QualityResult describing pass/warn/fail status.
    """
    result = QualityResult(
        status=QualityStatus.PASS,
        ticker=ticker,
        n_rows=len(df),
    )

    if df.empty:
        result.checks_failed.append("empty_dataframe")
    else:
        # Custom checks (always run, no extra dependency)
        _check_ohlcv_structure(df, ticker, result)
        _check_no_zero_prices(df, result)
        _check_no_negative_prices(df, result)
        _check_high_ge_low(df, result)
        _check_close_within_range(df, result)
        _check_price_spikes(df, result)
        _check_volume(df, result)
        _check_timestamps_monotonic(df, result)
        _check_no_null_prices(df, result)

        # Pandera schema (optional)
        if use_pandera:
            global _PANDERA_SCHEMA
            if _PANDERA_SCHEMA is None:
                _PANDERA_SCHEMA = _build_pandera_schema()
            if _PANDERA_SCHEMA is not None:
                try:
                    _PANDERA_SCHEMA.validate(df, lazy=True)
                except Exception as exc:
                    result.checks_warned.append(f"pandera:{str(exc)[:120]}")

    # Determine final status
    if result.checks_failed:
        result.status = QualityStatus.FAIL
    elif result.checks_warned:
        result.status = QualityStatus.WARN

    # Quarantine
    if result.status == QualityStatus.FAIL and quarantine_dir:
        _quarantine(df, ticker, quarantine_dir, result)

    level = (
        logging.ERROR
        if result.blocked
        else (logging.WARNING if result.status == QualityStatus.WARN else logging.DEBUG)
    )
    logger.log(
        level,
        "QualityGate %s [%s] rows=%d fails=%s warns=%s",
        ticker,
        result.status.value,
        result.n_rows,
        result.checks_failed,
        result.checks_warned,
    )

    return result


def _quarantine(
    df: pd.DataFrame, ticker: str, quarantine_dir: str, result: QualityResult
) -> None:
    """Save bad data to quarantine directory."""
    import os
    from datetime import datetime, timezone

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(quarantine_dir, ticker)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ts}_quarantine.parquet")
    try:
        df.to_parquet(path)
        result.metadata["quarantine_path"] = path
        logger.warning("Quarantined bad data for %s → %s", ticker, path)
    except Exception as exc:
        logger.error("Quarantine write failed for %s: %s", ticker, exc)


__all__ = ["QualityResult", "QualityStatus", "validate_ohlcv"]
