"""DataQualityGate — central validation entry-point. From 37_DATA_QUALITY_GATE.md §2.2."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .checks.missing_bars import detect_missing_trading_days
from .checks.price_spike import detect_price_spikes
from .checks.splits import detect_unadjusted_splits
from .checks.volume import detect_volume_anomalies
from .schemas.ohlcv import OHLCVSchema

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when a batch fails schema validation."""


class DataQualityGate:
    """Validate OHLCV data before it enters the feature pipeline.

    Usage::

        gate = DataQualityGate()
        clean_df, meta = gate.validate_ohlcv(raw_df, source="alpaca", batch_id="2026-04-28")
        anomalies = gate.run_anomaly_checks(clean_df)
    """

    def __init__(
        self,
        quarantine_path: str | Path = "data/quarantine",
        raise_on_schema_error: bool = True,
    ) -> None:
        self.quarantine_path = Path(quarantine_path)
        self.raise_on_schema_error = raise_on_schema_error

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        source: str,
        batch_id: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run Pandera schema checks. Returns (clean_df, metadata).

        On failure: quarantines batch + raises DataQualityError
        (if raise_on_schema_error=True) or returns empty df + metadata.
        """
        try:
            from pandera.errors import SchemaErrors

            clean_df = OHLCVSchema.validate(df, lazy=True)
            meta: dict[str, Any] = {
                "status": "pass",
                "rows_in": len(df),
                "rows_out": len(clean_df),
                "source": source,
                "batch_id": batch_id,
                "error_count": 0,
            }
            logger.debug(
                "[OK] DQ schema %s/%s: %d rows", source, batch_id, len(clean_df)
            )
            return clean_df, meta
        except Exception as exc:
            # Catch both SchemaErrors and SchemaError
            failures: pd.DataFrame
            try:
                from pandera.errors import SchemaErrors

                if isinstance(exc, SchemaErrors):
                    failures = exc.failure_cases
                else:
                    failures = pd.DataFrame([{"check": str(exc)}])
            except ImportError:
                failures = pd.DataFrame([{"check": str(exc)}])

            self._quarantine(df, source, batch_id, failures)
            meta = {
                "status": "fail",
                "rows_in": len(df),
                "rows_out": 0,
                "source": source,
                "batch_id": batch_id,
                "error_count": len(failures),
                "failures_sample": failures.head(5).to_dict("records"),
            }
            logger.warning(
                "[WARN] DQ schema FAIL %s/%s: %d errors",
                source,
                batch_id,
                len(failures),
            )
            if self.raise_on_schema_error:
                raise DataQualityError(
                    f"Batch {batch_id!r} from {source!r} failed validation "
                    f"with {len(failures)} errors. Quarantined."
                ) from exc
            return pd.DataFrame(), meta

    # ------------------------------------------------------------------
    # Anomaly checks (non-blocking by default)
    # ------------------------------------------------------------------

    def run_anomaly_checks(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        timestamp_col: str = "timestamp",
        price_col: str = "close",
        calendar: str = "NYSE",
    ) -> dict[str, pd.DataFrame]:
        """Run all dynamic anomaly checks. Returns dict of findings DataFrames.

        These are advisory — they do not block the pipeline. Callers decide
        whether to drop flagged rows or raise.
        """
        results: dict[str, pd.DataFrame] = {}

        results["price_spikes"] = detect_price_spikes(
            df, ticker_col=ticker_col, timestamp_col=timestamp_col, price_col=price_col
        )
        results["volume_anomalies"] = detect_volume_anomalies(
            df, ticker_col=ticker_col, timestamp_col=timestamp_col
        )
        results["possible_splits"] = detect_unadjusted_splits(
            df, ticker_col=ticker_col, timestamp_col=timestamp_col, price_col=price_col
        )

        if timestamp_col in df.columns:
            try:
                results["missing_bars"] = detect_missing_trading_days(
                    df,
                    ticker_col=ticker_col,
                    timestamp_col=timestamp_col,
                    expected_market_calendar=calendar,
                )
            except Exception as e:
                logger.debug("missing_bars check skipped: %s", e)
                results["missing_bars"] = pd.DataFrame()

        total_issues = sum(len(v) for v in results.values())
        if total_issues:
            logger.warning(
                "[WARN] DQ anomaly checks: %d issues (%s)",
                total_issues,
                ", ".join(f"{k}={len(v)}" for k, v in results.items() if len(v)),
            )
        return results

    def summary(self, anomalies: dict[str, pd.DataFrame]) -> dict[str, int]:
        """Compact counts summary from run_anomaly_checks output."""
        return {k: len(v) for k, v in anomalies.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quarantine(
        self,
        df: pd.DataFrame,
        source: str,
        batch_id: str,
        failures: pd.DataFrame,
    ) -> None:
        qdir = self.quarantine_path / source / batch_id
        qdir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(qdir / "data.parquet")
            failures.to_csv(qdir / "failures.csv", index=False)
            meta = {
                "source": source,
                "batch_id": batch_id,
                "quarantined_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(df),
                "error_count": len(failures),
            }
            with open(qdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning("Quarantine write failed: %s", e)
