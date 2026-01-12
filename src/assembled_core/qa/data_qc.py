# src/assembled_core/qa/data_qc.py
"""Data Quality Control (QC) Module.

This module provides deterministic, side-effect-free quality checks for price panels.
QC is a data hygiene gate, not strategy evaluation.

Key Functions:
    - run_price_panel_qc(): Run all QC checks on a price panel
    - qc_report_to_dict(): Convert QcReport to dictionary
    - write_qc_report_json(): Write QC report to JSON file
    - write_qc_summary_md(): Write QC summary to Markdown file (ASCII-only)

Determinism:
    - Issues are sorted: severity (FAIL before WARN), then check, then symbol, then timestamp
    - All timestamps are UTC-normalized (tz-aware)
    - JSON output has deterministic key order
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_THRESHOLDS = {
    "missing_sessions_warn_pct": 5.0,  # Warn if >5% of trading days missing
    "missing_sessions_fail_pct": 20.0,  # Fail if >20% of trading days missing
    "stale_price_sessions": 3,  # Warn if price unchanged for >=3 sessions
    "outlier_return_warn": 0.20,  # Warn if abs(daily_return) >= 0.20 (20%)
    "outlier_return_fail": 0.30,  # Fail if abs(daily_return) >= 0.30 (30%)
    "zero_volume_warn_pct": 10.0,  # Warn if >10% of trading days have zero volume
    "zero_volume_fail_pct": 50.0,  # Fail if >50% of trading days have zero volume
}


@dataclass
class QcIssue:
    """A single QC issue.

    Attributes:
        check: Check name (e.g., "missing_sessions", "negative_price", "outlier_return")
        severity: Severity level ("WARN" or "FAIL")
        symbol: Symbol affected (None if issue applies to entire panel)
        timestamp: Timestamp affected (None if issue applies to entire panel or symbol)
        message: Human-readable message
        details: Additional details (JSON-serializable dict, keep small)
    """

    check: str
    severity: Literal["WARN", "FAIL"]
    symbol: str | None = None
    timestamp: pd.Timestamp | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize timestamp to UTC if provided."""
        if self.timestamp is not None:
            if self.timestamp.tz is None:
                self.timestamp = pd.to_datetime(self.timestamp, utc=True)
            elif self.timestamp.tz != pd.Timestamp.utcnow().tz:
                self.timestamp = self.timestamp.tz_convert("UTC")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        result = {
            "check": self.check,
            "severity": self.severity,
            "symbol": self.symbol,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp is not None else None
            ),
            "message": self.message,
            "details": self.details,
        }
        return result


@dataclass
class QcReport:
    """QC report for a price panel.

    Attributes:
        ok: True if no FAIL issues, False otherwise
        summary: Summary statistics (counts per check, counts WARN/FAIL, etc.)
        issues: List of QC issues (sorted deterministically)
        created_at_utc: Report creation timestamp (UTC)
    """

    ok: bool
    summary: dict[str, Any]
    issues: list[QcIssue]
    created_at_utc: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())

    def __post_init__(self) -> None:
        """Ensure created_at_utc is UTC-aware."""
        if self.created_at_utc.tz is None:
            self.created_at_utc = pd.to_datetime(self.created_at_utc, utc=True)
        elif self.created_at_utc.tz != pd.Timestamp.utcnow().tz:
            self.created_at_utc = self.created_at_utc.tz_convert("UTC")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            "ok": self.ok,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "created_at_utc": self.created_at_utc.isoformat(),
        }


def _sort_issues_deterministic(issues: list[QcIssue]) -> list[QcIssue]:
    """Sort issues deterministically: severity (FAIL before WARN), then check, then symbol, then timestamp.

    Args:
        issues: List of QC issues

    Returns:
        Sorted list of issues
    """
    # Define severity order (FAIL=0, WARN=1)
    severity_order = {"FAIL": 0, "WARN": 1}

    def sort_key(issue: QcIssue) -> tuple:
        return (
            severity_order.get(issue.severity, 2),  # FAIL before WARN
            issue.check,  # Then by check name
            issue.symbol or "",  # Then by symbol (None -> empty string)
            issue.timestamp or pd.Timestamp.min,  # Then by timestamp (None -> min)
        )

    return sorted(issues, key=sort_key)


def _check_invalid_prices(
    prices: pd.DataFrame, issues: list[QcIssue]
) -> list[QcIssue]:
    """Check for invalid prices (close <= 0 or NaN in required fields).

    Args:
        prices: Price DataFrame
        issues: List to append issues to

    Returns:
        Updated list of issues
    """
    # Check for NaN in required fields
    required_cols = ["timestamp", "symbol", "close"]
    for col in required_cols:
        if col not in prices.columns:
            issues.append(
                QcIssue(
                    check="missing_column",
                    severity="FAIL",
                    message=f"Required column '{col}' is missing",
                    details={"column": col},
                )
            )
            continue

        nan_mask = prices[col].isna()
        if nan_mask.any():
            nan_count = nan_mask.sum()
            issues.append(
                QcIssue(
                    check="missing_values",
                    severity="FAIL",
                    message=f"Column '{col}' has {nan_count} NaN values",
                    details={"column": col, "count": int(nan_count)},
                )
            )

    # Check for negative or zero close prices
    if "close" in prices.columns:
        invalid_mask = (prices["close"] <= 0) | prices["close"].isna()
        if invalid_mask.any():
            invalid_rows = prices[invalid_mask]
            for _, row in invalid_rows.iterrows():
                issues.append(
                    QcIssue(
                        check="negative_price",
                        severity="FAIL",
                        symbol=str(row.get("symbol", "")),
                        timestamp=row.get("timestamp"),
                        message=f"Invalid close price: {row.get('close', 'NaN')}",
                        details={"close": float(row.get("close", 0.0))},
                    )
                )

    return issues


def _check_duplicate_rows(
    prices: pd.DataFrame, issues: list[QcIssue]
) -> list[QcIssue]:
    """Check for duplicate rows (identical symbol,timestamp).

    Args:
        prices: Price DataFrame
        issues: List to append issues to

    Returns:
        Updated list of issues
    """
    if "symbol" not in prices.columns or "timestamp" not in prices.columns:
        return issues

    duplicates = prices[prices.duplicated(subset=["symbol", "timestamp"], keep=False)]
    if not duplicates.empty:
        # Group by symbol,timestamp to count duplicates
        dup_groups = duplicates.groupby(["symbol", "timestamp"]).size()
        for (symbol, timestamp), count in dup_groups.items():
            issues.append(
                QcIssue(
                    check="duplicate_rows",
                    severity="FAIL",
                    symbol=str(symbol),
                    timestamp=timestamp if isinstance(timestamp, pd.Timestamp) else None,
                    message=f"Duplicate rows: {count} rows with same symbol,timestamp",
                    details={"count": int(count)},
                )
            )

    return issues


def _check_missing_sessions(
    prices: pd.DataFrame,
    freq: str,
    issues: list[QcIssue],
    thresholds: dict[str, Any],
    as_of: pd.Timestamp | None = None,
) -> list[QcIssue]:
    """Check for missing trading sessions (1d frequency only for now).

    Args:
        prices: Price DataFrame
        freq: Frequency string ("1d" or "5min")
        issues: List to append issues to
        thresholds: Threshold dictionary
        as_of: Optional cutoff timestamp

    Returns:
        Updated list of issues
    """
    if freq != "1d":
        # TODO: Implement for intraday (5min) later
        return issues

    if "symbol" not in prices.columns or "timestamp" not in prices.columns:
        return issues

    try:
        from src.assembled_core.data.calendar import trading_sessions
    except ImportError:
        logger.warning(
            "exchange_calendars not installed - skipping missing_sessions check"
        )
        return issues

    # Get date range from prices
    if prices.empty:
        return issues

    min_date = prices["timestamp"].min().date()
    max_date = prices["timestamp"].max().date()

    # Get expected trading sessions
    try:
        expected_sessions = trading_sessions(min_date, max_date)
    except Exception as e:
        logger.warning(f"Failed to get trading sessions: {e}")
        return issues

    if expected_sessions.empty:
        return issues

    # Check each symbol
    for symbol in prices["symbol"].unique():
        symbol_data = prices[prices["symbol"] == symbol].copy()
        if symbol_data.empty:
            continue

        # Get actual sessions for this symbol
        actual_dates = pd.to_datetime(symbol_data["timestamp"]).dt.date.unique()
        expected_dates = pd.to_datetime(expected_sessions).date

        # Find missing dates
        missing_dates = set(expected_dates) - set(actual_dates)

        if missing_dates:
            missing_pct = (len(missing_dates) / len(expected_dates)) * 100.0

            # Determine severity based on threshold
            if missing_pct >= thresholds.get("missing_sessions_fail_pct", 20.0):
                severity = "FAIL"
            elif missing_pct >= thresholds.get("missing_sessions_warn_pct", 5.0):
                severity = "WARN"
            else:
                # Below threshold, skip
                continue

            issues.append(
                QcIssue(
                    check="missing_sessions",
                    severity=severity,
                    symbol=str(symbol),
                    message=f"Missing {len(missing_dates)} trading sessions ({missing_pct:.1f}%)",
                    details={
                        "missing_count": len(missing_dates),
                        "missing_pct": float(missing_pct),
                        "expected_count": len(expected_dates),
                        "actual_count": len(actual_dates),
                    },
                )
            )

    return issues


def _check_stale_prices(
    prices: pd.DataFrame,
    freq: str,
    issues: list[QcIssue],
    thresholds: dict[str, Any],
) -> list[QcIssue]:
    """Check for stale prices (price unchanged for >= N sessions).

    Args:
        prices: Price DataFrame
        freq: Frequency string ("1d" or "5min")
        issues: List to append issues to
        thresholds: Threshold dictionary

    Returns:
        Updated list of issues
    """
    if "close" not in prices.columns or "symbol" not in prices.columns:
        return issues

    stale_sessions = thresholds.get("stale_price_sessions", 3)

    # Check each symbol
    for symbol in prices["symbol"].unique():
        symbol_data = prices[prices["symbol"] == symbol].copy()
        if symbol_data.empty or len(symbol_data) < stale_sessions:
            continue

        symbol_data = symbol_data.sort_values("timestamp").reset_index(drop=True)

        # Find consecutive unchanged prices
        price_changes = symbol_data["close"].diff().abs()
        unchanged_mask = price_changes < 1e-6  # Tolerance for floating point

        # Find runs of unchanged prices
        run_length = 0
        for i, unchanged in enumerate(unchanged_mask):
            if unchanged:
                run_length += 1
            else:
                if run_length >= stale_sessions:
                    # Found stale price run
                    start_idx = i - run_length
                    end_idx = i - 1
                    start_ts = symbol_data.iloc[start_idx]["timestamp"]
                    end_ts = symbol_data.iloc[end_idx]["timestamp"]
                    price_val = symbol_data.iloc[start_idx]["close"]

                    issues.append(
                        QcIssue(
                            check="stale_price",
                            severity="WARN",
                            symbol=str(symbol),
                            timestamp=end_ts,
                            message=f"Price unchanged for {run_length} sessions: {price_val:.2f}",
                            details={
                                "sessions": int(run_length),
                                "price": float(price_val),
                                "start_timestamp": start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else str(start_ts),
                                "end_timestamp": end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else str(end_ts),
                            },
                        )
                    )
                run_length = 0

        # Check final run
        if run_length >= stale_sessions:
            start_idx = len(symbol_data) - run_length
            end_idx = len(symbol_data) - 1
            start_ts = symbol_data.iloc[start_idx]["timestamp"]
            end_ts = symbol_data.iloc[end_idx]["timestamp"]
            price_val = symbol_data.iloc[start_idx]["close"]

            issues.append(
                QcIssue(
                    check="stale_price",
                    severity="WARN",
                    symbol=str(symbol),
                    timestamp=end_ts,
                    message=f"Price unchanged for {run_length} sessions: {price_val:.2f}",
                    details={
                        "sessions": int(run_length),
                        "price": float(price_val),
                        "start_timestamp": start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else str(start_ts),
                        "end_timestamp": end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else str(end_ts),
                    },
                )
            )

    return issues


def _check_outlier_returns(
    prices: pd.DataFrame,
    issues: list[QcIssue],
    thresholds: dict[str, Any],
) -> list[QcIssue]:
    """Check for outlier returns (abs(daily_return) >= threshold).

    Args:
        prices: Price DataFrame
        issues: List to append issues to
        thresholds: Threshold dictionary

    Returns:
        Updated list of issues
    """
    if "close" not in prices.columns or "symbol" not in prices.columns:
        return issues

    warn_threshold = thresholds.get("outlier_return_warn", 0.20)
    fail_threshold = thresholds.get("outlier_return_fail", 0.30)

    # Check each symbol
    for symbol in prices["symbol"].unique():
        symbol_data = prices[prices["symbol"] == symbol].copy()
        if symbol_data.empty or len(symbol_data) < 2:
            continue

        symbol_data = symbol_data.sort_values("timestamp").reset_index(drop=True)

        # Calculate daily returns
        returns = symbol_data["close"].pct_change()

        # Find outliers
        for i, ret in enumerate(returns):
            if pd.isna(ret):
                continue

            abs_ret = abs(ret)

            if abs_ret >= fail_threshold:
                severity = "FAIL"
                threshold_used = fail_threshold
            elif abs_ret >= warn_threshold:
                severity = "WARN"
                threshold_used = warn_threshold
            else:
                continue

            timestamp = symbol_data.iloc[i]["timestamp"]
            price = symbol_data.iloc[i]["close"]
            prev_price = symbol_data.iloc[i - 1]["close"] if i > 0 else None

            issues.append(
                QcIssue(
                    check="outlier_return",
                    severity=severity,
                    symbol=str(symbol),
                    timestamp=timestamp,
                    message=f"Outlier return: {ret:.2%} (threshold: {threshold_used:.2%})",
                    details={
                        "return": float(ret),
                        "abs_return": float(abs_ret),
                        "threshold": float(threshold_used),
                        "price": float(price),
                        "prev_price": float(prev_price) if prev_price is not None else None,
                    },
                )
            )

    return issues


def _check_zero_volume(
    prices: pd.DataFrame,
    freq: str,
    issues: list[QcIssue],
    thresholds: dict[str, Any],
) -> list[QcIssue]:
    """Check for zero volume anomalies (volume==0 on trading days).

    Args:
        prices: Price DataFrame
        freq: Frequency string ("1d" or "5min")
        issues: List to append issues to
        thresholds: Threshold dictionary

    Returns:
        Updated list of issues
    """
    if "volume" not in prices.columns:
        # Volume not available, skip check
        return issues

    if freq != "1d":
        # TODO: Implement for intraday (5min) later
        return issues

    # Check each symbol
    for symbol in prices["symbol"].unique():
        symbol_data = prices[prices["symbol"] == symbol].copy()
        if symbol_data.empty:
            continue

        # Count zero volume days
        zero_volume_mask = (symbol_data["volume"] == 0) | symbol_data["volume"].isna()
        zero_volume_count = zero_volume_mask.sum()
        total_count = len(symbol_data)

        if zero_volume_count == 0:
            continue

        zero_volume_pct = (zero_volume_count / total_count) * 100.0

        # Determine severity based on threshold
        if zero_volume_pct >= thresholds.get("zero_volume_fail_pct", 50.0):
            severity = "FAIL"
        elif zero_volume_pct >= thresholds.get("zero_volume_warn_pct", 10.0):
            severity = "WARN"
        else:
            # Below threshold, skip
            continue

        issues.append(
            QcIssue(
                check="zero_volume",
                severity=severity,
                symbol=str(symbol),
                message=f"Zero volume on {zero_volume_count} days ({zero_volume_pct:.1f}%)",
                details={
                    "zero_volume_count": int(zero_volume_count),
                    "total_count": int(total_count),
                    "zero_volume_pct": float(zero_volume_pct),
                },
            )
        )

    return issues


def run_price_panel_qc(
    prices: pd.DataFrame,
    *,
    freq: str,
    calendar: str = "NYSE",
    as_of: pd.Timestamp | None = None,
    thresholds: dict[str, Any] | None = None,
) -> QcReport:
    """Run quality control checks on a price panel.

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, close, ... (volume optional)
        freq: Frequency string ("1d" or "5min")
        calendar: Calendar name (default: "NYSE", currently only NYSE supported)
        as_of: Optional cutoff timestamp for PIT-safe checks
        thresholds: Optional threshold dictionary (overrides defaults)

    Returns:
        QcReport with all issues found

    Example:
        >>> prices = pd.DataFrame({
        ...     "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        ...     "symbol": ["AAPL"] * 10,
        ...     "close": [150.0] * 10,
        ... })
        >>> report = run_price_panel_qc(prices, freq="1d")
        >>> print(f"QC OK: {report.ok}")
    """
    if calendar != "NYSE":
        raise ValueError(f"Only NYSE calendar is supported, got: {calendar}")

    # Merge thresholds with defaults
    merged_thresholds = {**DEFAULT_THRESHOLDS}
    if thresholds:
        merged_thresholds.update(thresholds)

    # Normalize timestamps to UTC
    if not prices.empty and "timestamp" in prices.columns:
        if prices["timestamp"].dt.tz is None:
            prices = prices.copy()
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        elif prices["timestamp"].dt.tz != pd.Timestamp.utcnow().tz:
            prices = prices.copy()
            prices["timestamp"] = prices["timestamp"].dt.tz_convert("UTC")

    # Collect issues
    issues: list[QcIssue] = []

    # Run checks
    issues = _check_invalid_prices(prices, issues)
    issues = _check_duplicate_rows(prices, issues)
    issues = _check_missing_sessions(prices, freq, issues, merged_thresholds, as_of)
    issues = _check_stale_prices(prices, freq, issues, merged_thresholds)
    issues = _check_outlier_returns(prices, issues, merged_thresholds)
    issues = _check_zero_volume(prices, freq, issues, merged_thresholds)

    # Sort issues deterministically
    issues = _sort_issues_deterministic(issues)

    # Build summary
    summary: dict[str, Any] = {
        "total_issues": len(issues),
        "fail_count": sum(1 for issue in issues if issue.severity == "FAIL"),
        "warn_count": sum(1 for issue in issues if issue.severity == "WARN"),
        "checks_run": [
            "invalid_prices",
            "duplicate_rows",
            "missing_sessions",
            "stale_prices",
            "outlier_returns",
            "zero_volume",
        ],
    }

    # Count issues per check
    check_counts: dict[str, int] = {}
    for issue in issues:
        check_counts[issue.check] = check_counts.get(issue.check, 0) + 1
    summary["issues_by_check"] = check_counts

    # Determine overall status
    ok = summary["fail_count"] == 0

    return QcReport(ok=ok, summary=summary, issues=issues)


def qc_report_to_dict(report: QcReport) -> dict[str, Any]:
    """Convert QC report to dictionary (JSON-serializable).

    Args:
        report: QC report

    Returns:
        Dictionary representation
    """
    return report.to_dict()


def write_qc_report_json(report: QcReport, path: Path | str) -> None:
    """Write QC report to JSON file (deterministic key order, NaN/inf -> null).

    Args:
        report: QC report
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    report_dict = qc_report_to_dict(report)

    # Custom JSON encoder to handle NaN/inf
    class JSONEncoder(json.JSONEncoder):
        def encode(self, obj: Any) -> str:
            if isinstance(obj, float):
                if pd.isna(obj):
                    return "null"
                if np.isinf(obj):
                    return "null"
            return super().encode(obj)

    # Write with deterministic key order (sorted)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, sort_keys=True, cls=JSONEncoder, ensure_ascii=False)


def write_qc_summary_md(report: QcReport, path: Path | str) -> None:
    """Write QC summary to Markdown file (ASCII-only).

    Args:
        report: QC report
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Data Quality Control Report")
    lines.append("")
    lines.append(f"**Status:** {'OK' if report.ok else 'FAILED'}")
    lines.append(f"**Created:** {report.created_at_utc.isoformat()} UTC")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total Issues: {report.summary['total_issues']}")
    lines.append(f"- FAIL: {report.summary['fail_count']}")
    lines.append(f"- WARN: {report.summary['warn_count']}")
    lines.append("")
    lines.append("## Issues by Check")
    lines.append("")
    for check, count in sorted(report.summary.get("issues_by_check", {}).items()):
        lines.append(f"- {check}: {count}")
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    if not report.issues:
        lines.append("No issues found.")
    else:
        for issue in report.issues:
            lines.append(f"### {issue.check} ({issue.severity})")
            lines.append("")
            if issue.symbol:
                lines.append(f"- Symbol: {issue.symbol}")
            if issue.timestamp:
                lines.append(f"- Timestamp: {issue.timestamp.isoformat()}")
            lines.append(f"- Message: {issue.message}")
            if issue.details:
                lines.append("- Details:")
                for key, value in sorted(issue.details.items()):
                    lines.append(f"  - {key}: {value}")
            lines.append("")

    # Write file (ASCII-only)
    content = "\n".join(lines)
    # Ensure ASCII-only (replace non-ASCII with ASCII equivalents)
    content_bytes = content.encode("ascii", errors="replace")
    content = content_bytes.decode("ascii")

    with open(path, "w", encoding="ascii") as f:
        f.write(content)
