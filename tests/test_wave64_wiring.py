"""Tests for wave-64 module wiring into trading_cycle.py.

Covers:
  Step 7.76 — compliance.audit_log (AuditLog / AuditEventType)
  Step 7.77 — compliance.otr_monitor (OTRMonitor / OTRSnapshot)
  Step 7.78 — compliance.regulatory_reports (generate_best_execution_report)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.compliance.audit_log import (
    AuditLog,
    AuditEventType,
    AuditEntry,
)
from src.assembled_core.compliance.otr_monitor import (
    OTRMonitor,
    OTRSnapshot,
    OTRAlertLevel,
)
from src.assembled_core.compliance.regulatory_reports import (
    generate_best_execution_report,
    BestExecutionReport,
)


# ---------------------------------------------------------------------------
# audit_log (Step 7.76)
# ---------------------------------------------------------------------------

def test_audit_log_creates_in_memory():
    log = AuditLog(log_path=None)
    assert isinstance(log, AuditLog)


def test_audit_log_append():
    log = AuditLog(log_path=None)
    log.append(event_type=AuditEventType.RECONCILIATION, payload={"n": 3})
    assert len(log._entries) == 1


def test_audit_log_entry_fields():
    log = AuditLog(log_path=None)
    log.append(event_type=AuditEventType.ORDER_CREATED, payload={"symbol": "AAPL"})
    entry = log._entries[0]
    assert entry.sequence >= 1
    assert entry.event_type == AuditEventType.ORDER_CREATED.value


def test_audit_event_type_values():
    assert AuditEventType.RECONCILIATION == "reconciliation"
    assert AuditEventType.KILL_SWITCH == "kill_switch"


def test_audit_log_hash_chain():
    log = AuditLog(log_path=None)
    log.append(AuditEventType.ORDER_CREATED, {"symbol": "AAPL"})
    log.append(AuditEventType.ORDER_FILLED, {"symbol": "AAPL"})
    assert len(log._entries) == 2
    # second entry's prev_hash should match first entry's hash
    assert log._entries[1].prev_hash == log._entries[0].entry_hash


# ---------------------------------------------------------------------------
# otr_monitor (Step 7.77)
# ---------------------------------------------------------------------------

def test_otr_monitor_creates():
    mon = OTRMonitor()
    assert isinstance(mon, OTRMonitor)


def test_otr_monitor_default_thresholds():
    mon = OTRMonitor()
    assert mon.warning_threshold > 0
    assert mon.breach_threshold > mon.critical_threshold


def test_otr_monitor_compute_otr_empty():
    mon = OTRMonitor()
    snap = mon.compute_otr()
    assert isinstance(snap, OTRSnapshot)
    assert snap.otr_ratio == 0.0


def test_otr_monitor_record_orders():
    mon = OTRMonitor()
    mon.record_order("AAPL", "submit")
    mon.record_order("AAPL", "fill")
    snap = mon.compute_otr()
    assert snap.orders_submitted == 1
    assert snap.orders_filled == 1


def test_otr_monitor_alert_level_normal():
    mon = OTRMonitor()
    snap = mon.compute_otr()
    assert snap.alert_level == OTRAlertLevel.NORMAL


# ---------------------------------------------------------------------------
# regulatory_reports (Step 7.78)
# ---------------------------------------------------------------------------

def test_generate_best_execution_report_empty():
    report = generate_best_execution_report(
        fills=pd.DataFrame(),
        period_start="2024-01-01",
        period_end="2024-12-31",
    )
    assert isinstance(report, BestExecutionReport)
    assert report.total_orders == 0


def test_generate_best_execution_report_fields():
    report = generate_best_execution_report(
        fills=pd.DataFrame(),
        period_start="2024-01-01",
        period_end="2024-12-31",
    )
    assert hasattr(report, "total_orders")
    assert hasattr(report, "total_fills")
    assert hasattr(report, "avg_slippage_bps")


def test_generate_best_execution_report_with_fills():
    fills = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "venue": ["NASDAQ", "NYSE"],
        "fill_price": [150.5, 300.2],
        "arrival_price": [150.0, 300.0],
        "fill_time_ms": [5.0, 3.5],
        "volume": [100, 50],
        "timestamp": ["2024-06-01T10:00:00Z", "2024-06-01T10:01:00Z"],
    })
    report = generate_best_execution_report(
        fills=fills,
        period_start="2024-01-01",
        period_end="2024-12-31",
    )
    assert isinstance(report, BestExecutionReport)
    assert report.total_fills == 2
