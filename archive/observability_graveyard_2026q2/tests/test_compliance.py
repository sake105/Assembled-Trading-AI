"""Tests for M28 Compliance & Audit modules.

Covers:
- Task 28.1: Hash-chained Audit Log
- Task 28.2: OTR Monitor
- Task 28.3: Regulatory Report Generator
"""

from __future__ import annotations


import pytest
import numpy as np
import pandas as pd

import pytest; pytest.importorskip('src.assembled_core.compliance.audit_log')
from src.assembled_core.compliance.audit_log import (
    AuditLog, AuditEntry, AuditEventType,
)
import pytest; pytest.importorskip('src.assembled_core.compliance.otr_monitor')
from src.assembled_core.compliance.otr_monitor import (
    OTRMonitor, OTRSnapshot,
)
import pytest; pytest.importorskip('src.assembled_core.compliance.regulatory_reports')
from src.assembled_core.compliance.regulatory_reports import (
    generate_best_execution_report,
    generate_transaction_cost_report,
    generate_risk_report,
    generate_model_inventory,
    BestExecutionReport,
    TransactionCostReport,
    RiskReport,
    ModelInventoryReport,
)


# ===========================================================================
# Task 28.1: Audit Log
# ===========================================================================

@pytest.mark.phase12
class TestAuditLog:
    def test_append_basic(self):
        log = AuditLog()
        entry = log.append(AuditEventType.ORDER_CREATED, {"symbol": "AAPL", "qty": 100})
        assert isinstance(entry, AuditEntry)
        assert entry.sequence == 1
        assert entry.event_type == "order_created"
        assert entry.payload["symbol"] == "AAPL"

    def test_hash_chain(self):
        log = AuditLog()
        e1 = log.append(AuditEventType.ORDER_CREATED, {"symbol": "AAPL"})
        e2 = log.append(AuditEventType.ORDER_FILLED, {"symbol": "AAPL"})
        assert e2.prev_hash == e1.entry_hash
        assert e1.prev_hash == AuditLog.GENESIS_HASH

    def test_verify_chain_valid(self):
        log = AuditLog()
        for i in range(10):
            log.append(AuditEventType.ORDER_CREATED, {"seq": i})
        valid, broken = log.verify_chain()
        assert valid
        assert broken == -1

    def test_verify_chain_tampered(self):
        log = AuditLog()
        log.append(AuditEventType.ORDER_CREATED, {"a": 1})
        log.append(AuditEventType.ORDER_FILLED, {"b": 2})
        log.append(AuditEventType.RISK_BREACH, {"c": 3})
        # Tamper with middle entry
        log._entries[1].payload["b"] = 999
        valid, broken = log.verify_chain()
        assert not valid
        assert broken == 2

    def test_get_entries_by_type(self):
        log = AuditLog()
        log.append(AuditEventType.ORDER_CREATED, {"s": "A"})
        log.append(AuditEventType.RISK_BREACH, {"s": "B"})
        log.append(AuditEventType.ORDER_CREATED, {"s": "C"})
        results = log.get_entries(event_type=AuditEventType.ORDER_CREATED)
        assert len(results) == 2

    def test_persistence(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        log1 = AuditLog(log_path=log_file)
        log1.append(AuditEventType.ORDER_CREATED, {"x": 1})
        log1.append(AuditEventType.ORDER_FILLED, {"x": 2})

        # Reload
        log2 = AuditLog(log_path=log_file)
        assert log2.length == 2
        valid, _ = log2.verify_chain()
        assert valid

    def test_length(self):
        log = AuditLog()
        assert log.length == 0
        log.append(AuditEventType.CONFIG_CHANGE, {})
        assert log.length == 1

    def test_entry_to_dict(self):
        log = AuditLog()
        entry = log.append(AuditEventType.KILL_SWITCH, {"reason": "drawdown"})
        d = entry.to_dict()
        assert "entry_hash" in d
        assert "prev_hash" in d
        restored = AuditEntry.from_dict(d)
        assert restored.entry_hash == entry.entry_hash

    def test_all_event_types(self):
        log = AuditLog()
        for et in AuditEventType:
            log.append(et, {"test": True})
        assert log.length == len(AuditEventType)
        valid, _ = log.verify_chain()
        assert valid


# ===========================================================================
# Task 28.2: OTR Monitor
# ===========================================================================

@pytest.mark.phase12
class TestOTRMonitor:
    def test_basic_normal(self):
        mon = OTRMonitor()
        mon.record_order("AAPL", "submit")
        mon.record_order("AAPL", "fill")
        snap = mon.compute_otr()
        assert isinstance(snap, OTRSnapshot)
        assert snap.otr_ratio == 1.0
        assert snap.alert_level == "normal"

    def test_warning_threshold(self):
        mon = OTRMonitor(warning_threshold=3.0)
        for _ in range(6):
            mon.record_order("AAPL", "submit")
        mon.record_order("AAPL", "fill")
        snap = mon.compute_otr()
        assert snap.otr_ratio == 6.0
        assert snap.alert_level in ("warning", "critical", "breach")

    def test_breach_threshold(self):
        mon = OTRMonitor(breach_threshold=5.0)
        for _ in range(20):
            mon.record_order("MSFT", "submit")
        mon.record_order("MSFT", "fill")
        snap = mon.compute_otr()
        assert snap.alert_level == "breach"

    def test_per_symbol_otr(self):
        mon = OTRMonitor()
        for _ in range(5):
            mon.record_order("AAPL", "submit")
        mon.record_order("AAPL", "fill")
        mon.record_order("MSFT", "submit")
        mon.record_order("MSFT", "fill")
        snap_aapl = mon.compute_otr(symbol="AAPL")
        snap_msft = mon.compute_otr(symbol="MSFT")
        assert snap_aapl.otr_ratio > snap_msft.otr_ratio

    def test_zero_fills(self):
        mon = OTRMonitor()
        mon.record_order("AAPL", "submit")
        snap = mon.compute_otr()
        assert snap.otr_ratio == 1.0  # 1 submit, 0 fills

    def test_cancel_tracking(self):
        mon = OTRMonitor()
        mon.record_order("AAPL", "submit")
        mon.record_order("AAPL", "cancel")
        snap = mon.compute_otr()
        assert snap.orders_cancelled == 1

    def test_reset(self):
        mon = OTRMonitor()
        mon.record_order("AAPL", "submit")
        mon.reset()
        snap = mon.compute_otr()
        assert snap.orders_submitted == 0

    def test_history(self):
        mon = OTRMonitor()
        mon.record_order("AAPL", "submit")
        mon.compute_otr()
        mon.compute_otr()
        assert len(mon.history) == 2

    def test_flagged_symbols(self):
        mon = OTRMonitor(warning_threshold=3.0)
        for _ in range(10):
            mon.record_order("BAD", "submit")
        mon.record_order("GOOD", "submit")
        mon.record_order("GOOD", "fill")
        snap = mon.compute_otr()
        assert "BAD" in snap.symbols_flagged


# ===========================================================================
# Task 28.3: Regulatory Reports
# ===========================================================================

@pytest.mark.phase12
class TestBestExecutionReport:
    def test_basic(self):
        fills = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "venue": ["NYSE", "NASDAQ", "NYSE"],
            "fill_price": [150.0, 300.0, 151.0],
            "arrival_price": [149.5, 300.5, 150.5],
            "fill_time_ms": [50, 80, 60],
            "volume": [100, 200, 150],
        })
        report = generate_best_execution_report(fills, "2026-01-01", "2026-03-31")
        assert isinstance(report, BestExecutionReport)
        assert report.total_fills == 3
        assert len(report.venues) == 2

    def test_empty_fills(self):
        report = generate_best_execution_report(pd.DataFrame(), "2026-01-01", "2026-03-31")
        assert report.total_fills == 0

    def test_to_dict(self):
        fills = pd.DataFrame({
            "symbol": ["AAPL"],
            "venue": ["NYSE"],
            "fill_price": [150.0],
            "arrival_price": [149.5],
            "fill_time_ms": [50],
            "volume": [100],
        })
        report = generate_best_execution_report(fills, "2026-01-01", "2026-03-31")
        d = report.to_dict()
        assert "avg_slippage_bps" in d


@pytest.mark.phase12
class TestTransactionCostReport:
    def test_basic(self):
        trades = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "cost_bps": [5.0, 8.0],
            "commission_bps": [2.0, 3.0],
            "spread_bps": [2.0, 3.0],
            "impact_bps": [1.0, 2.0],
        })
        report = generate_transaction_cost_report(trades, "2026-01-01", "2026-03-31")
        assert isinstance(report, TransactionCostReport)
        assert report.total_trades == 2
        assert report.total_cost_bps > 0

    def test_empty(self):
        report = generate_transaction_cost_report(pd.DataFrame(), "2026-01-01", "2026-03-31")
        assert report.total_trades == 0


@pytest.mark.phase12
class TestRiskReport:
    def test_basic(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.02, 252))
        report = generate_risk_report(returns)
        assert isinstance(report, RiskReport)
        assert report.var_95 < 0  # VaR should be negative
        assert report.max_drawdown < 0

    def test_with_positions(self):
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100))
        positions = {"AAPL": 0.3, "MSFT": 0.2, "GOOG": -0.1}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech"}
        report = generate_risk_report(returns, positions, sectors)
        assert report.gross_exposure == pytest.approx(0.6, abs=0.01)
        assert report.net_exposure == pytest.approx(0.4, abs=0.01)
        assert "Tech" in report.sector_exposures

    def test_insufficient_data(self):
        report = generate_risk_report(pd.Series([0.01]))
        assert report.summary == "Insufficient data."


@pytest.mark.phase12
class TestModelInventory:
    def test_basic(self):
        models = [
            {"name": "alpha_xgb", "type": "XGBoost", "version": "2.1",
             "training_date": "2026-03-01", "feature_count": 30,
             "metrics": {"sharpe": 1.5, "ic": 0.05}, "status": "active"},
            {"name": "alpha_ridge", "type": "Ridge", "version": "1.0",
             "training_date": "2025-12-01", "feature_count": 15,
             "metrics": {"sharpe": 0.8}, "status": "retired"},
        ]
        report = generate_model_inventory(models)
        assert isinstance(report, ModelInventoryReport)
        assert report.total_active == 1
        assert report.total_retired == 1
        assert len(report.models) == 2

    def test_empty(self):
        report = generate_model_inventory([])
        assert report.total_active == 0
