"""Chaos test: reconciliation drift detection (Sprint 4 / Plan C21).

The reconciler must flag every class of drift between ledger and broker:
cash drift, quantity drift, symbols only in ledger, and symbols only in
broker. Chaos here means: we inject synthetic drift into an otherwise
consistent snapshot and verify that the reconciler never silently
accepts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.reconciliation import (  # noqa: E402
    reconcile_ledger_vs_broker,
)


def _base_positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "qty": 10.0},
            {"symbol": "MSFT", "qty": 5.0},
            {"symbol": "NVDA", "qty": 3.0},
        ]
    )


def test_clean_snapshot_reconciles() -> None:
    ledger = _base_positions()
    broker = _base_positions()
    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=10_000.0,
        fail_fast=False,
    )
    assert report["ok"] is True
    assert report["cash_match"] is True
    assert report["position_diffs_df"].empty
    assert report["missing_in_ledger"] == []
    assert report["missing_in_broker"] == []


def test_qty_drift_on_single_symbol_is_detected() -> None:
    ledger = _base_positions()
    broker = _base_positions()
    # Broker reports 1 extra share of MSFT.
    broker.loc[broker["symbol"] == "MSFT", "qty"] = 6.0

    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=10_000.0,
        fail_fast=False,
    )
    assert report["ok"] is False
    diffs = report["position_diffs_df"]
    assert not diffs.empty
    assert "MSFT" in set(diffs["symbol"])
    msft = diffs[diffs["symbol"] == "MSFT"].iloc[0]
    assert msft["ledger_qty"] == pytest.approx(5.0)
    assert msft["broker_qty"] == pytest.approx(6.0)
    assert msft["diff_qty"] == pytest.approx(-1.0)


def test_cash_drift_is_detected() -> None:
    ledger = _base_positions()
    broker = _base_positions()
    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=9_997.25,  # $2.75 drift, well above cash_tol
        fail_fast=False,
    )
    assert report["ok"] is False
    assert report["cash_match"] is False
    assert report["cash_diff"] == pytest.approx(2.75)


def test_symbol_only_in_broker_is_flagged() -> None:
    ledger = _base_positions()
    broker = pd.concat(
        [
            _base_positions(),
            pd.DataFrame([{"symbol": "TSLA", "qty": 2.0}]),
        ],
        ignore_index=True,
    )
    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=10_000.0,
        fail_fast=False,
    )
    assert report["ok"] is False
    assert "TSLA" in report["missing_in_ledger"]
    assert report["missing_in_broker"] == []


def test_symbol_only_in_ledger_is_flagged() -> None:
    ledger = pd.concat(
        [
            _base_positions(),
            pd.DataFrame([{"symbol": "GOOG", "qty": 1.0}]),
        ],
        ignore_index=True,
    )
    broker = _base_positions()
    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=10_000.0,
        fail_fast=False,
    )
    assert report["ok"] is False
    assert "GOOG" in report["missing_in_broker"]
    assert report["missing_in_ledger"] == []


def test_multi_class_drift_combined() -> None:
    """Cash + qty + missing-both-sides at once.

    This is the realistic chaos scenario: a broker snapshot that drifted
    on multiple dimensions simultaneously. The reconciler must flag all
    four classes, not just the first one it encounters.
    """
    ledger = pd.concat(
        [
            _base_positions(),
            pd.DataFrame([{"symbol": "ONLY_LEDGER", "qty": 7.0}]),
        ],
        ignore_index=True,
    )
    broker = _base_positions()
    broker.loc[broker["symbol"] == "AAPL", "qty"] = 11.0  # qty drift
    broker = pd.concat(
        [broker, pd.DataFrame([{"symbol": "ONLY_BROKER", "qty": 4.0}])],
        ignore_index=True,
    )

    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=9_500.0,  # cash drift
        fail_fast=False,
    )
    assert report["ok"] is False
    assert report["cash_match"] is False
    assert report["cash_diff"] == pytest.approx(500.0)

    diffs = report["position_diffs_df"]
    assert "AAPL" in set(diffs["symbol"])

    assert "ONLY_LEDGER" in report["missing_in_broker"]
    assert "ONLY_BROKER" in report["missing_in_ledger"]


def test_drift_within_tolerance_is_ignored() -> None:
    """Qty drift below qty_tol and cash drift below cash_tol must not
    trip the reconciler — otherwise floating-point noise would alert
    on every reconciliation."""
    ledger = _base_positions()
    broker = _base_positions()
    # 1e-10 drift << qty_tol=1e-6
    broker.loc[broker["symbol"] == "NVDA", "qty"] = 3.0 + 1e-10

    report = reconcile_ledger_vs_broker(
        ledger_positions_df=ledger,
        ledger_cash=10_000.0,
        broker_positions_df=broker,
        broker_cash=10_000.0 + 1e-10,  # << cash_tol=1e-8
        fail_fast=False,
    )
    assert report["ok"] is True
    assert report["position_diffs_df"].empty


def test_fail_fast_raises_on_drift() -> None:
    ledger = _base_positions()
    broker = _base_positions()
    broker.loc[broker["symbol"] == "AAPL", "qty"] = 12.0

    with pytest.raises(ValueError):
        reconcile_ledger_vs_broker(
            ledger_positions_df=ledger,
            ledger_cash=10_000.0,
            broker_positions_df=broker,
            broker_cash=10_000.0,
            fail_fast=True,
        )
