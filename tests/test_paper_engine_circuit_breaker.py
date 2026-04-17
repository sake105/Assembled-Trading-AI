"""Phase 3 regression tests for the market-wide circuit breaker gate.

Covers:

* ``enable_circuit_breaker=False`` (default) is a noop
* SPY return below L1 (-7%) halts trading → orders returned empty
* SPY return below L2 (-13%) also halts (same effect, different reason)
* benchmark row missing → check is skipped (no false positive)
* benchmark-return extraction from open/close works as fallback
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    enable_circuit_breaker: bool = True,
    market_benchmark: str = "SPY",
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_circuit_breaker=enable_circuit_breaker,
        market_benchmark=market_benchmark,
        run_id="cb_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_circuit_breaker_disabled_noop(tmp_path: Path) -> None:
    """With the breaker disabled, risk controls pass through unchanged."""
    eng = _make_engine(tmp_path, enable_circuit_breaker=False)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    result = eng._apply_risk_controls(orders, market_return_today=-0.10)
    assert len(result) == 1


def test_circuit_breaker_l1_halt_rejects_all_orders(tmp_path: Path) -> None:
    """SPY -8% → L1 halt, all orders rejected."""
    eng = _make_engine(tmp_path, enable_circuit_breaker=True)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 5.0, "price": 50.0},
        ]
    )
    result = eng._apply_risk_controls(orders, market_return_today=-0.08)
    assert result.empty
    assert "CIRCUIT_BREAKER_L1" in getattr(eng, "_last_circuit_breaker_reason", "")


def test_circuit_breaker_l2_halt(tmp_path: Path) -> None:
    """SPY -14% → L2 halt."""
    eng = _make_engine(tmp_path, enable_circuit_breaker=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    result = eng._apply_risk_controls(orders, market_return_today=-0.14)
    assert result.empty
    assert "CIRCUIT_BREAKER_L2" in getattr(eng, "_last_circuit_breaker_reason", "")


def test_circuit_breaker_safe_return_passes(tmp_path: Path) -> None:
    """SPY -2% → no halt, orders pass through."""
    eng = _make_engine(tmp_path, enable_circuit_breaker=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    result = eng._apply_risk_controls(orders, market_return_today=-0.02)
    assert len(result) == 1


def test_circuit_breaker_none_market_return_is_skipped(tmp_path: Path) -> None:
    """If benchmark return is unknown, the check is skipped (no false halt)."""
    eng = _make_engine(tmp_path, enable_circuit_breaker=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    result = eng._apply_risk_controls(orders, market_return_today=None)
    assert len(result) == 1


def test_benchmark_return_extracted_from_open_close(tmp_path: Path) -> None:
    """``_extract_benchmark_return`` falls back to close/open-1 when no explicit return."""
    eng = _make_engine(tmp_path)
    prices = pd.DataFrame(
        [
            {"symbol": "SPY", "open": 100.0, "close": 92.0},
            {"symbol": "AAA", "open": 50.0, "close": 50.5},
        ]
    )
    ret = eng._extract_benchmark_return(prices)
    assert ret == pytest.approx(-0.08)


def test_benchmark_return_from_explicit_return_column(tmp_path: Path) -> None:
    """Explicit `return` column takes precedence over open/close fallback."""
    eng = _make_engine(tmp_path)
    prices = pd.DataFrame(
        [
            {"symbol": "SPY", "open": 100.0, "close": 100.0, "return": -0.09},
        ]
    )
    ret = eng._extract_benchmark_return(prices)
    assert ret == pytest.approx(-0.09)


def test_benchmark_missing_returns_none(tmp_path: Path) -> None:
    """Benchmark symbol absent → None (skip check, no false positive)."""
    eng = _make_engine(tmp_path)
    prices = pd.DataFrame(
        [{"symbol": "AAA", "open": 100.0, "close": 100.0}]
    )
    ret = eng._extract_benchmark_return(prices)
    assert ret is None
