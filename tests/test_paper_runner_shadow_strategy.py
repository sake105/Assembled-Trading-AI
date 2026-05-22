"""§9.6 (b) Phase 1 shadow-mode regression guards for paper_runner.

Verifies:
  - `trend_baseline` is registered as a strategy in _prd_make_strategy_fns
  - signal_fn produces direction/score schema matching the canonical contract
  - sizing_fn produces target_weight/target_qty schema for LONGs only
  - _prd_run_shadow_strategy persists signals + targets JSON, no broker side-effects
  - default OFF: when shadow_strategy.enabled is missing/False, nothing is written
  - graceful failure: exception in shadow path logs WARNING + does not abort cycle
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.ops.paper_runner import (
    _prd_make_strategy_fns,
    _prd_run_shadow_strategy,
)


def _make_prices_multi_year() -> pd.DataFrame:
    """Synthetic 200-day OHLCV for 3 symbols, enough for MA(60) to be defined."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D", tz="UTC")
    rows = []
    for sym in ["AAPL", "MSFT", "NVDA"]:
        base = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 50.0)
        for i, d in enumerate(dates):
            close = base * (1 + 0.001 * i)  # mild uptrend so LONG fires
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_trend_baseline_registered_in_strategy_dispatch():
    sig_fn, size_fn = _prd_make_strategy_fns(
        "trend_baseline",
        {"ma_fast": 20, "ma_slow": 60},
        ledger_state=None,
    )
    prices = _make_prices_multi_year()
    sig = sig_fn(prices)
    assert isinstance(sig, pd.DataFrame)
    # Canonical signal schema
    for col in ("timestamp", "symbol", "direction", "score"):
        assert col in sig.columns, f"missing column: {col}"
    # Uptrend fixture should produce some LONGs
    assert (sig["direction"] == "LONG").any()


def test_trend_baseline_sizing_equal_weight_on_longs():
    sig_fn, size_fn = _prd_make_strategy_fns(
        "trend_baseline",
        {"ma_fast": 20, "ma_slow": 60},
        ledger_state=None,
    )
    prices = _make_prices_multi_year()
    sig = sig_fn(prices)
    # Take latest bar per symbol so sizing sees one row per sym
    latest = sig.sort_values("timestamp").groupby("symbol").tail(1)
    tgt = size_fn(latest, 100_000.0)
    assert set(tgt.columns) == {"symbol", "target_weight", "target_qty"}
    if not tgt.empty:
        # Equal-weight: all weights identical
        weights = tgt["target_weight"].unique()
        assert len(weights) == 1
        # Sum to target_invested_pct (default 1.0)
        assert abs(tgt["target_weight"].sum() - 1.0) < 1e-9


def test_trend_baseline_sizing_respects_max_positions():
    sig_fn, size_fn = _prd_make_strategy_fns(
        "trend_baseline",
        {"ma_fast": 20, "ma_slow": 60, "max_positions": 2},
        ledger_state=None,
    )
    prices = _make_prices_multi_year()
    sig = sig_fn(prices)
    latest = sig.sort_values("timestamp").groupby("symbol").tail(1)
    tgt = size_fn(latest, 100_000.0)
    assert len(tgt) <= 2


def test_shadow_strategy_disabled_by_default_writes_nothing(tmp_path):
    paper_cfg: dict = {}  # no shadow_strategy key
    _prd_run_shadow_strategy(
        prices=_make_prices_multi_year(),
        paper_cfg=paper_cfg,
        output_dir=tmp_path,
        as_of_ts=pd.Timestamp.now("UTC"),
        primary_signals=None,
    )
    assert not (tmp_path / "shadow_signals.json").exists()
    assert not (tmp_path / "shadow_targets.json").exists()


def test_shadow_strategy_enabled_persists_signals_and_targets(tmp_path):
    paper_cfg = {
        "start_capital": 100_000.0,
        "shadow_strategy": {
            "enabled": True,
            "name": "trend_baseline",
            "ma_fast": 20,
            "ma_slow": 60,
        },
    }
    _prd_run_shadow_strategy(
        prices=_make_prices_multi_year(),
        paper_cfg=paper_cfg,
        output_dir=tmp_path,
        as_of_ts=pd.Timestamp("2024-07-19", tz="UTC"),
        primary_signals=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-07-19", tz="UTC")] * 2,
                "symbol": ["AAPL", "MSFT"],
                "direction": ["LONG", "LONG"],
                "score": [0.5, 0.6],
            }
        ),
    )
    sig_path = tmp_path / "shadow_signals.json"
    tgt_path = tmp_path / "shadow_targets.json"
    assert sig_path.exists()
    assert tgt_path.exists()

    sig_payload = json.loads(sig_path.read_text(encoding="utf-8"))
    tgt_payload = json.loads(tgt_path.read_text(encoding="utf-8"))
    assert sig_payload["strategy"] == "trend_baseline"
    assert sig_payload["n_long"] >= 0
    assert "config" in sig_payload
    assert tgt_payload["strategy"] == "trend_baseline"


def test_shadow_strategy_unknown_name_logs_warning_and_skips(tmp_path, caplog):
    """F-S1-M1 §9.6 (b) Phase-2 pre-cond (ii) closure: unknown shadow strategy
    name (or feature-dependent strategy like multifactor_*) is loud-skipped via
    whitelist + WARNING log. Pre-fix this silently wrote empty JSON payloads
    indistinguishable from genuine 'no signals today'. Now it produces neither
    artifact and surfaces an explicit operator-actionable WARN message.
    """
    paper_cfg = {
        "shadow_strategy": {"enabled": True, "name": "does_not_exist"},
    }
    with caplog.at_level("WARNING", logger="src.assembled_core.ops.paper_runner"):
        # Must not raise
        _prd_run_shadow_strategy(
            prices=_make_prices_multi_year(),
            paper_cfg=paper_cfg,
            output_dir=tmp_path,
            as_of_ts=pd.Timestamp.now("UTC"),
            primary_signals=None,
        )

    # No artifacts written for unsafe strategy
    assert not (tmp_path / "shadow_signals.json").exists()
    assert not (tmp_path / "shadow_targets.json").exists()

    # Loud WARN with explicit whitelist reference
    warn_messages = [
        rec.message for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert any("does_not_exist" in m and "whitelist" in m for m in warn_messages), (
        f"expected loud WARN about whitelist, got: {warn_messages}"
    )


def test_shadow_strategy_feature_dependent_name_blocked_by_whitelist(tmp_path, caplog):
    """multifactor_v2 as shadow would silently degrade to zero-factor signals
    on raw prices (no feature pipeline upstream). Whitelist blocks it.
    """
    paper_cfg = {
        "shadow_strategy": {"enabled": True, "name": "multifactor_v2"},
    }
    with caplog.at_level("WARNING", logger="src.assembled_core.ops.paper_runner"):
        _prd_run_shadow_strategy(
            prices=_make_prices_multi_year(),
            paper_cfg=paper_cfg,
            output_dir=tmp_path,
            as_of_ts=pd.Timestamp.now("UTC"),
            primary_signals=None,
        )
    assert not (tmp_path / "shadow_signals.json").exists()
    assert any(
        "multifactor_v2" in rec.message and "whitelist" in rec.message
        for rec in caplog.records
        if rec.levelname == "WARNING"
    )


def test_shadow_strategy_enabled_but_no_name_skips(tmp_path):
    paper_cfg = {"shadow_strategy": {"enabled": True}}  # missing name
    _prd_run_shadow_strategy(
        prices=_make_prices_multi_year(),
        paper_cfg=paper_cfg,
        output_dir=tmp_path,
        as_of_ts=pd.Timestamp.now("UTC"),
        primary_signals=None,
    )
    assert not (tmp_path / "shadow_signals.json").exists()
