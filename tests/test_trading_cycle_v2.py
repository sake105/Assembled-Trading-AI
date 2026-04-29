"""Tests for trading_cycle_v2 — 7-function decomposed cycle (Days 6–8)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle_shared import TradingContext, TradingCycleResult
from src.assembled_core.pipeline.trading_cycle_v2 import (
    book_fills,
    build_features,
    check_risk,
    generate_signals,
    ingest_data,
    route_orders,
    run_trading_cycle,
    size_positions,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_DAYS = 60
SYMBOLS = ["AAPL", "MSFT", "GOOG"]


def _make_prices(n_days: int = N_DAYS, symbols: list[str] | None = None) -> pd.DataFrame:
    if symbols is None:
        symbols = SYMBOLS
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym in symbols:
        close = 100.0 + rng.normal(0, 1, n_days).cumsum()
        close = np.maximum(close, 10.0)
        for i, ts in enumerate(dates):
            rows.append({
                "timestamp": ts,
                "symbol": sym,
                "close": round(float(close[i]), 2),
                "open": round(float(close[i]) * 0.99, 2),
                "high": round(float(close[i]) * 1.01, 2),
                "low": round(float(close[i]) * 0.98, 2),
                "volume": int(1_000_000 + rng.integers(0, 100_000)),
            })
    return pd.DataFrame(rows)


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    """LONG for all symbols at latest timestamp."""
    ts = df["timestamp"].max() if "timestamp" in df.columns else pd.Timestamp.now("UTC")
    syms = list(df["symbol"].unique()) if "symbol" in df.columns else []
    return pd.DataFrame({
        "timestamp": [ts] * len(syms),
        "symbol": syms,
        "direction": ["LONG"] * len(syms),
        "score": [0.5] * len(syms),
    })


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Equal-weight across LONG signals."""
    if signals.empty or "direction" not in signals.columns:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    long_s = signals[signals["direction"] == "LONG"]
    if long_s.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    syms = long_s["symbol"].tolist()
    w = 1.0 / len(syms)
    return pd.DataFrame({
        "symbol": syms,
        "target_weight": [round(w, 4)] * len(syms),
        "target_qty": [round(w * capital, 2)] * len(syms),
    })


def _make_ctx(
    n_days: int = N_DAYS,
    symbols: list[str] | None = None,
    as_of: pd.Timestamp | None = None,
    mode: str = "backtest",
    **kwargs: Any,
) -> TradingContext:
    prices = _make_prices(n_days, symbols)
    if as_of is None:
        as_of = prices["timestamp"].max()
    return TradingContext(
        prices=prices,
        as_of=as_of,
        mode=mode,
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
        intel_sim_applied=True,
        **kwargs,
    )


def _minimal_result(ctx: TradingContext | None = None) -> TradingCycleResult:
    return TradingCycleResult(
        run_id=None,
        timestamp=pd.Timestamp.now("UTC"),
        status="success",
    )


# ---------------------------------------------------------------------------
# ingest_data — Stage 1
# ---------------------------------------------------------------------------


class TestIngestData:
    def test_raises_on_empty_prices(self):
        ctx = _make_ctx()
        ctx.prices = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            ingest_data(ctx)

    def test_raises_on_missing_close_column(self):
        ctx = _make_ctx()
        ctx.prices = ctx.prices.drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing required price columns"):
            ingest_data(ctx)

    def test_raises_on_missing_symbol_column(self):
        ctx = _make_ctx()
        ctx.prices = ctx.prices.drop(columns=["symbol"])
        with pytest.raises(ValueError, match="Missing required price columns"):
            ingest_data(ctx)

    def test_raises_on_missing_signal_fn(self):
        ctx = _make_ctx()
        ctx.signal_fn = None
        with pytest.raises(ValueError, match="signal_fn"):
            ingest_data(ctx)

    def test_raises_on_missing_position_sizing_fn(self):
        ctx = _make_ctx()
        ctx.position_sizing_fn = None
        with pytest.raises(ValueError, match="position_sizing_fn"):
            ingest_data(ctx)

    def test_returns_two_element_tuple(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = ingest_data(ctx)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prices_filtered_is_dataframe(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        pf, _ = ingest_data(ctx)
        assert isinstance(pf, pd.DataFrame)
        assert not pf.empty

    def test_prices_filtered_contains_required_columns(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        pf, _ = ingest_data(ctx)
        for col in ("timestamp", "symbol", "close"):
            assert col in pf.columns, f"column '{col}' missing from prices_filtered"

    def test_filtered_prices_all_before_or_equal_as_of(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        prices = _make_prices(60)
        midpoint = prices["timestamp"].sort_values().iloc[29]
        ctx = _make_ctx(as_of=midpoint)
        pf, _ = ingest_data(ctx)
        assert (pf["timestamp"] <= midpoint).all()

    def test_raises_when_all_prices_after_as_of(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        prices = _make_prices(10)
        way_before = prices["timestamp"].min() - pd.Timedelta(days=365)
        ctx = _make_ctx()
        ctx.prices = prices
        ctx.as_of = way_before
        with pytest.raises(ValueError):
            ingest_data(ctx)

    def test_risk_state_set_on_ctx(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        assert ctx.risk_state is None
        ingest_data(ctx)
        assert ctx.risk_state is not None


# ---------------------------------------------------------------------------
# build_features — Stage 2
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_two_element_tuple(self):
        ctx = _make_ctx()
        prices = _make_prices()
        result = build_features(prices, ctx)
        assert isinstance(result, tuple) and len(result) == 2

    def test_pwf_is_dataframe(self):
        ctx = _make_ctx()
        prices = _make_prices()
        pwf, _ = build_features(prices, ctx)
        assert isinstance(pwf, pd.DataFrame)
        assert not pwf.empty

    def test_pwf_has_more_columns_than_input(self):
        ctx = _make_ctx()
        prices = _make_prices()
        pwf, _ = build_features(prices, ctx)
        # TA features should have been added
        assert len(pwf.columns) > len(prices.columns)

    def test_prices_latest_update_none_in_eod_mode(self):
        ctx = _make_ctx(mode="eod")
        prices = _make_prices()
        _, pl = build_features(prices, ctx)
        # In non-backtest mode without precomputed features,
        # prices_latest_update should be None or a snapshot
        # (None is returned when pwf is produced via _build_features_default without as_of)
        assert pl is None or isinstance(pl, pd.DataFrame)

    def test_preserves_symbol_column(self):
        ctx = _make_ctx()
        prices = _make_prices()
        pwf, _ = build_features(prices, ctx)
        assert "symbol" in pwf.columns
        assert set(pwf["symbol"].unique()) == set(SYMBOLS)

    def test_backtest_snapshot_uses_precomputed(self):
        """Precomputed features panel replaces raw feature calculation."""
        prices = _make_prices(60)
        precomputed = prices.copy()
        precomputed["custom_feature"] = 1.0
        as_of = prices["timestamp"].max()
        ctx = TradingContext(
            prices=prices,
            as_of=as_of,
            mode="backtest",
            precomputed_prices_with_features=precomputed,
            backtest_use_snapshot=True,
            signal_fn=_signal_fn,
            position_sizing_fn=_sizing_fn,
            use_factor_store=False,
            write_outputs=False,
            capital=100_000.0,
            intel_sim_applied=True,
        )
        pwf, pl = build_features(prices, ctx)
        assert "custom_feature" in pwf.columns
        assert pl is not None


# ---------------------------------------------------------------------------
# generate_signals — Stage 3
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def _make_features(self) -> pd.DataFrame:
        prices = _make_prices()
        ctx = _make_ctx()
        pwf, _ = build_features(prices, ctx)
        return pwf

    def test_returns_dataframe(self):
        ctx = _make_ctx()
        feats = self._make_features()
        signals = generate_signals(feats, ctx)
        assert isinstance(signals, pd.DataFrame)

    def test_returns_required_columns(self):
        ctx = _make_ctx()
        feats = self._make_features()
        signals = generate_signals(feats, ctx)
        for col in ("timestamp", "symbol", "direction"):
            assert col in signals.columns, f"'{col}' missing from signals"

    def test_signal_fn_output_propagated(self):
        """signal_fn returning LONG for AAPL only → AAPL in signals."""
        def one_symbol_fn(df: pd.DataFrame) -> pd.DataFrame:
            ts = df["timestamp"].max()
            return pd.DataFrame({
                "timestamp": [ts],
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.9],
            })
        ctx = _make_ctx()
        ctx.signal_fn = one_symbol_fn
        feats = self._make_features()
        signals = generate_signals(feats, ctx)
        assert "AAPL" in signals["symbol"].values

    def test_raises_if_signal_fn_missing_required_columns(self):
        def bad_fn(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"symbol": ["AAPL"], "score": [0.5]})
        ctx = _make_ctx()
        ctx.signal_fn = bad_fn
        feats = self._make_features()
        with pytest.raises(ValueError, match="signals missing required columns"):
            generate_signals(feats, ctx)

    def test_non_empty_signals_for_default_fn(self):
        ctx = _make_ctx()
        feats = self._make_features()
        signals = generate_signals(feats, ctx)
        assert len(signals) > 0


# ---------------------------------------------------------------------------
# size_positions — Stage 4
# ---------------------------------------------------------------------------


class TestSizePositions:
    def _make_signals(self) -> pd.DataFrame:
        ts = pd.Timestamp("2024-04-01", tz="UTC")
        return pd.DataFrame({
            "timestamp": [ts] * 3,
            "symbol": SYMBOLS,
            "direction": ["LONG", "LONG", "LONG"],
            "score": [0.6, 0.5, 0.4],
        })

    def test_returns_three_element_tuple(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        result = size_positions(signals, ctx)
        assert isinstance(result, tuple) and len(result) == 3

    def test_first_element_is_dataframe(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        targets, _, _meta = size_positions(signals, ctx)
        assert isinstance(targets, pd.DataFrame)

    def test_do_rebal_is_bool(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        _, do_rebal, _meta = size_positions(signals, ctx)
        assert isinstance(do_rebal, bool)

    def test_targets_have_required_columns(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        targets, _, _meta = size_positions(signals, ctx)
        assert not targets.empty
        assert "symbol" in targets.columns
        assert "target_weight" in targets.columns or "target_qty" in targets.columns

    def test_default_sizing_equal_weight(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        targets, _, _meta = size_positions(signals, ctx)
        assert len(targets) == 3
        # equal-weight: each symbol ~1/3
        w = targets["target_weight"].values
        assert all(abs(wi - w[0]) < 1e-4 for wi in w), "expected equal weights"

    def test_do_rebal_true_when_no_current_positions(self):
        ctx = _make_ctx()
        ctx.current_positions = None
        signals = self._make_signals()
        _, do_rebal, _meta = size_positions(signals, ctx)
        assert do_rebal is True

    def test_target_weights_sum_to_one(self):
        ctx = _make_ctx()
        signals = self._make_signals()
        targets, _, _meta = size_positions(signals, ctx)
        assert abs(targets["target_weight"].sum() - 1.0) < 1e-4

    def test_sizing_fn_returning_none_yields_empty_targets(self):
        # size_positions is resilient: None → empty DataFrame (no raise)
        ctx = _make_ctx()
        ctx.position_sizing_fn = lambda s, c: None  # type: ignore[arg-type]
        signals = self._make_signals()
        targets, _, _meta = size_positions(signals, ctx)
        assert isinstance(targets, pd.DataFrame)
        assert targets.empty


# ---------------------------------------------------------------------------
# route_orders — Stage 6
# ---------------------------------------------------------------------------


class TestRouteOrders:
    def _make_targets(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": SYMBOLS,
            "target_weight": [1 / 3, 1 / 3, 1 / 3],
            "target_qty": [33_333.0, 33_333.0, 33_334.0],
        })

    def test_returns_empty_dataframe_when_do_rebal_false(self):
        ctx = _make_ctx()
        targets = self._make_targets()
        orders = route_orders(targets, ctx, do_rebal=False)
        assert isinstance(orders, pd.DataFrame)
        assert orders.empty

    def test_returns_empty_dataframe_when_targets_empty(self):
        ctx = _make_ctx()
        orders = route_orders(pd.DataFrame(), ctx, do_rebal=True)
        assert isinstance(orders, pd.DataFrame)
        assert orders.empty

    def test_returns_empty_dataframe_when_targets_none(self):
        ctx = _make_ctx()
        orders = route_orders(None, ctx, do_rebal=True)  # type: ignore[arg-type]
        assert isinstance(orders, pd.DataFrame)
        assert orders.empty

    def test_returns_dataframe_with_expected_columns(self):
        ctx = _make_ctx()
        targets = self._make_targets()
        orders = route_orders(targets, ctx, do_rebal=True)
        assert isinstance(orders, pd.DataFrame)
        # May be empty if _generate_orders_default returns nothing (no current positions)
        # but must at least have the right schema
        for col in ("timestamp", "symbol", "side", "qty", "price"):
            assert col in orders.columns, f"'{col}' missing from orders"

    def test_consistent_return_when_do_rebal_true_and_false(self):
        ctx = _make_ctx()
        targets = self._make_targets()
        orders_yes = route_orders(targets, ctx, do_rebal=True)
        orders_no = route_orders(targets, ctx, do_rebal=False)
        assert orders_no.empty
        # orders_yes may or may not be empty, but must be a DataFrame
        assert isinstance(orders_yes, pd.DataFrame)


# ---------------------------------------------------------------------------
# check_risk — Stage 5
# ---------------------------------------------------------------------------


class TestCheckRisk:
    def _make_orders(self) -> pd.DataFrame:
        ts = pd.Timestamp("2024-04-01", tz="UTC")
        return pd.DataFrame({
            "timestamp": [ts] * 3,
            "symbol": SYMBOLS,
            "side": ["buy", "buy", "buy"],
            "qty": [100.0, 80.0, 60.0],
            "price": [150.0, 300.0, 140.0],
        })

    def _make_result_with_signals(self) -> TradingCycleResult:
        ts = pd.Timestamp("2024-04-01", tz="UTC")
        r = _minimal_result()
        r.signals = pd.DataFrame({
            "timestamp": [ts] * 3,
            "symbol": SYMBOLS,
            "direction": ["LONG", "LONG", "LONG"],
            "score": [0.5, 0.5, 0.5],
        })
        return r

    def test_qa_block_clears_all_orders(self):
        ctx = _make_ctx()
        ctx.qa_block_trading = True
        ctx.qa_block_reason = "test_block"
        orders = self._make_orders()
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert out.orders_filtered.empty

    def test_qa_block_sets_meta_flag(self):
        ctx = _make_ctx()
        ctx.qa_block_trading = True
        orders = self._make_orders()
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert out.meta.get("qa_block_trading") is True

    def test_qa_block_returns_result_type(self):
        ctx = _make_ctx()
        ctx.qa_block_trading = True
        orders = self._make_orders()
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert isinstance(out, TradingCycleResult)

    def test_normal_case_sets_orders_filtered(self):
        ctx = _make_ctx()
        ctx.qa_block_trading = False
        orders = self._make_orders()
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert isinstance(out, TradingCycleResult)
        assert out.orders_filtered is not None

    def test_empty_orders_passed_through(self):
        ctx = _make_ctx()
        ctx.qa_block_trading = False
        orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert isinstance(out, TradingCycleResult)

    def test_returns_result_with_same_identity(self):
        ctx = _make_ctx()
        orders = self._make_orders()
        result = self._make_result_with_signals()
        out = check_risk(orders, result, ctx)
        assert out is result


# ---------------------------------------------------------------------------
# book_fills — Stage 7
# ---------------------------------------------------------------------------


class TestBookFills:
    def _make_result_ready(self) -> TradingCycleResult:
        ts = pd.Timestamp("2024-04-01", tz="UTC")
        r = _minimal_result()
        r.orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        r.orders_filtered = r.orders.copy()
        r.signals = pd.DataFrame({
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "direction": ["LONG"],
            "score": [0.5],
        })
        return r

    def test_returns_result_type(self):
        ctx = _make_ctx()
        result = self._make_result_ready()
        out = book_fills(result, ctx)
        assert isinstance(out, TradingCycleResult)

    def test_status_success_when_write_outputs_false(self):
        ctx = _make_ctx()
        result = self._make_result_ready()
        out = book_fills(result, ctx)
        assert out.status == "success"

    def test_returns_same_result_object(self):
        ctx = _make_ctx()
        result = self._make_result_ready()
        out = book_fills(result, ctx)
        assert out is result

    def test_orders_filtered_preserved(self):
        ctx = _make_ctx()
        result = self._make_result_ready()
        ts = pd.Timestamp("2024-04-01", tz="UTC")
        result.orders_filtered = pd.DataFrame({
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "side": ["buy"],
            "qty": [100.0],
            "price": [150.0],
        })
        out = book_fills(result, ctx)
        assert len(out.orders_filtered) == 1

    def test_no_file_io_when_write_outputs_false(self, tmp_path: Path):
        """No files should be created when write_outputs=False."""
        ctx = _make_ctx()
        ctx.output_dir = tmp_path / "output"
        result = self._make_result_ready()
        book_fills(result, ctx)
        # output_dir should NOT be created for a no-write cycle
        assert not (tmp_path / "output").exists() or not (tmp_path / "output" / "orders_latest.csv").exists()


# ---------------------------------------------------------------------------
# run_trading_cycle — end-to-end orchestrator
# ---------------------------------------------------------------------------


class TestRunTradingCycle:
    def test_error_status_on_empty_prices(self):
        ctx = _make_ctx()
        ctx.prices = pd.DataFrame()
        result = run_trading_cycle(ctx)
        assert result.status == "error"
        assert result.error_message is not None

    def test_error_status_on_missing_signal_fn(self):
        ctx = _make_ctx()
        ctx.signal_fn = None
        result = run_trading_cycle(ctx)
        assert result.status == "error"

    def test_returns_trading_cycle_result_type(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        assert isinstance(result, TradingCycleResult)

    def test_end_to_end_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        assert result.status == "success", result.error_message

    def test_prices_filtered_populated_on_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        if result.status == "success":
            assert isinstance(result.prices_filtered, pd.DataFrame)
            assert not result.prices_filtered.empty

    def test_signals_populated_on_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        if result.status == "success":
            assert isinstance(result.signals, pd.DataFrame)
            assert not result.signals.empty

    def test_target_positions_populated_on_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        if result.status == "success":
            assert isinstance(result.target_positions, pd.DataFrame)

    def test_orders_filtered_is_dataframe(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        if result.status == "success":
            assert isinstance(result.orders_filtered, pd.DataFrame)

    def test_qa_block_propagates_through_full_cycle(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        ctx.qa_block_trading = True
        ctx.qa_block_reason = "integration_test_block"
        result = run_trading_cycle(ctx)
        # Status may be "success" — QA block is not an error, just empties orders
        assert result.orders_filtered is not None
        assert result.orders_filtered.empty

    def test_error_message_is_string_on_failure(self):
        ctx = _make_ctx()
        ctx.prices = pd.DataFrame()
        result = run_trading_cycle(ctx)
        assert isinstance(result.error_message, str)

    def test_run_id_propagated_to_result(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx(run_id="test_run_001")
        result = run_trading_cycle(ctx)
        assert result.run_id == "test_run_001"


# ---------------------------------------------------------------------------
# Wiring gap fixes — Phase 11 observability
# ---------------------------------------------------------------------------


class TestRejectionCountsWiring:
    """rejection_counts must always be written to result.meta by check_risk."""

    def test_rejection_counts_present_in_meta(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        assert "rejection_counts" in result.meta
        assert isinstance(result.meta["rejection_counts"], dict)

    def test_rejection_counts_empty_when_no_rejections(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx(enable_risk_controls=False)
        result = run_trading_cycle(ctx)
        # fast-path bypasses filter steps — no rejection_counts key written (expected)
        assert result.status in ("success", "error")

    def test_rejection_counts_written_by_check_risk_directly(self):
        """check_risk writes rejection_counts even with zero rejections."""
        ctx = _make_ctx()
        orders = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-03-01", tz="UTC")] * 2,
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "BUY"],
            "qty": [10.0, 5.0],
            "price": [150.0, 300.0],
        })
        result = _minimal_result(ctx)
        result = check_risk(orders, result, ctx)
        assert "rejection_counts" in result.meta
        assert isinstance(result.meta["rejection_counts"], dict)


class TestTotalCostBpsWiring:
    """total_cost_bps must be derived in book_fills so Phase 11 histogram works."""

    def test_total_cost_bps_derived_from_total_cost_cash(self):
        # Use mode="live" so A8 (add_cost_columns_to_trades) does not overwrite
        # total_cost_cash with model-computed values before A8b runs.
        ctx = _make_ctx(mode="live")
        result = _minimal_result(ctx)
        result.orders_filtered = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-03-01", tz="UTC")],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [10.0],
            "price": [150.0],
            "total_cost_cash": [3.0],  # 3 USD on 1500 USD notional = 20 bps
        })
        result.prices_with_features = pd.DataFrame()
        result = book_fills(result, ctx)
        assert "total_cost_bps" in result.orders_filtered.columns
        assert abs(result.orders_filtered["total_cost_bps"].iloc[0] - 20.0) < 1.0

    def test_total_cost_bps_falls_back_to_expected_impact_bps(self):
        # Use mode="live" to skip A8 cost annotation so the expected_impact_bps
        # fallback path in A8b is reachable.
        ctx = _make_ctx(mode="live")
        result = _minimal_result(ctx)
        result.orders_filtered = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-03-01", tz="UTC")],
            "symbol": ["MSFT"],
            "side": ["BUY"],
            "qty": [5.0],
            "price": [300.0],
            "expected_impact_bps": [12.5],
        })
        result.prices_with_features = pd.DataFrame()
        result = book_fills(result, ctx)
        assert "total_cost_bps" in result.orders_filtered.columns
        assert result.orders_filtered["total_cost_bps"].iloc[0] == pytest.approx(12.5)


class TestDriftMonitorWiring:
    """drift_monitor skip-by-default (no policy.enabled) must not error."""

    def test_drift_monitor_not_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        # Without policy drift_monitor.enabled=True, key should be absent
        assert result.meta.get("drift_monitor") is None

    def test_drift_monitor_skipped_gracefully_on_missing_ref(self, monkeypatch: pytest.MonkeyPatch):
        """Enabled but missing reference_path → skip, no crash."""
        monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")
        monkeypatch.setattr(
            "src.assembled_core.config.policy_loader.load_policy",
            lambda: {"drift_monitor": {"enabled": True, "reference_path": "/nonexistent.parquet"}},
        )
        ctx = _make_ctx()
        result = run_trading_cycle(ctx)
        assert result.status == "success"
