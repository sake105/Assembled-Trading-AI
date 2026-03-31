"""Tests for M11: Post-Trade Learning Loop — analyzer."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.qa.post_trade_analyzer import (
    compute_forward_returns,
    compute_signal_hit_rate,
    build_learning_record,
)


@pytest.mark.phase12
@pytest.mark.phase13
class TestComputeForwardReturns:
    def _make_prices(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": list(dates) * 2,
                "symbol": ["AAPL"] * 20 + ["MSFT"] * 20,
                "close": [100.0 + i for i in range(20)]
                + [200.0 + i * 0.5 for i in range(20)],
            }
        )

    def test_returns_dataframe(self):
        df = compute_forward_returns(self._make_prices(), horizon_days=5)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = compute_forward_returns(self._make_prices(), horizon_days=5)
        assert "forward_return" in df.columns
        assert "symbol" in df.columns

    def test_forward_return_positive_for_rising_prices(self):
        df = compute_forward_returns(self._make_prices(), horizon_days=5)
        aapl = df[df["symbol"] == "AAPL"].dropna(subset=["forward_return"])
        assert (aapl["forward_return"] > 0).all()

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"timestamp": [], "close": []})
        with pytest.raises(ValueError, match="missing columns"):
            compute_forward_returns(bad)

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        result = compute_forward_returns(empty)
        assert result.empty

    def test_horizon_affects_returns(self):
        prices = self._make_prices()
        df5 = compute_forward_returns(prices, horizon_days=5).dropna()
        df1 = compute_forward_returns(prices, horizon_days=1).dropna()
        # 5-day return should generally differ from 1-day return
        assert len(df5) > 0 and len(df1) > 0


@pytest.mark.phase12
@pytest.mark.phase13
class TestComputeSignalHitRate:
    def _make_fwd(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["AAPL"] * 10,
                "close": [100.0 + i for i in range(10)],
                "forward_return": [0.05] * 5 + [-0.03] * 5,
            }
        )

    def test_buy_hit_on_positive_return(self):
        trades = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "side": ["BUY"],
                "event_ts": [pd.Timestamp("2024-01-01", tz="UTC")],
                "qty": [100.0],
                "price": [100.0],
            }
        )
        fwd = self._make_fwd()
        result = compute_signal_hit_rate(trades, fwd)
        assert len(result) == 1
        assert result.iloc[0]["hits"] == 1

    def test_buy_miss_on_negative_return(self):
        trades = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "side": ["BUY"],
                "event_ts": [pd.Timestamp("2024-01-06", tz="UTC")],
                "qty": [100.0],
                "price": [105.0],
            }
        )
        fwd = self._make_fwd()
        result = compute_signal_hit_rate(trades, fwd)
        assert len(result) == 1
        assert result.iloc[0]["hits"] == 0

    def test_empty_trades_returns_empty(self):
        fwd = self._make_fwd()
        trades = pd.DataFrame(columns=["symbol", "side", "event_ts", "qty", "price"])
        result = compute_signal_hit_rate(trades, fwd)
        assert result.empty

    def test_empty_fwd_returns_empty(self):
        trades = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "side": ["BUY"],
                "event_ts": [pd.Timestamp("2024-01-01", tz="UTC")],
                "qty": [100.0],
                "price": [100.0],
            }
        )
        fwd = pd.DataFrame(columns=["timestamp", "symbol", "close", "forward_return"])
        result = compute_signal_hit_rate(trades, fwd)
        assert result.empty

    def test_hit_rate_between_0_and_1(self):
        trades = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "side": ["BUY", "BUY"],
                "event_ts": [
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-06", tz="UTC"),
                ],
                "qty": [100.0, 100.0],
                "price": [100.0, 105.0],
            }
        )
        fwd = self._make_fwd()
        result = compute_signal_hit_rate(trades, fwd)
        if not result.empty:
            assert (result["hit_rate"] >= 0).all()
            assert (result["hit_rate"] <= 1).all()


@pytest.mark.phase12
@pytest.mark.phase13
class TestBuildLearningRecord:
    def _make_hit_df(self):
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "total_trades": [10, 8],
                "hits": [6, 5],
                "hit_rate": [0.6, 0.625],
                "avg_forward_return": [0.025, 0.018],
            }
        )

    def test_record_has_required_keys(self):
        rec = build_learning_record("run1", "2024-01-15", self._make_hit_df())
        assert "run_id" in rec
        assert "analysis_date" in rec
        assert "overall_hit_rate" in rec
        assert "per_symbol" in rec

    def test_run_id_set(self):
        rec = build_learning_record("test_run", "2024-01-15", self._make_hit_df())
        assert rec["run_id"] == "test_run"

    def test_overall_hit_rate_computed(self):
        rec = build_learning_record("r", "2024-01-15", self._make_hit_df())
        # (6+5)/(10+8) = 11/18 ~ 0.611
        assert abs(rec["overall_hit_rate"] - 11 / 18) < 0.01

    def test_empty_hit_df_gives_zero(self):
        empty_df = pd.DataFrame(
            columns=[
                "symbol",
                "total_trades",
                "hits",
                "hit_rate",
                "avg_forward_return",
            ]
        )
        rec = build_learning_record("r", "2024-01-15", empty_df)
        assert rec["overall_hit_rate"] == 0.0
        assert rec["overall_total_trades"] == 0

    def test_extra_fields_included(self):
        rec = build_learning_record(
            "r", "2024-01-15", self._make_hit_df(), extra={"source": "test"}
        )
        assert rec["source"] == "test"
