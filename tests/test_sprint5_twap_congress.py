"""Tests for Sprint 5: TWAP wiring + Congress feature integration.

Covers:
  - TWAP scheduler produces valid slices
  - TWAP slice quantities sum to total
  - Congress features add expected columns
  - Congress features with empty events
  - FeatureConfig includes congress fields
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# TWAP Scheduler tests
# ---------------------------------------------------------------------------


def test_twap_scheduler_produces_correct_count() -> None:
    from src.assembled_core.execution.algo_execution import TWAPScheduler

    scheduler = TWAPScheduler(n_slices=5, randomize=False)
    now = datetime.utcnow()
    slices = scheduler.schedule(
        symbol="AAPL",
        total_qty=500.0,
        side="BUY",
        start_time=now,
        end_time=now + timedelta(hours=1),
    )
    assert len(slices) == 5
    assert all(s.symbol == "AAPL" for s in slices)
    assert all(s.side == "BUY" for s in slices)
    assert all(s.algo == "TWAP" for s in slices)


def test_twap_scheduler_quantity_sums_to_total() -> None:
    from src.assembled_core.execution.algo_execution import TWAPScheduler

    scheduler = TWAPScheduler(n_slices=10, randomize=True)
    now = datetime.utcnow()
    slices = scheduler.schedule(
        symbol="MSFT",
        total_qty=1000.0,
        side="SELL",
        start_time=now,
        end_time=now + timedelta(hours=2),
        random_seed=42,
    )
    total = sum(s.quantity for s in slices)
    assert total == pytest.approx(1000.0, abs=1e-6)


def test_twap_scheduler_to_dict() -> None:
    from src.assembled_core.execution.algo_execution import TWAPScheduler

    scheduler = TWAPScheduler(n_slices=3, randomize=False)
    now = datetime.utcnow()
    slices = scheduler.schedule(
        symbol="GOOGL",
        total_qty=300.0,
        side="BUY",
        start_time=now,
        end_time=now + timedelta(minutes=30),
    )
    for s in slices:
        d = s.to_dict()
        assert "symbol" in d
        assert "quantity" in d
        assert "scheduled_time" in d
        assert d["algo"] == "TWAP"


def test_twap_scheduler_rejects_zero_qty() -> None:
    from src.assembled_core.execution.algo_execution import TWAPScheduler

    scheduler = TWAPScheduler(n_slices=5)
    now = datetime.utcnow()
    with pytest.raises(ValueError, match="positive"):
        scheduler.schedule("AAPL", 0.0, "BUY", now, now + timedelta(hours=1))


# ---------------------------------------------------------------------------
# VWAP Scheduler tests
# ---------------------------------------------------------------------------


def test_vwap_scheduler_fallback_to_equal() -> None:
    from src.assembled_core.execution.algo_execution import VWAPScheduler

    scheduler = VWAPScheduler(n_slices=4)
    now = datetime.utcnow()
    slices = scheduler.schedule(
        symbol="AAPL",
        total_qty=400.0,
        side="BUY",
        start_time=now,
        end_time=now + timedelta(hours=1),
        volume_profile=None,
    )
    assert len(slices) == 4
    total = sum(s.quantity for s in slices)
    assert total == pytest.approx(400.0, abs=1e-6)
    assert all(s.algo == "VWAP" for s in slices)


# ---------------------------------------------------------------------------
# Implementation Shortfall Model
# ---------------------------------------------------------------------------


def test_is_model_basic_cost() -> None:
    from src.assembled_core.execution.algo_execution import ImplementationShortfallModel

    model = ImplementationShortfallModel(kyle_lambda=0.1)
    cost = model.estimate_cost(
        quantity=1000, adv=100_000, daily_vol=0.02, price=150.0,
    )
    assert cost["total_cost_bps"] > 0
    assert cost["market_impact_bps"] > 0
    assert cost["total_cost_notional"] > 0


def test_is_model_zero_adv() -> None:
    from src.assembled_core.execution.algo_execution import ImplementationShortfallModel

    model = ImplementationShortfallModel()
    cost = model.estimate_cost(quantity=100, adv=0, daily_vol=0.02, price=50.0)
    assert cost["market_impact_bps"] == 0.0


# ---------------------------------------------------------------------------
# Congress Features integration
# ---------------------------------------------------------------------------


def test_congress_features_add_columns() -> None:
    from src.assembled_core.data.congress_trades_ingest import load_congress_sample
    from src.assembled_core.features.congress_features import add_congress_features

    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0 + i for i in range(10)],
    })
    events = load_congress_sample()  # dummy data

    result = add_congress_features(prices, events)
    assert "congress_trade_count_60d" in result.columns
    assert "congress_total_amount_60d" in result.columns
    assert "congress_trade_count_90d" in result.columns
    assert "congress_total_amount_90d" in result.columns
    assert len(result) == len(prices)


def test_congress_features_empty_events() -> None:
    from src.assembled_core.features.congress_features import add_congress_features

    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC"),
        "symbol": ["MSFT"] * 5,
        "close": [400.0] * 5,
    })
    empty_events = pd.DataFrame(columns=["timestamp", "symbol", "amount"])

    result = add_congress_features(prices, empty_events)
    assert (result["congress_trade_count_60d"] == 0).all()


def test_congress_net_buy_score() -> None:
    from src.assembled_core.features.congress_features import compute_congress_net_buy_score

    trades = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "amount": [100_000, 50_000, 200_000],
        "type": ["buy", "sell", "purchase"],
        "disclosure_date": pd.date_range("2024-01-01", periods=3, freq="7D"),
        "member_id": ["m1", "m2", "m3"],
    })

    scores = compute_congress_net_buy_score(trades)
    assert "AAPL" in scores
    assert "MSFT" in scores
    assert scores["AAPL"] == pytest.approx(50_000.0)  # 100k buy - 50k sell
    assert scores["MSFT"] > 0  # pure buy


# ---------------------------------------------------------------------------
# FeatureConfig includes congress
# ---------------------------------------------------------------------------


def test_feature_config_has_congress_field() -> None:
    from src.assembled_core.config.models import FeatureConfig

    cfg = FeatureConfig(include_congress=True, congress_data_path="data/congress.csv")
    assert cfg.include_congress is True
    assert cfg.congress_data_path == "data/congress.csv"

    cfg_default = FeatureConfig()
    assert cfg_default.include_congress is False
    assert cfg_default.congress_data_path is None
