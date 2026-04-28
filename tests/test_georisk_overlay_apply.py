from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle
from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.risk.georisk_overlay import apply_exposure_multiplier_to_targets


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_apply_exposure_multiplier_to_targets_basic() -> None:
    """Basic scaling of weights and quantities without cash row."""
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "target_weight": [0.6, 0.4],
            "target_qty": [60.0, 40.0],
        }
    )
    multiplier = 0.5

    scaled = apply_exposure_multiplier_to_targets(df, multiplier)

    assert pytest.approx(scaled["target_weight"].sum()) == 0.5
    assert all(w > 0 for w in scaled["target_weight"])
    assert (
        pytest.approx(scaled.loc[scaled["symbol"] == "AAA", "target_weight"].item())
        == 0.3
    )
    assert (
        pytest.approx(scaled.loc[scaled["symbol"] == "BBB", "target_weight"].item())
        == 0.2
    )
    assert pytest.approx(scaled["target_qty"].sum()) == 50.0


def test_apply_exposure_multiplier_handles_no_cash_mask_cleanly() -> None:
    """No CASH symbol present: all rows treated as risky, no crash, correct scaling."""
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "target_weight": [0.7, 0.3],
            "target_qty": [70.0, 30.0],
        }
    )
    multiplier = 0.5

    scaled = apply_exposure_multiplier_to_targets(df, multiplier)

    assert pytest.approx(scaled["target_weight"].sum()) == 0.5
    assert (
        pytest.approx(scaled.loc[scaled["symbol"] == "AAA", "target_weight"].item())
        == 0.35
    )
    assert (
        pytest.approx(scaled.loc[scaled["symbol"] == "BBB", "target_weight"].item())
        == 0.15
    )
    assert pytest.approx(scaled["target_qty"].sum()) == 50.0


def test_apply_exposure_multiplier_qty_non_numeric_does_not_crash() -> None:
    """Non-numeric target_qty values are coerced to NaN and do not crash."""
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "target_weight": [0.6, 0.4],
            "target_qty": [60.0, ""],  # non-numeric qty
        }
    )
    multiplier = 0.5

    scaled = apply_exposure_multiplier_to_targets(df, multiplier)

    # AAA qty scaled, BBB becomes NaN
    assert (
        pytest.approx(scaled.loc[scaled["symbol"] == "AAA", "target_qty"].item())
        == 30.0
    )
    assert pd.isna(scaled.loc[scaled["symbol"] == "BBB", "target_qty"].item())


def test_georisk_overlay_applied_in_trading_cycle(monkeypatch: Any) -> None:
    """GeoRisk overlay scales final target weights in trading cycle."""

    def dummy_signal_fn(prices_with_features: pd.DataFrame) -> pd.DataFrame:
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": [ts, ts],
                "symbol": ["AAA", "BBB"],
                "direction": ["LONG", "LONG"],
                "score": [1.0, 1.0],
            }
        )

    def dummy_position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions(signals, total_capital=1.0, equal_weight=True)

    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
            "symbol": ["AAA", "BBB"],
            "close": [100.0, 200.0],
        }
    )

    ctx = TradingContext(
        prices=prices,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        capital=1.0,
    )

    # Monkeypatch overlay components at call site in trading_cycle_v2
    import src.assembled_core.pipeline.trading_cycle_v2 as tc

    def fake_load_policy(path: str = "configs/policy.yaml") -> dict:
        return {"georisk_overlay": {"enabled": True}}

    def fake_compute_exposure_multiplier(_ctx: TradingContext, _policy: dict) -> float:
        return 0.5

    monkeypatch.setattr(tc, "load_policy", fake_load_policy)
    monkeypatch.setattr(
        tc, "compute_exposure_multiplier", fake_compute_exposure_multiplier
    )

    result = run_trading_cycle(ctx)

    weights = result.target_positions["target_weight"].tolist()
    assert pytest.approx(sum(weights)) == 0.5
    assert all(w > 0 for w in weights)
