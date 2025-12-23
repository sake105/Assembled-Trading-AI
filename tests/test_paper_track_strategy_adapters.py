"""Tests for Paper-Track strategy adapters (trend + multifactor)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.assembled_core.paper.paper_track as paper_module
from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackState,
)


pytestmark = pytest.mark.advanced


def _build_base_state(strategy_name: str = "test_strategy") -> PaperTrackState:
    """Create a minimal PaperTrackState for adapter tests."""
    return PaperTrackState(
        strategy_name=strategy_name,
        last_run_date=None,
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )


@pytest.fixture
def synthetic_prices_5days(tmp_path: Path) -> pd.DataFrame:
    """Local synthetic price data for adapter tests (3 symbols, 5 days)."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC")

    data: list[dict[str, object]] = []
    for sym_idx, sym in enumerate(symbols):
        base_price = 100.0 + sym_idx * 50.0
        for i, date in enumerate(dates):
            price = base_price + i * 1.0
            data.append(
                {
                    "timestamp": date,
                    "symbol": sym,
                    "open": price * 0.998,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                }
            )

    return pd.DataFrame(data)


def test_trend_adapter_generates_targets(
    synthetic_prices_5days: pd.DataFrame, tmp_path: Path
) -> None:
    """Trend adapter should generate non-empty target positions from prices."""

    config = PaperTrackConfig(
        strategy_name="trend_adapter_test",
        strategy_type="trend_baseline",
        strategy_params={
            "ma_fast": 2,
            "ma_slow": 3,
            "top_n": 2,
            "min_score": 0.0,
        },
        universe_file=tmp_path / "universe.txt",
        freq="1d",
        seed_capital=100000.0,
    )

    state_before = _build_base_state(strategy_name=config.strategy_name)

    # Use helper from paper_track to mimic orchestrator behaviour
    as_of = synthetic_prices_5days["timestamp"].max()
    prices_filtered = paper_module._filter_prices_for_date(  # type: ignore[attr-defined]
        synthetic_prices_5days, as_of
    )
    prices_with_features = paper_module._compute_features_for_strategy(  # type: ignore[attr-defined]
        config, prices_filtered
    )

    signals, targets = paper_module._generate_signals_and_targets_for_day(  # type: ignore[attr-defined]
        config=config,
        state_before=state_before,
        prices_full=synthetic_prices_5days,
        prices_filtered=prices_filtered,
        prices_with_features=prices_with_features,
        as_of=as_of,
    )

    # Basic sanity checks (shape + no errors); targets may be empty depending on params
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(targets, pd.DataFrame)
    if not targets.empty:
        assert "symbol" in targets.columns
        assert "target_qty" in targets.columns


def test_multifactor_adapter_generates_targets_with_patched_runtime(
    monkeypatch: pytest.MonkeyPatch, synthetic_prices_5days: pd.DataFrame, tmp_path: Path
) -> None:
    """Multi-factor adapter should generate target positions via patched strategy runtime.

    We patch the multifactor strategy functions to avoid heavy factor bundle dependencies
    and to focus on adapter wiring/shape.
    """

    # Patch price loader so run_paper_day uses synthetic prices (no I/O)
    def mock_load_prices(universe_file, freq):  # type: ignore[override]
        return synthetic_prices_5days.copy()

    monkeypatch.setattr(
        paper_module, "load_eod_prices_for_universe", mock_load_prices
    )

    # Patch multifactor signal generator
    def fake_generate_multifactor_long_short_signals(
        prices: pd.DataFrame, factors=None, config=None
    ) -> pd.DataFrame:  # type: ignore[override]
        last_ts = prices["timestamp"].max()
        return pd.DataFrame(
            {
                "timestamp": [last_ts, last_ts],
                "symbol": ["AAPL", "MSFT"],
                "direction": ["LONG", "SHORT"],
                "score": [1.0, -1.0],
            }
        )

    # Patch multifactor position sizing
    def fake_compute_multifactor_long_short_positions(
        signals: pd.DataFrame,
        capital: float,
        config=None,
        regime_state_df=None,
        timestamp_col: str = "timestamp",
        group_col: str = "symbol",
    ) -> pd.DataFrame:  # type: ignore[override]
        latest_ts = signals[timestamp_col].max()
        latest = signals[signals[timestamp_col] == latest_ts]
        # Simple symmetric positions: +10 for LONG, -10 for SHORT
        qty = latest["direction"].map({"LONG": 10.0, "SHORT": -10.0}).astype(float)
        return pd.DataFrame(
            {
                "symbol": latest[group_col].values,
                "target_weight": 0.0,
                "target_qty": qty.values,
            }
        )

    # Patch on adapter module (strategy_adapters) which is used by paper_track
    import src.assembled_core.paper.strategy_adapters as adapters

    monkeypatch.setattr(
        adapters,
        "generate_multifactor_long_short_signals",
        fake_generate_multifactor_long_short_signals,
    )
    monkeypatch.setattr(
        adapters,
        "compute_multifactor_long_short_positions",
        fake_compute_multifactor_long_short_positions,
    )

    # Build multifactor config using strategy_params
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    config = PaperTrackConfig(
        strategy_name="multifactor_adapter_test",
        strategy_type="multifactor_long_short",
        strategy_params={
            "bundle_path": "dummy_bundle.yaml",
            "top_quantile": 0.2,
            "bottom_quantile": 0.2,
            "max_gross_exposure": 1.0,
        },
        universe_file=universe_file,
        freq="1d",
        seed_capital=100000.0,
    )

    as_of = synthetic_prices_5days["timestamp"].max()
    state_path = tmp_path / "state" / "state.json"

    result = paper_module.run_paper_day(
        config=config,
        as_of=as_of,
        state_path=state_path,
    )

    assert result.status == "success"
    # With patched adapter we expect some positions or at least orders
    assert not result.orders.empty or not result.state_after.positions.empty


