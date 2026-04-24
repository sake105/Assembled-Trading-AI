"""Tests for M28: Behavioral Finance Signals."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.signals.behavioral_finance")
from src.assembled_core.signals.behavioral_finance import (
    BehavioralConfig,
    BehavioralSignal,
    compute_disposition_score,
    compute_anchoring_score,
    compute_herding_score,
    compute_overreaction_score,
    generate_behavioral_signals,
)


@pytest.fixture
def rising_prices():
    """Steadily rising price series."""
    return np.linspace(50, 100, 120)


@pytest.fixture
def falling_prices():
    """Steadily falling price series."""
    return np.linspace(100, 50, 120)


@pytest.fixture
def mean_reverting_prices():
    """Oscillating price series."""
    rng = np.random.default_rng(42)
    base = 100.0
    noise = np.cumsum(rng.normal(0, 0.5, 120))
    # Mean-revert
    prices = base + noise - 0.05 * np.cumsum(noise)
    return np.maximum(prices, 10)


@pytest.fixture
def sample_prices_df(rising_prices):
    """Multi-symbol DataFrame."""
    rng = np.random.default_rng(42)
    n = len(rising_prices)
    falling = np.linspace(100, 60, n)

    rows = []
    for i in range(n):
        rows.append({"timestamp": i, "symbol": "AAPL", "close": rising_prices[i],
                      "volume": 1_000_000 + rng.integers(-200_000, 200_000)})
        rows.append({"timestamp": i, "symbol": "MSFT", "close": falling[i],
                      "volume": 800_000 + rng.integers(-100_000, 100_000)})
    return pd.DataFrame(rows)


@pytest.mark.phase12
class TestDispositionScore:
    def test_at_highs_negative(self, rising_prices):
        score = compute_disposition_score(rising_prices)
        assert score < 0  # near highs -> profit-taking pressure

    def test_at_lows_positive(self, falling_prices):
        score = compute_disposition_score(falling_prices)
        assert score > 0  # near lows -> selling exhaustion

    def test_short_series_zero(self):
        score = compute_disposition_score(np.array([100.0, 101.0, 99.0]))
        assert score == 0.0

    def test_with_volume(self, falling_prices):
        volumes = np.ones(len(falling_prices)) * 1_000_000
        # Declining volume at lows strengthens the signal
        volumes[-5:] = 500_000
        score = compute_disposition_score(falling_prices, volumes)
        assert score > 0

    def test_bounded(self, rising_prices):
        score = compute_disposition_score(rising_prices)
        assert -1.0 <= score <= 1.0


@pytest.mark.phase12
class TestAnchoringScore:
    def test_near_52week_high(self):
        prices = np.linspace(80, 100, 252)
        score = compute_anchoring_score(prices)
        assert score > 0  # near 52-week high -> bullish

    def test_near_round_number(self):
        # Price just above 100
        prices = np.full(60, 99.0)
        prices[-5:] = 101.0
        score = compute_anchoring_score(prices, round_levels=[100])
        assert score > 0  # just above round number

    def test_short_series_zero(self):
        score = compute_anchoring_score(np.array([50.0, 51.0]))
        assert score == 0.0

    def test_bounded(self):
        prices = np.linspace(50, 150, 300)
        score = compute_anchoring_score(prices)
        assert -1.0 <= score <= 1.0


@pytest.mark.phase12
class TestHerdingScore:
    def test_panic_selling_contrarian_buy(self):
        rng = np.random.default_rng(42)
        n = 60
        volumes = np.ones(n) * 1_000_000
        returns = rng.normal(0, 0.01, n)
        # Spike volume with negative returns = panic
        volumes[-5:] = 5_000_000
        returns[-5:] = -0.03
        score = compute_herding_score(volumes, returns)
        assert score > 0  # contrarian buy on panic

    def test_euphoria_contrarian_sell(self):
        rng = np.random.default_rng(42)
        n = 60
        volumes = np.ones(n) * 1_000_000
        returns = rng.normal(0, 0.01, n)
        volumes[-5:] = 5_000_000
        returns[-5:] = 0.03
        score = compute_herding_score(volumes, returns)
        assert score < 0  # contrarian sell on euphoria

    def test_normal_volume_no_signal(self):
        volumes = np.ones(60) * 1_000_000
        returns = np.zeros(60)
        score = compute_herding_score(volumes, returns)
        assert score == 0.0

    def test_short_series_zero(self):
        score = compute_herding_score(np.array([1.0, 2.0]), np.array([0.01, -0.01]))
        assert score == 0.0


@pytest.mark.phase12
class TestOverreactionScore:
    def test_large_drop_reversal(self):
        returns = np.zeros(30)
        returns[-5:] = -0.03  # -15% cumulative
        score = compute_overreaction_score(returns, lookback=5, threshold=0.05)
        assert score > 0  # expect reversal upward

    def test_large_rally_reversal(self):
        returns = np.zeros(30)
        returns[-5:] = 0.03  # +15% cumulative
        score = compute_overreaction_score(returns, lookback=5, threshold=0.05)
        assert score < 0  # expect reversal downward

    def test_moderate_move_continuation(self):
        returns = np.zeros(30)
        returns[-5:] = 0.005  # +2.5% cumulative, moderate
        score = compute_overreaction_score(returns, lookback=5, threshold=0.05)
        assert score > 0  # underreaction, expect continuation upward

    def test_no_move_no_signal(self):
        returns = np.zeros(30)
        score = compute_overreaction_score(returns, lookback=5, threshold=0.05)
        assert score == 0.0

    def test_bounded(self):
        returns = np.zeros(30)
        returns[-5:] = -0.10  # extreme
        score = compute_overreaction_score(returns, lookback=5, threshold=0.05)
        assert -1.0 <= score <= 1.0


@pytest.mark.phase12
class TestGenerateBehavioralSignals:
    def test_basic_generation(self, sample_prices_df):
        signals = generate_behavioral_signals(sample_prices_df)
        assert len(signals) == 2  # AAPL and MSFT
        assert all(isinstance(s, BehavioralSignal) for s in signals)

    def test_composite_bounded(self, sample_prices_df):
        signals = generate_behavioral_signals(sample_prices_df)
        for s in signals:
            assert -1.0 <= s.composite_score <= 1.0

    def test_all_sub_scores_present(self, sample_prices_df):
        signals = generate_behavioral_signals(sample_prices_df)
        for s in signals:
            assert hasattr(s, "disposition_score")
            assert hasattr(s, "anchoring_score")
            assert hasattr(s, "herding_score")
            assert hasattr(s, "overreaction_score")

    def test_custom_config(self, sample_prices_df):
        cfg = BehavioralConfig(
            disposition_lookback=30,
            overreaction_threshold=0.10,
            blend_weights={
                "disposition": 1.0, "anchoring": 0.0,
                "herding": 0.0, "overreaction": 0.0,
            },
        )
        signals = generate_behavioral_signals(sample_prices_df, config=cfg)
        assert len(signals) == 2

    def test_without_volume_column(self):
        df = pd.DataFrame({
            "timestamp": list(range(60)) * 2,
            "symbol": ["A"] * 60 + ["B"] * 60,
            "close": list(np.linspace(50, 80, 60)) + list(np.linspace(80, 50, 60)),
        })
        signals = generate_behavioral_signals(df)
        assert len(signals) == 2

    def test_short_symbol_excluded(self):
        df = pd.DataFrame({
            "timestamp": list(range(5)),
            "symbol": ["X"] * 5,
            "close": [100, 101, 102, 101, 100],
            "volume": [1000] * 5,
        })
        signals = generate_behavioral_signals(df)
        assert len(signals) == 0  # too short
