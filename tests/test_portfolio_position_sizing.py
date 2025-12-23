# tests/test_portfolio_position_sizing.py
"""Sprint 11.1: Unit tests for position sizing.

This module tests the position sizing functions in src/assembled_core/portfolio/position_sizing.py:
- compute_target_positions: Compute target positions from signals
- compute_target_positions_from_trend_signals: Convenience function for trend signals

Tests cover:
- Happy path scenarios
- Equal weight vs score-based weighting
- Top-N selection
- Empty signals / no LONG signals
- Missing columns/required fields
- Edge cases (single signal, multiple signals, zero scores)
- Output stability (column names, data types, weight sums)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions,
    compute_target_positions_from_trend_signals,
)

pytestmark = pytest.mark.phase11


@pytest.fixture
def sample_signals():
    """Create sample trading signals for testing."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "direction": ["LONG", "LONG", "FLAT", "LONG", "FLAT"],
            "score": [0.8, 0.6, 0.0, 0.4, 0.0],
        }
    )


@pytest.fixture
def sample_trend_signals():
    """Create sample trend signals for testing."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "direction": ["LONG", "LONG", "LONG", "LONG", "FLAT"],
            "score": [0.9, 0.7, 0.5, 0.3, 0.0],
        }
    )


class TestComputeTargetPositions:
    """Tests for compute_target_positions function."""

    def test_compute_target_positions_happy_path(self, sample_signals):
        """Test position sizing with valid signals."""
        result = compute_target_positions(sample_signals, total_capital=10000.0)

        assert "symbol" in result.columns
        assert "target_weight" in result.columns
        assert "target_qty" in result.columns
        # Should only include LONG signals
        assert len(result) == 3  # AAPL, MSFT, AMZN (all LONG)
        assert all(result["symbol"].isin(["AAPL", "MSFT", "AMZN"]))

    def test_compute_target_positions_empty_signals(self):
        """Test that empty signals return empty DataFrame."""
        empty_signals = pd.DataFrame(columns=["symbol", "direction"])
        result = compute_target_positions(empty_signals)

        assert result.empty
        assert list(result.columns) == ["symbol", "target_weight", "target_qty"]

    def test_compute_target_positions_no_long_signals(self):
        """Test that signals with no LONG direction return empty DataFrame."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "direction": ["FLAT", "FLAT"],
            }
        )
        result = compute_target_positions(signals)

        assert result.empty

    def test_compute_target_positions_missing_columns(self):
        """Test that ValueError is raised when required columns are missing."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_target_positions(signals)

    def test_compute_target_positions_equal_weight(self, sample_signals):
        """Test equal weight allocation."""
        result = compute_target_positions(sample_signals, equal_weight=True)

        # With 3 LONG signals, each should have weight 1/3
        assert len(result) == 3
        expected_weight = 1.0 / 3.0
        import numpy as np

        assert np.allclose(result["target_weight"], expected_weight)
        # Weights should sum to 1.0
        assert result["target_weight"].sum() == pytest.approx(1.0)

    def test_compute_target_positions_score_based_weight(self, sample_signals):
        """Test score-based weight allocation."""
        result = compute_target_positions(sample_signals, equal_weight=False)

        assert len(result) == 3
        # Weights should be proportional to scores
        # AAPL: 0.8, MSFT: 0.6, AMZN: 0.4
        # Total score: 1.8
        # AAPL weight: 0.8 / 1.8 ≈ 0.444
        # MSFT weight: 0.6 / 1.8 ≈ 0.333
        # AMZN weight: 0.4 / 1.8 ≈ 0.222
        assert result["target_weight"].sum() == pytest.approx(1.0)
        # Highest score should have highest weight
        aapl_weight = result[result["symbol"] == "AAPL"]["target_weight"].iloc[0]
        amzn_weight = result[result["symbol"] == "AMZN"]["target_weight"].iloc[0]
        assert aapl_weight > amzn_weight

    def test_compute_target_positions_top_n(self, sample_signals):
        """Test top-N selection."""
        result = compute_target_positions(sample_signals, top_n=2)

        assert len(result) == 2
        # Should select top 2 by score: AAPL (0.8) and MSFT (0.6)
        assert set(result["symbol"]) == {"AAPL", "MSFT"}

    def test_compute_target_positions_top_n_without_scores(self):
        """Test top-N selection when scores are not available."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL"],
                "direction": ["LONG", "LONG", "LONG"],
            }
        )
        result = compute_target_positions(signals, top_n=2)

        assert len(result) == 2

    def test_compute_target_positions_total_capital(self, sample_signals):
        """Test that total_capital affects target_qty."""
        result1 = compute_target_positions(sample_signals, total_capital=10000.0)
        result2 = compute_target_positions(sample_signals, total_capital=20000.0)

        # target_qty should be proportional to total_capital
        assert result2["target_qty"].iloc[0] == pytest.approx(
            result1["target_qty"].iloc[0] * 2.0
        )

    def test_compute_target_positions_zero_scores_fallback(self):
        """Test that zero scores fall back to equal weight."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL"],
                "direction": ["LONG", "LONG", "LONG"],
                "score": [0.0, 0.0, 0.0],
            }
        )
        result = compute_target_positions(signals, equal_weight=False)

        # Should fall back to equal weight when all scores are zero
        expected_weight = 1.0 / 3.0
        import numpy as np

        assert np.allclose(result["target_weight"], expected_weight)

    def test_compute_target_positions_no_scores_fallback(self):
        """Test that missing scores fall back to equal weight."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL"],
                "direction": ["LONG", "LONG", "LONG"],
            }
        )
        result = compute_target_positions(signals, equal_weight=False)

        # Should fall back to equal weight when no scores
        expected_weight = 1.0 / 3.0
        import numpy as np

        assert np.allclose(result["target_weight"], expected_weight)

    def test_compute_target_positions_sorted_by_symbol(self, sample_signals):
        """Test that output is sorted by symbol."""
        result = compute_target_positions(sample_signals)

        assert result["symbol"].is_monotonic_increasing

    def test_compute_target_positions_single_signal(self):
        """Test position sizing with single signal."""
        signals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.8],
            }
        )
        result = compute_target_positions(signals)

        assert len(result) == 1
        assert result["target_weight"].iloc[0] == pytest.approx(1.0)


class TestComputeTargetPositionsFromTrendSignals:
    """Tests for compute_target_positions_from_trend_signals convenience function."""

    def test_compute_target_positions_from_trend_signals_happy_path(
        self, sample_trend_signals
    ):
        """Test convenience function with valid trend signals."""
        result = compute_target_positions_from_trend_signals(sample_trend_signals)

        assert "symbol" in result.columns
        assert "target_weight" in result.columns
        assert "target_qty" in result.columns
        # Should only include LONG signals
        assert len(result) == 4  # AAPL, MSFT, GOOGL, AMZN (all LONG)

    def test_compute_target_positions_from_trend_signals_min_score(
        self, sample_trend_signals
    ):
        """Test filtering by minimum score."""
        result = compute_target_positions_from_trend_signals(
            sample_trend_signals, min_score=0.6
        )

        # Should only include signals with score >= 0.6: AAPL (0.9), MSFT (0.7)
        assert len(result) == 2
        assert set(result["symbol"]) == {"AAPL", "MSFT"}

    def test_compute_target_positions_from_trend_signals_top_n(
        self, sample_trend_signals
    ):
        """Test top-N selection."""
        result = compute_target_positions_from_trend_signals(
            sample_trend_signals, top_n=2
        )

        assert len(result) == 2
        # Should select top 2 by score: AAPL (0.9) and MSFT (0.7)
        assert set(result["symbol"]) == {"AAPL", "MSFT"}

    def test_compute_target_positions_from_trend_signals_score_based(
        self, sample_trend_signals
    ):
        """Test that convenience function uses score-based weighting."""
        result = compute_target_positions_from_trend_signals(sample_trend_signals)

        # Should use score-based weighting (not equal weight)
        # Weights should be proportional to scores
        assert result["target_weight"].sum() == pytest.approx(1.0)
        # Highest score should have highest weight
        aapl_weight = result[result["symbol"] == "AAPL"]["target_weight"].iloc[0]
        amzn_weight = result[result["symbol"] == "AMZN"]["target_weight"].iloc[0]
        assert aapl_weight > amzn_weight

    def test_compute_target_positions_from_trend_signals_total_capital(
        self, sample_trend_signals
    ):
        """Test that total_capital is respected."""
        result = compute_target_positions_from_trend_signals(
            sample_trend_signals, total_capital=50000.0
        )

        # target_qty should reflect total_capital
        assert result["target_qty"].sum() == pytest.approx(50000.0)
