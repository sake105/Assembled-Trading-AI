# tests/test_portfolio_position_sizing.py
"""Tests for position sizing."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions,
    compute_target_positions_from_trend_signals
)


def create_sample_signals() -> pd.DataFrame:
    """Create sample signals for testing."""
    data = [
        {"symbol": "AAPL", "direction": "LONG", "score": 0.8},
        {"symbol": "MSFT", "direction": "LONG", "score": 0.6},
        {"symbol": "GOOGL", "direction": "LONG", "score": 0.4},
        {"symbol": "TSLA", "direction": "FLAT", "score": 0.2},
    ]
    return pd.DataFrame(data)


def test_compute_target_positions_equal_weight():
    """Test equal weight position sizing."""
    signals = create_sample_signals()
    targets = compute_target_positions(signals, total_capital=1.0, equal_weight=True)
    
    # Should have 3 LONG positions (TSLA is FLAT, excluded)
    assert len(targets) == 3
    assert set(targets["symbol"].values) == {"AAPL", "MSFT", "GOOGL"}
    
    # All weights should be equal (1/3)
    assert (targets["target_weight"] == pytest.approx(1.0 / 3.0)).all()
    
    # Weights should sum to 1.0
    assert targets["target_weight"].sum() == pytest.approx(1.0)
    
    # Quantities should equal weights (when total_capital=1.0)
    assert (targets["target_qty"] == targets["target_weight"]).all()


def test_compute_target_positions_score_based():
    """Test score-based position sizing."""
    signals = create_sample_signals()
    targets = compute_target_positions(signals, total_capital=1.0, equal_weight=False)
    
    # Should have 3 LONG positions
    assert len(targets) == 3
    
    # Weights should be proportional to scores
    # AAPL: 0.8, MSFT: 0.6, GOOGL: 0.4
    # Total: 1.8
    # AAPL weight: 0.8 / 1.8 ≈ 0.444
    # MSFT weight: 0.6 / 1.8 ≈ 0.333
    # GOOGL weight: 0.4 / 1.8 ≈ 0.222
    
    aapl_weight = targets[targets["symbol"] == "AAPL"]["target_weight"].iloc[0]
    msft_weight = targets[targets["symbol"] == "MSFT"]["target_weight"].iloc[0]
    googl_weight = targets[targets["symbol"] == "GOOGL"]["target_weight"].iloc[0]
    
    assert aapl_weight > msft_weight > googl_weight, "Weights should follow score order"
    assert targets["target_weight"].sum() == pytest.approx(1.0)


def test_compute_target_positions_top_n():
    """Test top-N position selection."""
    signals = create_sample_signals()
    targets = compute_target_positions(signals, total_capital=1.0, top_n=2, equal_weight=True)
    
    # Should have only top 2 by score (AAPL, MSFT)
    assert len(targets) == 2
    assert set(targets["symbol"].values) == {"AAPL", "MSFT"}
    
    # Equal weights: 1/2 each
    assert (targets["target_weight"] == pytest.approx(0.5)).all()


def test_compute_target_positions_empty():
    """Test with empty signals."""
    signals = pd.DataFrame(columns=["symbol", "direction"])
    targets = compute_target_positions(signals)
    
    assert targets.empty
    assert list(targets.columns) == ["symbol", "target_weight", "target_qty"]


def test_compute_target_positions_no_long():
    """Test with no LONG signals."""
    signals = pd.DataFrame([
        {"symbol": "AAPL", "direction": "FLAT", "score": 0.5},
        {"symbol": "MSFT", "direction": "FLAT", "score": 0.3},
    ])
    targets = compute_target_positions(signals)
    
    assert targets.empty


def test_compute_target_positions_missing_columns():
    """Test that ValueError is raised when required columns are missing."""
    signals = pd.DataFrame([{"symbol": "AAPL"}])  # Missing "direction"
    
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_target_positions(signals)


def test_compute_target_positions_from_trend_signals():
    """Test convenience function for trend signals."""
    signals = create_sample_signals()
    targets = compute_target_positions_from_trend_signals(
        signals,
        total_capital=1.0,
        top_n=2,
        min_score=0.5
    )
    
    # Should filter by min_score=0.5 (AAPL=0.8, MSFT=0.6) and take top_n=2
    assert len(targets) == 2
    assert set(targets["symbol"].values) == {"AAPL", "MSFT"}
    
    # Should use score-based weighting (not equal weight)
    aapl_weight = targets[targets["symbol"] == "AAPL"]["target_weight"].iloc[0]
    msft_weight = targets[targets["symbol"] == "MSFT"]["target_weight"].iloc[0]
    
    # AAPL score (0.8) > MSFT score (0.6), so AAPL weight > MSFT weight
    assert aapl_weight > msft_weight

