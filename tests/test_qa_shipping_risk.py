"""Tests for shipping and systemic risk analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.qa.shipping_risk import (
    compute_shipping_exposure,
    compute_systemic_risk_flags,
)

pytestmark = pytest.mark.phase8


@pytest.fixture
def synthetic_portfolio() -> pd.DataFrame:
    """Create synthetic portfolio positions."""
    return pd.DataFrame(
        {"symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"], "weight": [0.4, 0.3, 0.2, 0.1]}
    )


@pytest.fixture
def synthetic_shipping_features() -> pd.DataFrame:
    """Create synthetic shipping features with varying congestion scores."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
            "shipping_congestion_score": [
                80.0,
                45.0,
                30.0,
                75.0,
            ],  # AAPL and TSLA have high congestion
            "shipping_ships_count": [15, 8, 5, 12],
            "route_id": ["US-CN-001", "US-EU-002", "US-AS-003", "US-CN-004"],
            "port_from": ["LAX", "NYC", "SEA", "LAX"],
            "port_to": ["SHG", "HAM", "SIN", "SHG"],
        }
    )


def test_compute_shipping_exposure_basic(
    synthetic_portfolio, synthetic_shipping_features
):
    """Test basic shipping exposure computation."""
    exposure = compute_shipping_exposure(
        synthetic_portfolio, synthetic_shipping_features
    )

    # Check that all expected keys are present
    assert "avg_shipping_congestion" in exposure
    assert "high_congestion_weight" in exposure
    assert "top_routes" in exposure
    assert "exposed_symbols" in exposure

    # Compute expected weighted average: 0.4*80 + 0.3*45 + 0.2*30 + 0.1*75 = 32 + 13.5 + 6 + 7.5 = 59.0
    expected_avg = 0.4 * 80.0 + 0.3 * 45.0 + 0.2 * 30.0 + 0.1 * 75.0
    assert abs(exposure["avg_shipping_congestion"] - expected_avg) < 0.01

    # High congestion (>70): AAPL (0.4) + TSLA (0.1) = 0.5
    assert abs(exposure["high_congestion_weight"] - 0.5) < 0.01

    # Exposed symbols should be AAPL and TSLA
    assert set(exposure["exposed_symbols"]) == {"AAPL", "TSLA"}


def test_compute_shipping_exposure_high_congestion(
    synthetic_portfolio, synthetic_shipping_features
):
    """Test shipping exposure with high congestion threshold."""
    # Use higher threshold (85 instead of 70) - only TSLA (75) and below, so none should be high
    # But let's use 75.0 to catch TSLA
    exposure = compute_shipping_exposure(
        synthetic_portfolio, synthetic_shipping_features, congestion_threshold=75.0
    )

    # Only AAPL (80) has congestion > 75, so high_congestion_weight should be 0.4
    assert abs(exposure["high_congestion_weight"] - 0.4) < 0.01
    assert exposure["exposed_symbols"] == ["AAPL"]


def test_compute_shipping_exposure_top_routes(
    synthetic_portfolio, synthetic_shipping_features
):
    """Test that top routes are correctly identified."""
    exposure = compute_shipping_exposure(
        synthetic_portfolio, synthetic_shipping_features
    )

    # Top routes should include routes for exposed symbols (AAPL, TSLA)
    assert len(exposure["top_routes"]) > 0
    assert "US-CN-001" in exposure["top_routes"]  # AAPL's route
    assert "US-CN-004" in exposure["top_routes"]  # TSLA's route


def test_compute_shipping_exposure_empty_portfolio():
    """Test shipping exposure with empty portfolio."""
    empty_portfolio = pd.DataFrame(columns=["symbol", "weight"])
    shipping_features = pd.DataFrame(
        {"symbol": ["AAPL"], "shipping_congestion_score": [80.0]}
    )

    exposure = compute_shipping_exposure(empty_portfolio, shipping_features)

    assert exposure["avg_shipping_congestion"] == 0.0
    assert exposure["high_congestion_weight"] == 0.0
    assert exposure["top_routes"] == []
    assert exposure["exposed_symbols"] == []


def test_compute_shipping_exposure_empty_shipping():
    """Test shipping exposure with empty shipping features."""
    portfolio = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.6, 0.4]})
    empty_shipping = pd.DataFrame(columns=["symbol", "shipping_congestion_score"])

    exposure = compute_shipping_exposure(portfolio, empty_shipping)

    assert exposure["avg_shipping_congestion"] == 0.0
    assert exposure["high_congestion_weight"] == 0.0
    assert exposure["top_routes"] == []
    assert exposure["exposed_symbols"] == []


def test_compute_shipping_exposure_missing_symbols(synthetic_portfolio):
    """Test shipping exposure when some portfolio symbols have no shipping data."""
    shipping_features = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],  # Missing GOOGL and TSLA
            "shipping_congestion_score": [80.0, 45.0],
        }
    )

    exposure = compute_shipping_exposure(synthetic_portfolio, shipping_features)

    # Missing symbols should be treated as 0 congestion
    # Expected: 0.4*80 + 0.3*45 + 0.2*0 + 0.1*0 = 32 + 13.5 = 45.5
    expected_avg = 0.4 * 80.0 + 0.3 * 45.0
    assert abs(exposure["avg_shipping_congestion"] - expected_avg) < 0.01

    # Only AAPL has high congestion
    assert abs(exposure["high_congestion_weight"] - 0.4) < 0.01
    assert exposure["exposed_symbols"] == ["AAPL"]


def test_compute_shipping_exposure_value_column():
    """Test shipping exposure with value column instead of weight."""
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "value": [6000.0, 4000.0],  # Total: 10000, weights: 0.6, 0.4
        }
    )
    shipping_features = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"], "shipping_congestion_score": [80.0, 50.0]}
    )

    exposure = compute_shipping_exposure(portfolio, shipping_features)

    # Expected: 0.6*80 + 0.4*50 = 48 + 20 = 68.0
    expected_avg = 0.6 * 80.0 + 0.4 * 50.0
    assert abs(exposure["avg_shipping_congestion"] - expected_avg) < 0.01


def test_compute_shipping_exposure_no_weight_column():
    """Test shipping exposure with no weight or value column (equal weights)."""
    portfolio = pd.DataFrame({"symbol": ["AAPL", "MSFT", "GOOGL"]})
    shipping_features = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "shipping_congestion_score": [80.0, 50.0, 30.0],
        }
    )

    exposure = compute_shipping_exposure(portfolio, shipping_features)

    # Equal weights: 1/3 each
    # Expected: (1/3)*80 + (1/3)*50 + (1/3)*30 = 26.67 + 16.67 + 10 = 53.33
    expected_avg = (80.0 + 50.0 + 30.0) / 3.0
    assert abs(exposure["avg_shipping_congestion"] - expected_avg) < 0.01


def test_compute_systemic_risk_flags_high_risk():
    """Test systemic risk flags with high risk exposure."""
    exposure = {
        "avg_shipping_congestion": 75.0,
        "high_congestion_weight": 0.4,
        "exposed_symbols": ["AAPL", "TSLA"],
        "top_routes": ["US-CN-001"],
    }

    flags = compute_systemic_risk_flags(exposure)

    assert flags["high_shipping_risk"] is True
    assert flags["exposed_to_blockade_routes"] is True
    assert flags["risk_level"] == "HIGH"
    assert "congestion" in flags["risk_reason"].lower()


def test_compute_systemic_risk_flags_medium_risk():
    """Test systemic risk flags with medium risk exposure."""
    exposure = {
        "avg_shipping_congestion": 65.0,  # Below threshold
        "high_congestion_weight": 0.35,  # Above threshold
        "exposed_symbols": ["AAPL"],
        "top_routes": [],
    }

    flags = compute_systemic_risk_flags(exposure)

    assert flags["high_shipping_risk"] is True
    assert flags["exposed_to_blockade_routes"] is True
    assert flags["risk_level"] == "MEDIUM"
    assert "exposure" in flags["risk_reason"].lower()


def test_compute_systemic_risk_flags_low_risk():
    """Test systemic risk flags with low risk exposure."""
    exposure = {
        "avg_shipping_congestion": 45.0,
        "high_congestion_weight": 0.1,
        "exposed_symbols": [],
        "top_routes": [],
    }

    flags = compute_systemic_risk_flags(exposure)

    assert flags["high_shipping_risk"] is False
    assert flags["exposed_to_blockade_routes"] is False
    assert flags["risk_level"] == "LOW"
    assert "low" in flags["risk_reason"].lower()


def test_compute_systemic_risk_flags_custom_thresholds():
    """Test systemic risk flags with custom thresholds."""
    exposure = {
        "avg_shipping_congestion": 60.0,
        "high_congestion_weight": 0.25,
        "exposed_symbols": ["AAPL"],
        "top_routes": [],
    }

    # Use lower thresholds
    flags = compute_systemic_risk_flags(
        exposure, high_congestion_threshold=50.0, high_exposure_threshold=0.2
    )

    # Should be HIGH risk with lower thresholds
    assert flags["high_shipping_risk"] is True
    assert flags["risk_level"] in ["HIGH", "MEDIUM"]


def test_compute_systemic_risk_flags_no_exposed_symbols():
    """Test systemic risk flags when no symbols are exposed."""
    exposure = {
        "avg_shipping_congestion": 50.0,
        "high_congestion_weight": 0.0,
        "exposed_symbols": [],
        "top_routes": [],
    }

    flags = compute_systemic_risk_flags(exposure)

    assert flags["exposed_to_blockade_routes"] is False
    assert flags["risk_level"] == "LOW"


def test_compute_shipping_exposure_with_timestamp():
    """Test shipping exposure with timestamp in portfolio positions."""
    portfolio = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "symbol": ["AAPL", "MSFT"],
            "weight": [0.6, 0.4],
        }
    )
    shipping_features = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT"], "shipping_congestion_score": [80.0, 50.0]}
    )

    exposure = compute_shipping_exposure(portfolio, shipping_features)

    # Should work the same way (timestamp is ignored)
    expected_avg = 0.6 * 80.0 + 0.4 * 50.0
    assert abs(exposure["avg_shipping_congestion"] - expected_avg) < 0.01
