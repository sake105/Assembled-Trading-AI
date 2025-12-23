# tests/test_api_monitoring.py
"""Tests for monitoring API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app

pytestmark = pytest.mark.phase10


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestMonitoringQAStatus:
    """Tests for GET /api/v1/monitoring/qa_status endpoint."""

    def test_qa_status_summary_invalid_freq(self, client: TestClient):
        """Test that invalid frequency returns 400."""
        response = client.get("/api/v1/monitoring/qa_status?freq=invalid")
        assert response.status_code == 400
        assert "Unsupported frequency" in response.json()["detail"]

    def test_qa_status_summary_no_data(self, client: TestClient):
        """Test that missing equity files returns 404."""
        response = client.get("/api/v1/monitoring/qa_status?freq=1d")
        # Should return 404 if no equity files exist
        # If files exist, should return 200 with data
        assert response.status_code in [200, 404]
        if response.status_code == 404:
            assert "No equity file found" in response.json()["detail"]

    def test_qa_status_summary_valid_response(self, client: TestClient):
        """Test that valid response has correct structure."""
        response = client.get("/api/v1/monitoring/qa_status?freq=1d")

        if response.status_code == 200:
            data = response.json()
            assert "overall_result" in data
            assert "gate_counts" in data
            assert "key_metrics" in data
            assert "last_updated" in data

            # Check gate_counts structure
            assert isinstance(data["gate_counts"], dict)
            assert "ok" in data["gate_counts"]
            assert "warning" in data["gate_counts"]
            assert "block" in data["gate_counts"]

            # Check key_metrics structure
            assert isinstance(data["key_metrics"], dict)
            # Should have sharpe_ratio, max_drawdown_pct, total_return, cagr (may be None)
            assert "sharpe_ratio" in data["key_metrics"]
            assert "max_drawdown_pct" in data["key_metrics"]
            assert "total_return" in data["key_metrics"]
            assert "cagr" in data["key_metrics"]


class TestMonitoringRiskStatus:
    """Tests for GET /api/v1/monitoring/risk_status endpoint."""

    def test_risk_status_summary_invalid_freq(self, client: TestClient):
        """Test that invalid frequency returns 400."""
        response = client.get("/api/v1/monitoring/risk_status?freq=invalid")
        assert response.status_code == 400
        assert "Unsupported frequency" in response.json()["detail"]

    def test_risk_status_summary_no_data(self, client: TestClient):
        """Test that missing equity files returns 404."""
        response = client.get("/api/v1/monitoring/risk_status?freq=1d")
        # Should return 404 if no equity files exist
        assert response.status_code in [200, 404]
        if response.status_code == 404:
            assert "No equity file found" in response.json()["detail"]

    def test_risk_status_summary_valid_response(self, client: TestClient):
        """Test that valid response has correct structure."""
        response = client.get("/api/v1/monitoring/risk_status?freq=1d")

        if response.status_code == 200:
            data = response.json()
            assert "sharpe_ratio" in data
            assert "max_drawdown_pct" in data
            assert "volatility" in data
            assert "var_95" in data
            assert "current_drawdown" in data
            assert "last_updated" in data

            # All fields should be present (may be None)
            assert isinstance(data.get("sharpe_ratio"), (float, type(None)))
            assert isinstance(data.get("max_drawdown_pct"), (float, type(None)))
            assert isinstance(data.get("volatility"), (float, type(None)))
            assert isinstance(data.get("var_95"), (float, type(None)))
            assert isinstance(data.get("current_drawdown"), (float, type(None)))


class TestMonitoringDriftStatus:
    """Tests for GET /api/v1/monitoring/drift_status endpoint."""

    def test_drift_status_summary_invalid_freq(self, client: TestClient):
        """Test that invalid frequency returns 400."""
        response = client.get("/api/v1/monitoring/drift_status?freq=invalid")
        assert response.status_code == 400
        assert "Unsupported frequency" in response.json()["detail"]

    def test_drift_status_summary_default_params(self, client: TestClient):
        """Test that default parameters work."""
        response = client.get("/api/v1/monitoring/drift_status?freq=1d")
        assert response.status_code == 200
        data = response.json()

        assert "overall_severity" in data
        assert "features_with_drift" in data
        assert "total_features_checked" in data
        assert "last_updated" in data

        # Should return valid structure even if no drift analysis available
        assert isinstance(data["overall_severity"], str)
        assert data["overall_severity"] in ["NONE", "MODERATE", "SEVERE"]
        assert isinstance(data["features_with_drift"], list)
        assert isinstance(data["total_features_checked"], int)

    def test_drift_status_summary_with_top_n(self, client: TestClient):
        """Test that top_n parameter works."""
        response = client.get("/api/v1/monitoring/drift_status?freq=1d&top_n=5")
        assert response.status_code == 200
        data = response.json()

        assert "features_with_drift" in data
        # Should respect top_n limit (if features exist)
        assert len(data["features_with_drift"]) <= 5

    def test_drift_status_summary_invalid_top_n(self, client: TestClient):
        """Test that invalid top_n values are handled."""
        # top_n too large (max is 50)
        response = client.get("/api/v1/monitoring/drift_status?freq=1d&top_n=100")
        assert response.status_code == 422  # Validation error

        # top_n too small (min is 1)
        response = client.get("/api/v1/monitoring/drift_status?freq=1d&top_n=0")
        assert response.status_code == 422  # Validation error

    def test_drift_status_summary_feature_structure(self, client: TestClient):
        """Test that feature drift items have correct structure."""
        response = client.get("/api/v1/monitoring/drift_status?freq=1d")
        assert response.status_code == 200
        data = response.json()

        # If features_with_drift has items, check their structure
        if len(data["features_with_drift"]) > 0:
            feature = data["features_with_drift"][0]
            assert "feature" in feature
            assert "psi" in feature
            assert "drift_flag" in feature

            assert isinstance(feature["feature"], str)
            assert isinstance(feature["psi"], (int, float))
            assert feature["drift_flag"] in ["NONE", "MODERATE", "SEVERE"]


class TestMonitoringIntegration:
    """Integration tests for monitoring endpoints."""

    def test_all_monitoring_endpoints_accessible(self, client: TestClient):
        """Test that all monitoring endpoints are accessible."""
        endpoints = [
            "/api/v1/monitoring/qa_status?freq=1d",
            "/api/v1/monitoring/risk_status?freq=1d",
            "/api/v1/monitoring/drift_status?freq=1d",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return 404 (endpoint not found) or 500 (server error)
            assert response.status_code in [200, 400, 404]

            if response.status_code == 400:
                # Invalid parameter
                assert "detail" in response.json()
            elif response.status_code == 404:
                # No data found (expected if no files exist)
                assert "detail" in response.json()
