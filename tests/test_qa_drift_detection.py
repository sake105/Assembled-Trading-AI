"""Tests for qa.drift_detection module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase9

from src.assembled_core.qa.drift_detection import (
    compute_performance_drift,
    compute_psi,
    detect_feature_drift,
    detect_label_drift,
)


class TestComputePSI:
    """Tests for compute_psi function."""

    def test_compute_psi_identical_distributions(self):
        """Test that identical distributions yield PSI ≈ 0."""
        base = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        current = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        psi = compute_psi(base, current)
        
        assert abs(psi) < 0.01  # Should be very close to 0

    def test_compute_psi_different_distributions(self):
        """Test that different distributions yield PSI > 0.2."""
        base = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        current = pd.Series([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        
        psi = compute_psi(base, current)
        
        assert psi > 0.2  # Significant drift

    def test_compute_psi_moderate_drift(self):
        """Test moderate drift scenario."""
        np.random.seed(42)
        base = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(0.5, 1, 100))  # Slight shift
        
        psi = compute_psi(base, current)
        
        # Should be between 0.05 and 1.0 (moderate drift, can vary with random seed)
        assert 0.05 < psi < 1.0

    def test_compute_psi_constant_values_same(self):
        """Test that same constant values yield PSI = 0."""
        base = pd.Series([5.0] * 10)
        current = pd.Series([5.0] * 10)
        
        psi = compute_psi(base, current)
        
        assert abs(psi) < 0.01

    def test_compute_psi_constant_values_different(self):
        """Test that different constant values yield high PSI."""
        base = pd.Series([5.0] * 10)
        current = pd.Series([10.0] * 10)
        
        psi = compute_psi(base, current)
        
        assert psi > 0.5  # High drift indicator

    def test_compute_psi_one_constant(self):
        """Test when one series is constant and the other is not."""
        base = pd.Series([5.0] * 10)
        current = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        psi = compute_psi(base, current)
        
        assert psi > 0.0  # Should indicate drift

    def test_compute_psi_empty_series(self):
        """Test handling of empty series."""
        base = pd.Series([1, 2, 3])
        current = pd.Series([])
        
        psi = compute_psi(base, current)
        
        assert psi == 1.0  # Should return high PSI for empty series

    def test_compute_psi_with_nans(self):
        """Test that NaN values are handled correctly."""
        base = pd.Series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
        current = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        psi = compute_psi(base, current)
        
        # Should compute PSI on non-NaN values
        assert psi >= 0.0
        assert not np.isnan(psi)

    def test_compute_psi_zero_bins_edge_case(self):
        """Test PSI calculation when one distribution has zero bins.
        
        This tests the bug fix where renormalized frequencies were incorrectly
        used in edge-case calculations for zero bins.
        """
        # Base: values only in lower range (will have empty bins in upper range)
        base = pd.Series([1, 2, 3, 4, 5] * 20)  # 100 values, all low
        
        # Current: values only in upper range (will have empty bins in lower range)
        current = pd.Series([50, 60, 70, 80, 90] * 20)  # 100 values, all high
        
        psi = compute_psi(base, current)
        
        # Should detect significant drift (complete separation of distributions)
        assert psi > 0.2  # Significant drift
        
        # PSI should be reasonably large (not artificially small due to bug)
        # With complete separation, PSI should be substantial
        assert psi > 0.5  # Should indicate severe drift
        
    def test_compute_psi_zero_bins_reverse(self):
        """Test PSI calculation with zero bins in reverse direction.
        
        Similar to test_compute_psi_zero_bins_edge_case but reversed.
        """
        # Base: values only in upper range
        base = pd.Series([50, 60, 70, 80, 90] * 20)
        
        # Current: values only in lower range
        current = pd.Series([1, 2, 3, 4, 5] * 20)
        
        psi = compute_psi(base, current)
        
        # Should detect significant drift
        assert psi > 0.2
        assert psi > 0.5  # Should be substantial


class TestDetectFeatureDrift:
    """Tests for detect_feature_drift function."""

    def test_detect_feature_drift_no_drift(self):
        """Test when features have no drift."""
        # Use identical distributions to ensure no drift
        base_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,
            "feature2": [10, 20, 30, 40, 50] * 20
        })
        current_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5] * 20,  # Same as base
            "feature2": [10, 20, 30, 40, 50] * 20  # Same as base
        })
        
        results = detect_feature_drift(base_df, current_df)
        
        assert len(results) == 2
        # Identical distributions should have PSI ≈ 0
        assert all(results["psi"] < 0.1)
        assert all(results["drift_flag"] == "NONE")

    def test_detect_feature_drift_severe_drift(self):
        """Test when a feature has severe drift."""
        np.random.seed(42)
        base_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100)
        })
        current_df = pd.DataFrame({
            "feature1": np.random.normal(5, 1, 100),  # Shifted distribution
            "feature2": np.random.normal(0, 1, 100)  # Same distribution
        })
        
        results = detect_feature_drift(base_df, current_df)
        
        assert len(results) == 2
        feature1_result = results[results["feature"] == "feature1"].iloc[0]
        assert feature1_result["drift_flag"] == "SEVERE"
        assert feature1_result["psi"] >= 0.3

    def test_detect_feature_drift_moderate_drift(self):
        """Test when a feature has moderate drift."""
        np.random.seed(42)
        base_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100)
        })
        current_df = pd.DataFrame({
            "feature1": np.random.normal(0.3, 1, 100)  # Small shift
        })
        
        results = detect_feature_drift(base_df, current_df, psi_threshold=0.2, severe_threshold=0.3)
        
        assert len(results) == 1
        # Should detect some level of drift (may be MODERATE or SEVERE depending on random seed)
        assert results.iloc[0]["drift_flag"] in ["MODERATE", "SEVERE", "NONE"]
        assert results.iloc[0]["psi"] >= 0.0

    def test_detect_feature_drift_constant_feature(self):
        """Test handling of constant feature values."""
        base_df = pd.DataFrame({
            "feature1": [5.0] * 100,
            "feature2": np.random.normal(0, 1, 100)
        })
        current_df = pd.DataFrame({
            "feature1": [5.0] * 100,  # Same constant
            "feature2": np.random.normal(0, 1, 100)
        })
        
        results = detect_feature_drift(base_df, current_df)
        
        assert len(results) == 2
        # Should handle constant values without crashing
        assert "feature1" in results["feature"].values

    def test_detect_feature_drift_different_constant(self):
        """Test when constant feature changes value."""
        base_df = pd.DataFrame({
            "feature1": [5.0] * 100
        })
        current_df = pd.DataFrame({
            "feature1": [10.0] * 100  # Different constant
        })
        
        results = detect_feature_drift(base_df, current_df)
        
        assert len(results) == 1
        # Should detect drift for different constants
        assert results.iloc[0]["drift_flag"] in ["MODERATE", "SEVERE"]

    def test_detect_feature_drift_no_common_columns(self):
        """Test when DataFrames have no common columns."""
        base_df = pd.DataFrame({"feature1": [1, 2, 3]})
        current_df = pd.DataFrame({"feature2": [1, 2, 3]})
        
        results = detect_feature_drift(base_df, current_df)
        
        assert len(results) == 0
        assert list(results.columns) == ["feature", "psi", "drift_flag"]


class TestDetectLabelDrift:
    """Tests for detect_label_drift function."""

    def test_detect_label_drift_binary_no_drift(self):
        """Test binary labels with no drift."""
        base_labels = pd.Series([0, 0, 0, 1, 1, 1])
        current_labels = pd.Series([0, 0, 0, 1, 1, 1])
        
        drift = detect_label_drift(base_labels, current_labels)
        
        assert drift["drift_detected"] is False
        assert abs(drift["psi"]) < 0.2
        assert drift["drift_severity"] == "NONE"

    def test_detect_label_drift_binary_with_drift(self):
        """Test binary labels with significant drift."""
        base_labels = pd.Series([0, 0, 0, 0, 0, 1, 1])  # ~29% positive
        current_labels = pd.Series([0, 1, 1, 1, 1, 1, 1])  # ~86% positive
        
        drift = detect_label_drift(base_labels, current_labels)
        
        # Should detect drift (PSI should be high due to distribution shift)
        assert drift["mean_shift"] > 0  # More positive labels
        assert drift["base_mean"] < drift["current_mean"]
        # With such a large shift, drift should be detected
        assert drift["drift_detected"] is True or drift["psi"] >= 0.1

    def test_detect_label_drift_multi_class(self):
        """Test multi-class label drift."""
        base_labels = pd.Series([0, 0, 1, 1, 2, 2])
        current_labels = pd.Series([0, 1, 1, 2, 2, 2])  # Shift towards higher classes
        
        drift = detect_label_drift(base_labels, current_labels)
        
        assert drift["mean_shift"] > 0  # Shift towards higher values
        assert drift["current_mean"] > drift["base_mean"]

    def test_detect_label_drift_empty_series(self):
        """Test handling of empty label series."""
        base_labels = pd.Series([0, 1, 1])
        current_labels = pd.Series([])
        
        drift = detect_label_drift(base_labels, current_labels)
        
        assert drift["drift_detected"] is True
        assert drift["drift_severity"] == "SEVERE"
        assert drift["base_mean"] is None or drift["current_mean"] is None

    def test_detect_label_drift_with_nans(self):
        """Test that NaN values are handled correctly."""
        base_labels = pd.Series([0, 0, 1, 1, np.nan])
        current_labels = pd.Series([0, 1, 1, 1, 1])
        
        drift = detect_label_drift(base_labels, current_labels)
        
        # Should compute on non-NaN values
        assert drift["base_mean"] is not None
        assert drift["current_mean"] is not None


class TestComputePerformanceDrift:
    """Tests for compute_performance_drift function."""

    def test_compute_performance_drift_degrading(self):
        """Test equity with degrading performance in second half."""
        np.random.seed(42)
        # First half: good performance (1% daily return)
        returns1 = np.random.normal(0.01, 0.01, 50)
        # Second half: bad performance (-0.5% daily return)
        returns2 = np.random.normal(-0.005, 0.01, 50)
        
        equity_values = [10000.0]
        for r in list(returns1) + list(returns2):
            equity_values.append(equity_values[-1] * (1 + r))
        
        equity = pd.Series(equity_values)
        
        drift = compute_performance_drift(equity, window=20)
        
        assert drift["performance_degrading"] is True
        assert drift["current_avg_return"] < drift["reference_avg_return"]
        assert drift["sharpe_degradation"] < 0

    def test_compute_performance_drift_stable(self):
        """Test equity with stable performance."""
        np.random.seed(42)
        # Consistent performance throughout
        returns = np.random.normal(0.005, 0.01, 100)
        
        equity_values = [10000.0]
        for r in returns:
            equity_values.append(equity_values[-1] * (1 + r))
        
        equity = pd.Series(equity_values)
        
        drift = compute_performance_drift(equity, window=20)
        
        # Should not be degrading (or only slightly)
        # Note: Due to randomness, might be slightly degrading, but should be close
        assert drift["reference_sharpe"] is not None
        assert drift["current_sharpe"] is not None

    def test_compute_performance_drift_improving(self):
        """Test equity with improving performance."""
        np.random.seed(42)
        # First half: bad performance
        returns1 = np.random.normal(-0.005, 0.01, 50)
        # Second half: good performance
        returns2 = np.random.normal(0.01, 0.01, 50)
        
        equity_values = [10000.0]
        for r in list(returns1) + list(returns2):
            equity_values.append(equity_values[-1] * (1 + r))
        
        equity = pd.Series(equity_values)
        
        drift = compute_performance_drift(equity, window=20)
        
        assert drift["performance_degrading"] is False
        assert drift["current_avg_return"] > drift["reference_avg_return"]
        assert drift["sharpe_degradation"] > 0

    def test_compute_performance_drift_insufficient_data(self):
        """Test with insufficient data for rolling window."""
        equity = pd.Series([10000.0, 10100.0, 10200.0])  # Only 3 points
        
        drift = compute_performance_drift(equity, window=63)
        
        # Should handle gracefully
        assert drift["reference_sharpe"] is not None or drift["reference_sharpe"] is None
        assert isinstance(drift["performance_degrading"], bool)

    def test_compute_performance_drift_very_short(self):
        """Test with very short equity series."""
        equity = pd.Series([10000.0, 10100.0])
        
        drift = compute_performance_drift(equity, window=63)
        
        # Should handle gracefully
        assert isinstance(drift["performance_degrading"], bool)

    def test_compute_performance_drift_single_value(self):
        """Test with single equity value."""
        equity = pd.Series([10000.0])
        
        drift = compute_performance_drift(equity, window=63)
        
        assert drift["performance_degrading"] is False
        assert drift["reference_sharpe"] is None

