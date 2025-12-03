"""Tests for ensemble layer combining rule-based signals with meta-model confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.ensemble import apply_meta_filter, apply_meta_scaling


# Check if sklearn and joblib are available
try:
    import sklearn
    import joblib
    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False

# Create a fake MetaModel for testing (works without sklearn)
class FakeMetaModel:
    """Fake MetaModel for testing without sklearn dependency."""
    
    def __init__(self, feature_names: list[str], confidence_scores: pd.Series | None = None):
        self.feature_names = feature_names
        self.label_name = "label"
        self._confidence_scores = confidence_scores
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return deterministic confidence scores for testing."""
        if self._confidence_scores is not None:
            if len(self._confidence_scores) == len(X):
                return self._confidence_scores
            else:
                # Repeat or truncate as needed
                n = len(X)
                if len(self._confidence_scores) >= n:
                    return self._confidence_scores.iloc[:n]
                else:
                    # Repeat last value
                    repeated = list(self._confidence_scores.values) + [self._confidence_scores.iloc[-1]] * (n - len(self._confidence_scores))
                    return pd.Series(repeated[:n], index=X.index, name="confidence_score")
        else:
            # Default: return 0.5 for all
            return pd.Series([0.5] * len(X), index=X.index, name="confidence_score")


@pytest.mark.phase7
class TestApplyMetaFilter:
    """Tests for apply_meta_filter function."""
    
    @pytest.fixture
    def sample_signals(self) -> pd.DataFrame:
        """Create sample signals DataFrame."""
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "symbol": ["AAPL"] * 10,
            "direction": ["LONG", "LONG", "FLAT", "LONG", "FLAT", "LONG", "LONG", "FLAT", "LONG", "FLAT"],
            "score": [0.8, 0.9, 0.0, 0.7, 0.0, 0.6, 0.5, 0.0, 0.4, 0.0],
        })
    
    @pytest.fixture
    def sample_features(self) -> pd.DataFrame:
        """Create sample features DataFrame."""
        return pd.DataFrame({
            "feature_1": np.random.randn(10),
            "feature_2": np.random.randn(10),
            "ma_20": np.random.randn(10),
        })
    
    def test_apply_meta_filter_filters_low_confidence(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that signals with confidence < min_confidence are filtered (set to FLAT)."""
        # Create fake meta-model that returns varying confidence scores
        confidence_scores = pd.Series([0.9, 0.3, 0.5, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1, 0.5])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2", "ma_20"], confidence_scores=confidence_scores)
        
        # Apply filter with min_confidence=0.5
        result = apply_meta_filter(
            signals=sample_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.5
        )
        
        # Check that meta_confidence column was added
        assert "meta_confidence" in result.columns
        
        # Check that signals with confidence < 0.5 were set to FLAT
        # Original LONG signals at indices 0, 1, 3, 5, 6, 8
        # Confidence scores: 0.9, 0.3, 0.7, 0.6, 0.4, 0.1
        # Signals at indices 1 (0.3), 6 (0.4), 8 (0.1) should be filtered
        assert result.loc[0, "direction"] == "LONG"  # 0.9 >= 0.5
        assert result.loc[1, "direction"] == "FLAT"  # 0.3 < 0.5 (was LONG)
        assert result.loc[3, "direction"] == "LONG"  # 0.7 >= 0.5
        assert result.loc[5, "direction"] == "LONG"  # 0.6 >= 0.5
        assert result.loc[6, "direction"] == "FLAT"  # 0.4 < 0.5 (was LONG)
        assert result.loc[8, "direction"] == "FLAT"  # 0.1 < 0.5 (was LONG)
        
        # FLAT signals should remain FLAT
        assert result.loc[2, "direction"] == "FLAT"
        assert result.loc[4, "direction"] == "FLAT"
    
    def test_apply_meta_filter_preserves_index(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that index/keys are preserved after filtering."""
        confidence_scores = pd.Series([0.6] * len(sample_signals))
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2", "ma_20"], confidence_scores=confidence_scores)
        
        result = apply_meta_filter(
            signals=sample_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.5
        )
        
        # Check that all original rows are present
        assert len(result) == len(sample_signals)
        assert all(result["timestamp"] == sample_signals["timestamp"])
        assert all(result["symbol"] == sample_signals["symbol"])
    
    def test_apply_meta_filter_with_join_keys(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that join_keys work correctly."""
        confidence_scores = pd.Series([0.6] * len(sample_signals))
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2", "ma_20"], confidence_scores=confidence_scores)
        
        # Add timestamp and symbol to features for join
        features_with_keys = sample_features.copy()
        features_with_keys["timestamp"] = sample_signals["timestamp"].values
        features_with_keys["symbol"] = sample_signals["symbol"].values
        
        result = apply_meta_filter(
            signals=sample_signals,
            meta_model=meta_model,
            features=features_with_keys,
            min_confidence=0.5,
            join_keys=["timestamp", "symbol"]
        )
        
        assert "meta_confidence" in result.columns
        assert len(result) == len(sample_signals)
    
    def test_apply_meta_filter_empty_signals(self, sample_features: pd.DataFrame):
        """Test that empty signals DataFrame is handled gracefully."""
        empty_signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2", "ma_20"])
        
        result = apply_meta_filter(
            signals=empty_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.5
        )
        
        assert result.empty
        assert "meta_confidence" in result.columns or len(result) == 0


@pytest.mark.phase7
class TestApplyMetaScaling:
    """Tests for apply_meta_scaling function."""
    
    @pytest.fixture
    def sample_signals(self) -> pd.DataFrame:
        """Create sample signals DataFrame."""
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": ["AAPL"] * 5,
            "direction": ["LONG", "LONG", "LONG", "LONG", "FLAT"],
            "score": [0.8, 0.9, 0.7, 0.6, 0.0],
        })
    
    @pytest.fixture
    def sample_features(self) -> pd.DataFrame:
        """Create sample features DataFrame."""
        return pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
        })
    
    def test_apply_meta_scaling_scales_scores(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that scores are scaled by confidence."""
        confidence_scores = pd.Series([0.9, 0.5, 0.7, 0.3, 0.8])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2"], confidence_scores=confidence_scores)
        
        result = apply_meta_scaling(
            signals=sample_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.0,  # Don't filter
            max_scaling=1.0,
            scale_score=True
        )
        
        assert "meta_confidence" in result.columns
        assert "final_score" in result.columns
        
        # Check that final_score = original_score * confidence
        assert result.loc[0, "final_score"] == pytest.approx(0.8 * 0.9, rel=1e-5)
        assert result.loc[1, "final_score"] == pytest.approx(0.9 * 0.5, rel=1e-5)
        assert result.loc[2, "final_score"] == pytest.approx(0.7 * 0.7, rel=1e-5)
        assert result.loc[3, "final_score"] == pytest.approx(0.6 * 0.3, rel=1e-5)
    
    def test_apply_meta_scaling_respects_max_scaling(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that max_scaling limits the scaling factor."""
        confidence_scores = pd.Series([0.9, 0.5, 0.7, 0.3, 0.8])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2"], confidence_scores=confidence_scores)
        
        result = apply_meta_scaling(
            signals=sample_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.0,
            max_scaling=0.5,  # Limit scaling to 0.5
            scale_score=True
        )
        
        # meta_confidence itself is not clipped (it's the raw confidence)
        # But final_score should be <= original_score * max_scaling
        # Since confidence=0.9 > 0.5, final_score should be <= 0.8 * 0.5 = 0.4
        assert result.loc[0, "final_score"] <= 0.8 * 0.5
        # For confidence=0.5 (which is <= max_scaling), final_score should be <= 0.9 * 0.5 = 0.45
        assert result.loc[1, "final_score"] <= 0.9 * 0.5
    
    def test_apply_meta_scaling_filters_low_confidence(self, sample_signals: pd.DataFrame, sample_features: pd.DataFrame):
        """Test that signals with confidence < min_confidence are still filtered."""
        confidence_scores = pd.Series([0.9, 0.3, 0.7, 0.2, 0.8])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2"], confidence_scores=confidence_scores)
        
        result = apply_meta_scaling(
            signals=sample_signals,
            meta_model=meta_model,
            features=sample_features,
            min_confidence=0.5,  # Filter signals with confidence < 0.5
            max_scaling=1.0,
            scale_score=True
        )
        
        # Signals at indices 1 (0.3) and 3 (0.2) should be filtered
        assert result.loc[1, "direction"] == "FLAT"  # 0.3 < 0.5
        assert result.loc[3, "direction"] == "FLAT"  # 0.2 < 0.5
    
    def test_apply_meta_scaling_creates_score_if_missing(self, sample_features: pd.DataFrame):
        """Test that final_score is created from confidence if original score is missing."""
        signals_no_score = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "symbol": ["AAPL"] * 3,
            "direction": ["LONG", "LONG", "FLAT"],
        })
        
        confidence_scores = pd.Series([0.8, 0.6, 0.4])
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2"], confidence_scores=confidence_scores)
        
        result = apply_meta_scaling(
            signals=signals_no_score,
            meta_model=meta_model,
            features=sample_features.iloc[:3],
            min_confidence=0.0,
            max_scaling=1.0,
            scale_score=True
        )
        
        assert "final_score" in result.columns
        # Final score should equal confidence (since no original score)
        assert result.loc[0, "final_score"] == pytest.approx(0.8, rel=1e-5)


@pytest.mark.phase7
class TestEnsembleIntegration:
    """Integration tests for ensemble layer."""
    
    def test_ensemble_preserves_signal_structure(self):
        """Test that ensemble functions preserve signal DataFrame structure."""
        signals = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": ["AAPL", "MSFT", "GOOGL", "AAPL", "MSFT"],
            "direction": ["LONG", "LONG", "FLAT", "LONG", "FLAT"],
            "score": [0.8, 0.9, 0.0, 0.7, 0.0],
        })
        
        features = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
        })
        
        meta_model = FakeMetaModel(feature_names=["feature_1", "feature_2"])
        
        result = apply_meta_filter(
            signals=signals,
            meta_model=meta_model,
            features=features,
            min_confidence=0.5
        )
        
        # Check that original columns are preserved
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "direction" in result.columns
        assert "score" in result.columns
        assert "meta_confidence" in result.columns
        
        # Check that all rows are present
        assert len(result) == len(signals)

