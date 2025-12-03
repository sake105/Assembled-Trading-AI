"""Tests for meta-model training, saving, loading, and evaluation."""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

# Check if sklearn and joblib are available
try:
    import sklearn
    import joblib
    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ML_DEPS_AVAILABLE,
    reason="scikit-learn and joblib not installed. Install with: pip install scikit-learn joblib"
)

from src.assembled_core.qa.ml_evaluation import evaluate_meta_model, plot_calibration_curve
from src.assembled_core.signals.meta_model import (
    MetaModel,
    load_meta_model,
    save_meta_model,
    train_meta_model,
)


@pytest.mark.phase7
class TestTrainMetaModel:
    """Tests for train_meta_model function."""
    
    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create a small synthetic dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create features
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "ma_20": np.random.randn(n_samples),
            "rsi_14": np.random.uniform(0, 100, n_samples),
        })
        
        # Create labels (binary, with some correlation to features)
        # Label = 1 if feature_1 + feature_2 > 0 (simplified)
        y = ((X["feature_1"] + X["feature_2"]) > 0).astype(int)
        
        # Combine into dataset
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "symbol": ["AAPL"] * n_samples,
            "label": y,
            **X.to_dict(orient="series"),
        })
        
        return df
    
    def test_train_meta_model_happy_path(self, sample_dataset: pd.DataFrame):
        """Test training meta-model with valid dataset."""
        model = train_meta_model(
            df=sample_dataset,
            feature_cols=None,  # Auto-detect
            label_col="label",
            model_type="gradient_boosting",
            random_state=42
        )
        
        assert isinstance(model, MetaModel)
        assert len(model.feature_names) > 0
        assert model.label_name == "label"
        assert model.model is not None
    
    def test_train_meta_model_predict_proba_values(self, sample_dataset: pd.DataFrame):
        """Test that predict_proba returns values in [0, 1] and no NaNs."""
        model = train_meta_model(
            df=sample_dataset,
            feature_cols=None,
            label_col="label",
            model_type="gradient_boosting",
            random_state=42
        )
        
        X = sample_dataset[model.feature_names]
        y_prob = model.predict_proba(X)
        
        assert len(y_prob) == len(sample_dataset)
        assert y_prob.min() >= 0.0
        assert y_prob.max() <= 1.0
        assert not y_prob.isnull().any()
    
    def test_train_meta_model_random_forest(self, sample_dataset: pd.DataFrame):
        """Test training with random_forest model type."""
        model = train_meta_model(
            df=sample_dataset,
            feature_cols=None,
            label_col="label",
            model_type="random_forest",
            random_state=42
        )
        
        assert isinstance(model, MetaModel)
        assert model.model is not None
        
        # Test prediction
        X = sample_dataset[model.feature_names]
        y_prob = model.predict_proba(X)
        assert len(y_prob) == len(sample_dataset)
        assert (y_prob >= 0).all() and (y_prob <= 1).all()
    
    def test_train_meta_model_explicit_feature_cols(self, sample_dataset: pd.DataFrame):
        """Test training with explicitly provided feature columns."""
        feature_cols = ["feature_1", "feature_2", "ma_20"]
        
        model = train_meta_model(
            df=sample_dataset,
            feature_cols=feature_cols,
            label_col="label",
            model_type="gradient_boosting",
            random_state=42
        )
        
        assert set(model.feature_names) == set(feature_cols)
    
    def test_train_meta_model_missing_label_column(self, sample_dataset: pd.DataFrame):
        """Test that missing label column raises ValueError."""
        df_no_label = sample_dataset.drop(columns=["label"])
        
        with pytest.raises(ValueError, match="Label column 'label' not found"):
            train_meta_model(
                df=df_no_label,
                feature_cols=None,
                label_col="label",
                model_type="gradient_boosting"
            )
    
    def test_train_meta_model_unsupported_model_type(self, sample_dataset: pd.DataFrame):
        """Test that unsupported model type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            train_meta_model(
                df=sample_dataset,
                feature_cols=None,
                label_col="label",
                model_type="unsupported_type"
            )
    
    def test_train_meta_model_single_class(self, sample_dataset: pd.DataFrame):
        """Test that single-class dataset raises ValueError."""
        df_single_class = sample_dataset.copy()
        df_single_class["label"] = 1  # All labels are 1
        
        with pytest.raises(ValueError, match="Need at least 2 classes"):
            train_meta_model(
                df=df_single_class,
                feature_cols=None,
                label_col="label",
                model_type="gradient_boosting"
            )


@pytest.mark.phase7
class TestSaveLoadMetaModel:
    """Tests for save_meta_model and load_meta_model functions."""
    
    @pytest.fixture
    def trained_model(self) -> MetaModel:
        """Create a trained model for testing."""
        np.random.seed(42)
        n_samples = 50
        
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
        })
        y = ((X["feature_1"] + X["feature_2"]) > 0).astype(int)
        
        df = pd.DataFrame({
            "label": y,
            **X.to_dict(orient="series"),
        })
        
        return train_meta_model(
            df=df,
            feature_cols=["feature_1", "feature_2"],
            label_col="label",
            model_type="gradient_boosting",
            random_state=42
        )
    
    def test_save_and_load_meta_model(self, trained_model: MetaModel, tmp_path: pathlib.Path):
        """Test saving and loading a meta-model."""
        model_path = tmp_path / "test_model.joblib"
        
        # Save
        save_meta_model(trained_model, model_path)
        assert model_path.exists()
        
        # Load
        loaded_model = load_meta_model(model_path)
        
        assert isinstance(loaded_model, MetaModel)
        assert loaded_model.feature_names == trained_model.feature_names
        assert loaded_model.label_name == trained_model.label_name
    
    def test_save_and_load_predictions_match(self, trained_model: MetaModel, tmp_path: pathlib.Path):
        """Test that predictions before and after save/load are identical."""
        model_path = tmp_path / "test_model.joblib"
        
        # Create test data
        test_X = pd.DataFrame({
            "feature_1": [0.5, -0.5, 1.0],
            "feature_2": [0.3, -0.3, 0.8],
        })
        
        # Predict before saving
        y_prob_before = trained_model.predict_proba(test_X)
        
        # Save and load
        save_meta_model(trained_model, model_path)
        loaded_model = load_meta_model(model_path)
        
        # Predict after loading
        y_prob_after = loaded_model.predict_proba(test_X)
        
        # Compare (allowing small numerical differences)
        pd.testing.assert_series_equal(
            y_prob_before,
            y_prob_after,
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_load_meta_model_file_not_found(self, tmp_path: pathlib.Path):
        """Test that loading non-existent file raises FileNotFoundError."""
        non_existent_path = tmp_path / "non_existent.joblib"
        
        with pytest.raises(FileNotFoundError):
            load_meta_model(non_existent_path)


@pytest.mark.phase7
class TestEvaluateMetaModel:
    """Tests for evaluate_meta_model function."""
    
    def test_evaluate_meta_model_happy_path(self):
        """Test evaluation with valid inputs."""
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_prob = pd.Series([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.75, 0.15, 0.9])
        
        metrics = evaluate_meta_model(y_true, y_prob)
        
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        
        # Check that values are numeric (not NaN for roc_auc and log_loss if we have both classes)
        assert isinstance(metrics["roc_auc"], float)
        assert isinstance(metrics["brier_score"], float)
        assert isinstance(metrics["log_loss"], float)
        
        # Brier score should be in [0, 1]
        assert 0.0 <= metrics["brier_score"] <= 1.0
    
    def test_evaluate_meta_model_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_prob = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0])
        
        metrics = evaluate_meta_model(y_true, y_prob)
        
        assert metrics["roc_auc"] == 1.0
        assert metrics["brier_score"] == 0.0
        assert metrics["log_loss"] == 0.0
    
    def test_evaluate_meta_model_single_class(self):
        """Test evaluation with single class (should handle gracefully)."""
        y_true = pd.Series([1, 1, 1, 1, 1])
        y_prob = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
        
        metrics = evaluate_meta_model(y_true, y_prob)
        
        # ROC-AUC and Log Loss should be NaN (can't calculate with single class)
        assert pd.isna(metrics["roc_auc"])
        assert pd.isna(metrics["log_loss"])
        # Brier score should still be calculable
        assert isinstance(metrics["brier_score"], float)
    
    def test_evaluate_meta_model_different_lengths(self):
        """Test that different lengths raise ValueError."""
        y_true = pd.Series([0, 1, 1])
        y_prob = pd.Series([0.5, 0.6])
        
        with pytest.raises(ValueError, match="must have same length"):
            evaluate_meta_model(y_true, y_prob)
    
    def test_evaluate_meta_model_invalid_probabilities(self):
        """Test that probabilities outside [0, 1] raise ValueError."""
        y_true = pd.Series([0, 1, 1])
        y_prob = pd.Series([0.5, 1.5, 0.8])  # 1.5 is invalid
        
        with pytest.raises(ValueError, match="must contain values in \\[0, 1\\]"):
            evaluate_meta_model(y_true, y_prob)


@pytest.mark.phase7
class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""
    
    def test_plot_calibration_curve_creates_file(self, tmp_path: pathlib.Path):
        """Test that calibration curve plot is created."""
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 10)  # More samples for better plot
        y_prob = pd.Series([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.75, 0.15, 0.9] * 10)
        
        output_path = tmp_path / "calibration.png"
        
        plot_calibration_curve(y_true, y_prob, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File is not empty
    
    def test_plot_calibration_curve_different_lengths(self, tmp_path: pathlib.Path):
        """Test that different lengths raise ValueError."""
        y_true = pd.Series([0, 1, 1])
        y_prob = pd.Series([0.5, 0.6])
        output_path = tmp_path / "calibration.png"
        
        with pytest.raises(ValueError, match="must have same length"):
            plot_calibration_curve(y_true, y_prob, output_path)
    
    def test_plot_calibration_curve_invalid_probabilities(self, tmp_path: pathlib.Path):
        """Test that probabilities outside [0, 1] raise ValueError."""
        y_true = pd.Series([0, 1, 1])
        y_prob = pd.Series([0.5, 1.5, 0.8])
        output_path = tmp_path / "calibration.png"
        
        with pytest.raises(ValueError, match="must contain values in \\[0, 1\\]"):
            plot_calibration_curve(y_true, y_prob, output_path)


@pytest.mark.phase7
class TestMetaModelPredictProba:
    """Tests for MetaModel.predict_proba method."""
    
    @pytest.fixture
    def trained_model(self) -> MetaModel:
        """Create a trained model for testing."""
        np.random.seed(42)
        n_samples = 50
        
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
        })
        y = ((X["feature_1"] + X["feature_2"]) > 0).astype(int)
        
        df = pd.DataFrame({
            "label": y,
            **X.to_dict(orient="series"),
        })
        
        return train_meta_model(
            df=df,
            feature_cols=["feature_1", "feature_2"],
            label_col="label",
            model_type="gradient_boosting",
            random_state=42
        )
    
    def test_predict_proba_missing_features(self, trained_model: MetaModel):
        """Test that missing features raise ValueError."""
        X_missing = pd.DataFrame({
            "feature_1": [0.5, -0.5],
            # feature_2 is missing
        })
        
        with pytest.raises(ValueError, match="Missing required feature columns"):
            trained_model.predict_proba(X_missing)
    
    def test_predict_proba_handles_nan(self, trained_model: MetaModel):
        """Test that NaN values in features are handled (filled with 0)."""
        X_with_nan = pd.DataFrame({
            "feature_1": [0.5, np.nan, 1.0],
            "feature_2": [0.3, 0.4, np.nan],
        })
        
        # Should not raise (NaN filled with 0)
        y_prob = trained_model.predict_proba(X_with_nan)
        
        assert len(y_prob) == 3
        assert (y_prob >= 0).all() and (y_prob <= 1).all()
        assert not y_prob.isnull().any()

