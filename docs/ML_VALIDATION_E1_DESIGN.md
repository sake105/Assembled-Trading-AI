# ML Validation & Model Comparison – Design Document (E1)

## Overview

**Sprint:** E1 – ML Validation & Model Comparison  
**Phase:** E – ML Validation, Explainability & TCA  
**Status:** 📋 Design Phase

**Goal:** Erweitere die bestehende Factor-Analyse um Machine-Learning-basierte Prädiktionsmodelle für Forward-Returns. Ermöglicht systematischen Vergleich von linearen und nicht-linearen Modellen (Random Forest, etc.) sowie robuste Validation mit Time-Series-Cross-Validation.

**Integration:** Baut direkt auf Phase C1/C2 (IC Engine, Factor Portfolio Returns) und Phase B1/B2 (Alt-Data Factors) auf. Nutzt das gleiche Factor-Panel-Format, keine neuen Data Contracts.

---

## Scope E1

### Kern-Funktionalität

1. **ML-Modell-Training auf Factor-Panels:**
   - Input: Factor-Panel mit `fwd_return_{horizon}d` als Label
   - Models: Linear (Ridge, Lasso), Tree-based (RandomForest)
   - Output: Vorhersagen (`y_pred`) für Forward-Returns

2. **Time-Series-Cross-Validation:**
   - Expanding Window: Training-Daten wachsen über Zeit
   - Rolling Window: Feste Trainings-Größe, rollt über Zeit
   - Strict Train/Test-Split: Kein Data Leakage (keine Zukunfts-Daten im Training)

3. **Evaluations-Metriken:**
   - **Klassische ML-Metriken**: MSE, MAE, R²
   - **Factor-spezifische Metriken**: IC/Rank-IC zwischen Predicted Alpha & Realized Returns
   - **Portfolio-Metriken**: Long/Short-Portfolio aus Predictions (Deciles/Quantiles) + Sharpe

4. **Model Comparison:**
   - Vergleich mehrerer Modelle auf gleichen Daten
   - Tracking von Metriken pro Modell, Horizon, Split
   - Identifikation von Overfitting (Train vs. Test Performance Gap)

### Keine Live-APIs

- **Nur lokale Factor-Panels**: Alle Daten aus bereits vorhandenen Factor-Analyse-Outputs
- **Keine Online-Fetches**: Keine neuen Daten-Downloads im ML-Workflow
- **Wiederverwendung**: Nutzt Factor-Panels aus `analyze_factors` oder Research-Skripten

---

## Data Contracts

### Input: Factor-Panel (bestehendes Format)

**Kein neuer Data Contract** – nutzt das gleiche Format wie C1/C2:

```python
factor_panel_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - fwd_return_{horizon}d: float  # Label-Spalte (z.B. fwd_return_20d)
    - factor_*: float  # Feature-Spalten (z.B. returns_12m, trend_strength_200, rv_20, ...)
    - Optional: weitere Metadaten-Spalten (z.B. close für Normalisierung)

Format:
    - Panel-Format (nicht Wide-Format)
    - Kein MultiIndex
    - Sortiert nach symbol, dann timestamp (aufsteigend)
    - Missing Values: NaN für fehlende Faktoren oder Forward-Returns
```

**Beispiel:**
```
timestamp              symbol  fwd_return_20d  returns_12m  trend_strength_200  rv_20  earnings_eps_surprise_last
2020-01-01 00:00:00+00:00  AAPL      0.02            0.15          0.5           0.20   0.05
2020-01-01 00:00:00+00:00  MSFT      0.01            0.12          0.3           0.18   0.03
2020-01-01 00:00:00+00:00  GOOG      0.03            0.18          0.7           0.22   NaN
2020-01-02 00:00:00+00:00  AAPL      0.015           0.16          0.52          0.21   0.05
...
```

### Output: ML Predictions

```python
ml_predictions_df: pd.DataFrame
    - timestamp: pd.Timestamp
    - symbol: str
    - y_true: float  # Tatsächliche Forward-Returns
    - y_pred: float  # Modell-Vorhersage
    - y_resid: float (optional)  # Residual = y_true - y_pred
    - split_index: int (optional)  # Index des CV-Splits
    - model_name: str (optional)  # Name des Modells
```

**Beispiel:**
```
timestamp              symbol  y_true   y_pred   y_resid   split_index  model_name
2020-06-01 00:00:00+00:00  AAPL   0.02    0.018   0.002    0            ridge_20d
2020-06-01 00:00:00+00:00  MSFT   0.01    0.012   -0.002   0            ridge_20d
...
```

### Output: ML Metrics

```python
ml_metrics_df: pd.DataFrame
    - model_name: str  # z.B. "ridge_20d", "random_forest_20d"
    - horizon_days: int  # z.B. 20
    - split_index: int  # Index des CV-Splits (-1 für global/aggregiert)
    - mse: float
    - mae: float
    - r2: float
    - ic_mean: float | None  # Mean IC zwischen y_pred und y_true (cross-sectional)
    - ic_ir: float | None  # IC Information Ratio
    - rank_ic_mean: float | None  # Mean Rank-IC
    - rank_ic_ir: float | None  # Rank-IC Information Ratio
    - ls_sharpe: float | None  # Long/Short Portfolio Sharpe (aus Predictions)
    - ls_annualized_return: float | None
    - n_train_samples: int
    - n_test_samples: int
    - train_start: pd.Timestamp
    - train_end: pd.Timestamp
    - test_start: pd.Timestamp
    - test_end: pd.Timestamp
```

**Beispiel:**
```
model_name        horizon_days  split_index  mse      mae      r2       ic_mean  ic_ir   ls_sharpe
ridge_20d         20            -1           0.0005   0.015    0.12     0.08     0.45    0.85
ridge_20d         20            0            0.0006   0.016    0.10     0.07     0.40    0.78
random_forest_20d 20            -1           0.0004   0.014    0.18     0.12     0.60    1.05
random_forest_20d 20            0            0.0007   0.018    0.08     0.06     0.35    0.72
```

### Output: Portfolio Metrics (Optional)

```python
ml_portfolio_metrics_df: pd.DataFrame
    - timestamp: pd.Timestamp
    - model_name: str
    - quantile: int  # z.B. 1 (bottom) bis 5 (top)
    - portfolio_return: float  # Gleichgewichteter Portfolio-Return
    - n_symbols: int
```

**Nutzung:** Für detaillierte Portfolio-Analyse, ähnlich wie in C2 (`build_factor_portfolio_returns()`).

---

## Geplante Funktionen (Interfaces)

### Modul: `src/assembled_core/ml/factor_models.py` (neu)

```python
from dataclasses import dataclass
from typing import Literal
import pandas as pd
from sklearn.base import BaseEstimator

@dataclass
class MLModelConfig:
    """Configuration for an ML model.
    
    Attributes:
        name: Model name (e.g., "ridge_20d", "random_forest_20d")
        model_type: Type of model ("linear", "ridge", "lasso", "random_forest", "gradient_boosting")
        params: Model-specific hyperparameters (dict)
        sklearn_model: Optional pre-instantiated sklearn model (if provided, overrides model_type and params)
    """
    name: str
    model_type: str  # "linear", "ridge", "lasso", "random_forest", "gradient_boosting"
    params: dict = None  # Model-specific hyperparameters
    sklearn_model: BaseEstimator | None = None  # Optional: pre-instantiated model


@dataclass
class MLExperimentConfig:
    """Configuration for an ML experiment.
    
    Attributes:
        label_col: Name of label column (e.g., "fwd_return_20d")
        feature_cols: List of feature column names (None = auto-detect factor_* columns)
        test_start: Optional start date for test set (None = use all data, split by time)
        test_end: Optional end date for test set
        n_splits: Number of time-series CV splits (for expanding/rolling window)
        train_size: Training window size in days (for rolling window, None = expanding)
        step_size: Days to step forward between splits (default: test_size if train_size provided)
        standardize: Whether to standardize features (default: True)
        min_train_samples: Minimum number of samples required for training (default: 100)
    """
    label_col: str
    feature_cols: list[str] | None = None  # None = auto-detect factor_* columns
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    n_splits: int = 5  # Number of CV splits
    train_size: int | None = None  # None = expanding window, int = rolling window size
    step_size: int | None = None  # Days to step forward (default: test_size for rolling, train_size for expanding)
    standardize: bool = True
    min_train_samples: int = 100


def prepare_ml_dataset(
    factor_panel_df: pd.DataFrame,
    experiment: MLExperimentConfig,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter time range, select features & label column, return X, y.
    
    Args:
        factor_panel_df: Factor panel DataFrame (panel format: timestamp, symbol, factor_*, fwd_return_*)
        experiment: MLExperimentConfig with label_col, feature_cols, etc.
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")
    
    Returns:
        Tuple of (X, y):
        - X: DataFrame with features (rows = samples, columns = features)
        - y: Series with labels (same index as X)
        
    Raises:
        ValueError: If label column or required features are missing
        ValueError: If insufficient data after filtering
    """


def run_time_series_cv(
    factor_panel_df: pd.DataFrame,
    experiment: MLExperimentConfig,
    model_cfg: MLModelConfig,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> dict:
    """
    Run time-series cross-validation.
    
    Splits:
    - Expanding Window: Train grows over time (train_size = None)
    - Rolling Window: Fixed train size, rolls over time (train_size = int)
    
    For each split:
    1. Filter train/test data by time range
    2. Prepare X_train, y_train, X_test, y_test
    3. Standardize features (if experiment.standardize = True)
    4. Train model on train set
    5. Predict on test set
    6. Compute metrics (MSE, MAE, R², IC, Rank-IC, Portfolio Sharpe)
    
    Args:
        factor_panel_df: Factor panel DataFrame
        experiment: MLExperimentConfig
        model_cfg: MLModelConfig
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")
    
    Returns:
        Dictionary with:
        - predictions_df: DataFrame with columns: timestamp, symbol, y_true, y_pred, y_resid, split_index
        - metrics_df: DataFrame with columns: model_name, horizon_days, split_index, mse, mae, r2, ic_mean, ic_ir, ...
        - global_metrics: Dictionary with aggregated metrics across all splits (split_index = -1)
        - per_split_metrics: List of dictionaries, one per split
        
    Raises:
        ValueError: If insufficient data for CV splits
    """


def evaluate_ml_predictions(
    predictions_df: pd.DataFrame,
    horizon_days: int,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    """
    Compute evaluation metrics from ML predictions.
    
    Metrics:
    1. **Classical ML Metrics:**
       - MSE, MAE, R² (from y_true vs. y_pred)
    
    2. **Factor-Specific Metrics:**
       - IC (Pearson correlation between y_pred and y_true, cross-sectional per timestamp)
       - Rank-IC (Spearman correlation, cross-sectional per timestamp)
       - IC-IR (mean IC / std IC)
       - Rank-IC-IR (mean Rank-IC / std Rank-IC)
    
    3. **Portfolio Metrics:**
       - Long/Short Portfolio Returns (Top Quintile - Bottom Quintile based on y_pred)
       - Sharpe Ratio of L/S Portfolio
       - Annualized Return of L/S Portfolio
       - Max Drawdown of L/S Portfolio
    
    Args:
        predictions_df: DataFrame with columns: timestamp, symbol, y_true, y_pred (and optional split_index)
        horizon_days: Forward return horizon (for annualization, metadata)
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")
        y_true_col: Name of true labels column (default: "y_true")
        y_pred_col: Name of predicted labels column (default: "y_pred")
    
    Returns:
        Dictionary with metrics:
        - mse: float
        - mae: float
        - r2: float
        - ic_mean: float | None
        - ic_ir: float | None
        - rank_ic_mean: float | None
        - rank_ic_ir: float | None
        - ls_sharpe: float | None
        - ls_annualized_return: float | None
        - ls_max_drawdown: float | None
        - n_samples: int
        - n_timestamps: int
        
    Note:
        - IC/Rank-IC are computed cross-sectionally (per timestamp) and then aggregated
        - Portfolio metrics are computed by sorting symbols by y_pred each timestamp,
          then constructing equal-weighted portfolios in top/bottom quintiles
    """


def compare_ml_models(
    factor_panel_df: pd.DataFrame,
    experiment: MLExperimentConfig,
    model_configs: list[MLModelConfig],
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    Compare multiple ML models on the same dataset.
    
    Runs time-series CV for each model and aggregates results.
    
    Args:
        factor_panel_df: Factor panel DataFrame
        experiment: MLExperimentConfig
        model_configs: List of MLModelConfig (one per model)
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")
    
    Returns:
        DataFrame with comparison metrics:
        - model_name: str
        - horizon_days: int
        - mse, mae, r2: float
        - ic_mean, ic_ir: float | None
        - rank_ic_mean, rank_ic_ir: float | None
        - ls_sharpe: float | None
        - train_test_gap_r2: float  # R² on train - R² on test (overfitting indicator)
        - n_splits: int
        - avg_train_samples: float
        - avg_test_samples: float
        
    Example:
        >>> models = [
        ...     MLModelConfig(name="linear", model_type="linear", params={}),
        ...     MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 0.1}),
        ...     MLModelConfig(name="rf", model_type="random_forest", params={"n_estimators": 100}),
        ... ]
        >>> comparison = compare_ml_models(factor_panel_df, experiment, models)
        >>> print(comparison.sort_values("ls_sharpe", ascending=False))
    """
```

---

## Integration Points

### Wiederverwendung bestehender Module

1. **Factor-Panel-Format**: Gleicher Data Contract wie `qa/factor_analysis.py`
   - Forward Returns bereits vorhanden (via `add_forward_returns()`)
   - Keine neuen Data-Preprocessing-Schritte nötig

2. **IC/Rank-IC-Berechnung**: Wiederverwendung aus `qa/factor_analysis.py`
   - `compute_factor_ic()` / `compute_factor_rank_ic()` können wiederverwendet werden
   - Statt einzelner Faktoren: `y_pred` als "Faktor" verwenden

3. **Portfolio-Metriken**: Ähnlich zu C2 (`qa/factor_analysis.py`)
   - `build_long_short_portfolio_returns()` als Inspiration
   - Statt Faktor-Werten: `y_pred` für Sortierung verwenden

4. **Time-Series-Splitting**: Ähnlich zu `qa/walk_forward.py`
   - `WalkForwardConfig` als Inspiration für Train/Test-Splits
   - Angepasst für ML-Workflow (kein Backtest, nur Feature/Label-Splitting)

### Neue Abhängigkeiten

- **scikit-learn**: Für Ridge, Lasso, RandomForest, StandardScaler
- **Optional**: XGBoost, LightGBM (für GradientBoosting-Modelle)

---

## Implementation Steps (E1.1 – E1.4)

### E1.1: ML Dataset Preparation & Model Config

**Tasks:**
- Erstelle `src/assembled_core/ml/__init__.py`
- Erstelle `src/assembled_core/ml/factor_models.py`
- Implementiere `MLModelConfig`, `MLExperimentConfig` (dataclasses)
- Implementiere `prepare_ml_dataset()`:
  - Feature-Auto-Detection (alle `factor_*` Spalten)
  - Time-Range-Filtering
  - NaN-Handling (drop rows mit missing labels, optional: impute features)
  - X, y Trennung

**Tests:**
- `tests/test_ml_factor_models.py`:
  - `test_prepare_ml_dataset_basic`: Korrekte X/y-Trennung
  - `test_prepare_ml_dataset_auto_detect_features`: Auto-Detection von `factor_*` Spalten
  - `test_prepare_ml_dataset_time_filtering`: Time-Range-Filtering funktioniert

---

### E1.2: Time-Series Cross-Validation

**Tasks:**
- Implementiere `run_time_series_cv()`:
  - Expanding Window Splitting
  - Rolling Window Splitting
  - Model Training (Ridge, Lasso, RandomForest)
  - Feature Standardization (StandardScaler)
  - Prediction auf Test-Set
  - Basic Metrics (MSE, MAE, R²)

**Tests:**
- `test_run_time_series_cv_expanding`: Expanding Window funktioniert
- `test_run_time_series_cv_rolling`: Rolling Window funktioniert
- `test_run_time_series_cv_no_leakage`: Kein Data Leakage (Test-Daten nicht im Training)

---

### E1.3: Factor-Specific & Portfolio Metrics

**Tasks:**
- Implementiere `evaluate_ml_predictions()`:
  - IC/Rank-IC-Berechnung (Wiederverwendung aus `qa/factor_analysis.py`)
  - Long/Short Portfolio aus Predictions (Top-Quintil vs. Bottom-Quintil)
  - Portfolio Sharpe, Annualized Return, Max Drawdown

**Tests:**
- `test_evaluate_ml_predictions_ic`: IC-Metriken stimmen
- `test_evaluate_ml_predictions_portfolio`: Portfolio-Metriken stimmen
- `test_evaluate_ml_predictions_consistency`: Konsistenz mit `compute_factor_ic()`

---

### E1.4: Model Comparison & CLI Integration

**Tasks:**
- Implementiere `compare_ml_models()`
- Erstelle `scripts/train_factor_models.py` (CLI-Script)
- CLI-Integration in `scripts/cli.py`:
  - Subcommand: `train_ml_model` oder `validate_ml_model`
  - Arguments: `--factor-panel-file`, `--label-col`, `--model-type`, `--n-splits`, etc.
- Output: `ml_predictions.csv`, `ml_metrics.csv`, `ml_model_comparison.csv`

**Tests:**
- `tests/test_cli_train_factor_models.py`:
  - `test_cli_train_ml_model_basic`: CLI funktioniert
  - `test_cli_model_comparison`: Model Comparison funktioniert

---

## Usage Examples (geplant)

### Basic ML Model Training

```python
from src.assembled_core.ml.factor_models import (
    MLModelConfig,
    MLExperimentConfig,
    run_time_series_cv,
)

# Load factor panel (from analyze_factors output)
factor_panel_df = pd.read_parquet("output/factor_analysis/ai_tech_factors.parquet")

# Configure experiment
experiment = MLExperimentConfig(
    label_col="fwd_return_20d",
    feature_cols=None,  # Auto-detect factor_* columns
    n_splits=5,
    train_size=None,  # Expanding window
    standardize=True,
)

# Configure model
model_cfg = MLModelConfig(
    name="ridge_20d",
    model_type="ridge",
    params={"alpha": 0.1},
)

# Run CV
result = run_time_series_cv(
    factor_panel_df=factor_panel_df,
    experiment=experiment,
    model_cfg=model_cfg,
)

# Access results
predictions_df = result["predictions_df"]
metrics_df = result["metrics_df"]
global_metrics = result["global_metrics"]

print(f"R²: {global_metrics['r2']:.4f}")
print(f"IC-IR: {global_metrics['ic_ir']:.4f}")
print(f"L/S Sharpe: {global_metrics['ls_sharpe']:.4f}")
```

### Model Comparison

```python
from src.assembled_core.ml.factor_models import compare_ml_models

# Define multiple models
models = [
    MLModelConfig(name="linear", model_type="linear", params={}),
    MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 0.1}),
    MLModelConfig(name="lasso", model_type="lasso", params={"alpha": 0.01}),
    MLModelConfig(
        name="random_forest",
        model_type="random_forest",
        params={"n_estimators": 100, "max_depth": 10}
    ),
]

# Compare
comparison_df = compare_ml_models(
    factor_panel_df=factor_panel_df,
    experiment=experiment,
    model_configs=models,
)

# Sort by L/S Sharpe
print(comparison_df.sort_values("ls_sharpe", ascending=False))
```

### CLI Usage (geplant)

```powershell
# Train single model
python scripts/cli.py train_ml_model `
  --factor-panel-file output/factor_analysis/ai_tech_factors.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --n-splits 5 `
  --output-dir output/ml_models/ridge_20d

# Compare multiple models
python scripts/cli.py compare_ml_models `
  --factor-panel-file output/factor_analysis/ai_tech_factors.parquet `
  --label-col fwd_return_20d `
  --models "linear,ridge,random_forest" `
  --n-splits 5 `
  --output-dir output/ml_models/comparison_20d
```

---

## Dependencies & Requirements

### Python Packages

- **scikit-learn** (>= 1.0.0): Ridge, Lasso, RandomForest, StandardScaler
- **pandas** (bereits vorhanden)
- **numpy** (bereits vorhanden)
- **Optional**: XGBoost, LightGBM (für erweiterte Gradient-Boosting-Modelle)

### Keine neuen Data-Dependencies

- Nutzt vorhandene Factor-Panels (aus `analyze_factors` oder Research-Skripten)
- Keine Live-APIs
- Keine neuen Data-Ingestion-Pipelines

---

## E2 – Explainability & Feature Importance (✅ Completed)

**Status:** Implemented and integrated with E1 workflows

### Overview

Phase E2 extends E1 with explainability tools to understand which factors are most important for ML model predictions. This enables transparency and interpretability of ML-driven signals, and helps bridge the gap between classical factor rankings and ML-based predictions.

### Implemented Functions

**Module:** `src/assembled_core/ml/explainability.py`

1. **`compute_model_feature_importance(model, feature_names)`**
   - Extracts feature importance directly from trained models
   - Supports linear models (uses absolute coefficients)
   - Supports tree models (uses `feature_importances_`)
   - Returns DataFrame with: `feature`, `importance`, `raw_value`, `direction`

2. **`compute_permutation_importance(model, X, y, scoring, n_repeats, random_state)`**
   - Model-agnostic feature importance via permutation
   - Measures how much model performance decreases when a feature is shuffled
   - Returns DataFrame with: `feature`, `importance_mean`, `importance_std`, `importance_median`

3. **`summarize_feature_importance_global(feature_importance_dfs)`**
   - Aggregates feature importance across multiple models (e.g., from Model Zoo)
   - Returns consolidated DataFrame with: `feature`, `importance_mean`, `importance_median`, `importance_max`, `n_models`

### Integration with E1

**Automatic Feature Importance Computation:**
- After `run_time_series_cv()`, the last trained model and feature names are returned
- `run_ml_factor_validation.py` automatically computes feature importance and saves:
  - `ml_feature_importance_{model_type}_{label_col}.csv`
  - Optional: `ml_permutation_importance_{model_type}_{label_col}.csv` (on subsample for cost control)
- Markdown reports include "Top 10 Features" tables with interpretation hints

**Model Zoo Integration:**
- `model_zoo_factor_validation.py` collects feature importance for all models
- Creates global summary: `ml_model_zoo_feature_importance_summary.csv`
- Enables comparison of feature importance across different model types

### Output Files

**Single Model Validation:**
- `ml_feature_importance_{model_type}_{label_col}.csv`
- `ml_permutation_importance_{model_type}_{label_col}.csv` (optional)
- Markdown report includes feature importance section

**Model Zoo:**
- `ml_model_zoo_feature_importance_summary.csv` (aggregated across all models)

### Benefits

- **Transparency:** Understand which factors drive ML predictions
- **Validation:** Compare ML-derived importance with classical factor rankings
- **Feature Selection:** Identify redundant or less important features
- **Model Comparison:** See how different models weight the same features

### Limitations

- Currently supports only sklearn-compatible models
- Permutation importance can be computationally expensive (subsampling recommended)
- No local explanations (SHAP) yet (planned for future)

### References

- Implementation: `src/assembled_core/ml/explainability.py`
- Integration: `scripts/run_ml_factor_validation.py`, `research/ml/model_zoo_factor_validation.py`
- Tests: `tests/test_ml_explainability.py`
- Workflow documentation: [ML Validation & Model Comparison Workflows](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)

---

## Future Enhancements (beyond E2)

### E3: Advanced Models
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks (optional, für komplexe nicht-lineare Patterns)
- Ensemble Methods (Stacking, Blending)

### E4: Hyperparameter Tuning
- Grid Search / Random Search
- Bayesian Optimization (Optuna)
- Nested CV für robuste Hyperparameter-Selection

---

## Integration with Existing Backend

### Phase E in Advanced Analytics Roadmap

- **E1**: ML Validation & Model Comparison (✅ Completed)
- **E2**: Model Explainability & Feature Importance (✅ Completed)
- **E3**: Transaction Cost Analysis (TCA) (Planned)

### Related Backend Phases

- **Phase 7 (ML Meta-Layer)**: Erweitert Meta-Model-Framework um Factor-Model-Validation
- **Phase 9 (Model Governance)**: Nutzt ML-Validation für Model-Approval-Prozess
- **Phase C (Factor Analysis)**: ML-Modelle als Alternative zu IC-basierten Factor-Rankings

---

## References

- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md): Gesamt-Roadmap
- [Factor Analysis Workflows](WORKFLOWS_FACTOR_ANALYSIS.md): Wie man Factor-Panels erzeugt
- [Risk Metrics & Attribution](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md): Risk-Analyse für ML-Modelle (zukünftig)

