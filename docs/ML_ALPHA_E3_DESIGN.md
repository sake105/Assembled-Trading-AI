# ML Alpha Factor & Strategy Integration - Design Document (E3)

## Overview

**Sprint:** E3 - ML Alpha Factor & Strategy Integration  
**Phase:** E - ML Validation, Explainability & TCA  
**Status:** 📋 Design Phase

**Goal:** Integrate ML model predictions as alpha factors into the multi-factor strategy framework. Generate `ml_alpha_{model_type}_{horizon}d` columns from trained ML models and make them available as factors in factor bundles, enabling seamless integration with existing multi-factor strategies.

**Integration:** Builds on Phase E1 (ML Validation & Model Comparison) and E2 (Explainability). Uses existing factor panel format and multi-factor strategy infrastructure. No new strategy types required—only new factor bundles and export workflow.

---

## Scope E3

### Core Functionality

1. **ML Alpha Factor Generation:**
   - Input: Factor panel with `fwd_return_{horizon}d` (from `export_factor_panel_for_ml.py`)
   - Process: Train ML model via `run_time_series_cv()` (E1)
   - Output: Factor panel with new column `ml_alpha_{model_type}_{horizon}d` containing predictions (`y_pred`)

2. **Factor Panel Enhancement:**
   - Merge ML predictions back into original factor panel
   - Preserve existing factor columns and forward returns
   - Ensure temporal alignment (timestamp, symbol)

3. **Factor Bundle Integration:**
   - Create new factor bundles that include ML alpha factors
   - Options:
     - **Pure ML bundles:** 100% ML alpha (e.g., `ml_alpha_only`)
     - **Mixed bundles:** Core/Alt factors + ML alpha (e.g., `core+ml`, `core+alt+ml`)

4. **Strategy Integration:**
   - Use existing `multifactor_long_short` strategy
   - Load ML-alpha bundles via `load_factor_bundle()`
   - No changes to strategy code required—ML alpha treated as regular factor

### Keine Live-APIs

- **Nur lokale Factor-Panels:** Alle Daten aus bereits vorhandenen Factor-Panels
- **Wiederverwendung:** Nutzt `run_time_series_cv()` und `export_factor_panel_for_ml.py` aus E1
- **Bestehende Strategie:** Nutzt `multifactor_long_short` ohne Modifikation

---

## Data Contracts

### Input: Factor Panel (from E1)

**Format:** Same as used in E1 (`export_factor_panel_for_ml.py`):

```python
factor_panel_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - fwd_return_{horizon}d: float  # Label column (e.g., fwd_return_20d)
    - factor_*: float  # Feature columns (e.g., returns_12m, trend_strength_200, rv_20, ...)
    - Optional: Alt-Data factor columns (earnings_*, insider_*, news_*, macro_*)

Format:
    - Panel format (not Wide format)
    - Sorted by timestamp, then symbol (ascending)
    - UTC timezone for all timestamps
```

**Example:**
```
timestamp              symbol  fwd_return_20d  returns_12m  trend_strength_200  rv_20
2020-01-01 00:00:00+00:00  AAPL      0.02            0.15          0.5           0.20
2020-01-01 00:00:00+00:00  MSFT      0.01            0.12          0.3           0.18
2020-01-02 00:00:00+00:00  AAPL      0.015           0.16          0.52          0.21
...
```

### Intermediate: ML Predictions (from `run_time_series_cv()`)

**Format:** Output from `run_time_series_cv()` (E1):

```python
cv_result: dict
    - predictions_df: pd.DataFrame
        - timestamp: pd.Timestamp
        - symbol: str
        - y_true: float  # Actual forward returns
        - y_pred: float  # ML model predictions (this becomes ml_alpha_*)
        - split_index: int (optional)
    - metrics_df: pd.DataFrame
    - global_metrics: dict
    - last_model: trained sklearn model (optional, for feature importance)
    - feature_names: list[str] (optional)
```

**Example:**
```
timestamp              symbol  y_true   y_pred   split_index
2020-06-01 00:00:00+00:00  AAPL   0.02    0.018   0
2020-06-01 00:00:00+00:00  MSFT   0.01    0.012   0
...
```

### Output: ML Alpha Factor Panel

**Format:**

```python
ml_alpha_panel_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - ml_alpha_{model_type}_{horizon}d: float  # ML predictions (y_pred merged from predictions_df)
    - fwd_return_{horizon}d: float  # Original label column (optional, for validation)
    - factor_*: float  # Original factor columns (preserved)
    - Optional: Alt-Data factor columns (preserved)

Format:
    - Panel format (same as input)
    - Sorted by timestamp, then symbol (ascending)
    - UTC timezone for all timestamps
```

**Example:**
```
timestamp              symbol  ml_alpha_ridge_20d  fwd_return_20d  returns_12m  trend_strength_200
2020-06-01 00:00:00+00:00  AAPL      0.018            0.02           0.15          0.5
2020-06-01 00:00:00+00:00  MSFT      0.012            0.01           0.12          0.3
...
```

**Naming Convention:**
- `ml_alpha_{model_type}_{horizon}d`
- Examples:
  - `ml_alpha_ridge_20d` (Ridge regression, 20-day horizon)
  - `ml_alpha_random_forest_20d` (Random Forest, 20-day horizon)
  - `ml_alpha_ridge_60d` (Ridge regression, 60-day horizon)

---

## Planned Components

### E3.1: ML Alpha Export Script

**File:** `research/ml/export_ml_alpha_factor.py`

**Purpose:** Generate ML alpha factor panels from factor panels and ML models.

**Workflow:**

1. **Load Factor Panel:**
   - Input: Parquet/CSV file (from `export_factor_panel_for_ml.py`)
   - Validate format (timestamp, symbol, `fwd_return_{horizon}d`, factor columns)

2. **Build ML Configuration:**
   - `MLExperimentConfig`: From CLI args or defaults
     - `label_col`: Auto-detect from `fwd_return_*` columns or CLI arg
     - `feature_cols`: Auto-detect factor columns (default: None)
     - `n_splits`: Default 5, configurable
     - `train_size`: Default None (expanding window), configurable
     - `standardize`: Default True
   - `MLModelConfig`: From CLI args
     - `model_type`: "linear", "ridge", "lasso", "random_forest"
     - `params`: Model hyperparameters (dict)

3. **Run Time-Series Cross-Validation:**
   - Call `run_time_series_cv(factor_panel_df, experiment_config, model_config)`
   - Get `predictions_df` from result
   - Extract `y_pred` column (this becomes the ML alpha factor)

4. **Merge Predictions into Panel:**
   - Join `predictions_df` with original `factor_panel_df` on `timestamp` and `symbol`
   - Create new column: `ml_alpha_{model_type}_{horizon}d` = `y_pred`
   - Preserve all original columns (factors, forward returns, etc.)

5. **Handle Missing Predictions:**
   - Some rows may not have predictions (e.g., insufficient training data in early periods)
   - Option 1: Keep as NaN (recommended for transparency)
   - Option 2: Forward-fill or interpolate (optional, configurable)

6. **Save Output:**
   - Save as Parquet: `output/ml_alpha_factors/ml_alpha_{model_type}_{horizon}d_{universe}_{freq}.parquet`
   - Optional: Save as CSV for inspection
   - Include metadata file (JSON) with model config, experiment config, metrics summary

**CLI Arguments:**

```bash
python research/ml/export_ml_alpha_factor.py \
  --factor-panel-file output/factor_panels/factor_panel_macro_world_etfs_core_20d_1d.parquet \
  --model-type ridge \
  --model-param alpha=0.1 \
  --n-splits 5 \
  --output-file output/ml_alpha_factors/ml_alpha_ridge_20d_macro_world_etfs.parquet \
  --label-col fwd_return_20d  # Optional: auto-detect if not provided
```

**Output Files:**

- `ml_alpha_{model_type}_{horizon}d_{universe}_{freq}.parquet`: Enhanced factor panel with ML alpha column
- `ml_alpha_{model_type}_{horizon}d_{universe}_{freq}_metadata.json`: Model config, experiment config, metrics summary

---

### E3.2: Factor Bundle Configuration

**Location:** `config/factor_bundles/`

**New Bundle Files:**

1. **`ai_tech_ml_alpha_bundle.yaml`** (100% ML Alpha):
```yaml
universe: universe_ai_tech
factor_set: ml_alpha_only
horizon_days: 20

factors:
  - name: ml_alpha_ridge_20d
    weight: 1.0
    direction: positive  # Higher predictions = better (long)

options:
  winsorize: true
  winsorize_limits: [0.01, 0.99]
  zscore: true
  neutralize_by: null
```

2. **`ai_tech_core_ml_bundle.yaml`** (Core + ML Alpha Mix):
```yaml
universe: universe_ai_tech
factor_set: core+ml
horizon_days: 20

factors:
  - name: returns_12m
    weight: 0.3
    direction: positive
  - name: trend_strength_200
    weight: 0.2
    direction: positive
  - name: rv_20
    weight: 0.2
    direction: negative  # Lower volatility = better
  - name: ml_alpha_ridge_20d
    weight: 0.3
    direction: positive

options:
  winsorize: true
  winsorize_limits: [0.01, 0.99]
  zscore: true
  neutralize_by: null
```

3. **`macro_world_etfs_ml_alpha_bundle.yaml`** (Macro ETFs, 100% ML Alpha):
```yaml
universe: macro_world_etfs
factor_set: ml_alpha_only
horizon_days: 20

factors:
  - name: ml_alpha_ridge_20d
    weight: 1.0
    direction: positive

options:
  winsorize: true
  winsorize_limits: [0.01, 0.99]
  zscore: true
  neutralize_by: null
```

**Bundle Structure:**
- Same format as existing bundles (`core_ta_price.yaml`, `ai_tech_core_alt_bundle.yaml`)
- ML alpha factors follow same naming convention: `ml_alpha_{model_type}_{horizon}d`
- Direction: Always `positive` (higher predictions = better returns = long position)
- Weights: Must sum to 1.0 (same validation as existing bundles)

---

### E3.3: Strategy Integration

**No Changes Required:** The existing `multifactor_long_short` strategy already supports ML alpha factors through the factor bundle system.

**Workflow:**

1. **Generate ML Alpha Factor Panel:**
   ```bash
   python research/ml/export_ml_alpha_factor.py \
     --factor-panel-file output/factor_panels/factor_panel_ai_tech_core+alt_full_20d_1d.parquet \
     --model-type ridge \
     --model-param alpha=0.1
   ```

2. **Create/Use ML Alpha Bundle:**
   - Bundle file: `config/factor_bundles/ai_tech_ml_alpha_bundle.yaml`
   - References: `ml_alpha_ridge_20d` as factor

3. **Run Strategy with ML Alpha Bundle:**
   ```bash
   python scripts/run_backtest_strategy.py \
     --strategy multifactor_long_short \
     --freq 1d \
     --factor-bundle config/factor_bundles/ai_tech_ml_alpha_bundle.yaml \
     --start-date 2015-01-01 \
     --end-date 2025-12-03
   ```

**How It Works:**

1. Strategy loads factor bundle via `load_factor_bundle()`
2. Strategy loads factor panel (must include `ml_alpha_ridge_20d` column)
3. `build_multifactor_signal()` processes ML alpha factor:
   - Winsorizes if enabled
   - Z-scores cross-sectionally (per timestamp)
   - Applies direction (positive = no inversion)
   - Computes weighted score (in pure ML bundle: score = z-scored ML alpha)
4. Strategy selects top/bottom quantiles based on `mf_score`
5. Strategy generates positions and runs backtest

**Key Points:**
- ML alpha factors are treated exactly like regular factors
- Same processing pipeline (winsorization, z-scoring, weighting)
- Same backtest infrastructure
- No special handling required in strategy code

---

## Implementation Steps

### E3.1: Export Script & Core Logic

**Tasks:**
1. Create `research/ml/export_ml_alpha_factor.py`
2. Implement factor panel loading and validation
3. Implement ML configuration building (from CLI args)
4. Integrate `run_time_series_cv()` from E1
5. Implement predictions merging logic
6. Implement output file writing (Parquet + metadata JSON)
7. Add CLI argument parsing
8. Add error handling and logging

**Acceptance Criteria:**
- Script successfully generates ML alpha factor panels
- Output panel contains `ml_alpha_{model_type}_{horizon}d` column
- Original factor columns and forward returns are preserved
- Metadata JSON contains model and experiment configuration

---

### E3.2: Factor Bundle Configuration

**Tasks:**
1. Create `config/factor_bundles/ai_tech_ml_alpha_bundle.yaml`
2. Create `config/factor_bundles/ai_tech_core_ml_bundle.yaml` (optional)
3. Create `config/factor_bundles/macro_world_etfs_ml_alpha_bundle.yaml` (optional)
4. Validate bundle structure (weights sum to 1.0, valid directions)
5. Test bundle loading via `load_factor_bundle()`

**Acceptance Criteria:**
- Bundles load successfully via `load_factor_bundle()`
- Bundle validation passes (weight sum, direction checks)
- Bundle metadata (universe, factor_set, horizon_days) is correct

---

### E3.3: Tests

**Tasks:**
1. Create `tests/test_ml_alpha_export.py`
   - Test factor panel loading
   - Test ML configuration building
   - Test predictions merging
   - Test output file generation
   - Test metadata JSON structure
2. Create integration test:
   - Generate ML alpha factor panel
   - Load ML alpha bundle
   - Run `build_multifactor_signal()` with ML alpha factor
   - Verify signal generation works correctly

**Acceptance Criteria:**
- All unit tests pass
- Integration test confirms ML alpha factors work in signal generation
- Edge cases handled (missing predictions, NaN values)

---

### E3.4: Documentation & Examples

**Tasks:**
1. Update `docs/WORKFLOWS_STRATEGIES_MULTIFACTOR.md`:
   - Add section "Using ML Alpha Factors"
   - Include CLI examples for ML alpha export
   - Include examples of ML alpha bundles
   - Include backtest examples with ML alpha bundles
2. Create example workflow document:
   - End-to-end example: Factor panel → ML alpha export → Bundle → Backtest
   - Comparison: Pure ML alpha vs. Core factors vs. Mixed bundle
3. Update `README.md`:
   - Add ML Alpha Factor Generation to usage examples

**Acceptance Criteria:**
- Documentation is complete and accurate
- Examples are runnable and tested
- Workflow is clearly explained

---

## Integration with Existing System

### Data Flow

```
┌─────────────────────────────────┐
│ Factor Panel Export (E1)        │
│ export_factor_panel_for_ml.py   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Factor Panel                    │
│ (timestamp, symbol,             │
│  fwd_return_20d, factor_*)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ ML Alpha Export (E3.1)          │
│ export_ml_alpha_factor.py       │
│ - run_time_series_cv()          │
│ - Merge y_pred → ml_alpha_*     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ ML Alpha Factor Panel           │
│ (timestamp, symbol,             │
│  ml_alpha_ridge_20d,            │
│  fwd_return_20d, factor_*)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Factor Bundle (E3.2)            │
│ ai_tech_ml_alpha_bundle.yaml    │
│ - ml_alpha_ridge_20d: weight=1.0│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Multi-Factor Strategy           │
│ multifactor_long_short          │
│ - build_multifactor_signal()    │
│ - Select top/bottom quantiles   │
│ - Generate positions            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Backtest Results                │
│ - Equity curve                  │
│ - Positions                     │
│ - Performance metrics           │
└─────────────────────────────────┘
```

### Module Dependencies

**E3 uses:**
- **E1 (`src/assembled_core/ml/factor_models.py`):**
  - `run_time_series_cv()`: Train ML models and get predictions
  - `MLModelConfig`, `MLExperimentConfig`: Model and experiment configuration
  - `prepare_ml_dataset()`: Dataset preparation (internal to `run_time_series_cv()`)

- **E1 (`research/factors/export_factor_panel_for_ml.py`):**
  - Generates input factor panels with forward returns
  - Reuses factor computation logic from `run_factor_analysis.py`

- **Existing (`src/assembled_core/config/factor_bundles.py`):**
  - `load_factor_bundle()`: Load bundle configurations
  - `FactorBundleConfig`: Bundle data structure

- **Existing (`src/assembled_core/signals/multifactor_signal.py`):**
  - `build_multifactor_signal()`: Process factors according to bundle
  - `MultiFactorSignalResult`: Signal generation result

- **Existing (`src/assembled_core/strategies/multifactor_long_short.py`):**
  - `MultiFactorStrategyConfig`: Strategy configuration
  - `generate_multifactor_long_short_signals()`: Generate trading signals
  - `compute_multifactor_long_short_positions()`: Compute positions from signals

**E3 provides:**
- `research/ml/export_ml_alpha_factor.py`: ML alpha factor generation script
- `config/factor_bundles/*_ml_alpha_bundle.yaml`: ML alpha factor bundle configurations
- Enhanced factor panels with `ml_alpha_*` columns

---

## Usage Examples (Planned)

### Example 1: Pure ML Alpha Strategy

**Step 1: Generate Factor Panel**
```bash
python research/factors/export_factor_panel_for_ml.py \
  --freq 1d \
  --universe config/universe_ai_tech_tickers.txt \
  --factor-set core+alt_full \
  --horizon-days 20 \
  --start-date 2015-01-01 \
  --end-date 2025-12-03
```

**Step 2: Generate ML Alpha Factor**
```bash
python research/ml/export_ml_alpha_factor.py \
  --factor-panel-file output/factor_panels/factor_panel_universe_ai_tech_core+alt_full_20d_1d.parquet \
  --model-type ridge \
  --model-param alpha=0.1 \
  --n-splits 5 \
  --label-col fwd_return_20d
```

**Step 3: Run Strategy with ML Alpha Bundle**
```bash
python scripts/run_backtest_strategy.py \
  --strategy multifactor_long_short \
  --freq 1d \
  --factor-bundle config/factor_bundles/ai_tech_ml_alpha_bundle.yaml \
  --factor-file output/ml_alpha_factors/ml_alpha_ridge_20d_universe_ai_tech_1d.parquet \
  --start-date 2015-01-01 \
  --end-date 2025-12-03
```

---

### Example 2: Mixed Bundle (Core + ML Alpha)

**Step 1-2: Same as Example 1**

**Step 3: Run Strategy with Mixed Bundle**
```bash
python scripts/run_backtest_strategy.py \
  --strategy multifactor_long_short \
  --freq 1d \
  --factor-bundle config/factor_bundles/ai_tech_core_ml_bundle.yaml \
  --factor-file output/ml_alpha_factors/ml_alpha_ridge_20d_universe_ai_tech_1d.parquet \
  --start-date 2015-01-01 \
  --end-date 2025-12-03
```

**Note:** The mixed bundle requires both ML alpha factors and core factors in the same panel. The factor panel from `export_ml_alpha_factor.py` includes both.

---

## Future Enhancements (beyond E3)

### E3.1: Multiple Model Types

- Generate multiple ML alpha factors from different models (Ridge, Lasso, Random Forest)
- Create bundles that combine multiple ML alpha factors
- Example: `ml_alpha_ensemble_20d` = weighted average of `ml_alpha_ridge_20d` + `ml_alpha_rf_20d`

### E3.2: Model Selection & Retraining

- Automatic model selection based on validation metrics (IC-IR, Sharpe)
- Periodic retraining (e.g., quarterly) with updated data
- Model versioning and tracking

### E3.3: Real-Time Prediction

- Generate ML alpha factors for live data
- Integration with paper-trading and live trading systems
- Streaming predictions via API

---

## References

- [ML Validation & Model Comparison Design (E1)](ML_VALIDATION_E1_DESIGN.md)
- [ML Validation Workflows](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)
- [Multi-Factor Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md)
- [Factor Ranking Overview](FACTOR_RANKING_OVERVIEW.md)
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

