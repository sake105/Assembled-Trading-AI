# ML Validation & Model Comparison Workflows (E1)

**Last Updated:** 2025-12-13  
**Status:** Active Workflows for ML Model Validation on Factor Panels

---

## Overview

This document describes workflows for ML validation and model comparison on factor panels. This functionality is part of **Phase E: ML Validation, Explainability & TCA** in the [Advanced Analytics Factor Labs Plan](ADVANCED_ANALYTICS_FACTOR_LABS.md).

### How E1 Fits into the Factor Labs Roadmap

**E1 (ML Validation & Model Comparison)** builds on top of existing factor analysis infrastructure:

- **Prerequisites:**
  - **Phase C1/C2**: IC-based and portfolio-based factor evaluation
  - **Phase B1/B2**: Alt-Data factors (Earnings, Insider, News, Macro)
  - **Phase A1/A2**: Core TA/Price and Volatility/Liquidity factors

- **Goal:** 
  - Train ML models to predict forward returns from factor panels
  - Compare different model types (Linear, Ridge, Lasso, Random Forest)
  - Evaluate model performance using time-series cross-validation
  - Assess predictive power using IC/Rank-IC and Long/Short portfolio metrics

- **Future Integration:**
  - **E2/E3 (Future)**: Regime-specific ML models, explainability tools
  - **D3 (Future)**: Use ML predictions for adaptive factor selection
  - **Multi-Factor Strategy**: Integrate ML predictions into signal generation

### Key Features

1. **Time-Series Cross-Validation:**
   - Expanding or rolling window splits
   - Prevents data leakage by respecting temporal order
   - Per-split and global metrics

2. **Model Comparison:**
   - Multiple model types: Linear, Ridge, Lasso, Random Forest
   - Hyperparameter tuning via CLI arguments
   - Systematic comparison of model performance

3. **Comprehensive Metrics:**
   - **Classical ML Metrics:** MSE, MAE, R², Directional Accuracy
   - **Factor-Specific Metrics:** IC, Rank-IC, IC-IR, Rank-IC-IR
   - **Portfolio Metrics:** Long/Short Sharpe, Return, Volatility

---

## Inputs & Prerequisites

### Required: Factor Panels

**Factor Panel Format:**
- **Columns:**
  - `timestamp` (datetime, UTC)
  - `symbol` (string)
  - `fwd_return_{horizon}d` (e.g., `fwd_return_20d`) - Label column
  - `factor_*` columns (e.g., `factor_mom`, `factor_value`, `factor_quality`)
  - Optional: Alt-Data factor columns (`returns_*`, `earnings_*`, `insider_*`, `news_*`, `macro_*`)

- **Structure:**
  - Panel format (one row per timestamp × symbol combination)
  - Sorted by `symbol`, then `timestamp`
  - No MultiIndex required

**Sources for Factor Panels:**

1. **From `analyze_factors` Workflow:**
   - Factor panels are generated automatically during factor analysis
   - Can be saved and reused for ML validation
   - See [Workflows – Factor Analysis](WORKFLOWS_FACTOR_ANALYSIS.md)

2. **From Research Scripts:**
   - Custom factor panels created in research notebooks
   - Export to Parquet format for ML validation

3. **Dedicated Panel Export Scripts:**
   - `research/factors/export_factor_panel_for_ml.py`: Export factor panels for ML experiments
   - Once integrated, factor panels can also be loaded from the factor store (P2) using `load_factors()` from `src.assembled_core.data.factor_store`, avoiding recomputation of factors for ML validation runs. See [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md) for details.

### Important: Only Local Data

**All data processing is offline:**
- ✅ Local factor panel files (Parquet/CSV)
- ✅ Local alt-data snapshots (Parquet files)
- ❌ **No live price APIs** (Yahoo Finance, Twelve Data, etc.)
- ❌ **No online fetches** in the ML validation workflow itself

The ML validation workflow is fully **offline** and works only with pre-computed factor panels stored locally.

---

## Standard Workflows

### 1. Basic ML Validation Run (Single Model)

**Use Case:** Validate a single model on a factor panel.

**Example:**
```powershell
# Using a factor panel from macro_world_etfs analysis
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/factor_analysis_core_20d_1d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type linear `
  --n-splits 5 `
  --output-dir output/ml_validation
```

**Output Files:**
- `ml_metrics_linear_fwd_return_20d.csv`: Per-split and global metrics
- `ml_predictions_sample_linear_fwd_return_20d.parquet`: Sample predictions (max 10k rows)
- `ml_portfolio_metrics_linear_fwd_return_20d.csv`: Portfolio metrics (IC, Sharpe, etc.)
- `ml_validation_report_linear_fwd_return_20d.md`: Markdown report with interpretation

**Report Contents:**
- Classical ML metrics (MSE, MAE, R², Directional Accuracy)
- Factor-specific metrics (IC, Rank-IC, IC-IR, Rank-IC-IR)
- Portfolio metrics (L/S Sharpe, L/S Return)
- Interpretation guidelines

---

### 2. Model Zoo – Mehrere Modelle auf einem Panel

**Use Case:** Automatisch mehrere Modelle auf demselben Factor-Panel vergleichen.

**Was das Model-Zoo-Script macht:**
- Führt automatisch eine vordefinierte Liste von ML-Modellen aus (Linear, Ridge mit verschiedenen Alpha-Werten, Lasso, Random Forest)
- Nutzt für alle Modelle dieselbe `MLExperimentConfig` (fairer Vergleich)
- Sammelt Metriken für jedes Modell
- Erstellt eine Vergleichstabelle (CSV + optional Markdown)
- Sortiert Modelle nach IC-IR (oder R²)

**Vorteil gegenüber manuellem Multi-Model-Run:**
- Ein einziger Command statt mehrerer separater Runs
- Garantiert gleiche Experiment-Konfiguration für alle Modelle
- Automatische Aggregation und Sortierung

**Example:**
```powershell
python scripts/cli.py ml_model_zoo `
  --factor-panel-file output/factor_panels/factor_panel_custom_AAPL_MSFT_core+vol_liquidity_20d_1d.parquet `
  --label-col fwd_return_20d `
  --output-dir output/ml_model_zoo/ai_tech_core_alt
```

**Mit optionalen Parametern:**
```powershell
python scripts/cli.py ml_model_zoo `
  --factor-panel-file output/factor_panels/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --n-splits 10 `
  --test-start 2020-01-01 `
  --test-end 2024-12-31 `
  --output-dir output/ml_model_zoo/custom_run
```

**Output Files:**
- `ml_model_zoo_summary.csv`: Vergleichstabelle mit allen Modellen (eine Zeile pro Modell)
- `ml_model_zoo_summary.md`: Markdown-Report mit Top Models, Vergleichstabelle, und Interpretation

**Interpreting the Summary Table:**

Die `ml_model_zoo_summary.csv` enthält pro Modell:

- **`model_name`**: Name des Modells (z.B. `ridge_0_1`, `rf_depth_5`)
- **`model_type`**: Modell-Typ (`linear`, `ridge`, `lasso`, `random_forest`)
- **`test_r2_mean`**: Durchschnittlicher Test-R² über alle CV-Splits
  - **Vergleich:** Höhere R² = bessere Modell-Performance (aber bei Returns oft niedrig)
- **`ic_mean`**: Durchschnittlicher Information Coefficient
- **`ic_ir`**: IC Information Ratio (IC-StdDev / IC-Mean)
  - **Vergleich:** IC-IR > 0.5 = konsistenter Signal über Zeit
  - **Best Practice:** Sortiere nach IC-IR für beste Modelle
- **`rank_ic_mean`**, **`rank_ic_ir`**: Rank-IC Metriken (robust gegen Outliers)
- **`ls_sharpe`**: Long/Short Portfolio Sharpe Ratio (falls genug Symbole vorhanden)
  - **Vergleich:** L/S Sharpe > 1.0 = gutes Trading-Signal
  - **Hinweis:** Kann `NaN` sein, wenn zu wenige Symbole für Portfolio-Konstruktion
- **`train_test_gap_r2`**: Gap zwischen Train- und Test-R²
  - **Vergleich:** Großer Gap (> 0.1) deutet auf Overfitting hin

**Empfohlene Analyse:**
1. **Sortiere nach IC-IR** (wird automatisch gemacht)
2. **Prüfe L/S Sharpe** für Portfolio-Relevanz
3. **Vergleiche Train/Test Gap** für Overfitting-Detection
4. **Wähle Modell mit:**
   - Hohem IC-IR (> 0.5)
   - Realistischem L/S Sharpe (falls verfügbar)
   - Kleinem Train/Test Gap (< 0.1)

**Wichtig:** Alle verwendeten Factor-Panels sind **lokal** (Parquet/CSV), keine Live-APIs werden im Model-Zoo-Workflow verwendet.

---

### 3. Manual Model Comparison (Alternative)

**Use Case:** Manueller Vergleich mehrerer Modelle mit unterschiedlichen Konfigurationen.

**Workflow:**

1. **Run multiple models individually:**
   ```powershell
   # Linear model
   python scripts/cli.py ml_validate_factors `
     --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
     --label-col fwd_return_20d `
     --model-type linear `
     --output-dir output/ml_validation
   
   # Ridge regression
   python scripts/cli.py ml_validate_factors `
     --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
     --label-col fwd_return_20d `
     --model-type ridge `
     --model-param alpha=0.1 `
     --output-dir output/ml_validation
   
   # Lasso regression
   python scripts/cli.py ml_validate_factors `
     --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
     --label-col fwd_return_20d `
     --model-type lasso `
     --model-param alpha=0.01 `
     --output-dir output/ml_validation
   
   # Random Forest
   python scripts/cli.py ml_validate_factors `
     --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
     --label-col fwd_return_20d `
     --model-type random_forest `
     --model-param n_estimators=100 `
     --model-param max_depth=10 `
     --output-dir output/ml_validation
   ```

2. **Compare results manually:**
   - Review `ml_portfolio_metrics_*.csv` files from each run
   - Rank models by IC-IR or L/S Sharpe
   - Check consistency across CV splits

**Note:** For standardized comparison, prefer the **Model Zoo** workflow (Section 2) over manual runs.

---

### 4. Time Range Filtering

**Use Case:** Validate models on specific time periods.

**Example:**
```powershell
# Focus on recent data (2020-2024)
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --test-start 2020-01-01 `
  --test-end 2024-12-31 `
  --n-splits 10 `
  --output-dir output/ml_validation
```

**Use Cases:**
- Out-of-sample validation on recent data
- Regime-specific validation (e.g., bull vs. bear markets)
- Sensitivity analysis across different time periods

---

### 5. Integration with Factor Ranking & Regimes

**Current State (Conceptual):**

While ML validation and factor ranking are currently separate workflows, they can be interpreted together:

1. **Cross-Validation:**
   - Identify factors that rank highly in both classical IC/IR analysis AND have high feature importance in ML models
   - Check for consistency between factor rankings and ML model coefficients/importances

2. **Portfolio Construction:**
   - Use ML predictions as additional signal alongside factor bundles
   - Combine ML-based predictions with multifactor signal generation

**Future Integration (E2/E3):**

- **Regime-Specific ML Models:**
  - Train separate ML models for each identified regime (Bull, Bear, Crisis, Neutral)
  - Compare model performance across regimes
  - Adaptive model selection based on current regime

- **ML Explainability:**
  - Feature importance analysis (permutation importance, SHAP values)
  - Factor contribution attribution
  - Model interpretability for regulatory compliance

---

## Interpretation & Best Practices

### Interpreting Models: Feature Importance & Explainability

**Feature Importance** helps understand which factors are most important for ML model predictions. This enables you to:

1. **Validate Factor Rankings:** Compare ML-derived feature importance with classical factor rankings (IC-IR, Deflated Sharpe)
2. **Identify Key Drivers:** See which factors dominate model predictions (e.g., momentum vs. value factors)
3. **Detect Redundancy:** Identify features that contribute little to predictions
4. **Compare Model Behavior:** Understand how different model types (Linear, Ridge, Random Forest) weight the same features

#### Feature Importance Outputs

**After running ML validation, you get:**

1. **`ml_feature_importance_{model_type}_{label_col}.csv`**
   - Columns: `feature`, `importance`, `raw_value`, `direction`
   - Sorted by importance (descending)
   - For linear models: `importance` = |coefficient|, `direction` = +1/-1
   - For tree models: `importance` = feature_importances_, `direction` = None

2. **`ml_permutation_importance_{model_type}_{label_col}.csv`** (optional)
   - Model-agnostic importance via permutation
   - Columns: `feature`, `importance_mean`, `importance_std`, `importance_median`
   - More computationally expensive, but works for any model

3. **Markdown Report Section: "Top 10 Features"**
   - Automatically generated table with top features
   - Interpretation hints (e.g., "Momentum factors dominate", "Volatility factors less important")

**Example from Report:**
```
## Top 10 Features (Model Coefficients / Feature Importances)

| Rank | Feature | Importance | Raw Value | Direction |
|------|---------|------------|-----------|----------|
| 1 | factor_mom | 0.854231 | 0.854231 | 1 |
| 2 | factor_value | 0.523412 | -0.523412 | -1 |
| 3 | factor_vol | 0.342156 | -0.342156 | -1 |
...

### Feature Importance Interpretation

- **Momentum factors** (factor_mom) are among the top features. This suggests momentum effects are important for return prediction.
- **Value factors** (factor_value) are prominent. Value-based signals contribute to the model.
```

#### Connecting Feature Importance with Factor Rankings

**Key Questions to Answer:**

1. **"Do the best ML models use the same factors as top-ranked factors?"**
   - Compare `ml_feature_importance_*.csv` with Factor Ranking outputs
   - Check if factors with high IC-IR also have high feature importance in ML models
   - Example: If `factor_mom` has IC-IR > 0.5 AND feature importance > 0.7 in Ridge model → strong consensus

2. **"Are certain Alt-Data factors (Earnings/Insider/News) particularly important?"**
   - Look for `earnings_*`, `insider_*`, `news_*` factors in top 10 features
   - Compare their importance with core TA/Price factors
   - If Alt-Data factors rank highly, they add value beyond traditional factors

3. **"How do different model types weight factors differently?"**
   - Use Model Zoo output: `ml_model_zoo_feature_importance_summary.csv`
   - Compare feature importance across Ridge, Lasso, Random Forest
   - Random Forest may discover non-linear factor interactions not visible in linear models

**Example Workflow:**

```powershell
# 1. Run Factor Ranking
python scripts/cli.py analyze_factors --universe macro_world_etfs --factor-set core+alt_full

# 2. Export Factor Panel
python research/factors/export_factor_panel_for_ml.py \
  --freq 1d \
  --symbols-file config/macro_world_etfs_tickers.txt \
  --factor-set core+alt_full \
  --horizon-days 20 \
  --start-date 2010-01-01 \
  --end-date 2025-12-03

# 3. Run ML Validation with Feature Importance
python scripts/cli.py ml_validate_factors \
  --factor-panel-file output/factor_panels/factor_panel_macro_world_etfs_core+alt_full_20d_1d.parquet \
  --label-col fwd_return_20d \
  --model-type ridge

# 4. Compare Results
# - Factor Ranking: output/factor_analysis/.../factor_rankings.csv
# - Feature Importance: output/ml_validation/ml_feature_importance_ridge_20d.csv
# - Look for overlap in top factors
```

#### Feature Importance vs. Factor Rankings

**Classical Factor Rankings (Phase C1/C2):**
- Based on IC/IC-IR and Deflated Sharpe Ratio
- Measures predictive power at the factor level (single-factor portfolios)
- Robust to model assumptions

**ML Feature Importance (Phase E2):**
- Based on model coefficients or permutation importance
- Measures contribution to multi-factor predictions
- Captures factor interactions and non-linear effects
- Model-specific (different models may weight factors differently)

**Best Practice:**
- Use both approaches together for comprehensive factor evaluation
- High IC-IR + High Feature Importance → Strong factor across multiple evaluation methods
- High IC-IR but Low Feature Importance → Factor may be redundant in multi-factor context
- Low IC-IR but High Feature Importance → Factor contributes through interactions

**Note:** All feature importance calculations are based on local factor panels. No live APIs are used in the explainability workflow.

---

### Understanding ML Metrics for Return Forecasting

#### Classical ML Metrics (MSE, MAE, R²)

**Important Context:**
- **R² for return forecasting is often low (< 0.1)**, but this does NOT mean the model is useless
- Even small R² values can indicate significant predictive power in finance
- Focus on **significance** (IC-IR > 0.5) rather than R² magnitude

**Interpretation:**
- **R² > 0.05**: Strong signal in finance context
- **R² 0.01 - 0.05**: Moderate signal, still valuable
- **R² < 0.01**: Weak signal, but check IC/Rank-IC metrics

**Why R² is low for returns:**
- Returns have high noise-to-signal ratio
- Idiosyncratic factors dominate individual stock returns
- Cross-sectional predictions (relative ranking) matter more than absolute predictions

#### Factor-Specific Metrics (IC, Rank-IC)

**Information Coefficient (IC):**
- Cross-sectional correlation between predictions and realized returns
- **IC > 0.05**: Strong predictive signal
- **IC-IR > 0.5**: Consistent signal over time

**Rank-IC (Spearman Correlation):**
- Robust to outliers
- Better suited for non-linear relationships
- **Rank-IC > IC**: Suggests non-linear factor relationships

**Best Practice:** Prioritize IC-IR and Rank-IC-IR over raw IC values.

#### Portfolio Metrics (Long/Short Sharpe)

**Long/Short Portfolio Construction:**
- Top quintile (20%) based on predictions → Long
- Bottom quintile (20%) based on predictions → Short
- Annualized Sharpe ratio of L/S portfolio

**Interpretation:**
- **L/S Sharpe > 1.0**: Good predictive signal
- **L/S Sharpe 0.5 - 1.0**: Moderate signal
- **L/S Sharpe < 0.5**: Weak signal

**Best Practice:** Compare L/S Sharpe to classical factor ranking Sharpe ratios for consistency.

---

### Detecting Overfitting

**Warning Signs:**
1. **Large discrepancy between CV splits:**
   - High variance in metrics across splits (e.g., R² ranges from -0.1 to 0.3)
   - Inconsistent IC values across time periods

2. **Train vs. Test Performance Gap:**
   - Train R² >> Test R² (e.g., Train R² = 0.5, Test R² = 0.01)
   - Train IC >> Test IC

3. **High Model Complexity:**
   - Random Forest with many trees and deep trees
   - Ridge/Lasso with very low regularization

**Best Practices:**
- Use time-series CV (not random splits)
- Compare train vs. test metrics in `ml_metrics_*.csv`
- Prefer simpler models if performance is similar
- Cross-validate hyperparameters

---

### Integrating ML Results with Existing Workflows

#### With Factor Bundles

**Current Workflow:**
1. Factor analysis → Factor rankings → Factor bundles → Multifactor signals
2. ML validation → ML predictions → (Future: ML-based signals)

**Integration Approach:**
- Use ML predictions as an additional factor in multifactor signal generation
- Combine ML predictions with classical factor bundles using ensemble methods
- Compare ML-based signals with bundle-based signals

**Example (Conceptual):**
```python
# Future: Combine ML predictions with factor bundles
ml_signal = ml_model.predict(factor_panel)
bundle_signal = build_multifactor_signal(factor_bundle_config)

# Ensemble approach
combined_signal = 0.6 * bundle_signal + 0.4 * ml_signal
```

#### With Multifactor Strategy

**Current:** Multifactor strategy uses factor bundles to generate signals.

**Future:** ML predictions can be:
- Used as an additional input to signal generation
- Used to weight factor bundles dynamically
- Used to filter or scale existing signals

---

## From ML Validation to ML Alpha Factors

After validating ML models and comparing their performance, you can generate ML alpha factors that can be used directly in trading strategies. This workflow bridges the gap between ML validation and strategy execution.

### Workflow Overview

1. **Export Factor Panel** (E1)
   - Use `export_factor_panel_for_ml.py` to create a factor panel with forward returns
   - Panel contains: `timestamp`, `symbol`, `fwd_return_{horizon}d`, `factor_*` columns

2. **ML Validation / Model Comparison** (E1/E2)
   - Run `ml_validate_factors` for single model validation
   - Or run `ml_model_zoo` for systematic model comparison
   - Review metrics (R², IC-IR, L/S Sharpe) and feature importance

3. **Select Best Model**
   - Based on validation metrics, choose the best-performing model
   - Consider factors like IC-IR, test R², overfitting (train/test gap), feature importance

4. **Export ML Alpha Factor** (E3)
   - Use `export_ml_alpha_factor.py` to generate ML alpha factor panel
   - Merges model predictions (`y_pred`) back into factor panel as `ml_alpha_{model_type}_{horizon}d`

5. **Use in Strategy**
   - Create factor bundle with ML alpha factor (e.g., `ai_tech_ml_alpha_bundle.yaml`)
   - Run backtest with `multifactor_long_short` strategy
   - Compare performance: ML-only vs. Mixed (Core+ML) vs. Core-only

### Export ML Alpha Factor

**Example: Export ML Alpha Factor from Factor Panel**

```powershell
# Step 1: Export factor panel (if not already done)
python research/factors/export_factor_panel_for_ml.py `
  --freq 1d `
  --symbols AAPL MSFT `
  --factor-set core+alt_full `
  --horizon-days 20 `
  --start-date 2015-01-01 `
  --end-date 2025-12-03

# Step 2: Run ML validation to evaluate model (optional, but recommended)
python scripts/cli.py ml_model_zoo `
  --factor-panel-file output/factor_panels/factor_panel_custom_AAPL_MSFT_core+alt_full_20d_1d.parquet `
  --label-col fwd_return_20d `
  --output-dir output/ml_validation/model_zoo_comparison

# Step 3: Export ML alpha factor using best model (e.g., Ridge with alpha=0.1)
python research/ml/export_ml_alpha_factor.py `
  --factor-panel-file output/factor_panels/factor_panel_custom_AAPL_MSFT_core+alt_full_20d_1d.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --model-param alpha=0.1 `
  --n-splits 5 `
  --output-dir output/ml_alpha_factors
```

**Output Files:**
- `ml_alpha_panel_{model_type}_{horizon}.parquet`: Enhanced factor panel with `ml_alpha_*` column
- `ml_alpha_predictions_{model_type}_{horizon}.parquet`: Raw predictions (for inspection)
- `ml_alpha_metadata_{model_type}_{horizon}.json`: Model config, experiment config, metrics

**Key Points:**
- Only test samples (out-of-sample from CV) have ML alpha predictions
- Training samples remain NaN (by design, to prevent look-ahead bias)
- Original factor columns are preserved in the output panel
- Panel format remains compatible with existing factor bundle infrastructure

**Note:** All data processing is based on local factor panels. No live APIs are used in the ML alpha export workflow.

### Next Steps: Using ML Alpha in Strategies

After exporting ML alpha factors, you can use them in trading strategies:

1. **Create Factor Bundle** (already provided):
   - `config/factor_bundles/ai_tech_ml_alpha_bundle.yaml` (100% ML alpha)
   - `config/factor_bundles/ai_tech_core_ml_bundle.yaml` (Mixed: Core + ML alpha)

2. **Run Backtest**:
   - Use `multifactor_long_short` strategy with ML alpha bundle
   - See [Multi-Factor Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) for details

3. **Compare Performance**:
   - ML-only vs. Core-only vs. Mixed bundles
   - Use Risk Reports to analyze regime-specific performance

**References:**
- Design: [ML Alpha Factor & Strategy Integration Design (E3)](ML_ALPHA_E3_DESIGN.md)
- Strategy Workflows: [Multi-Factor Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md)

---

## Troubleshooting

### Too Few Data Points

**Symptom:**
```
ValueError: Insufficient unique timestamps for CV with min_train_samples=100 and min_test_samples=20
```

**Solution:**
- Reduce `--n-splits` (e.g., from 5 to 3)
- Reduce `min_train_samples` in code (if using API directly)
- Use more data (extend time range or add more symbols)

**Example:**
```powershell
# Reduce splits for small dataset
python scripts/cli.py ml_validate_factors `
  --factor-panel-file small_panel.parquet `
  --label-col fwd_return_20d `
  --model-type linear `
  --n-splits 3  # Reduced from default 5
```

---

### NaN Problems

**Symptom:**
- Warnings about dropped rows due to NaN in features or labels
- Empty predictions DataFrame

**Solution:**
1. **Check factor panel for NaN:**
   ```python
   import pandas as pd
   df = pd.read_parquet("factor_panel.parquet")
   print(df.isna().sum())
   ```

2. **Handle NaN before ML validation:**
   - Forward-fill or backward-fill time-series NaNs
   - Drop rows with excessive NaN
   - Impute using median or mean (by symbol)

3. **Feature Engineering:**
   - Ensure factors are computed correctly
   - Check for data gaps in source data

---

### sklearn Not Installed

**Symptom:**
```
ImportError: scikit-learn not installed – please add it to your dependencies
```

**Solution:**
```bash
pip install scikit-learn
```

**Note:** `scikit-learn` is currently an optional dependency. Consider adding it to `requirements.txt` if ML validation is part of your regular workflow.

---

### Low Model Performance

**Symptoms:**
- R² < 0.01
- IC-IR < 0.3
- L/S Sharpe < 0.5

**Possible Causes:**
1. **Weak factors:** Factor panel does not contain predictive factors
2. **Noise-dominated returns:** Forward returns have high idiosyncratic noise
3. **Model mismatch:** Model type not suited for data (e.g., linear model for non-linear relationships)

**Solutions:**
- Try different factor sets (e.g., `core+alt_full` instead of `core`)
- Try different horizons (e.g., `fwd_return_5d` instead of `fwd_return_20d`)
- Try non-linear models (Random Forest) instead of linear models
- Check factor rankings: Use factors that rank highly in IC/IR analysis

---

---

## Putting it all together: AI/Tech Playbook

**End-to-End Automation:** Das Research-Playbook `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py` automatisiert den kompletten Workflow von der Factor-Panel-Erstellung über ML-Validation, Modellauswahl, ML-Alpha-Export, Backtests mit verschiedenen Bundles bis hin zu Risk-Reports und einem konsolidierten Research-Summary.

**Workflow-Übersicht:**
1. Export Factor Panel (mit Forward-Returns)
2. ML Model Zoo Comparison (systematischer Modellvergleich)
3. Best Model Selection (IC-IR, Test-R²)
4. ML Alpha Factor Export
5. Backtests (Core-only, Core+ML, ML-only)
6. Risk Reports (mit optionaler Regime-Attribution)
7. Research Summary (konsolidierter Markdown-Report)

**Vorteile:**
- Ein einziger Aufruf statt manueller Schritte
- Konsistente Konfiguration über alle Schritte
- Automatische Artefakt-Sammlung für Summary
- Reproduzierbare Research-Workflows

**Usage:**
```python
from research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook import main

main()  # Runs complete workflow with default AI/Tech config
```

**Output:** Konsolidierter Research-Summary mit ML-Vergleich, Backtest-Performance und Risk-Metriken in `output/risk_reports/research_summaries/`.

**Referenzen:**
- Implementation: `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py`
- Advanced Analytics: [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) (R1)

---

## References

- **Design Document:** [ML Validation E1 Design](ML_VALIDATION_E1_DESIGN.md)
- **ML Alpha Design:** [ML Alpha Factor & Strategy Integration Design (E3)](ML_ALPHA_E3_DESIGN.md)
- **Factor Analysis Workflows:** [Workflows – Factor Analysis](WORKFLOWS_FACTOR_ANALYSIS.md)
- **Multifactor Strategy Workflows:** [Workflows – Strategies: Multifactor](WORKFLOWS_STRATEGIES_MULTIFACTOR.md)
- **Risk Metrics Workflows:** [Workflows – Risk Metrics & Attribution](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md)
- **Factor Labs Roadmap:** [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

## CLI Reference

### Command: `ml_model_zoo`

**Full Syntax:**
```powershell
python scripts/cli.py ml_model_zoo `
  --factor-panel-file FILE `
  --label-col COL `
  [--output-dir DIR] `
  [--n-splits N] `
  [--train-size DAYS] `
  [--standardize BOOL] `
  [--min-train-samples N] `
  [--test-start YYYY-MM-DD] `
  [--test-end YYYY-MM-DD] `
  [--no-markdown]
```

**Arguments:**
- `--factor-panel-file`: Path to factor panel file (Parquet or CSV) - **Required**
- `--label-col`: Label column name (e.g., `fwd_return_20d`) - **Required**
- `--output-dir`: Output directory (default: `output/ml_model_zoo`)
- `--n-splits`: Number of CV splits (default: 5)
- `--train-size`: Training window size in days (default: None = expanding window)
- `--standardize`: Whether to standardize features (default: True)
- `--min-train-samples`: Minimum training samples (default: 252)
- `--test-start`: Test start date in YYYY-MM-DD format (optional)
- `--test-end`: Test end date in YYYY-MM-DD format (optional)
- `--no-markdown`: Skip Markdown report generation (CSV only)

**Examples:**
```powershell
# Basic model zoo comparison
python scripts/cli.py ml_model_zoo `
  --factor-panel-file output/factor_panels/core_20d_factors.parquet `
  --label-col fwd_return_20d

# With custom CV splits and time filter
python scripts/cli.py ml_model_zoo `
  --factor-panel-file output/factor_panels/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --n-splits 10 `
  --test-start 2020-01-01 `
  --test-end 2024-12-31 `
  --output-dir output/ml_model_zoo/custom_run
```

**Output:**
- `ml_model_zoo_summary.csv`: Comparison table with all models (sorted by IC-IR)
- `ml_model_zoo_summary.md`: Markdown report with top models and interpretation (unless `--no-markdown` is set)

---

### Command: `ml_validate_factors`

**Full Syntax:**
```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file FILE `
  --label-col COL `
  --model-type {linear,ridge,lasso,random_forest} `
  [--model-param KEY=VALUE ...] `
  [--n-splits N] `
  [--test-start YYYY-MM-DD] `
  [--test-end YYYY-MM-DD] `
  [--output-dir DIR]
```

**Arguments:**
- `--factor-panel-file`: Path to factor panel file (Parquet or CSV) - **Required**
- `--label-col`: Label column name (e.g., `fwd_return_20d`) - **Required**
- `--model-type`: Model type - **Required** (choices: `linear`, `ridge`, `lasso`, `random_forest`)
- `--model-param`: Model hyperparameter in format `key=value` (can be specified multiple times)
- `--n-splits`: Number of time-series CV splits (default: 5)
- `--test-start`: Test start date in YYYY-MM-DD format (optional)
- `--test-end`: Test end date in YYYY-MM-DD format (optional)
- `--output-dir`: Output directory (default: `output/ml_validation`)

**Examples:**
```powershell
# Basic validation with Ridge model
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type ridge

# With custom hyperparameters
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --model-param alpha=0.1 `
  --model-param max_iter=1000

# Random Forest with time filter
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type random_forest `
  --n-splits 10 `
  --test-start 2020-01-01 `
  --test-end 2024-12-31
```

