# ML Validation Experiments - Quick Start Guide

**Status:** Ready for first real-world experiments  
**Last Updated:** 2025-12-13

---

## Prerequisites

### What You Need for Each Experiment

1. **Factor Panel as File**
   - Format: `timestamp`, `symbol`, `fwd_return_20d`, `factor_*` columns
   - See [ML Validation E1 Design](ML_VALIDATION_E1_DESIGN.md) for data contract details

2. **CLI Command**
   ```powershell
   python scripts/cli.py ml_validate_factors `
     --factor-panel-file <YOUR_FACTOR_PANEL> `
     --label-col fwd_return_20d `
     --model-type <MODEL_TYPE> `
     --output-dir output/ml_validation/<EXPERIMENT_NAME>
   ```

---

## Experiment 1: Baseline – Core Factors, Linear Model (Macro World ETFs)

**Goal:** Check how much "explainable structure" exists in Core factors without Alt-Data, on a clean ETF universe.

### Setup

- **Universe:** `macro_world_etfs`
- **Factor-Set:** `core` or `core+vol_liquidity`
- **Horizon/Label:** `fwd_return_20d`
- **Time Range:** 2010-01-01 to 2025-12-03 (adjust as needed)
- **Model:** `linear`

### Command

```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_panels/macro_world_etfs_core_1d_2010_2025.parquet `
  --label-col fwd_return_20d `
  --model-type linear `
  --n-splits 5 `
  --output-dir output/ml_validation/macro_world_etfs_core_linear
```

**Note:** Adjust file path to your actual panel file.

### Dealing with the Factor Zoo

When testing multiple models, parameter combinations, or factors, you need to account for multiple testing bias. The "Factor Zoo" problem occurs when many variants are tested, and some appear good by chance alone.

**Always check:**
- `n_tests`: Number of variants tested (models, parameters, factors)
- `ls_sharpe_raw`: Raw Sharpe Ratio (before adjustment)
- `ls_sharpe_deflated`: Deflated Sharpe Ratio (adjusted for multiple testing)

**Guidelines:**
- `sharpe_deflated < 0.5`: Likely a gimmick, do not pursue further
- `0.5 <= sharpe_deflated < 1.0`: Only pursue with additional evidence (Walk-Forward, Regime Analysis)
- `sharpe_deflated >= 1.0`: Serious candidate for further research rounds

**Additional Validation:**
- Walk-Forward Analysis (B3): Out-of-sample stability across multiple time windows
- Regime Analysis (B3): Consistent performance across different market regimes
- Transaction Cost Analysis (E4): Ensure costs do not destroy performance

**References:**
- [Deflated Sharpe B4 Design](../docs/DEFLATED_SHARPE_B4_DESIGN.md)
- [Walk-Forward & Regime Analysis B3 Design](../docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md)

### What to Look For

**ML Metrics:**
- R² (will likely be low – this is normal for returns)
- MSE, MAE

**Finance Metrics (from report):**
- IC / Rank-IC between `y_pred` and `fwd_return_20d`
- L/S Sharpe (Top vs. Bottom Quintile based on predictions)

**Success Criteria:**
- Stable positive ICs → Good sign that Core factors carry ML value
- L/S Sharpe > 0.5 → Predictive signal exists

---

## Experiment 2: Core + Alt (B1/B2) vs. Core Only (AI/Tech)

**Goal:** Compare how much additional power Alt-Data factors (Earnings, Insider, News, Macro) provide, and whether some regularization (Ridge) helps.

### Setup

Run two experiments on the same universe:

- **Universe:** `universe_ai_tech`
- **Horizon/Label:** `fwd_return_20d`
- **Model:** `ridge`
- **Time Range:** 2015-01-01 to 2025-12-03 (adjust as needed)

### Commands

#### a) Core Factors Only

```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_panels/ai_tech_core_1d_2015_2025.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --model-param alpha=1.0 `
  --n-splits 5 `
  --output-dir output/ml_validation/ai_tech_core_ridge
```

#### b) Core + All Alt-Data Factors

```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_panels/ai_tech_core_alt_full_1d_2015_2025.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --model-param alpha=1.0 `
  --n-splits 5 `
  --output-dir output/ml_validation/ai_tech_core_alt_full_ridge
```

### What to Compare

**Between the two runs:**
- **IC & IC-IR** of predictions
- **L/S Sharpe** and **L/S Return**
- Number of features actually used (visible in report, or via Feature-Importance if implemented later)

**Success Criteria:**
- If Core+Alt shows significantly better IC/Sharpe → Clear evidence that B1/B2 add real value
- If Core+Alt only slightly better → Alt-Data may be redundant or requires better feature engineering

---

## Experiment 3: Non-Linear Model (Random Forest) on AI/Tech with Core+Alt

**Goal:** Check if a non-linear model (Random Forest) can extract additional value from factors – or if it just overfits.

### Setup

- **Universe:** `universe_ai_tech`
- **Factor-Set:** `core+alt_full`
- **Horizon/Label:** `fwd_return_20d`
- **Model:** `random_forest`
- **Time Range:** 2015-01-01 to 2025-12-03

### Command

```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_panels/ai_tech_core_alt_full_1d_2015_2025.parquet `
  --label-col fwd_return_20d `
  --model-type random_forest `
  --model-param n_estimators=300 `
  --model-param max_depth=5 `
  --model-param min_samples_leaf=10 `
  --n-splits 5 `
  --output-dir output/ml_validation/ai_tech_core_alt_full_rf
```

### What to Compare

Compare Random Forest with the previous Ridge run on the same panel:

**Overfitting Indicators:**
- **Train R² vs. Test R²:** Large gap = overfitting
- **IC / Rank-IC:** Should be stable across CV splits
- **L/S Sharpe:** Should be similar or slightly better than Ridge

**Success Criteria:**
- If RF shows Train R² >> Test R² and Sharpe not better → **Classic overfitting**
- If RF Test metrics (IC / Sharpe) are realistically better → **Non-linear combinations add value**

---

## Getting Factor Panels

### Export via Script (Recommended)

Factor-Panels werden über ein dediziertes Helper-Script erzeugt, das auf der bestehenden `run_factor_analysis`-Logik aufsetzt und **nur lokale Preis-Snapshots** verwendet:

```bash
python research/factors/export_factor_panel_for_ml.py \
  --freq 1d \
  --symbols-file config/macro_world_etfs_tickers.txt \
  --data-source local \
  --factor-set core+vol_liquidity \
  --horizon-days 20 \
  --start-date 2010-01-01 \
  --end-date 2025-12-03
```

**Output:**

Das Script erzeugt ein Parquet-File unter `output/factor_panels/`, z.B.:

```
output/factor_panels/factor_panel_macro_world_etfs_core+vol_liquidity_20d_1d.parquet
```

**Usage:**

Dieses File kann direkt im `ml_validate_factors` Command verwendet werden:

```bash
python scripts/cli.py ml_validate_factors \
  --factor-panel-file output/factor_panels/factor_panel_macro_world_etfs_core+vol_liquidity_20d_1d.parquet \
  --label-col fwd_return_20d \
  --model-type ridge
```

**Benefits:**
- Clean separation of concerns
- Reusable for multiple experiments
- Uses existing factor computation logic
- Only local data (no live API calls)

---

## Interpretation Guidelines

### R² is Low? → Normal!

**Context for Return Forecasting:**
- R² < 0.1 is **typical** for return forecasting
- R² 0.01-0.05 can still indicate **significant predictive power**

**Focus on Finance Metrics Instead:**
1. **IC-IR > 0.5** → Consistent signal
2. **Rank-IC-IR > 0.5** → Robust signal
3. **L/S Sharpe > 1.0** → Good trading signal

### Detecting Overfitting

**Warning Signs:**
- Large discrepancy between CV splits (e.g., R² ranges from -0.1 to 0.3)
- Train R² >> Test R² (e.g., Train R² = 0.5, Test R² = 0.01)

**Best Practices:**
- Compare train vs. test metrics in `ml_metrics_*.csv`
- Prefer simpler models if performance is similar
- Use regularization (Ridge/Lasso) for high-dimensional feature spaces

---

## Expected Output Structure

Each experiment produces:

```
output/ml_validation/<EXPERIMENT_NAME>/
├── ml_metrics_<model_type>_fwd_return_20d.csv        # Per-split & global metrics
├── ml_predictions_sample_<model_type>_fwd_return_20d.parquet  # Sample predictions
├── ml_portfolio_metrics_<model_type>_fwd_return_20d.csv       # Portfolio metrics
└── ml_validation_report_<model_type>_fwd_return_20d.md        # Markdown report
```

---

## Next Steps After Experiments

1. **Compare Results:**
   - Aggregate metrics across experiments
   - Identify best model type and hyperparameters
   - Check for consistent signals across universes

2. **Model Zoo Comparison (Automated):**
   
   Nach den Single-Model-Experimenten kannst du mit dem **Model-Zoo** mehrere Modelle automatisch auf demselben Panel vergleichen:
   
   ```powershell
   python scripts/cli.py ml_model_zoo `
     --factor-panel-file output/factor_panels/core_20d_factors.parquet `
     --label-col fwd_return_20d `
     --output-dir output/ml_model_zoo/comparison
   ```
   
   **Vorteil:**
   - Ein Command statt mehrerer separater Runs
   - Garantiert gleiche Experiment-Konfiguration für alle Modelle
   - Automatische Aggregation und Sortierung nach IC-IR
   - Vergleichstabelle (CSV + Markdown) mit allen Modellen
   
   **Interpretation:**
   - Siehe [Workflows – ML Validation & Model Comparison](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md) Abschnitt "Model Zoo – Mehrere Modelle auf einem Panel"
   - Fokussiere auf IC-IR und L/S Sharpe für Modell-Auswahl
   
   **Wichtig:** Auch der Model-Zoo verwendet nur lokale Factor-Panels (Parquet/CSV), keine Live-APIs.

3. **Feature Analysis (Future E2):**
   - Feature importance analysis
   - Identify which factors contribute most
   - Compare with classical factor rankings

4. **Feature Analysis (E2 - Completed):**
   - Feature importance analysis (automatically included in ML validation reports)
   - Permutation importance for model-agnostic insights
   - Compare with classical factor rankings (see [Factor Ranking Overview](FACTOR_RANKING_OVERVIEW.md))

5. **Turn Best Model into ML Alpha Factor (E3 - Completed):**
   
   Nachdem du mehrere Modelle verglichen und das beste Modell identifiziert hast, kannst du die ML-Vorhersagen als Alpha-Faktor in deine Trading-Strategien integrieren.
   
   **Workflow: ML Alpha Factor Export**
   
   **1. Wähle das beste Modell aus Model-Zoo:**
   
   Basierend auf den Metriken aus `ml_model_zoo_summary.csv`:
   - Höchster IC-IR oder Test-R²
   - Niedrigste Train/Test-Gap (wenig Overfitting)
   - Konsistente Performance über alle CV-Splits
   
   **2. Exportiere ML-Alpha-Faktor:**
   
   ```powershell
   # Verwende das beste Modell aus dem Model-Zoo-Vergleich
   # Beispiel: Ridge mit alpha=0.1 hatte den besten IC-IR
   
   python research/ml/export_ml_alpha_factor.py `
     --factor-panel-file output/factor_panels/factor_panel_universe_ai_tech_core+alt_full_20d_1d.parquet `
     --label-col fwd_return_20d `
     --model-type ridge `
     --model-param alpha=0.1 `
     --n-splits 5 `
     --output-dir output/ml_alpha_factors
   ```
   
   **3. Verwende ML-Alpha-Faktor in Strategie:**
   
   Siehe [Multi-Factor Strategy Workflows – Using ML Alpha as a Factor](WORKFLOWS_STRATEGIES_MULTIFACTOR.md#using-ml-alpha-as-a-factor) für vollständige Anleitung.
   
   **Vorteil:**
   - ML-Vorhersagen werden wie normale Faktoren behandelt
   - Kann in Pure-ML-Bundles (100% ML-Alpha) oder Mixed-Bundles (Core + ML-Alpha) verwendet werden
   - Direkter Vergleich: ML-only vs. Core-only vs. Mixed möglich
   
   **Referenzen:**
   - Design: [ML Alpha Factor & Strategy Integration Design (E3)](ML_ALPHA_E3_DESIGN.md)
   - Workflow: [ML Validation & Model Comparison Workflows – From ML Validation to ML Alpha Factors](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md#from-ml-validation-to-ml-alpha-factors)

---

## References

- **Design Document:** [ML Validation E1 Design](ML_VALIDATION_E1_DESIGN.md)
- **ML Alpha Design:** [ML Alpha Factor & Strategy Integration Design (E3)](ML_ALPHA_E3_DESIGN.md)
- **Workflow Guide:** [Workflows – ML Validation & Model Comparison](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)
- **Strategy Workflows:** [Multi-Factor Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md)
- **Factor Analysis:** [Workflows – Factor Analysis](WORKFLOWS_FACTOR_ANALYSIS.md)

