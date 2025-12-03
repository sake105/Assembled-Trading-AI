# Workflows – ML Meta-Models & Experiments

## 1. Overview

**Goal:** From raw data to ML-ready dataset, train a meta-model, evaluate it, and use it in backtests.

**Workflow:**
1. Build ML dataset from strategy backtest
2. Train meta-model on dataset
3. Evaluate model performance
4. Use model in backtests via ensemble layer
5. Track experiments for reproducibility

---

## 2. Building an ML Dataset

### 2.1 Basic Dataset Building

```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-path output/ml_datasets/trend_baseline_1d.parquet
```

### 2.2 Important Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--strategy` | `{trend_baseline,event_insider_shipping}` | ✅ Yes | Strategy name |
| `--freq` | `{1d,5min}` | ✅ Yes | Trading frequency |
| `--start-date` | `YYYY-MM-DD` | ❌ No | Start date (default: use all available data) |
| `--end-date` | `YYYY-MM-DD` | ❌ No | End date (default: use all available data) |
| `--output-path` | `FILE` | ❌ No | Output path (default: `output/ml_datasets/{strategy}_{freq}.parquet`) |
| `--price-file` | `FILE` | ❌ No | Explicit path to price file |
| `--universe` | `FILE` | ❌ No | Path to universe file (default: `watchlist.txt`) |
| `--symbols` | `SYMBOL ...` | ❌ No | List of symbols (overrides `--universe`) |
| `--label-horizon-days` | `DAYS` | ❌ No | Label horizon in days (default: 10) |
| `--success-threshold` | `THRESHOLD` | ❌ No | Success threshold percentage (default: 0.02 = 2%) |
| `--label-type` | `{binary_absolute,binary_outperformance,multi_class}` | ❌ No | Label type (default: `binary_absolute`) |
| `--format` | `{parquet,csv}` | ❌ No | Output format (default: `parquet`) |

### 2.3 What Goes Into the Dataset

**Features:**
- Technical indicators (EMA, RSI, ATR, etc.)
- Event features (insider trading, shipping congestion) - if using event strategy
- Price/volume features
- All features are prefixed consistently (e.g., `ta_*`, `insider_*`, `shipping_*`)

**Labels:**
- `label`: Binary label (1 = successful, 0 = unsuccessful)
- `realized_return`: Actual return achieved within horizon
- `entry_price`: Price at signal generation
- `exit_price`: Price at end of horizon (or when target met)
- `horizon_days`: Horizon used for labeling

**Label Types:**
- **`binary_absolute`**: Label = 1 if price increases by `success_threshold` within `horizon_days`
- **`binary_outperformance`**: (Future) Label = 1 if outperforming benchmark
- **`multi_class`**: (Future) Multi-class labels (e.g., strong buy, buy, hold, sell, strong sell)

### 2.4 Examples

**Basic dataset:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d
```

**With custom labeling parameters:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --label-horizon-days 5 \
  --success-threshold 0.03 \
  --label-type binary_absolute
```

**With specific symbols:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --symbols AAPL MSFT GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

**Event strategy dataset:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy event_insider_shipping \
  --freq 1d \
  --output-path output/ml_datasets/event_insider_shipping_1d.parquet
```

---

## 3. Training a Meta-Model

### 3.1 Basic Training

**From existing dataset:**
```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet \
  --model-type gradient_boosting \
  --output-model-path models/meta/trend_baseline_meta_model.joblib
```

**Build dataset and train in one step:**
```bash
python scripts/cli.py train_meta_model \
  --strategy trend_baseline \
  --freq 1d \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --model-type gradient_boosting
```

### 3.2 Important Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--dataset-path` | `FILE` | ❌ No* | Path to existing ML dataset (*required if not building on-the-fly) |
| `--strategy` | `{trend_baseline,event_insider_shipping}` | ❌ No* | Strategy name (*required if `--dataset-path` not provided) |
| `--freq` | `{1d,5min}` | ❌ No* | Trading frequency (*required if `--dataset-path` not provided) |
| `--start-date` | `YYYY-MM-DD` | ❌ No* | Start date (*required if `--dataset-path` not provided) |
| `--end-date` | `YYYY-MM-DD` | ❌ No* | End date (*required if `--dataset-path` not provided) |
| `--model-type` | `{gradient_boosting,random_forest}` | ❌ No | Model type (default: `gradient_boosting`) |
| `--output-model-path` | `FILE` | ❌ No | Output path (default: `models/meta/{strategy}_meta_model.joblib`) |
| `--label-horizon-days` | `DAYS` | ❌ No | Label horizon (for on-the-fly building, default: 10) |
| `--success-threshold` | `THRESHOLD` | ❌ No | Success threshold (for on-the-fly building, default: 0.05 = 5%) |
| `--symbols` | `SYMBOL ...` | ❌ No | List of symbols (for on-the-fly building) |

### 3.3 Optional Dependencies

**ML dependencies are optional:**
```bash
# Install with ML extras
pip install -e .[ml]
```

This installs:
- `scikit-learn` (for gradient boosting and random forest)
- `joblib` (for model serialization)

**Note:** If ML dependencies are not installed, the training command will fail with a clear error message.

### 3.4 Logged Metrics

After training, the following metrics are logged:

- **ROC-AUC**: Area under the ROC curve (higher is better, max: 1.0)
- **Brier Score**: Calibration metric (lower is better, min: 0.0)
- **Log Loss**: Logarithmic loss (lower is better, min: 0.0)

**Example output:**
```
Training meta-model...
Model trained: GradientBoostingClassifier
Evaluation metrics:
  ROC-AUC: 0.7234
  Brier Score: 0.1845
  Log Loss: 0.5123
Model saved to: models/meta/trend_baseline_meta_model.joblib
```

### 3.5 Calibration Plot

**Location:** `output/calibration_plots/calibration_{model_name}_{timestamp}.png`

**Note:** Calibration plots are generated if matplotlib is available. They show:
- Predicted probability vs. actual frequency
- Calibration curve (ideal vs. actual)
- Bins for probability ranges

---

## 4. Using Experiment Tracking for ML Runs

### 4.1 Enabling Experiment Tracking

```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet \
  --track-experiment \
  --experiment-name "meta_trend_gb_v1" \
  --experiment-tags "meta,trend,gb"
```

### 4.2 What is Logged

**Metrics:**
- `roc_auc`: ROC-AUC score
- `brier_score`: Brier score
- `log_loss`: Log loss
- Training configuration (model type, dataset path, etc.)

**Artifacts:**
- Trained model file (copied to `experiments/{run_id}/artifacts/`)
- Calibration plot (if generated)

**Configuration:**
- All CLI arguments saved to `experiments/{run_id}/run.json`

### 4.3 Inspecting Experiment Runs

**List all runs:**
```bash
ls experiments/
```

**View run details:**
```bash
cat experiments/20250115_143022_a1b2c3d4/run.json
```

**View metrics:**
```bash
cat experiments/20250115_143022_a1b2c3d4/metrics.csv
```

**View artifacts:**
```bash
ls experiments/20250115_143022_a1b2c3d4/artifacts/
```

---

## 5. Connecting to Research Notebooks

### 5.1 Loading Datasets

**In Jupyter Notebook:**
```python
import pandas as pd

# Load ML dataset
dataset = pd.read_parquet("output/ml_datasets/trend_baseline_1d.parquet")

# Inspect structure
print(dataset.head())
print(dataset.columns.tolist())
print(f"Label distribution: {dataset['label'].value_counts()}")
```

### 5.2 Loading Experiment Metrics

**In Jupyter Notebook:**
```python
import pandas as pd
import json

# Load experiment run
run_id = "20250115_143022_a1b2c3d4"
run_dir = f"experiments/{run_id}"

# Load run metadata
with open(f"{run_dir}/run.json") as f:
    run_meta = json.load(f)

# Load metrics
metrics = pd.read_csv(f"{run_dir}/metrics.csv")
print(metrics)
```

### 5.3 Loading Artifacts

**In Jupyter Notebook:**
```python
from pathlib import Path
from src.assembled_core.signals.meta_model import load_meta_model

# Load trained model
model_path = Path("experiments/20250115_143022_a1b2c3d4/artifacts/trend_baseline_meta_model.joblib")
meta_model = load_meta_model(model_path)

# Use model for predictions
# (see Phase 7 documentation for details)
```

### 5.4 Research Notebook Templates

**Location:** `research/`

**Available templates:**
- `research/trend/trend_baseline_experiments.ipynb` - Trend strategy experiments
- `research/meta/meta_model_calibration.ipynb` - Meta-model calibration analysis
- `research/altdata/insider_congress_shipping_exploration.ipynb` - Alternative data exploration
- `research/risk/scenario_and_risk_experiments.ipynb` - Risk scenario experiments

**See:** [Research Roadmap](RESEARCH_ROADMAP.md) for more details.

---

## 6. Complete Workflow Example

### 6.1 End-to-End Example

**1. Build dataset:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-path output/ml_datasets/trend_baseline_1d.parquet
```

**2. Train meta-model:**
```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet \
  --model-type gradient_boosting \
  --output-model-path models/meta/trend_baseline_meta_model.joblib \
  --track-experiment \
  --experiment-name "meta_trend_gb_v1" \
  --experiment-tags "meta,trend,gb"
```

**3. Use in backtest:**
```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --use-meta-model \
  --meta-model-path models/meta/trend_baseline_meta_model.joblib \
  --meta-ensemble-mode filter \
  --meta-min-confidence 0.5 \
  --generate-report \
  --track-experiment \
  --experiment-name "trend_meta_filter_0.5" \
  --experiment-tags "trend,meta,filter"
```

**4. Analyze results:**
- Compare metrics from experiment runs
- Review QA reports
- Load datasets and models in research notebooks for deeper analysis

---

## 7. Troubleshooting

### 7.1 Common Issues

**Dataset is empty:**
- Check that price data exists for the specified date range
- Verify universe file contains valid symbols
- Check logs for data loading errors

**Model training fails:**
- Ensure ML dependencies are installed: `pip install -e .[ml]`
- Check that dataset contains both features and labels
- Verify dataset is not empty

**Model not found in backtest:**
- Check model path is correct
- Use `--meta-model-path` explicitly if auto-detection fails
- Verify model file exists: `ls models/meta/`

### 7.2 Getting Help

- Check existing documentation:
  - [Phase 7 Meta Layer](PHASE7_META_LAYER.md) - Meta-layer architecture
  - [Workflows – Backtests & Meta-Model Ensemble](WORKFLOWS_BACKTEST_AND_ENSEMBLE.md) - Using meta-models in backtests
  - [Research Roadmap](RESEARCH_ROADMAP.md) - Research strategy

---

## 8. Related Documentation

- [Workflows – Backtests & Meta-Model Ensemble](WORKFLOWS_BACKTEST_AND_ENSEMBLE.md) - Using meta-models in backtests
- [Phase 7 Meta Layer](PHASE7_META_LAYER.md) - Meta-layer architecture
- [Phase 9 Model Governance](PHASE9_MODEL_GOVERNANCE.md) - Model validation
- [Research Roadmap](RESEARCH_ROADMAP.md) - Research strategy and backlog

