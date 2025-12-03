# Workflows – Backtests & Meta-Model Ensemble

## 1. Overview

**Goal:** Run a backtest for a strategy, optionally with the ML meta-model ensemble.

**Components:**
- Strategy signals (e.g., trend baseline, event_insider_shipping)
- Portfolio / position sizing
- Meta-model (optional)
- Ensemble layer (filter or scaling)
- Performance & QA metrics

---

## 2. Basic Backtest Without Meta-Model

### 2.1 Running a Basic Backtest

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --start-date 2020-01-01 \
  --end-date 2023-12-31
```

### 2.2 Important Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--freq` | `{1d,5min}` | ✅ Yes | Trading frequency |
| `--strategy` | `{trend_baseline,event_insider_shipping}` | ❌ No | Strategy name (default: `trend_baseline`) |
| `--price-file` | `FILE` | ❌ No | Explicit path to price file |
| `--universe` | `FILE` | ❌ No | Path to universe file (default: `watchlist.txt`) |
| `--start-date` | `YYYY-MM-DD` | ❌ No | Start date for backtest |
| `--end-date` | `YYYY-MM-DD` | ❌ No | End date for backtest |
| `--start-capital` | `AMOUNT` | ❌ No | Starting capital (default: 10000.0) |
| `--with-costs` | Flag | ❌ No | Include transaction costs (default: True) |
| `--no-costs` | Flag | ❌ No | Disable transaction costs |
| `--commission-bps` | `BPS` | ❌ No | Commission in basis points |
| `--spread-w` | `WEIGHT` | ❌ No | Spread weight for cost model |
| `--impact-w` | `WEIGHT` | ❌ No | Market impact weight |
| `--generate-report` | Flag | ❌ No | Generate QA report after backtest |

### 2.3 Strategy Options

**Trend Baseline (`trend_baseline`):**
- EMA-based trend following strategy
- Uses fast and slow moving averages (default: 20/50 for 1d, 10/30 for 5min)
- Generates LONG signals on EMA crossover

**Event Insider Shipping (`event_insider_shipping`):**
- Event-based strategy using insider trading and shipping congestion data
- Combines multiple event signals
- Requires sample event data in `data/sample/events/`

### 2.4 Outputs

**Performance Metrics:**
- Total return, CAGR, Sharpe ratio
- Max drawdown, volatility
- Total trades, win rate

**Output Files:**
- **Equity Curve:** `output/equity_curve_{freq}.csv`
- **Performance Report:** `output/reports/performance_report_{freq}.md` (if `--generate-report` is used)
- **QA Report:** `output/reports/qa_report_{strategy}_{freq}_{date}.md` (if `--generate-report` is used)

**Example Output:**
```
Backtest completed: 1000 equity points
Total Return: 15.23%
CAGR: 4.85%
Sharpe Ratio: 1.2345
Max Drawdown: -12.5%
Total Trades: 45
```

---

## 3. Backtest with Meta-Model Ensemble

### 3.1 Prerequisites

Before using the meta-model ensemble, you need:

1. **Trained Meta-Model:**
   - Train a meta-model using `train_meta_model` subcommand
   - Model should be saved in `models/meta/` directory
   - Default naming: `{strategy}_meta_model.joblib`

2. **ML Dataset:**
   - Build an ML dataset using `build_ml_dataset` subcommand
   - Dataset should contain features and labels matching your strategy

See [Workflows – ML Meta-Models & Experiments](WORKFLOWS_ML_AND_EXPERIMENTS.md) for details.

### 3.2 Using Filter Mode

Filter mode removes signals below the confidence threshold:

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --use-meta-model \
  --meta-model-path models/meta/trend_baseline_meta_model.joblib \
  --meta-ensemble-mode filter \
  --meta-min-confidence 0.5
```

**How it works:**
- Meta-model calculates `confidence_score` (0-1) for each signal
- Signals with `confidence_score < meta_min_confidence` are filtered out (set to FLAT)
- Only high-confidence signals proceed to position sizing

**Output:**
```
Meta-model ensemble applied:
  Original signals: 150 (LONG: 120)
  After filtering: 95 (LONG: 75)
  Dropped signals: 45
  Mode: filter, Min confidence: 0.5
```

### 3.3 Using Scaling Mode

Scaling mode adjusts position sizes based on confidence:

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --use-meta-model \
  --meta-model-path models/meta/trend_baseline_meta_model.joblib \
  --meta-ensemble-mode scaling \
  --meta-min-confidence 0.3
```

**How it works:**
- Meta-model calculates `confidence_score` (0-1) for each signal
- Position sizes are scaled by `confidence_score` (up to `max_scaling`, default: 1.0)
- Signals with `confidence_score < meta_min_confidence` are still filtered out
- Higher confidence → larger positions

**Output:**
```
Meta-model ensemble applied:
  Original signals: 150 (LONG: 120)
  After scaling: 120 (LONG: 120)
  Mode: scaling, Min confidence: 0.3, Max scaling: 1.0
```

### 3.4 Difference Between Filter vs. Scaling

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Filter** | Removes low-confidence signals entirely | When you want fewer, higher-quality trades |
| **Scaling** | Adjusts position sizes by confidence | When you want to keep all signals but size by confidence |

**Recommendation:**
- Start with **filter mode** to understand signal quality
- Use **scaling mode** if you want to maintain signal diversity while reducing risk

### 3.5 Auto-Detection of Model Path

If `--meta-model-path` is not provided, the system tries to auto-detect:

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --use-meta-model \
  --meta-ensemble-mode filter
```

The system looks for: `models/meta/{strategy}_meta_model.joblib`

If not found, an error is raised with instructions.

---

## 4. Experiment Tracking for Backtests

### 4.1 Enabling Experiment Tracking

Track backtest runs for research and comparison:

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --track-experiment \
  --experiment-name "trend_baseline_v1" \
  --experiment-tags "trend,baseline,ma20_50"
```

### 4.2 What is Logged

**Metrics:**
- `final_pf`: Final portfolio value
- `total_return`: Total return percentage
- `cagr`: Compound annual growth rate
- `sharpe_ratio`: Sharpe ratio
- `max_drawdown_pct`: Maximum drawdown percentage
- `total_trades`: Total number of trades
- `qa_overall_result`: QA gate result (OK/WARNING/BLOCK)

**Artifacts:**
- QA report (if `--generate-report` is used)
- Saved to `experiments/{run_id}/artifacts/qa_report.md`

**Configuration:**
- All CLI arguments are saved to `experiments/{run_id}/run.json`

### 4.3 Experiment Run Structure

**Location:** `experiments/{run_id}/`

**Files:**
- `run.json`: Run metadata, configuration, tags
- `metrics.csv`: Time-series metrics (step, timestamp, metric_name, metric_value)
- `artifacts/`: Copied files (reports, plots, etc.)

**Example `run.json`:**
```json
{
  "run_id": "20250115_143022_a1b2c3d4",
  "name": "trend_baseline_v1",
  "created_at": "2025-01-15T14:30:22Z",
  "status": "finished",
  "config": {
    "freq": "1d",
    "strategy": "trend_baseline",
    "start_capital": 10000.0,
    "use_meta_model": true,
    "meta_ensemble_mode": "filter",
    "meta_min_confidence": 0.5
  },
  "tags": ["trend", "baseline", "ma20_50"]
}
```

### 4.4 Inspecting Experiment Runs

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

## 5. Recommended Workflow for Strategy Research

### 5.1 Step-by-Step Research Process

**1. Define Hypothesis**
- Example: "EMA crossover with 20/50 periods should outperform baseline on daily data"

**2. Run Baseline Backtest**
```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --generate-report \
  --track-experiment \
  --experiment-name "baseline_ma20_50" \
  --experiment-tags "baseline,ma20_50"
```

**3. Build ML Dataset**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-path output/ml_datasets/trend_baseline_1d.parquet
```

**4. Train Meta-Model**
```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet \
  --model-type gradient_boosting \
  --output-model-path models/meta/trend_baseline_meta_model.joblib \
  --track-experiment \
  --experiment-name "meta_trend_gb_v1" \
  --experiment-tags "meta,trend,gb"
```

**5. Run Backtest with Meta-Model Ensemble**
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
  --experiment-name "trend_baseline_meta_filter_0.5" \
  --experiment-tags "trend,meta,filter"
```

**6. Compare Metrics & Distributions**
- Compare Sharpe ratios, drawdowns, trade counts
- Review QA reports side-by-side
- Analyze experiment metrics in `experiments/` directories

**7. Document Results**
- Create a research notebook in `research/trend/`
- Document hypothesis, results, and conclusions
- Reference experiment run IDs

### 5.2 Iterative Improvement

**Vary Parameters:**
- Try different `meta_min_confidence` thresholds (0.3, 0.5, 0.7)
- Compare `filter` vs. `scaling` modes
- Test different model types (`gradient_boosting` vs. `random_forest`)

**Track Everything:**
- Always use `--track-experiment` for reproducibility
- Use descriptive experiment names and tags
- Document parameter choices in research notebooks

---

## 6. Examples

### 6.1 Basic Backtest with Report

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --universe watchlist.txt \
  --start-capital 50000 \
  --generate-report
```

### 6.2 Event Strategy Backtest

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy event_insider_shipping \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --generate-report
```

### 6.3 Meta-Model Ensemble with Experiment Tracking

```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --use-meta-model \
  --meta-model-path models/meta/trend_baseline_meta_model.joblib \
  --meta-ensemble-mode scaling \
  --meta-min-confidence 0.4 \
  --generate-report \
  --track-experiment \
  --experiment-name "trend_meta_scaling_0.4" \
  --experiment-tags "trend,meta,scaling"
```

---

## 7. Backtests on Alt-Daten Snapshot

### 7.1 Overview

The Alt-Daten snapshot contains historical price data (2000-01-01 to 2025-12-03) stored locally as Parquet files.
This allows running comprehensive backtests without relying on online data sources.

### 7.2 Downloading Alt-Daten for Universes

Before running backtests, you need to download historical data for your universes:

**Download Macro World ETFs:**
```powershell
python scripts/download_historical_snapshot.py `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start 2000-01-01 `
  --end 2025-12-03 `
  --interval 1d `
  --target-root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --sleep-seconds 2.0
```

**Note:** Some ETFs (e.g., GLD, DBC, HYG, ACWI, VT) were launched after 2000, so earlier dates may not have data for all tickers. This is expected behavior.

For other universes, use the same command pattern with the respective `--symbols-file` argument.

See [Download Strategy](DOWNLOAD_STRATEGY.md) for detailed guidance on handling Yahoo Finance rate limits.

### 7.3 Setup for Alt-Daten Mode

**Environment Configuration:**
```powershell
# Set data source to local
$env:ASSEMBLED_DATA_SOURCE = "local"

# Set path to Alt-Daten snapshot
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
```

**Data Structure:**
- Parquet files stored as: `<local_data_root>/<freq>/<SYMBOL>.parquet`
- Example: `F:\...\stand 3-12-2025\1d\NVDA.parquet`
- Format: `timestamp` (UTC), `symbol`, `open`, `high`, `low`, `close`, `adj_close`, `volume`

### 7.4 Running Backtests on Alt-Daten Universes

**1. Universe AI Tech (24 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2000-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_ai_tech_2000_2025" `
  --experiment-tags "trend,ai_tech,altdata"
```

**2. Healthcare Biotech (4 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/healthcare_biotech_tickers.txt `
  --start-date 2000-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_healthcare_2000_2025" `
  --experiment-tags "trend,healthcare,altdata"
```

**3. Energy Resources Cyclicals (7 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/energy_resources_cyclicals_tickers.txt `
  --start-date 2000-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_energy_2000_2025" `
  --experiment-tags "trend,energy,altdata"
```

**4. Defense Security Aero (11 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/defense_security_aero_tickers.txt `
  --start-date 2000-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_defense_2000_2025" `
  --experiment-tags "trend,defense,altdata"
```

**5. Consumer Financial Misc (3 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/consumer_financial_misc_tickers.txt `
  --start-date 2000-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_consumer_2000_2025" `
  --experiment-tags "trend,consumer,altdata"
```

**6. Macro World ETFs (10 Symbols):**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy trend_baseline `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start-date 2003-01-01 `
  --end-date 2025-12-02 `
  --data-source local `
  --track-experiment `
  --experiment-name "trend_baseline_macro_etfs_2003_2025" `
  --experiment-tags "trend,macro,etfs,altdata"
```

**Note:** The start date for Macro ETFs is set to 2003-01-01 because some ETFs (e.g., GLD, DBC, HYG, ACWI, VT) were launched later than 2000. Earlier dates may not have data for all tickers, which is expected behavior.

### 7.5 Validating Alt-Daten Snapshot

Before running backtests, validate the snapshot:
```powershell
python scripts/validate_altdata_snapshot.py `
  --root "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025" `
  --interval 1d
```

This shows:
- Number of files per symbol
- Date ranges (min/max)
- Missing columns or data gaps
- Overall validation status

### 7.6 Summarizing Backtest Experiments

After running backtests, summarize results across universes:
```powershell
python scripts/summarize_backtest_experiments.py `
  --experiments-root experiments `
  --filter-tag "altdata" `
  --output-csv output/experiment_summaries/altdata_backtests.csv
```

This generates a comparison table with:
- Universe name
- CAGR, Sharpe ratio, Max drawdown
- Total trades, win rate
- Date ranges covered

---

## 8. Related Documentation

- [Workflows – ML Meta-Models & Experiments](WORKFLOWS_ML_AND_EXPERIMENTS.md) - Building datasets and training meta-models
- [Phase 7 Meta Layer](PHASE7_META_LAYER.md) - Meta-layer architecture
- [Phase 9 Model Governance](PHASE9_MODEL_GOVERNANCE.md) - Model validation
- [Research Roadmap](RESEARCH_ROADMAP.md) - Research strategy and backlog

