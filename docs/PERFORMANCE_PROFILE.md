# Performance Profile - Standard Benchmark Runs

**Purpose:** Define standard benchmark runs for performance profiling and optimization tracking.

**Status:** Baseline (P1) - Initial definition, runtime data to be collected.

---

## Standard Benchmark Runs

### EOD_SMALL

**Description:** Small-scale EOD pipeline run for daily regression testing.

**Command:**
```bash
python scripts/run_daily.py
```

**Note:** `run_daily.py` uses frequency "1d" by default (EOD daily run).

**Parameters:**
- **Time range:** 2 years (default date range from settings)
- **Universe:** 100 symbols (from watchlist.txt or default_universe)
- **Frequency:** 1d (daily)
- **Strategy:** Default EMA trend (via run_daily)

**Expected artifacts:**
- `output/orders_1d.csv`
- `output/performance_report_1d.md`
- `output/equity_curve_1d.csv`

**Use case:** Quick smoke test, regression validation, daily CI checks.

---

### BACKTEST_MEDIUM

**Description:** Medium-scale backtest run for performance benchmarking.

**Command:**
```bash
python scripts/run_backtest_strategy.py --freq 1d --strategy trend_baseline --start-capital 10000
```

**Parameters:**
- **Time range:** 10 years (full available data)
- **Universe:** 200 symbols (from watchlist.txt or default_universe)
- **Frequency:** 1d (daily)
- **Strategy:** trend_baseline (EMA-based trend following)
- **Start capital:** 10000.0
- **Costs:** Default cost model (commission_bps=0.5, spread_w=0.25, impact_w=0.5)

**Expected artifacts:**
- `output/orders_1d.csv` (if include_trades=True)
- `output/performance_report_1d.md` (via backtest engine)
- `output/equity_curve_1d.csv` (via backtest engine)

**Use case:** Performance optimization baseline, backtest engine profiling, TradingCycle integration validation.

---

### ML_JOB

**Description:** Machine learning dataset build and meta-model training.

**Command:**
```bash
# Step 1: Build ML dataset
python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d

# Step 2: Train meta-model
python scripts/cli.py train_meta_model --strategy trend_baseline --freq 1d
```

**Parameters:**
- **Dataset build:**
  - Time range: Full available data (typically 5-10 years)
  - Universe: All symbols from watchlist (typically 50-200 symbols)
  - Frequency: 1d (daily)
  - Features: TA features (MA, ATR, RSI, log returns)
  - Labels: Signal outcomes (from trend_baseline strategy)
- **Meta-model training:**
  - Strategy: trend_baseline (or other base strategy)
  - Model type: Gradient Boosting (XGBoost/LightGBM)
  - Validation: Time-series cross-validation

**Expected artifacts:**
- `data/ml/dataset_*.parquet` (feature + label dataset)
- `models/meta/{strategy}_meta_model.joblib` (trained meta-model)
- `output/ml_training_report.md` (training metrics, feature importance)

**Use case:** ML pipeline performance, feature engineering bottlenecks, model training time.

---

## Performance Tracking Table

### Runtime Benchmarks

| Run ID | Date | Runtime (s) | Memory Peak (MB) | Notes |
|--------|------|------------|------------------|-------|
| EOD_SMALL | - | - | - | Baseline to be collected |
| BACKTEST_MEDIUM | - | - | - | Baseline to be collected |
| ML_JOB | - | - | - | Baseline to be collected |

### Top-3 Hotspots (per run)

**EOD_SMALL:**
1. (to be profiled)
2. (to be profiled)
3. (to be profiled)

**BACKTEST_MEDIUM:**
1. (to be profiled)
2. (to be profiled)
3. (to be profiled)

**ML_JOB:**
1. (to be profiled)
2. (to be profiled)
3. (to be profiled)

---

## Profiling Workflow

### Collecting Baseline Data

1. **EOD_SMALL:**
   ```bash
   python scripts/profile_jobs.py --run eod_small --iterations 3
   ```

2. **BACKTEST_MEDIUM:**
   ```bash
   python scripts/profile_jobs.py --run backtest_medium --iterations 3
   ```

3. **ML_JOB:**
   ```bash
   python scripts/profile_jobs.py --run ml_job --iterations 1
   ```

### Updating This Document

After profiling:
1. Update runtime benchmarks table with median runtime
2. Identify top-3 hotspots from profiler output (cProfile, py-spy, etc.)
3. Document optimization opportunities

---

## Before/After P3 Performance Comparison

**Purpose:** Track performance improvements from P3 optimizations (P3.1-P3.4).

**P3 Optimizations:**
- P3.1: Backtest Engine Refactoring (loop extraction)
- P3.2: Vectorization (NumPy-based calculations)
- P3.3: Numba Integration (optional JIT compilation)
- P3.4: Factor Store Caching (precomputed features)

### Performance Tracking Table

| Run ID | Phase | Date | Runtime (s) | Top-1 Hotspot | Top-2 Hotspot | Top-3 Hotspot | Notes |
|--------|-------|------|------------|---------------|---------------|---------------|-------|
| BACKTEST_MEDIUM | Before P3 | - | - | - | - | - | Baseline to be collected |
| BACKTEST_MEDIUM | After P3 | - | - | - | - | - | Post-optimization to be collected |
| EOD_SMALL | Before P3 | - | - | - | - | - | Baseline to be collected |
| EOD_SMALL | After P3 | - | - | - | - | - | Post-optimization to be collected |
| ML_JOB | Before P3 | - | - | - | - | - | Baseline to be collected |
| ML_JOB | After P3 | - | - | - | - | - | Post-optimization to be collected |

### Profiling Commands

**Before P3 (Baseline):**
```bash
# Profile BACKTEST_MEDIUM
python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler cprofile --update-doc

# Profile EOD_SMALL
python scripts/profile_job.py --job EOD_SMALL --profiler cprofile --update-doc

# Profile ML_JOB
python scripts/profile_job.py --job ML_JOB --profiler cprofile --update-doc
```

**After P3 (Post-Optimization):**
```bash
# Repeat the same commands after P3 optimizations are complete
python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler cprofile --update-doc
python scripts/profile_job.py --job EOD_SMALL --profiler cprofile --update-doc
python scripts/profile_job.py --job ML_JOB --profiler cprofile --update-doc
```

**Note:** The `--update-doc` flag automatically appends hotspots and runtime to this document. To mark a run as "Before P3" or "After P3", manually edit the phase identifier in the appended section headers (or enhance the script with a `--phase` flag).

---

## Related Documents

- `docs/BACKTEST_ENGINE_OPTIMIZATION_P3.md` - Backtest engine optimization details
- `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` - Factor labs and performance work
- `scripts/profile_job.py` - Profiling harness

---

**Last Updated:** 2025-01-XX (P1 baseline created, P3 tracking section added)
