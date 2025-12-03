# Workflows – Daily EOD Pipeline & QA

## 1. Overview

**Goal:** Run the end-of-day pipeline, generate signals/orders, and review QA reports.

**Main components:**
- Data loading & resampling
- Feature generation
- Signal generation
- Position sizing & order file (SAFE CSV)
- Daily QA report
- Logging + optional experiment tracking

---

## 2. Prerequisites

### Python Version
- Python 3.10+ (as specified in `pyproject.toml`)

### Installation
```bash
pip install -e .[dev]
```

This installs:
- Core dependencies (pandas, numpy, fastapi, yfinance, etc.)
- Dev dependencies (pytest, ruff, black, mypy)

### Environment Configuration (Optional)

You can configure defaults via environment variables. Create a `.env` file in the repository root:

```bash
# .env file
ASSEMBLED_DATA_SOURCE=yahoo          # Default: "local"
ASSEMBLED_RUNTIME_PROFILE=DEV         # Default: "DEV"
```

**Available settings:**
- `ASSEMBLED_DATA_SOURCE`: `"local"` (offline) or `"yahoo"` (live)
- `ASSEMBLED_RUNTIME_PROFILE`: `"BACKTEST"`, `"PAPER"`, or `"DEV"`
- `ASSEMBLED_KILL_SWITCH`: `"1"` or `"true"` to block all orders (Phase 10)

See `src/assembled_core/config/settings.py` for all available settings.

---

## 3. Standard EOD Commands

### 3.1 Offline / Local Mode (Default)

**Basic command:**
```bash
python scripts/cli.py run_daily --freq 1d --data-source local
```

**What it does:**
- Uses existing local Parquet files from `output/aggregates/`
- Complete pipeline: Features → Signals → Orders → QA
- Good for repro tests and offline development
- No internet connection required

### 3.2 Live / Yahoo Mode (Current Market Data)

**Basic command:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --data-source yahoo \
  --symbols AAPL MSFT GOOGL \
  --end-date today
```

**What it does:**
- Fetches fresh OHLCV data from Yahoo Finance
- Builds features, signals, positions, orders
- Caches live data to `output/aggregates/{freq}_live_cache.parquet`
- Clear error messages if rate-limited

**Note:** Yahoo Finance has rate limits. If you get rate-limited:
- Use fewer symbols
- Add delays between requests
- Use local data source as fallback

### 3.3 With Experiment Tracking (Recommended)

**Command:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --data-source yahoo \
  --symbols AAPL MSFT GOOGL \
  --end-date today \
  --track-experiment \
  --experiment-name "daily_yahoo_smoke_2025-12-03" \
  --experiment-tags "daily,live,yahoo"
```

**What you get:**
- Experiment run in `experiments/{run_id}/`:
  - `run.json` → Config + Tags
  - `metrics.csv` → Logged metrics
  - `artifacts/` → Reports, plots (if available)
- Parallel outputs: Orders, QA reports, logs (`logs/{run_id}.log`)

### 3.4 Using Universe Files

Instead of specifying `--symbols` each time, use a universe file:

**Create universe file** (`config/universe_live_equities.txt`):
```
AAPL
MSFT
GOOGL
AMZN
NVDA
# Comments start with #
```

**Command:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --universe config/universe_live_equities.txt \
  --end-date today \
  --track-experiment \
  --experiment-name "daily_live_default_universe" \
  --experiment-tags "daily,live,default"
```

### 3.5 Quick Setup for Daily Development

**1. Set default data source in `.env`:**
```bash
ASSEMBLED_DATA_SOURCE=yahoo
```

**2. Create universe file** (`config/universe_live_equities.txt`) with your standard symbols (10-20 for development).

**3. Your standard daily command:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --universe config/universe_live_equities.txt \
  --end-date today \
  --track-experiment \
  --experiment-name "daily_live_default_universe" \
  --experiment-tags "daily,live,default"
```

Since `ASSEMBLED_DATA_SOURCE=yahoo` is set, you can even omit `--data-source`:
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --universe config/universe_live_equities.txt \
  --end-date today
```

---

## 4. Running the Daily EOD Pipeline (Detailed Reference)

### 3.4.1 Using the Console Script

The recommended way to run the EOD pipeline:

```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --universe watchlist.txt \
  --start-date 2024-01-01 \
  --end-date 2024-01-31
```

### 3.4.2 Important Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--freq` | `{1d,5min}` | ✅ Yes | Trading frequency: `1d` for daily or `5min` for 5-minute bars |
| `--universe` | `FILE` | ❌ No | Path to universe file (default: `watchlist.txt` in repo root) |
| `--price-file` | `FILE` | ❌ No | Explicit path to price file (overrides default path) |
| `--start-date` | `YYYY-MM-DD` | ❌ No | Start date for price data filtering |
| `--end-date` | `YYYY-MM-DD` or `today` | ❌ No | End date for price data filtering. Use `today` for live data. |
| `--data-source` | `{local,yahoo}` | ❌ No | Data source type: `local` (Parquet files) or `yahoo` (Yahoo Finance API). Default: from `settings.data_source` |
| `--symbols` | `SYMBOL ...` | ❌ No | List of symbols to load (e.g., `--symbols AAPL MSFT GOOGL`). Overrides universe file. |
| `--start-capital` | `AMOUNT` | ❌ No | Starting capital in USD (default: 10000.0) |
| `--skip-backtest` | Flag | ❌ No | Skip backtest step in pipeline |
| `--skip-portfolio` | Flag | ❌ No | Skip portfolio simulation step |
| `--skip-qa` | Flag | ❌ No | Skip QA checks step |
| `--commission-bps` | `BPS` | ❌ No | Commission in basis points (overrides default cost model) |
| `--spread-w` | `WEIGHT` | ❌ No | Spread weight for cost model (overrides default) |
| `--impact-w` | `WEIGHT` | ❌ No | Market impact weight for cost model (overrides default) |
| `--out` | `DIR` | ❌ No | Output directory (default: `output/` from config) |
| `--profile` | `{BACKTEST,PAPER,DEV}` | ❌ No | Runtime profile (default: DEV) |

### 3.4.3 Examples

**Basic daily run:**
```bash
python scripts/cli.py run_daily --freq 1d
```

**With custom universe and capital:**
```bash
python scripts/cli.py run_daily --freq 1d --universe watchlist.txt --start-capital 50000
```

**With explicit price file:**
```bash
python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet
```

**With date range:**
```bash
python scripts/cli.py run_daily --freq 1d --start-date 2024-01-01 --end-date 2024-12-31
```

**With live data from Yahoo Finance:**
```bash
python scripts/cli.py run_daily --freq 1d --data-source yahoo --symbols AAPL MSFT GOOGL --end-date today
```

---

## 3.5 Using Live Data Sources

The EOD pipeline supports both **local** (historical/offline) and **live** (online) data sources.

### Local Data Source (Default)

The default data source is `local`, which loads price data from Parquet files in `output/aggregates/`:

```bash
python scripts/cli.py run_daily --freq 1d --data-source local
```

This is the traditional mode that works with pre-downloaded and resampled price data.

### Yahoo Finance Data Source (Live)

To use live/current market data from Yahoo Finance:

```bash
python scripts/cli.py run_daily --freq 1d --data-source yahoo --symbols AAPL MSFT GOOGL --end-date today
```

**Requirements:**
- `yfinance` package (already included in dependencies)
- Internet connection
- Valid stock symbols

**Features:**
- Automatically fetches current market data
- Supports both daily (`1d`) and intraday (`5min`) frequencies
- Caches fetched data to `output/aggregates/{freq}_live_cache.parquet` for future use
- Handles date ranges and symbol filtering

**Configuration:**

You can set the default data source in `settings.py` or via environment variable:

```bash
# Set default data source via environment variable
export ASSEMBLED_DATA_SOURCE=yahoo

# Or in .env file
ASSEMBLED_DATA_SOURCE=yahoo
```

**Example: Fetch today's data for specific symbols:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --data-source yahoo \
  --symbols AAPL MSFT GOOGL \
  --start-date 2024-01-01 \
  --end-date today
```

**Note:** When using `yahoo` data source, the `--price-file` argument is ignored. The system fetches data directly from Yahoo Finance API.

---

## 4. Outputs & File Structure

The pipeline writes outputs to the `output/` directory (configurable via `--out` or `ASSEMBLED_OUTPUT_DIR` environment variable).

### 4.1 Order Files (SAFE CSV)

**Path:** `output/orders_{freq}.csv`

**Column Structure:**
- `timestamp`: Order timestamp (UTC)
- `symbol`: Stock symbol
- `side`: Order side (`BUY` or `SELL`)
- `qty`: Quantity (shares)
- `price`: Order price

**Example:**
```csv
timestamp,symbol,side,qty,price
2024-01-15 00:00:00+00:00,AAPL,BUY,10.0,150.25
2024-01-15 00:00:00+00:00,MSFT,BUY,5.0,380.50
```

### 4.2 Equity Curves

**Backtest Equity (without costs):**
- **Path:** `output/equity_curve_{freq}.csv`
- **Columns:** `timestamp`, `equity`

**Portfolio Equity (with costs):**
- **Path:** `output/portfolio_equity_{freq}.csv`
- **Columns:** `timestamp`, `equity`

### 4.3 Performance Reports

**Backtest Report:**
- **Path:** `output/performance_report_{freq}.md`
- Contains: Sharpe ratio, CAGR, max drawdown, total trades, etc.

**Portfolio Report:**
- **Path:** `output/portfolio_report_{freq}.md`
- Contains: Portfolio metrics with transaction costs

### 4.4 QA Reports

**Path:** `output/reports/qa_report_{strategy}_{freq}_{date}.md`

**Contents:**
- Performance metrics summary
- QA gate results (OK/WARNING/BLOCK)
- Equity curve reference
- Data status
- Configuration details

### 4.5 Aggregated Data

**Path:** `output/aggregates/{freq}.parquet`

Contains resampled price data with features.

---

## 5. Logging

### 5.1 Log Files

**Location:** `logs/{run_id}.log`

**Run ID Format:** `{prefix}_{YYYYMMDD}_{HHMMSS}_{uuid8}`

Example: `eod_20250115_143022_a1b2c3d4.log`

### 5.2 Finding Logs for a Specific Run

The Run ID is printed at the start of each pipeline execution. To find logs:

```bash
# List all logs
ls logs/

# Search for a specific run ID
grep -r "run_id_here" logs/

# View latest log
tail -f logs/eod_*.log
```

### 5.3 Log Format

**Console Output:**
```
[INFO] Pipeline started
[INFO] Run-ID: eod_20250115_143022_a1b2c3d4
```

**File Output:**
```
2025-01-15 14:30:22 | INFO     | scripts.cli              | [eod_20250115_143022_a1b2c3d4] Pipeline started
```

---

## 6. Daily QA Workflow

### 6.1 Generating QA Reports

QA reports are automatically generated when running the EOD pipeline (unless `--skip-qa` is used).

**Manual QA Report Generation:**

The QA report is generated as part of the pipeline. To inspect it:

```bash
# View the latest QA report
cat output/reports/qa_report_*.md

# Or open in your editor
code output/reports/qa_report_trend_baseline_1d_20250115.md
```

### 6.2 Typical QA Checks

The QA report includes:

1. **Performance Metrics:**
   - Total return, CAGR, Sharpe ratio
   - Max drawdown, volatility
   - Total trades, win rate

2. **QA Gates:**
   - **OK**: All checks passed
   - **WARNING**: Some metrics outside ideal range
   - **BLOCK**: Critical issues detected

3. **Data Status:**
   - Data range, frequency
   - Missing data indicators

4. **Configuration:**
   - Strategy parameters
   - Cost model settings
   - Capital allocation

### 6.3 Interpreting QA Results

**OK Status:**
- All gates passed
- Performance metrics within acceptable ranges
- No data quality issues

**WARNING Status:**
- Some metrics may be suboptimal (e.g., low Sharpe ratio, high drawdown)
- Review configuration and data quality
- Consider adjusting strategy parameters

**BLOCK Status:**
- Critical issues detected (e.g., excessive drawdown, data gaps)
- Trading should be paused until issues are resolved
- Review logs and data sources

---

## 7. Optional – Experiment Tracking for EOD Runs

### 7.1 Enabling Experiment Tracking

Currently, experiment tracking is primarily available for backtests (`run_backtest` subcommand). For EOD runs, you can manually track runs by:

1. Noting the Run ID from logs
2. Saving relevant outputs (orders, reports) to a tracking system
3. Documenting configuration in a research notebook

**Future Enhancement:** Direct experiment tracking support for `run_daily` is planned.

### 7.2 Manual Experiment Documentation

For now, document EOD runs manually:

```markdown
# EOD Run: 2025-01-15

**Run ID:** eod_20250115_143022_a1b2c3d4

**Configuration:**
- Frequency: 1d
- Universe: watchlist.txt
- Start Capital: 10000.0
- Date Range: 2024-01-01 to 2024-12-31

**Results:**
- Orders: output/orders_1d.csv (45 orders)
- QA Report: output/reports/qa_report_trend_baseline_1d_20250115.md
- Sharpe Ratio: 1.23
- Max Drawdown: -12.5%

**Notes:**
- All QA gates passed
- Performance within acceptable range
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**No data found / empty universe:**
```bash
# Check universe file exists and has symbols
cat watchlist.txt

# Check price data exists
ls data/raw/1min/*.parquet
ls data/sample/eod_sample.parquet
```

**Missing output directories:**
- The pipeline automatically creates output directories
- If issues persist, check file permissions:
  ```bash
  mkdir -p output/reports
  chmod -R 755 output/
  ```

**File not found in QA report test:**
- Ensure the pipeline completed successfully
- Check that all required steps ran (execute, backtest, portfolio, QA)
- Review logs for errors:
  ```bash
  tail -n 100 logs/eod_*.log
  ```

### 8.2 Debugging Using Logs

**View latest log:**
```bash
tail -f logs/eod_*.log
```

**Search for errors:**
```bash
grep -i "error\|exception\|failed" logs/eod_*.log
```

**Check specific Run ID:**
```bash
grep "run_id_here" logs/*.log
```

### 8.3 Getting Help

- Check existing documentation:
  - `docs/CLI_REFERENCE.md` - CLI command reference
  - `docs/ARCHITECTURE_BACKEND.md` - Architecture overview
  - `docs/KNOWN_ISSUES.md` - Known issues and workarounds

- Run with verbose logging:
  - Logs are automatically written to `logs/`
  - Check log files for detailed error messages

---

## 9. Related Documentation

- [CLI Reference](CLI_REFERENCE.md) - Complete CLI command reference
- [Architecture Backend](ARCHITECTURE_BACKEND.md) - System architecture
- [Phase 8 Risk Engine](PHASE8_RISK_ENGINE.md) - Risk management
- [Phase 9 Model Governance](PHASE9_MODEL_GOVERNANCE.md) - Model validation
- [Known Issues](KNOWN_ISSUES.md) - Known issues and workarounds

