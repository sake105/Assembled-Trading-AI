# Operations & Monitoring Runbook

**Last Updated:** 2025-01-XX  
**Status:** Active

This runbook provides daily and weekly operational procedures for monitoring the health of the backend trading system. It covers health checks, status interpretation, troubleshooting, and automation recommendations.

---

## 1. Daily Checklist

**Note:** For Paper Track, you can enable strict PIT checks by setting the `PAPER_TRACK_STRICT_PIT_CHECKS=true` environment variable. This helps detect look-ahead bias in feature computation.

### Step-by-Step Procedure

1. **Check EOD-Runs Status**
   - Navigate to `output/backtests/` directory
   - Identify the latest backtest experiment directory (by modification time or naming convention)
   - Verify expected files exist:
     - `equity_curve.csv` or `equity_curve.parquet`
     - `risk_report.md` (optional but recommended)
     - `risk_summary.csv` (optional but recommended)
   - Check for error logs or warnings in recent pipeline runs

2. **Run Health Check**
   ```bash
   python scripts/cli.py check_health --backtests-root output/backtests/
   ```
   Or use the standalone script:
   ```bash
   python scripts/check_health.py --backtests-root output/backtests/
   ```

3. **Interpret Health Status**
   - Review the output (`health_summary.md` or console output)
   - Check overall status: `OK`, `WARN`, `CRITICAL`, or `SKIP`
   - Review individual checks for details

4. **Paper Track Catch-Up (if applicable)**
   ```bash
   # Run paper track for yesterday (or catch up missing days)
   python scripts/cli.py paper_track \
       --config-file configs/paper_track/strategy_core_ai_tech.yaml \
       --as-of $(date -d "yesterday" +%Y-%m-%d)  # Linux/Mac
   # Or on Windows PowerShell:
   # --as-of $((Get-Date).AddDays(-1).ToString("yyyy-MM-dd"))
   ```
   Or catch up a date range:
   ```bash
   python scripts/cli.py paper_track \
       --config-file configs/paper_track/strategy_core_ai_tech.yaml \
       --start-date 2025-01-15 \
       --end-date 2025-01-20
   ```

5. **Check Paper Track Status (if applicable)**
   ```bash
   python scripts/cli.py check_health \
       --backtests-root output/backtests/ \
       --paper-track-root output/paper_track/ \
       --paper-track-days 3
   ```
   Or skip if paper track is not set up:
   ```bash
   python scripts/cli.py check_health \
       --backtests-root output/backtests/ \
       --skip-paper-track-if-missing
   ```
   This checks:
   - Paper track freshness (last run within 3 days by default)
   - Artefacts presence (state file, daily summaries)
   - Metrics plausibility (daily PnL spikes, drawdown bounds)

6. **Optional: Review Detailed Reports**
   - Open `risk_report.md` for comprehensive risk metrics
   - Check `tca_report.md` for transaction cost analysis (if available)
   - Review `factor_exposures_summary.csv` for factor attribution (if available)
   - Review paper track daily summaries: `output/paper_track/{strategy}/runs/{YYYYMMDD}/daily_summary.md`

### Expected Outputs

After running the health check, you should see:
- `output/health/health_summary.json`: Machine-readable health status
- `output/health/health_summary.md`: Human-readable report

---

## 2. Weekly Checklist

1. **Review Walk-Forward / Regime Reports**
   - Update walk-forward analysis if new data is available
   - Review regime breakdowns and performance by market regime
   - Check for regime shifts that might impact strategy performance

2. **Batch Backtests Review**
   - Review results from batch backtest runs (if applicable)
   - Compare performance across different parameter sets
   - Identify top-performing configurations

3. **Storage & Logs Audit**
   - Check disk space usage in `output/` directory
   - Review log files for recurring errors or warnings
   - Archive or clean up old backtest outputs if needed

4. **Factor & Model Validation**
   - Review ML validation reports (`output/ml_validation/`)
   - Check factor ranking summaries
   - Review deflated Sharpe ratios to identify robust factors

---

## 3. Health Check Usage

### Basic Usage

```bash
# Basic health check with default settings
python scripts/cli.py check_health --backtests-root output/backtests/

# With custom output directory
python scripts/cli.py check_health --backtests-root output/backtests/ --output-dir output/custom_health/

# JSON output format (for automation)
python scripts/cli.py check_health --backtests-root output/backtests/ --format json
```

### Advanced Usage

```bash
# With benchmark correlation check
python scripts/cli.py check_health \
    --backtests-root output/backtests/ \
    --benchmark-symbol SPY \
    --benchmark-file data/benchmarks/spy_returns.csv

# With stricter thresholds
python scripts/cli.py check_health \
    --backtests-root output/backtests/ \
    --min-sharpe 0.5 \
    --max-drawdown-min -0.30 \
    --max-turnover 5.0

# With custom lookback window
python scripts/cli.py check_health \
    --backtests-root output/backtests/ \
    --days 30
```

### Command-Line Arguments

- `--backtests-root`: Root directory containing backtest outputs (default: `output/backtests/`)
- `--days`: Lookback window in days for freshness checks (default: 60)
- `--benchmark-symbol`: Benchmark symbol for correlation checks (optional)
- `--benchmark-file`: Path to benchmark returns file (optional)
- `--output-dir`: Output directory for health reports (default: `output/health/`)
- `--format`: Output format - `text`, `json`, or `both` (default: `text`)
- `--min-sharpe`: Minimum acceptable Sharpe ratio (default: 0.0)
- `--max-drawdown-min`: Minimum acceptable max drawdown, more negative = worse (default: -0.40)
- `--max-drawdown-max`: Maximum acceptable max drawdown, less negative = better (default: 0.0)
- `--max-turnover`: Maximum acceptable turnover (default: 10.0)
- `-v, --verbose`: Enable verbose logging

### Status Levels & Actions

#### OK Status
- **Meaning**: All checks passed within expected ranges
- **Action**: No immediate action required
- **Frequency**: Continue daily monitoring

#### WARN Status
- **Meaning**: One or more checks are outside expected ranges but not critical
- **Typical Causes**:
  - Risk report missing (non-critical but recommended)
  - Metrics slightly outside thresholds (e.g., Sharpe slightly below target)
  - Backtest data slightly stale (within 2x lookback window)
- **Actions**:
  - Review detailed check output in `health_summary.md`
  - Check recent backtest runs and market events
  - Review risk metrics in detail reports
  - Monitor trends over next few days

#### CRITICAL Status
- **Meaning**: One or more critical checks failed
- **Typical Causes**:
  - Equity curve file missing or empty
  - Backtest data extremely stale (> 2x lookback window)
  - Severe drawdowns exceeding thresholds
  - Negative Sharpe ratios far below minimum
- **Actions**:
  - **Immediate**: Review EOD/backtest pipeline logs for errors
  - Check if daily runs completed successfully
  - Review recent trades and positions for anomalies
  - Consider pausing automated strategies if issues persist
  - Review market conditions and regime changes
  - Check system resources (disk space, memory, network)

#### SKIP Status
- **Meaning**: Some checks were skipped (e.g., optional checks with missing data)
- **Action**: Review which checks were skipped and why
- **Note**: SKIP is not an error condition, but indicates optional data is missing

---

## 4. Troubleshooting

### Problem: Missing Equity Curve File

**Symptoms**: `equity_curve_exists` check returns CRITICAL

**Possible Causes**:
- Backtest pipeline failed before writing equity curve
- File path mismatch between pipeline and health check
- Permission issues preventing file creation

**Solution Steps**:
1. Check backtest pipeline logs for errors
2. Verify backtest directory exists and contains expected files
3. Re-run backtest if necessary
4. Check file permissions on output directory

### Problem: Stale Backtest Data

**Symptoms**: `backtest_freshness` check returns WARN or CRITICAL

**Possible Causes**:
- Daily EOD pipeline not running
- Pipeline errors preventing completion
- Network/data source issues

**Solution Steps**:
1. Check if EOD pipeline is scheduled and running
2. Review pipeline logs for errors
3. Verify data source availability
4. Check system resources (disk space, network connectivity)

### Problem: Risk Report Missing

**Symptoms**: `risk_report_exists` check returns WARN

**Possible Causes**:
- Risk report generation not enabled in pipeline
- Report generation failed silently
- Path mismatch between generation and health check

**Solution Steps**:
1. Verify risk report generation is enabled in pipeline configuration
2. Manually generate risk report: `python scripts/cli.py risk_report --backtest-dir <dir>`
3. Check for errors in risk report generation logs

### Problem: Metrics Outside Expected Ranges

**Symptoms**: Individual metric checks (Sharpe, Drawdown, Turnover) return WARN or CRITICAL

**Possible Causes**:
- Market regime change affecting strategy performance
- Strategy parameter drift or configuration errors
- Data quality issues
- Legitimate strategy underperformance

**Solution Steps**:
1. Review detailed risk metrics in `risk_report.md`
2. Check market conditions and regime analysis
3. Compare current metrics to historical averages
4. Review recent trades and positions for anomalies
5. Consider walk-forward analysis to validate strategy robustness
6. Review factor exposures for unexpected changes

### Problem: Benchmark Correlation Outside Range

**Symptoms**: `benchmark_correlation` check returns WARN

**Possible Causes**:
- Strategy behavior changed (legitimately or due to bug)
- Benchmark data quality issues
- Market regime shift affecting correlation

**Solution Steps**:
1. Verify benchmark data is correct and up-to-date
2. Review factor exposures for changes
3. Check if strategy parameters were modified
4. Review regime analysis for context

### Problem: High Turnover

**Symptoms**: `turnover` check returns WARN

**Possible Causes**:
- Position sizing logic issues
- Signal generation producing excessive trading
- Rebalancing frequency too high

**Solution Steps**:
1. Review position sizing configuration
2. Check signal generation frequency and logic
3. Review transaction costs impact (TCA report)
4. Consider adjusting rebalancing parameters

---

## 5. Integration & Automation

### Daily Automation

**Recommended**: Run health checks automatically after daily EOD pipeline completion.

**Cron Example** (Linux/Mac):
```bash
# Run health check daily at 9:00 AM UTC (after EOD pipeline)
0 9 * * * cd /path/to/repo && .venv/bin/python scripts/check_health.py --backtests-root output/backtests/ --format json --output-dir output/health/ >> logs/health_check.log 2>&1
```

**Task Scheduler Example** (Windows):
- Create scheduled task to run `check_health.py` daily
- Set working directory to repository root
- Capture output to log file

### Integration with Job Profiling

If `scripts/profile_jobs.py` is extended with `OPERATIONS_HEALTH_CHECK` job type:

```bash
# Profile health check performance
python scripts/profile_jobs.py --job-type OPERATIONS_HEALTH_CHECK --iterations 5
```

### Alerting

**Recommended Setup**:
- Parse `health_summary.json` for overall status
- Send alerts (email, Slack, etc.) if status is CRITICAL
- Optionally alert on WARN status (configurable)

**Example Alert Script** (pseudo-code):
```python
import json
import sys

with open("output/health/health_summary.json") as f:
    data = json.load(f)

if data["overall_status"] == "CRITICAL":
    # Send alert
    send_alert("Health check CRITICAL", data)
    sys.exit(2)
elif data["overall_status"] == "WARN":
    # Optional: send warning
    send_warning("Health check WARN", data)
    sys.exit(1)
```

---

## 6. References

### Design Documents
- [Operations Backend Design (A3)](OPERATIONS_BACKEND_A3_DESIGN.md): Detailed design document for health check system

### Workflow Documentation
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md): Comprehensive guide to risk reporting and analysis
- [Batch Backtests & Parallelization](WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md): Guide to batch backtesting workflows

### Related Documentation
- [ML Validation Experiments](ML_VALIDATION_EXPERIMENTS.md): ML model validation and factor analysis
- [Walk-Forward & Regime Analysis Design (B3)](WALK_FORWARD_AND_REGIME_B3_DESIGN.md): Out-of-sample validation and regime analysis

### CLI Commands
- `scripts/cli.py check_health`: Health check CLI command
- `scripts/cli.py risk_report`: Risk report generation
- `scripts/cli.py run_backtest`: Strategy backtest execution
- `scripts/cli.py run_daily`: Daily EOD pipeline execution

---

## Appendix: Quick Reference

### Health Check Exit Codes
- `0`: OK or SKIP (no action needed)
- `1`: WARN (review recommended)
- `2`: CRITICAL (immediate action required)

### Default Thresholds
- Min Sharpe: 0.0
- Max Drawdown Range: [-0.40, 0.0]
- Max Turnover: 10.0
- Lookback Window: 60 days

### File Locations
- Health Reports: `output/health/`
- Backtest Outputs: `output/backtests/<experiment_id>/`
- Risk Reports: `output/backtests/<experiment_id>/` or `output/risk_reports/<experiment_id>/`
- Logs: Check pipeline-specific log directories

