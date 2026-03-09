# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Assembled Trading AI is a pure Python quantitative trading backend (no Docker, no external services required). Source code lives in `src/assembled_core/`, scripts in `scripts/`, tests in `tests/`. See `docs/cursor/CONTEXT_PACK.md` for full architecture and `README.md` for CLI usage.

### Environment

- **Python 3.12** with virtualenv at `.venv`
- Activate: `source /workspace/.venv/bin/activate`
- Install: `pip install -e ".[dev]" && pip install exchange-calendars`

### Running the Pipeline (E2E)

The full trading pipeline runs on local demo data. Seed data first, then run 3 steps:

```bash
# 1. Create 5min aggregated data from demo 1min CSV
python -c "
import pandas as pd
raw = pd.read_csv('data/raw/1min/demo_1min.csv')
raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
agg = raw.set_index('timestamp').groupby('symbol')['close'].resample('5min').last().dropna().reset_index()
agg = agg.sort_values(['timestamp','symbol'])[['symbol','timestamp','close']]
agg.to_parquet('output/aggregates/5min.parquet', index=False)
"

# 2. Execute → Backtest → Portfolio
python scripts/sprint9_execute.py --freq 5min
python scripts/sprint9_backtest.py --freq 5min --start-capital 10000
python scripts/sprint10_portfolio.py --freq 5min --start-capital 10000
```

**Important**: `load_prices()` in `pipeline/io.py` expects `output/aggregates/5min.parquet`.

### Lint / Test / Build

See `README.md` and `docs/TESTING_COMMANDS.md` for full details. Quick reference:

- **Lint**: `ruff check src tests scripts --exclude scripts/tools --exclude scripts/00_seed_demo_data.py` (76 pre-existing findings as of Sprint 13)
- **Tests**: `pytest -m "not advanced" -q --maxfail=3 --tb=short` (CI-equivalent command)
- **Key phase tests**: `pytest tests/test_cli.py tests/test_features_ta.py tests/test_qa_metrics.py tests/test_qa_gates.py tests/test_execution_kill_switch.py tests/test_qa_risk_metrics.py -v` (all pass)

### FastAPI Server

```bash
python scripts/run_api.py   # binds to 0.0.0.0:8000
```

20+ REST endpoints at `/api/v1/`. Test: `curl http://localhost:8000/api/v1/orders/5min`

### .gitignore Caveat

The `.gitignore` pattern `data/` also matches `src/assembled_core/data/`. When adding new files under `src/assembled_core/data/`, use `git add -f src/assembled_core/data/<file>`.

### Known Issues (pre-existing)

- ~19 test files fail collection due to incomplete stub functions in `src/assembled_core/data/` (tests expect functions beyond minimal stubs)
- Ruff reports 76 lint findings (unused imports, etc.) — CI also shows these
- Some test failures in `test_qa_backtest_engine.py` and others relate to evolving data contracts
