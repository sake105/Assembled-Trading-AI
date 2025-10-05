- `Sprint2_Update_v24/` (your Sprint 2 package, as provided)
  - All scripts auto-load `.env` for API keys via `python-dotenv`
  - `scripts/migrate_raw_layout.py`: migrates `data/raw/*.csv|*.parquet` -> `data/raw/<TICKER>/<YYYY>.parquet`
  - `requirements.txt` now includes `python-dotenv`
- `.env` (ALPHAVANTAGE_KEY, FINNHUB_KEY) — **sensitive**, do not share.
- `run_all_sprint2.ps1` — one-click runner for migration + screener + events + news
1. Create/activate venv, then install requirements:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -U -r .\Sprint2_Update_v24
equirements.txt
   ```
2. Place raw files (CSV/Parquet) under `data/raw/` or run your backfill first.
3. Run everything:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .
un_all_sprint2.ps1
   ```
- `.env` is loaded automatically by each Sprint-2 script — no code changes needed.
- Keep `.env` private. Rotate keys if the zip is shared externally.
