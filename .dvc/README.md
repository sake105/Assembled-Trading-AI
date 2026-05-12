# DVC (Data Version Control) Scaffold

> Audit C2-045 — version the raw price-data parquet so backtests stay
> reproducible across vendor data revisions (yfinance, Polygon, Alpaca
> all silently rewrite history occasionally).

Status today: **scaffold only**. The `dvc` Python package is not
currently in the project venv. To activate:

```bash
pip install 'dvc[s3]'
cp .dvc/config.example .dvc/config  # then fill in the placeholders
dvc remote add -d backblaze s3://<bucket>/dvc \
  --endpoint-url https://s3.eu-central-003.backblazeb2.com
dvc add data/raw/<panel>.parquet
git add data/raw/<panel>.parquet.dvc .gitignore
git commit -m "data(v1): pin initial panel"
dvc push
```

After that, every `dvc pull` on a fresh clone fetches the exact panel
the backtest expected. The `Data hash / DVC tag` field in
`docs/RESEARCH_REGISTER_TEMPLATE.md` becomes meaningful.

## What lives in this directory

- `config.example` — template config; the real `config` is **gitignored**.
- `.gitignore` — keeps the real config + cache out of git.

The DVC tracker files (`*.dvc`) themselves are committed to git — they
contain only hashes + remote paths, no data.

## What this scaffold is NOT

- Not actively used until `dvc` is installed and at least one parquet
  has been `dvc add`-ed.
- Not a replacement for the rclone-based audit-log backup (see
  `scripts/ops/setup_b2_backup.sh`); the two coexist:
  - rclone → audit JSONL replication, 10y retention
  - DVC    → data-panel versioning
