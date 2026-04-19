# Secrets Configuration

This directory holds secret-key **templates** only. Actual keys are injected via environment variables.

**NEVER commit actual API keys, tokens, or passwords here.**

## Pattern

1. Copy `.env.template` to `.env` in the repo root (already gitignored)
2. Set actual values in `.env`
3. Workers load via `src/assembled_core/config/secrets_loader.py`

## Required environment variables (by source)

| Variable | Source | Required for |
|----------|--------|-------------|
| `NEWSAPI_KEY` | newsapi.org | T5.1 multi-source news |
| `REUTERS_API_KEY` | Reuters Connect | T5.1 |
| `AP_API_KEY` | AP DataConnect | T5.1 |
| `ACLED_API_KEY` | ACLED | T5.5 (when implemented) |
| `WORLD_BANK_API_KEY` | World Bank | T5.9 (optional) |
| `IMF_API_KEY` | IMF Data | optional |
| `EIA_API_KEY` | EIA | optional |
| `ASSEMBLED_SLACK_WEBHOOK_URL` | Slack | alerts.sinks.slack |
| `ASSEMBLED_SMTP_HOST` | SMTP server | alerts.sinks.email |
| `ASSEMBLED_SMTP_USER` | SMTP user | alerts.sinks.email |
| `ASSEMBLED_SMTP_PASS` | SMTP password | alerts.sinks.email |

## Security rules (CLAUDE.md §20)

- `.env` is gitignored — never commit it
- If a key is committed by accident: rotate immediately, do NOT just delete the file
- Use `configs/secrets/` only for templates and documentation, never live keys
