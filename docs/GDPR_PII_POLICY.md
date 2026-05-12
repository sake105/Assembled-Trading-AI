# GDPR / DSGVO PII Retention Policy

> Audit C4-090 — the News-ingestion pipeline collects author names,
> handles, and bylines (Article 4(1) GDPR personal data). This policy
> defines retention, the Article-17 deletion contract, and the data-
> processor role.
>
> **Status today:** the system is operated by a private trader (Hans,
> single natural person) for own-account research; no third-party
> data subjects are downstream. The policy still applies because we
> *process* PII (Article 4(2)) when scraping news feeds — the
> retention rules below MUST be honoured even before any commercial
> setup.

## 1. PII surfaces in the codebase

| Source | Field | Article-9? | Retention cap |
|---|---|---|---|
| RSS news feeds | `author`, `byline`, `author_handle` | No | 30 days |
| GDELT-XML headlines | `source_publisher`, sometimes byline | No | 30 days |
| Polygon news API | `author` | No | 30 days |
| Twitter / X if added later (DO NOT collect without DSGVO review) | `user.screen_name`, `user.id_str` | No | 0 days — refuse |

## 2. Retention contract

- **Hard cap**: any field listed above is deleted from the canonical
  store no later than **30 calendar days** after first ingestion.
- **Sentiment / score derivatives** computed from the news text are
  NOT PII as long as they do not identify an author. They may be
  retained indefinitely.
- Implementation: `ASSEMBLED_NEWS_PII_RETENTION_DAYS` env (default
  30). A daily cron (`scripts/ops/purge_pii_aged.py`, future) MUST
  drop expired rows.

## 3. Article-17 deletion endpoint

When a data subject asks for deletion:

1. Operator receives the request via `dataprivacy@<domain>` (set up
   today; for now any incoming email goes to the same single inbox).
2. Within 30 days (GDPR Art. 12(3)), purge every row in the news
   tables WHERE `author` matches (case-insensitive) OR `author_handle`
   matches.
3. Confirm in writing including which tables/files were touched.
4. Append a record to `output/ops/gdpr_deletion_audit.jsonl`:
   ```json
   {"ts": "...", "subject_hash": "<sha256 of normalized name>",
    "rows_deleted": N, "tables": ["news_articles", "news_sentiment"],
    "operator": "..."}
   ```
   (Hash, not raw name, so the audit log itself is privacy-safe.)

A FastAPI endpoint stub (`POST /api/v1/gdpr/delete`) is the eventual
hosted form of the above — currently the runbook is the artefact.

## 4. Data-processor role

- **Controller** today: the operator (Hans).
- **Processors today**: yfinance, Alpaca, Polygon, FRED, SEC-EDGAR.
  These are not Article 28 processors in the legal sense — they are
  *data sources* that provide already-public data. We never ship PII
  upstream to them.
- **Processors if commercial**: if at any point a third party gets
  read-access to our news/PII tables (e.g. a partner analyst),
  Article 28 processor agreement is required BEFORE access.

## 5. Cross-border transfers

- Source data is fetched from US-based APIs (Alpaca, Polygon, SEC).
  GDPR Recital 41 / Schrems-II caveat: this is an inbound transfer
  to the EU, which is unrestricted; the inverse (EU → US) is what
  needs adequacy or SCCs.
- We do NOT send PII outbound to any non-EU recipient today.
- If we ever do (e.g. cloud-hosting outside the EU), Standard
  Contractual Clauses must be in place first.

## 6. Breach notification

- If PII is exfiltrated (e.g. via a `.env` leak that contains a
  privileged news-API token whose abuse could leak our PII tables),
  the operator notifies the supervisory authority within 72 hours
  (GDPR Art. 33).
- Records the breach in `output/ops/gdpr_breach_audit.jsonl`.

## 7. What this policy is NOT

- Not a substitute for legal counsel — if you go gewerblich, get a
  Datenschutzbeauftragter (DPO) involvement check (GDPR Art. 37).
- Not a complete data-protection impact assessment (DPIA) — DPIA
  is required if/when we cross the "systematic monitoring on a large
  scale" threshold (Art. 35).
- Not a substitute for a privacy notice — public-facing services
  need their own privacy policy.
