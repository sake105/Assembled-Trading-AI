# Source Trust Tiers — T0 through T3

**Date:** 2026-04-19  
**Status:** Active (T7.5)

---

## Tier Definitions

| Tier | Label | Criteria | Examples |
|------|-------|---------|---------|
| **T0** | Authoritative | Primary source, machine-readable, official disclosure | SEC EDGAR, Federal Register, official government portals |
| **T1** | Verified Wire | Established wire service, direct API, editorial standards | Reuters Connect, AP DataConnect, Bloomberg Terminal feed |
| **T2** | Secondary | Aggregators, regional outlets, curated databases | GDELT, NewsAPI aggregated, ACLED (event data) |
| **T3** | Social / Unverified | Social media, unvetted user-generated | Twitter/X, Reddit — excluded from confidence scoring |

---

## Tier Boost in Confidence Formula

The legacy confidence formula (until T2.8 flip):
```
confidence = min(1.0, n_events * 0.15 + tier_boost)
```

| Tier | Boost |
|------|-------|
| T0 | +0.40 |
| T1 | +0.30 |
| T2 | +0.15 |
| T3 | +0.00 |

---

## Admission Criteria

To add a new source to the registry (`configs/news/sources.yaml`):

1. **Tier assignment:** Confirm which tier based on criteria above
2. **PIT compliance:** Source must have a stable `published_at` timestamp; no backdating
3. **Reliability SLA:** Source must have ≥95% uptime over 30-day window before production use
4. **MNPI check:** Source must not provide material non-public information
5. **API key:** If key required, add to `.env.template` and `configs/secrets/README.md`

---

## Removal Criteria

A source is removed from the registry when:

- Uptime < 90% over 30-day rolling window (T6.3 SLA monitoring)
- Schema or API changes break parsing without a fix within 48h
- Confirmed MNPI risk identified
- T3 source shows repeated misinfo scores > 0.70 (policy: `evidence_engine.misinfo_risk_threshold`)

---

## Current Source Inventory

| Source | Tier | Status | Notes |
|--------|------|--------|-------|
| GDELT v2 | T2 | Active | Free, broad coverage, ~15min lag |
| SEC EDGAR Form 4 | T0 | Active | T+2 latency (Form-4 filing deadline) |
| House PTR | T0 | Active | Periodic disclosure; PDF parser pending (T5.3) |
| NewsAPI | T2 | Planned | Requires `NEWSAPI_KEY` (T5.1) |
| Reuters Connect | T1 | Planned | Requires contract + `REUTERS_API_KEY` |
| AP DataConnect | T1 | Planned | Requires contract + `AP_API_KEY` |
| ACLED | T2 | Stub | `fetch_acled.py` archived; reimplement when key available |

---

## MNPI Policy

All sources used by this system must provide **public information** only.

- SEC EDGAR: public filings, time-delayed (T+2 for Form 4)
- House PTR: public legislative disclosures, delayed publication
- News wires: public broadcast content

Trading signals derived from non-public information violate CLAUDE.md, Abschnitt „MNPI", and SEC Rule 10b-5.
