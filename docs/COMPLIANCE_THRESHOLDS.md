# Compliance Activation Thresholds

> Audit C4-095 — when does this private project become a regulated
> activity? This is the **single source of truth** for thresholds.
> Cross-references all other compliance docs.
>
> **Goal:** when you cross a threshold, you reach for this file
> FIRST, not Google. Each threshold names the regulator, the trigger,
> and the doc that activates.

## Quick map

```
                  PRIVATE TRADER (today)
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
   Eigengeschäft     >15h/week        Third-party data
   only             trading time      processing
                                            │
                                       Article-28 risk
                                            │
       │                  │                  │
       ▼                  ▼                  ▼
  ──── still
  private  ────►   GEWERBLICH       ────►  PROCESSOR
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
   Eigengeschäft     Client funds /     Public service
   for own UG       discretion          (Substack, paid)
                                            │
       │                  │                  │
       ▼                  ▼                  ▼
   UG-internal      INVESTMENT FIRM     PUBLIC FIRM
   only             (KWG §32)           (KWG §32 + KWG §2c)
                          │
                          ▼
                  RTS-6, RTS-28, MAR-STOR,
                  MiFID-II Art. 17, ESMA
                  algo-validation
```

## Threshold matrix

### T1 — `private` → `gewerblich`

| Sub-criterion | Trigger | Implication | Activates |
|---|---|---|---|
| Trading frequency | > 25 transactions/year (BMF guideline, case law) | trading is "nachhaltig" | Gewerbe-Anmeldung |
| Time commitment | trading is the main professional activity | dito | dito |
| Asset-class diversity | systematic trading of multiple uncorrelated instrument classes | "Vermögensverwaltung in Drittinteresse" hint | KWG §1 (1a) review |
| External funding | trading capital from non-personal source | strongly suggests commercial | dito |

When T1 fires:
- `docs/GOBD_WORM_POLICY.md` activates by operation of law (§ 257 HGB)
- `docs/AUDIT_LOG_RETENTION.md` 7y → confirm to ≥ 10y under HGB/AO
- `docs/RTS6_SELF_ASSESSMENT.md` becomes mandatory annual
- `docs/MAR_SURVEILLANCE_POLICY.md` written procedures required

### T2 — `gewerblich` → `investment firm` (KWG §32)

| Sub-criterion | Trigger | Implication |
|---|---|---|
| Discretionary management for others | yes/no | KWG §1 (1a) "Finanzportfolioverwaltung" → §32 lic |
| Holding client money | yes/no | KWG §1 (1a) "Eigenhandel als Dienstleistung" |
| Reception/transmission of orders | yes/no | KWG §1 (1a) "Anlagevermittlung" |
| Investment advice | yes/no | KWG §1 (1a) "Anlageberatung" |

When T2 fires:
- `docs/MIFID2_VENUE_REPORTS.md` (RTS-28) becomes mandatory annual
- `docs/RTS6_SELF_ASSESSMENT.md` becomes the live algo-validation doc
- BaFin license process (KWG §32) — typically 12-24 months
- Capital requirements (IFR / IFD)

### T3 — Public publication of results

| Sub-criterion | Trigger | Implication |
|---|---|---|
| Substack / blog publishing strategy results | first public post | `docs/RISK_DISCLOSURE_TEMPLATE.md` MUST appear on every post |
| Paid newsletter | first paying subscriber | possibly §1 (1a) "Anlageberatung" → §32 lic — check with counsel |
| Open-source release with backtest claims | first README boast | risk-disclosure in README, in repo |

### T4 — Processing third-party PII

| Sub-criterion | Trigger | Implication |
|---|---|---|
| User accounts | first user signs up | `docs/GDPR_PII_POLICY.md` formal DPO check if > 250 users (Art. 30) |
| API tokens issued | first non-operator API key | API-audit log retention extends to user-actions |
| Newsletter signups | first subscriber | retention + Art. 17 deletion machinery active |

### T5 — Capital scale

| Sub-criterion | Trigger | Implication |
|---|---|---|
| Own capital > 100k EUR | tax-relevance only | Steuerberater required |
| Own capital > 750k EUR | (private) → likely impacts Vermögensverwaltung classification | counsel review |
| Client capital (any) | T2 fires immediately | KWG §32 |

## What to do when ANY threshold fires

1. **STOP** the next scheduled change — do not push out new
   functionality the day you crossed a threshold.
2. **Read** the activated doc end-to-end. Each doc has a "What this
   policy is NOT" section that bounds your obligation.
3. **Date** the activation: append a line to this file's "Activation
   log" below.
4. **Engage counsel** for T1+T2+T3+T4. The first hour is the
   cheapest investment of the entire compliance journey.
5. **Update** `docs/RTS6_SELF_ASSESSMENT.md` §1 to reflect the new
   status.

## Activation log (append-only)

```
2026-05-12  none activated. system is purely Eigengeschäft, private trader.
```

## What this doc is NOT

- Not legal advice. Lists the *triggers* but not the *full
  obligations* — those live in the activated policy docs.
- Not exhaustive of German Steuer- / Wertpapierhandelsrecht; only
  the most operator-relevant thresholds appear here.
- Not a substitute for a Steuerberater (T5) or
  Aufsichtsrechtsanwalt (T1+T2).
