# MAR Surveillance Policy

> Audit C4-093 — Market Abuse Regulation (EU 596/2014) surveillance
> obligations apply to *any* person who places orders into a regulated
> market. The threshold for a formal written policy is `gewerblich`
> (commercial) operation, but the technical signals below are good
> hygiene immediately.

## 1. Scope

This policy covers detection and escalation of behaviour that could
constitute market abuse under MAR Art. 12, specifically:

- **Spoofing** (Art. 12(1)(a)): submitting orders with no intention of
  executing them, to mislead other participants.
- **Layering** (Art. 12(1)(a)): submitting multiple orders on one
  side of the book to create a false impression of supply/demand.
- **Wash trades** (Art. 12(1)(a)): buy + sell same instrument with no
  change in beneficial ownership.
- **Marking the close** (Art. 12(1)(a)): trading near close to move
  the official settlement price.
- **Insider dealing** (Art. 14): trading on non-public price-sensitive
  information.

## 2. What this codebase already does

| Control | Implementation | Audit ID |
|---|---|---|
| No insider data ingestion | We use ONLY public data — yfinance, Polygon, SEC EDGAR, FRED. CLAUDE.md, Abschnitt „MNPI", forbids MNPI-derived logic. | C4-090 |
| Order cancellation tracking | `execution/order_lifecycle.py` records every CREATED → CANCELLED transition. | C4-020 |
| Wash-trade detection (manual today) | Reconciler shows net qty change per symbol per day; a zero net + non-zero gross would surface. | C4-031 |
| News-source provenance | `events/news/dedupe.py` + `news_signal_bridge` always cite the source feed and timestamp. | — |

## 3. Concrete surveillance signals to monitor

These are the technical signatures of the listed abuse patterns; the
operator MUST review weekly (or on demand from the audit log):

| Signal | Threshold | Action |
|---|---|---|
| Order-to-trade ratio (OTR) per symbol per day | > 20 cancellations per 1 fill | manual review — could be spoofing if intentional |
| Same-day buy + sell same symbol same qty | any occurrence | manual review — wash trade signature |
| Cluster of opposite-side cancellations within 200 ms of an opposite-direction order | any occurrence | manual review — classic layering signature |
| Orders placed within 5 min of session close that move the print > 50 bps | > 1 per symbol per week | manual review — marking the close |
| Sudden directional trades immediately after fetching an `events/news/*` item | any "anomalously profitable" event | manual review — confirm the news was public for >5 min before any trade |

All of these are *manual* today — no automated surveillance pipeline.
A real system has tools like Eventus or NICE Actimize; for a solo
private trader the manual weekly review of order-blotter + news-ingest
audit is proportionate.

## 4. Escalation path

If a signal triggers:

1. Operator pauses the offending strategy via kill-switch.
2. Operator drafts a written explanation (template:
   `journal/<date>-MAR-review-<symbol>.md`) — what happened, why it
   is or isn't market abuse, what's changed in the system to prevent
   recurrence.
3. If the operator concludes the event WAS abuse: file a STOR
   (Suspicious Transaction and Order Report) with BaFin within 30
   days (MAR Art. 16). This applies even to own-account incidental
   abuse — accidental wash trades from a bug count.
4. The journal entry stays in git permanently as evidence of
   diligence.

## 5. Retention

- All order-blotter rows (`output/ops/orders_audit.jsonl` once wired)
  retained 7 years per `docs/AUDIT_LOG_RETENTION.md`.
- News-ingest audit retained 30 days (PII retention cap, see
  `docs/GDPR_PII_POLICY.md`).
- MAR-review journal entries retained 7 years.

## 6. Annual refresh

Once a year:
- Review the order-to-trade ratio per strategy.
- Review any anomalously profitable trade immediately after news.
- Document the refresh in `docs/RTS6_SELF_ASSESSMENT.md` §5.

## 7. What this policy is NOT

- Not a substitute for an Eventus / Actimize / NICE / Trillium
  monitoring stack — those become necessary if we ever take third-
  party flow.
- Not legal advice. STOR-filing is a regulatory obligation with case
  law; consult counsel before submitting.
- Not exhaustive: MAR Annex II lists many more patterns. The five
  above are the ones a long-only EOD strategy on regulated venues
  is most likely to accidentally produce.
