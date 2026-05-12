# Failure Mode and Effect Analysis (FMEA)

> Audit C2-009 — operational failure catalog for the Assembled-Trading-AI
> backend. Each entry has a Severity × Occurrence × Detection score; the
> product is the **RPN (Risk Priority Number)**. Higher = more attention.
>
> Scoring scale: 1 (negligible) → 10 (catastrophic). RPN ranges 1–1000.

This is a **living document**. Re-score after every drill / incident /
post-mortem.

| Scale | 1 | 5 | 10 |
|-------|---|---|----|
| Severity | cosmetic | 1-day outage | unrecoverable capital loss / regulatory breach |
| Occurrence | once/decade | once/year | weekly |
| Detection | guaranteed-instant | within hours | "we notice when clients call" |

---

## Top-12 by RPN (current snapshot)

| # | Failure mode | Trigger | Mitigation (current) | S | O | D | RPN | Open follow-ups |
|---|---|---|---|---|---|---|---|---|
| 1 | **Kill-switch state file corrupted at restart** | power loss / OS crash mid-write | atomic write tmp+rename+fsync; dir-fsync on POSIX. C4-014. | 9 | 2 | 3 | 54 | annual disaster-drill validates restore (C4-052) |
| 2 | **Audit log tampered post-hoc** | adversarial insider or process bug | hash-chained JSONL (prev_hash + sha256). verify_audit_chain() detects break. C4-016. | 8 | 1 | 2 | 16 | off-site replication (C4-016 long-tail) |
| 3 | **Broker disconnect during in-flight order** | network / API outage | tenacity-retry convention available; SUBMITTED-timeout detector marks stuck orders (C4-020); kill-switch escalation on >5 consecutive failures (planned). | 9 | 4 | 4 | 144 | wire automatic escalation in broker_adapter (C4-038) |
| 4 | **Reconciliation diff persists across 3 cycles** | broker/ledger drift not corrected | evaluate_reconcile_slo fires reconciliation_warn/fail; recon audit JSONL written; AlertManager cooldown rule. C4-031, C3-072. | 10 | 3 | 5 | 150 | wire 3-strikes-auto-killswitch (C4-031 long-tail) |
| 5 | **FastAPI multi-worker creates split-brain engine** | someone sets `--workers > 1` | docs note (C4-036) + paper engine is single-process singleton in api/routers/paper_trading.py. | 9 | 2 | 7 | 126 | enforce `--workers 1` in deploy script; doc note in OPERATIONS_BACKEND |
| 6 | **NTP clock drift > 100ms** | host VM time falls behind | not wired yet (C4-043) | 7 | 4 | 8 | 224 | chrony+monitoring required |
| 7 | **Disk fills up — silent audit/ledger write loss** | logs/parquet growth unmanaged | /ready now checks disk-quota >= 90% → 503. C4-040. | 8 | 3 | 4 | 96 | log rotation policy doc (C3-074) |
| 8 | **PIT violation in feature pipeline** | feature accidentally uses future data | property test test_pit_safety_* (E-005); PIT-guard audit log (data/pit_guard.py). | 10 | 3 | 5 | 150 | extend property test coverage to every feature module |
| 9 | **Secret committed to git history** | accidental `.env` add | detect-secrets + gitleaks pre-commit hooks; .gitignore /data/, /output/ anchored. .secrets.baseline checked in. | 10 | 1 | 3 | 30 | rotate keys quarterly even if no detected leak (C3-010) |
| 10 | **CPCV/Stacking leakage in ERWEITERUNG ML pipeline** | implementation bug | branch-isolated (not on main); documented in AUDIT_SWEEP §3.1. C4-001, C4-002. | 10 | 3 | 6 | 180 | fix before any cherry-pick to main (C3-026, C3-027) |
| 11 | **Promotion of overfit candidate strategy** | inadequate validation | quant_gates in policy.yaml block DSR<0, PSR<0.95, PBO>0.5. permutation_p_value < 0.01 required. correlation_promotion_gate. D-002/003/004, C2-016, C2-057. | 9 | 3 | 4 | 108 | wire all gates in batch_backtest exit path |
| 12 | **Rate-limit absent → DoS / cost runaway on data API** | misconfigured client / bug loop | rate_limit_middleware (per-IP, opt-in via env). C3-114. | 6 | 3 | 5 | 90 | enable in production deploy by default |

---

## How to use this document

1. **Before any architectural change**, scan rows where the changed
   subsystem appears in "Trigger" or "Mitigation". If RPN > 100, the
   change should include an explicit risk note in the PR description.

2. **After every drill / incident**:
   - Re-score the affected row.
   - Add a new row if the failure was not in the table.
   - File a follow-up task for any RPN > 150 with no scheduled mitigation.

3. **Quarterly review**: sort by RPN desc, attack the top-5 until each
   has a mitigation that reduces Detection to ≤ 3.

## What this document is NOT

- Not a regulatory artefact (yet — see RTS-6 self-assessment for that).
- Not a substitute for the existing per-subsystem runbooks.
- Not exhaustive — a true FMEA workshop would expand this to 50+ rows.
  The top-12 are the ones with current mitigations or open follow-ups.
